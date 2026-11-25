from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "data" / "diamond_knowledge.json"


class LLMServiceError(RuntimeError):
    pass


class JewelryKnowledgeAssistant:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = os.getenv("OPENAI_API_KEY", "") if api_key is None else api_key
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.documents = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(
            [
                f"{item['title']}. {' '.join(item.get('keywords', []))}. {item['text']}"
                for item in self.documents
            ]
        )

    def retrieve(self, question: str) -> list[dict[str, str]]:
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix)[0]
        ranked = scores.argsort()[::-1][:3]
        return [
            self.documents[index]
            for index in ranked
            if scores[index] >= 0.08
        ]

    def answer(self, question: str) -> dict[str, object]:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        documents = self.retrieve(question)
        if not documents:
            return {
                "answer": (
                    "I couldn't find relevant information in the diamond reference "
                    "notes. Try asking about carat, cut, color, clarity, valuation, "
                    "or unusual listings."
                ),
                "sources": [],
                "mode": "retrieval",
            }

        if not self.api_key:
            return {
                "answer": f"{documents[0]['title']}: {documents[0]['text']}",
                "sources": [item["source"] for item in documents],
                "mode": "retrieval",
            }

        return self._generate_answer(question, documents)

    def _generate_answer(
        self, question: str, documents: list[dict[str, str]]
    ) -> dict[str, object]:
        context = "\n\n".join(
            f"Source: {item['source']}\n{item['title']}: {item['text']}"
            for item in documents
        )
        request_body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a diamond jewelry advisor. Answer using only "
                            "the retrieved reference notes below. If they do not "
                            "contain the answer, say so. Do not invent grades, "
                            "prices, or citations. Explain that model valuations "
                            "are estimates, not certified appraisals."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Reference notes:\n{context}\n\nQuestion: {question}",
                    },
                ],
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read(500).decode("utf-8", errors="replace")
            raise LLMServiceError(
                f"LLM service returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise LLMServiceError(f"Could not reach the LLM service: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMServiceError("LLM service returned invalid JSON") from exc

        try:
            answer = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMServiceError(
                "LLM service response did not contain a chat answer"
            ) from exc
        if not isinstance(answer, str) or not answer.strip():
            raise LLMServiceError("LLM service returned an empty answer")

        return {
            "answer": answer.strip(),
            "sources": [item["source"] for item in documents],
            "mode": "custom_llm",
        }
