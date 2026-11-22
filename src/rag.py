from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KNOWLEDGE = [
    ("Carat", "Carat measures diamond weight. One carat equals 0.200 grams. Higher carat generally increases price, especially for otherwise comparable stones."),
    ("Cut", "Cut describes how skillfully a diamond's proportions and finish interact with light. Common grades are Fair, Good, Very Good, Premium, and Ideal."),
    ("Color", "Diamond color grades range from D to Z. D through F is colorless, G through J is near-colorless, and lower grades show progressively more yellow or brown tint."),
    ("Clarity", "Clarity describes internal inclusions and external blemishes. A common scale is I1, SI2, SI1, VS2, VS1, VVS2, VVS1, and IF."),
    ("Fair value", "Fair value is estimated from comparable diamond attributes using a trained pricing model. Compare the listed price with the estimate and understand that model estimates are not appraisals."),
    ("Anomalies", "An unusually low or high listing may indicate an exceptional stone, a data-quality issue, or a suspicious price. Always verify dimensions and certificate details."),
]


class JewelryKnowledgeAssistant:
    def __init__(self):
        texts = [f"{title}. {body}" for title, body in KNOWLEDGE]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    def answer(self, question: str) -> str:
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix)[0]
        index = int(scores.argmax())
        title, body = KNOWLEDGE[index]
        if scores[index] < 0.08:
            return "I do not have enough retrieved guidance for that question. Ask about carat, cut, color, clarity, fair value, or anomalies."
        return f"{title}: {body}"
