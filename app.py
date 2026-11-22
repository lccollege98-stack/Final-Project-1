from __future__ import annotations

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_pipeline import DiamondFeatureEngineer, clean_diamonds
from src.rag import JewelryKnowledgeAssistant, LLMServiceError


DATA_PATH = "data/diamonds.csv"
MODEL_PATH = "models/diamond_model.joblib"

st.set_page_config(page_title="Diamond Intelligence", layout="wide")
st.title("AI-Powered Diamond Valuation & Jewelry Intelligence")
st.caption("Decision support for diamond retailers and buyers - not a replacement for a certified appraisal.")


@st.cache_data
def load_data():
    return clean_diamonds(pd.read_csv(DATA_PATH))


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_assistant():
    return JewelryKnowledgeAssistant()


try:
    df = load_data()
    model = load_model()
    assistant = load_assistant()
except Exception as exc:
    st.error(f"Project artifacts are not ready: {exc}")
    st.info("Run `python src/train.py` after placing the dataset in `data/diamonds.csv`.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Market Dashboard", "Valuation", "Recommendations", "Jewelry Advisor"]
)

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Diamonds", f"{len(df):,}")
    col2.metric("Median Price", f"${df['price'].median():,.0f}")
    col3.metric("Median Carat", f"{df['carat'].median():.2f}")
    col4.metric("Average Premium Score", f"{DiamondFeatureEngineer().fit_transform(df)['premium_score'].mean():.1f}")

    left, right = st.columns(2)
    left.plotly_chart(px.histogram(df, x="price", nbins=60, title="Price Distribution"), use_container_width=True)
    right.plotly_chart(px.scatter(df.sample(min(5000, len(df)), random_state=42), x="carat", y="price", color="cut", title="Carat vs Price"), use_container_width=True)

    engineered = DiamondFeatureEngineer().fit_transform(df)
    heatmap_columns = ["price", "carat", "depth", "table", "x", "y", "z", "volume", "premium_score"]
    st.plotly_chart(
        px.imshow(engineered[heatmap_columns].corr(), text_auto=".2f", title="Feature Correlations"),
        use_container_width=True,
    )

with tab2:
    c1, c2, c3 = st.columns(3)
    carat = c1.number_input("Carat", 0.1, 10.0, 1.0, 0.01)
    cut = c1.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = c2.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = c2.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
    depth = c3.number_input("Depth %", 40.0, 80.0, 61.5, 0.1)
    table = c3.number_input("Table %", 40.0, 90.0, 57.0, 0.1)
    x = c1.number_input("Length x", 0.1, 20.0, 6.5, 0.1)
    y = c2.number_input("Width y", 0.1, 20.0, 6.5, 0.1)
    z = c3.number_input("Depth z", 0.1, 20.0, 4.0, 0.1)
    listed_price = st.number_input("Listed price (optional)", 0.0, 200000.0, 0.0, 100.0)

    if st.button("Estimate fair value", type="primary"):
        input_df = pd.DataFrame([{
            "carat": carat, "cut": cut, "color": color, "clarity": clarity,
            "depth": depth, "table": table, "x": x, "y": y, "z": z,
        }])
        estimate = float(model.predict(input_df)[0])
        st.metric("Estimated Market Value", f"${estimate:,.2f}")
        if listed_price > 0:
            difference = (listed_price - estimate) / estimate * 100
            if difference > 15:
                st.warning(f"Likely overpriced by {difference:.1f}%")
            elif difference < -15:
                st.success(f"Likely underpriced by {abs(difference):.1f}%")
            else:
                st.info(f"Listed price is within approximately 15% of fair value ({difference:+.1f}%).")

with tab3:
    budget = st.number_input("Maximum budget (USD)", 500.0, 200000.0, 5000.0, 500.0)
    if st.button("Find best-value diamonds"):
        candidates = df[df["price"] <= budget].copy()
        if candidates.empty:
            st.warning("No diamonds found within this budget.")
        else:
            candidate_features = candidates.drop(columns=["price"])
            candidates["estimated_price"] = model.predict(candidate_features)
            candidates["value_gap"] = candidates["estimated_price"] - candidates["price"]
            candidates = candidates.sort_values("value_gap", ascending=False).head(20)
            st.dataframe(
                candidates[["carat", "cut", "color", "clarity", "price", "estimated_price", "value_gap"]],
                use_container_width=True,
            )

with tab4:
    question = st.text_input("Ask a diamond question", "What does VS1 clarity mean?")
    if st.button("Ask advisor"):
        try:
            result = assistant.answer(question)
        except LLMServiceError as exc:
            st.error(f"Custom LLM request failed: {exc}")
        else:
            st.write(result["answer"])
            if result["mode"] == "retrieval":
                st.info(
                    "Showing the retrieved reference note. Set OPENAI_API_KEY to "
                    "enable generated answers."
                )
            st.caption("Sources: " + "; ".join(result["sources"]))
