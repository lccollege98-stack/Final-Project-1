from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import DiamondFeatureEngineer, clean_diamonds


DATA_PATH = ROOT / "data" / "diamonds.csv"
REPORT_DIR = ROOT / "reports"


def build_report() -> None:
    raw = pd.read_csv(DATA_PATH)
    df = clean_diamonds(raw)
    engineered = DiamondFeatureEngineer().fit_transform(df)

    missing_before = int(raw.isna().sum().sum())
    duplicate_count = int(raw.duplicated().sum())
    invalid_dimensions = int(
        ((raw[["x", "y", "z"]] <= 0).any(axis=1)).sum()
        if set(["x", "y", "z"]).issubset(raw.columns)
        else 0
    )

    parts = [
        "<h1>Diamond Dataset - EDA Report</h1>",
        f"<p><strong>Raw rows:</strong> {len(raw):,}</p>",
        f"<p><strong>Cleaned rows:</strong> {len(df):,}</p>",
        f"<p><strong>Missing values in raw data:</strong> {missing_before:,}</p>",
        f"<p><strong>Duplicate rows removed:</strong> {duplicate_count:,}</p>",
        f"<p><strong>Rows with zero or negative dimensions:</strong> {invalid_dimensions:,}</p>",
        "<h2>Numeric Summary</h2>",
        df.describe(include="number").round(2).to_html(),
        "<h2>Price Distribution</h2>",
        px.histogram(df, x="price", nbins=60).to_html(full_html=False),
        "<h2>Carat vs Price</h2>",
        px.scatter(
            df.sample(min(5000, len(df)), random_state=42),
            x="carat",
            y="price",
            color="cut",
            opacity=0.7,
        ).to_html(full_html=False),
        "<h2>Cut Quality vs Price</h2>",
        px.box(df, x="cut", y="price").to_html(full_html=False),
        "<h2>Correlation Heatmap</h2>",
        px.imshow(
            engineered[
                ["price", "carat", "depth", "table", "x", "y", "z", "volume", "premium_score"]
            ].corr(),
            text_auto=".2f",
        ).to_html(full_html=False),
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "diamond_eda.html").write_text(
        "\n".join(parts),
        encoding="utf-8",
    )
    (REPORT_DIR / "diamond_eda_summary.md").write_text(
        "\n".join(
            [
                "# Diamond Dataset - EDA Summary",
                "",
                f"- Raw rows: {len(raw):,}",
                f"- Cleaned rows: {len(df):,}",
                f"- Missing values in raw data: {missing_before:,}",
                f"- Duplicate rows removed: {duplicate_count:,}",
                f"- Rows with zero or negative dimensions: {invalid_dimensions:,}",
                f"- Median price: ${df['price'].median():,.2f}",
                f"- Median carat: {df['carat'].median():.2f}",
                "",
                "The HTML report contains price, carat, cut, and correlation visualizations.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Created {REPORT_DIR / 'diamond_eda.html'}")
    print(f"Created {REPORT_DIR / 'diamond_eda_summary.md'}")


if __name__ == "__main__":
    build_report()
