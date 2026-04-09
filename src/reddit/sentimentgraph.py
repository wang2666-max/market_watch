from __future__ import annotations

import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymongo import MongoClient

from src.utility.paths import get_img_dir


# ---------------------------
# Env / config
# ---------------------------

try:
    from src.common.env import load_env
except ImportError:
    # Fallback if run directly from weird paths
    def load_env():
        return None


load_env()


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


MONGO_URI = _required_env("MONGO_URI")

DB_NAME = "wallstreetbets"
COLLECTION_NAME = "wallstreetbets"
SAVE_FOLDER = get_img_dir()


def get_mongo_client() -> MongoClient:
    return MongoClient(MONGO_URI)


def fetch_df() -> pd.DataFrame:
    mongo_client = get_mongo_client()

    try:
        col = mongo_client[DB_NAME][COLLECTION_NAME]

        rows = []
        for post in col.find({}):
            post_created_at = post.get("post_created_at")
            if not post_created_at:
                continue

            for s in post.get("sentiment", []):
                ticker = s.get("ticker")
                sentiment_score = s.get("sentiment_score")

                if ticker is None or sentiment_score is None:
                    continue

                rows.append(
                    {
                        "ticker": ticker,
                        "sentiment_score": sentiment_score,
                        "post_created_at": pd.to_datetime(post_created_at),
                    }
                )

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        df = df.sort_values("post_created_at")
        df["time"] = df["post_created_at"].dt.floor("h")
        return df

    finally:
        mongo_client.close()


def plot_sentiment_heatmap(df: pd.DataFrame):
    if df is None or df.empty:
        print("⚠️ No data to plot!")
        return

    pivot = df.pivot_table(
        values="sentiment_score",
        index="ticker",
        columns="time",
        aggfunc="mean",
    )

    if pivot.empty:
        print("⚠️ Pivot table is empty. Nothing to plot.")
        return

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        pivot,
        cmap="coolwarm",
        center=0,
        annot=False,
    )

    plt.title("Ticker Sentiment Over Time (VADER)")
    plt.xlabel("Time")
    plt.ylabel("Ticker")
    plt.tight_layout()

    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = SAVE_FOLDER / f"sentiment_heatmap_{ts}.png"

    plt.savefig(filename)
    plt.close()

    print(f"✅ Saved heatmap to: {filename}")


if __name__ == "__main__":
    df = fetch_df()
    print(df.head())

    if df.empty:
        print("⚠️ No data to plot!")
    else:
        plot_sentiment_heatmap(df)