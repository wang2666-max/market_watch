from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Set

import nltk
import pandas as pd
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pymongo import MongoClient


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


REDDIT_CLIENT_ID = _required_env("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = _required_env("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = _required_env("REDDIT_USER_AGENT")
MONGO_URI = _required_env("MONGO_URI")

DB_NAME = "wallstreetbets"
COLLECTION_NAME = "wallstreetbets"
SUBREDDIT_NAME = "wallstreetbets"
POST_FETCH_LIMIT = 200
COMMENT_LIMIT = 10


# ---------------------------
# VADER init
# ---------------------------

try:
    vader = SentimentIntensityAnalyzer()
except LookupError:
    print("[info] vader_lexicon not found; downloading...")
    nltk.download("vader_lexicon")
    vader = SentimentIntensityAnalyzer()


# ---------------------------
# Fallback whitelist
# ---------------------------

FALLBACK_TICKERS: Set[str] = {
    "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META",
    "GOOGL", "GOOG", "AMD", "GME", "AMC",
    "SPY", "QQQ", "TLT"
}


def load_whitelist() -> Set[str]:
    return FALLBACK_TICKERS


def compile_whitelist_regex(symbols: Set[str]) -> re.Pattern:
    items = sorted((re.escape(x) for x in symbols), key=len, reverse=True)
    pattern = r"(?<![A-Za-z0-9])\$?(?:" + "|".join(items) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def get_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )


def get_mongo_client() -> MongoClient:
    return MongoClient(MONGO_URI)


def get_collection():
    mongo_client = get_mongo_client()
    collection = mongo_client[DB_NAME][COLLECTION_NAME]
    return mongo_client, collection


def fetch_and_store_reddit_posts(subreddit_name: str, post_limit: int, comment_limit: int):
    """
    Fetch hot posts + comments and store into MongoDB.
    """
    reddit = get_reddit_client()
    mongo_client, collection = get_collection()

    try:
        subreddit = reddit.subreddit(subreddit_name)

        for i, submission in enumerate(subreddit.hot(limit=None)):
            if i >= post_limit:
                break

            post_data = {
                "id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext,
                "url": submission.url,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "comments": [],
                "post_created_at": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                "sentiment": [],
            }

            submission.comments.replace_more(limit=0)
            for c in submission.comments.list()[:comment_limit]:
                post_data["comments"].append(
                    {
                        "id": c.id,
                        "body": c.body,
                        "score": getattr(c, "score", None),
                    }
                )

            collection.update_one(
                {"id": submission.id},
                {"$set": post_data},
                upsert=True,
            )
            print(f"✅ Stored Reddit post: {submission.title[:60]}...")

    finally:
        mongo_client.close()


def score_vader_finance(text: str) -> tuple[float, str]:
    scores = vader.polarity_scores(text)
    compound = scores["compound"]
    sentiment_score = round(compound * 10, 2)

    sentiment_type = "neutral"
    if compound > 0.05:
        sentiment_type = "bullish"
    elif compound < -0.05:
        sentiment_type = "bearish"

    return sentiment_score, sentiment_type


def run_vader_sentiment(subreddit_name: str, whitelist: Set[str]):
    """
    Reads stored posts from Mongo, extracts ticker mentions from whitelist,
    scores each text source with VADER, and pushes sentiment records back into Mongo.
    """
    mongo_client, collection = get_collection()

    try:
        regex = compile_whitelist_regex(whitelist)
        cursor = collection.find({})

        all_results = []

        for post in cursor:
            post_id = post["id"]

            text_sources = [
                f"{post.get('title', '')} {post.get('selftext', '')}"
            ] + [c.get("body", "") for c in post.get("comments", [])]

            sentiment_records = []

            for text in text_sources:
                if not text:
                    continue

                tickers = regex.findall(text)
                for raw in tickers:
                    ticker = raw.lstrip("$").upper()
                    score, s_type = score_vader_finance(text[:500])

                    record = {
                        "ticker": ticker,
                        "sentiment_score": score,
                        "sentiment_type": s_type,
                        "created_at": datetime.now(timezone.utc),
                    }

                    sentiment_records.append(record)
                    all_results.append(record)

            if sentiment_records:
                collection.update_one(
                    {"id": post_id},
                    {"$push": {"sentiment": {"$each": sentiment_records}}},
                )

        if not all_results:
            print("⚠️ No tickers found.")
            return

        df = pd.DataFrame(all_results)
        summary = (
            df.groupby("ticker")
            .agg(
                total_mentions=("sentiment_score", "count"),
                avg_score=("sentiment_score", "mean"),
                max_score=("sentiment_score", "max"),
                min_score=("sentiment_score", "min"),
            )
            .reset_index()
        )

        summary["avg_score"] = summary["avg_score"].round(2)
        summary = summary.sort_values("total_mentions", ascending=False)

        print("\n📈 ---- VADER Summary ----")
        print(summary.to_string(index=False))
        print("\n🔎 Sample saved records:")
        print(df.head(10).to_string(index=False))

    finally:
        mongo_client.close()


if __name__ == "__main__":
    fetch_and_store_reddit_posts(SUBREDDIT_NAME, POST_FETCH_LIMIT, COMMENT_LIMIT)
    whitelist = load_whitelist()
    run_vader_sentiment(SUBREDDIT_NAME, whitelist)
    print("✅ Finished sentiment analysis using VADER ✅")