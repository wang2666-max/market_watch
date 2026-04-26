"""
ApeWisdom crowd-attention signal scanner.
Computes momentum, rank jumps, re-entries, and persistent leaders
from historical daily ranking files + the rolling ticker_state.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REDDIT_DIR = Path("data") / "reddit"


# ---Data Loaders ────────────────────────────────────────────────────────────

def load_daily_rankings(n_days: int = 7, today: date | None = None) -> dict[date, pd.DataFrame]:
    """Return last n_days of apewisdom_top100.csv keyed by date."""
    today = today or date.today()
    result: dict[date, pd.DataFrame] = {}
    for folder in sorted(REDDIT_DIR.glob("????-??-??")):
        try:
            d = date.fromisoformat(folder.name)
        except ValueError:
            continue
        if d > today:
            continue
        csv_path = folder / "apewisdom_top100.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
            df = df.dropna(subset=["rank"]).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
            result[d] = df
        except Exception:
            continue
    sorted_dates = sorted(result.keys())
    keep = sorted_dates[-n_days:]
    return {d: result[d] for d in keep}


def load_today_raw(today: date | None = None) -> list[dict]:
    """Load today's apewisdom_raw.json results list."""
    today = today or date.today()
    raw_path = REDDIT_DIR / today.isoformat() / "apewisdom_raw.json"
    if not raw_path.exists():
        return []
    try:
        with raw_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        for r in results:
            r["ticker"] = str(r.get("ticker", "")).upper().strip()
        return results
    except Exception:
        return []


# ---Signal Computations ─────────────────────────────────────────────────────

def compute_first_appearances(state_df: pd.DataFrame, today: date, top_n: int = 10) -> list[dict]:
    """Tickers seeing their first top-100 appearance today."""
    today_ts = pd.Timestamp(today)
    df = state_df[
        (pd.to_datetime(state_df["first_seen"], errors="coerce") == today_ts) &
        (state_df["current_rank"].notna())
    ].sort_values("current_rank")
    return [{"ticker": r["ticker"], "rank": int(r["current_rank"])} for _, r in df.head(top_n).iterrows()]


def compute_rank_jumpers(raw_results: list[dict], top_n: int = 10) -> list[dict]:
    """1-day rank improvers using rank_24h_ago from the API payload."""
    jumpers = []
    for r in raw_results:
        ticker = r.get("ticker", "")
        try:
            today_rank = int(r["rank"])
            prev_rank = int(r["rank_24h_ago"])
        except (KeyError, TypeError, ValueError):
            continue
        jump = prev_rank - today_rank
        if jump > 0:
            jumpers.append({"ticker": ticker, "prev": prev_rank, "today": today_rank, "jump": jump})
    return sorted(jumpers, key=lambda x: -x["jump"])[:top_n]


def compute_3day_momentum(history: dict[date, pd.DataFrame], top_n: int = 10) -> list[dict]:
    """Best rank improvement over the last 3 available days."""
    dates = sorted(history.keys())[-3:]
    if len(dates) < 2:
        return []

    rank_by_date: dict[date, dict[str, int]] = {
        d: {row["ticker"]: int(row["rank"]) for _, row in df.iterrows() if pd.notna(row["rank"])}
        for d, df in {d: history[d] for d in dates}.items()
    }

    today_date = dates[-1]
    movers = []
    for ticker, today_rank in rank_by_date[today_date].items():
        trail = [rank_by_date[d].get(ticker) for d in dates]
        prior = [r for r in trail[:-1] if r is not None]
        if not prior:
            continue
        score = prior[0] - today_rank
        if score <= 0:
            continue
        movers.append({"ticker": ticker, "trail": trail, "dates": dates, "score": score, "today_rank": today_rank})
    return sorted(movers, key=lambda x: -x["score"])[:top_n]


def compute_persistent_leaders(history: dict[date, pd.DataFrame], window: int = 7, top_n: int = 10) -> list[dict]:
    """Names dominating top10/top25 discussion repeatedly over the window."""
    dates = sorted(history.keys())[-window:]
    if not dates:
        return []

    # Precompute rank lookups
    rank_lk: dict[date, dict[str, int]] = {
        d: {row["ticker"]: int(row["rank"]) for _, row in df.iterrows() if pd.notna(row["rank"])}
        for d, df in {d: history[d] for d in dates}.items()
    }

    today_date = dates[-1]
    all_tickers = {t for lk in rank_lk.values() for t in lk}

    result = []
    for ticker in all_tickers:
        today_rank = rank_lk[today_date].get(ticker, 999)
        if today_rank > 25:
            continue

        top10_days = sum(1 for d in dates if rank_lk[d].get(ticker, 999) <= 10)

        streak_10 = 0
        for d in sorted(dates, reverse=True):
            if rank_lk[d].get(ticker, 999) <= 10:
                streak_10 += 1
            else:
                break

        streak_25 = 0
        for d in sorted(dates, reverse=True):
            if rank_lk[d].get(ticker, 999) <= 25:
                streak_25 += 1
            else:
                break

        if top10_days >= 2 or streak_25 >= 3:
            result.append({
                "ticker": ticker,
                "today_rank": today_rank,
                "top10_streak": streak_10,
                "top25_streak": streak_25,
                "top10_days": top10_days,
            })

    return sorted(result, key=lambda x: (-x["top10_streak"], -x["top25_streak"], -x["top10_days"]))[:top_n]


def compute_reentries(history: dict[date, pd.DataFrame], today: date, min_absent_days: int = 7, top_n: int = 10) -> list[dict]:
    """Tickers back in top100 after ≥ min_absent_days absence."""
    dates = sorted(history.keys())
    if not dates or dates[-1] != today:
        return []

    today_df = history[today]
    today_rank_map = {
        row["ticker"]: int(row["rank"])
        for _, row in today_df.iterrows()
        if pd.notna(row["rank"])
    }
    prev_dates = sorted(dates[:-1], reverse=True)

    # Build last-seen-before-today per ticker
    last_seen_before: dict[str, date] = {}
    for t in today_rank_map:
        for d in prev_dates:
            if t in {row["ticker"] for _, row in history[d].iterrows()}:
                last_seen_before[t] = d
                break

    reentries = []
    for ticker, today_rank in today_rank_map.items():
        if ticker not in last_seen_before:
            continue  # first-ever appearance, handled by first_appearances
        absent_days = (today - last_seen_before[ticker]).days
        if absent_days >= min_absent_days:
            reentries.append({"ticker": ticker, "absent_days": absent_days, "today_rank": today_rank})
    return sorted(reentries, key=lambda x: x["today_rank"])[:top_n]


def compute_mention_surges(raw_results: list[dict], top_n: int = 10) -> list[dict]:
    """Tickers with biggest absolute mention increase vs 24h ago."""
    surges = []
    for r in raw_results:
        ticker = r.get("ticker", "")
        try:
            mentions = int(r["mentions"])
            prev = int(r["mentions_24h_ago"])
        except (KeyError, TypeError, ValueError):
            continue
        if prev <= 0:
            continue
        change = mentions - prev
        if change > 0:
            surges.append({"ticker": ticker, "prev": prev, "today": mentions, "change": change})
    return sorted(surges, key=lambda x: -x["change"])[:top_n]


# ---State Enrichment ────────────────────────────────────────────────────────

def enrich_state(
    state_df: pd.DataFrame,
    raw_results: list[dict],
    history: dict[date, pd.DataFrame],
    today: date,
) -> pd.DataFrame:
    """Append signal columns to state_df. Columns are recomputed each run."""
    df = state_df.copy()

    # prev_rank + rank_change_1d from API
    rank_24h_map = {}
    for r in raw_results:
        t = r.get("ticker", "")
        try:
            rank_24h_map[t] = int(r["rank_24h_ago"])
        except (KeyError, TypeError, ValueError):
            pass
    df["prev_rank"] = df["ticker"].map(rank_24h_map)
    df["rank_change_1d"] = df.apply(
        lambda row: int(row["prev_rank"] - row["current_rank"])
        if pd.notna(row.get("prev_rank")) and pd.notna(row.get("current_rank"))
        else pd.NA,
        axis=1,
    )

    # is_new_today
    today_ts = pd.Timestamp(today)
    df["is_new_today"] = (
        (pd.to_datetime(df["first_seen"], errors="coerce") == today_ts) &
        df["current_rank"].notna()
    )

    # Reentry columns
    dates = sorted(history.keys())
    today_tickers = set()
    if dates and dates[-1] == today:
        today_tickers = {row["ticker"] for _, row in history[today].iterrows()}
    prev_seen: dict[str, date] = {}
    for t in today_tickers:
        first_seen_val = df.loc[df["ticker"] == t, "first_seen"]
        if not first_seen_val.empty:
            fs = pd.to_datetime(first_seen_val.iloc[0], errors="coerce")
            if pd.notna(fs) and fs.date() == today:
                continue  # new today, not reentry
        for d in sorted([x for x in dates if x < today], reverse=True):
            if t in {row["ticker"] for _, row in history[d].iterrows()}:
                prev_seen[t] = d
                break

    days_absent_map = {
        t: (today - d).days
        for t, d in prev_seen.items()
        if (today - d).days >= 7
    }
    df["is_reentry_today"] = df["ticker"].map(lambda t: t in days_absent_map)
    df["days_absent_before_return"] = df["ticker"].map(days_absent_map).fillna(0).astype(int)

    # Streak + frequency columns from 7-day history window
    window_dates = sorted(history.keys())[-7:]
    rank_lk: dict[date, dict[str, int]] = {
        d: {row["ticker"]: int(row["rank"]) for _, row in history[d].iterrows() if pd.notna(row["rank"])}
        for d in window_dates
    }
    all_tickers = set(df["ticker"].tolist())

    streak_10_map: dict[str, int] = {}
    streak_25_map: dict[str, int] = {}
    seen_7d_map: dict[str, int] = {}

    for t in all_tickers:
        seen_7d_map[t] = sum(1 for d in window_dates if t in rank_lk.get(d, {}))

        s10 = 0
        for d in sorted(window_dates, reverse=True):
            if rank_lk.get(d, {}).get(t, 999) <= 10:
                s10 += 1
            else:
                break
        streak_10_map[t] = s10

        s25 = 0
        for d in sorted(window_dates, reverse=True):
            if rank_lk.get(d, {}).get(t, 999) <= 25:
                s25 += 1
            else:
                break
        streak_25_map[t] = s25

    df["top10_streak"] = df["ticker"].map(streak_10_map).fillna(0).astype(int)
    df["top25_streak"] = df["ticker"].map(streak_25_map).fillna(0).astype(int)
    df["times_seen_last_7d"] = df["ticker"].map(seen_7d_map).fillna(0).astype(int)

    return df


# ---Report Renderer ─────────────────────────────────────────────────────────

def _fmt_trail(trail: list) -> str:
    return " -> ".join(str(r) if r is not None else "unranked" for r in trail)


def render_signal_report(
    today: date,
    ts: str,
    first_appearances: list[dict],
    rank_jumpers: list[dict],
    momentum_3d: list[dict],
    persistent_leaders: list[dict],
    reentries: list[dict],
    mention_surges: list[dict],
    admin: dict,
) -> str:
    lines: list[str] = []

    lines += [f"APEWISDOM SIGNAL REPORT -{ts}", "", f"date: {today.isoformat()}", ""]

    lines.append("---FIRST APPEARANCE ALERTS ---")
    if first_appearances:
        for item in first_appearances:
            lines.append(f"  {item['ticker']}  #{item['rank']}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("---FASTEST MOVERS (1D) ---")
    if rank_jumpers:
        for item in rank_jumpers:
            lines.append(f"  {item['ticker']}  #{item['prev']} -> #{item['today']}  (+{item['jump']})")
    else:
        lines.append("  (insufficient data)")
    lines.append("")

    lines.append("---3-DAY MOMENTUM ---")
    if momentum_3d:
        for item in momentum_3d:
            lines.append(f"  {item['ticker']}  {_fmt_trail(item['trail'])}  (score +{item['score']})")
    else:
        lines.append("  (insufficient data)")
    lines.append("")

    lines.append("---PERSISTENT LEADERS ---")
    if persistent_leaders:
        for item in persistent_leaders:
            parts = []
            if item["top10_streak"] >= 2:
                parts.append(f"top10 streak: {item['top10_streak']}")
            if item["top25_streak"] >= 3:
                parts.append(f"top25 streak: {item['top25_streak']}")
            parts.append(f"top10 in {item['top10_days']}d of window")
            lines.append(f"  {item['ticker']}  #{item['today_rank']}  - {',  '.join(parts)}")
    else:
        lines.append("  (insufficient data)")
    lines.append("")

    lines.append("---RE-ENTRY ALERTS ---")
    if reentries:
        for item in reentries:
            lines.append(f"  {item['ticker']}  back after {item['absent_days']}d  (#{item['today_rank']})")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("---MENTION SURGES ---")
    if mention_surges:
        for item in mention_surges:
            lines.append(f"  {item['ticker']}  {item['prev']} -> {item['today']}  (+{item['change']})")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("---ADMIN ---")
    lines.append(f"  eligible_hot_today:        {admin.get('eligible_hot_today', 'N/A')}")
    lines.append(f"  eligible_candidates_today: {admin.get('eligible_candidates_today', 'N/A')}")
    lines.append(f"  active_hot:                {admin.get('active_hot', 'N/A')}")
    lines.append(f"  active_candidates:         {admin.get('active_candidates', 'N/A')}")
    lines.append("")

    lines.append(f"generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return "\n".join(lines)
