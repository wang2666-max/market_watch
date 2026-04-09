# jobs/premarket.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so "import src...." works when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from datetime import datetime
import traceback
from typing import List, Optional, Tuple

from src.common.env import load_env
from src.utility.paths import get_img_dir, get_txt_dir
from src.utility.constant import SECTOR_TICKER_NAME_MAP


# ---------------------------
# Helpers: output + artifacts
# ---------------------------

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def _reports_dir() -> Path:
    p = Path("data") / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _graphs_dir_candidates() -> List[Path]:
    # common places we’ve used across branches
    return [
        Path("data") / "reports" / "graphs",
        Path("data") / "reports" / "output_graphs",
        Path("output_graphs"),
    ]


def _write_text_report(filename: str, content: str) -> Path:
    out_dir = get_txt_dir()
    out = out_dir / filename
    out.write_text(content, encoding="utf-8")
    return out


def _find_latest_image() -> Optional[Path]:
    exts = (".png", ".jpg", ".jpeg")
    best: Optional[Path] = None
    best_mtime = -1.0

    d = get_img_dir()
    if d.exists():
        for p in d.glob("*"):
            if p.suffix.lower() not in exts:
                continue
            try:
                m = p.stat().st_mtime
                if m > best_mtime:
                    best_mtime = m
                    best = p
            except Exception:
                continue

    if best is None:
        for d in _graphs_dir_candidates():
            if not d.exists():
                continue
            for p in d.glob("*"):
                if p.suffix.lower() not in exts:
                    continue
                try:
                    m = p.stat().st_mtime
                    if m > best_mtime:
                        best_mtime = m
                        best = p
                except Exception:
                    continue
    return best


def _append_text(textlist: List[str], p: Path):
    # processor expects paths (per your design); store as strings
    textlist.append(str(p))


def _append_img(jpblist: List[str], p: Path):
    jpblist.append(str(p))


# ---------------------------
# Pipeline steps
# ---------------------------

@dataclass
class EquityResult:
    summary_df: object  # pandas DataFrame, but keep loose typing
    vol_spike: List[str]
    recent_abn: List[str]


def pipeline_equity(textlist: List[str], jpblist: List[str], market_date) -> EquityResult:
    """
    Runs:
      - update universe prices
      - generate equity abnormal-vol report
    Appends a compact text artifact to textlist.
    """
    from src.prices.polygon_client import save_universe_excel
    from src.prices.smf_process import generate_reports, generate_macro

    print("\n=== [Equity] Updating prices ===")
    updates = save_universe_excel(market_date=market_date)

    # Print updates compactly
    if isinstance(updates, dict):
        changed = 0
        for t, info in updates.items():
            mode = info.get("mode", "?")
            rows = info.get("rows", 0)
            if mode != "noop" or rows:
                print(f"{t}: {mode} (+{rows})")
                changed += 1
        if changed == 0:
            print("(no changes)")
    else:
        print(updates)

    print("\n=== [Equity] Generating macro summary ===")
    macro_df = generate_macro(asof_date=market_date)
    macro_df['name'] = macro_df['symbol'].map(SECTOR_TICKER_NAME_MAP).fillna('')

    print("\n=== [Equity] Generating report ===")
    summary_df, vol_spike, recent_abn = generate_reports(asof_date=market_date)

    # Build a compact text block for the processor
    ts = _now_tag()
    lines = []
    lines.append(f"EQUITY FLAGS — {ts}")
    lines.append("")
    if macro_df is not None and not macro_df.empty:
        lines.append("SECTOR OVERVIEW:")
        if "symbol" in macro_df.columns and "ret_d" in macro_df.columns:
            lines.append(macro_df[["symbol", "name", "ret_d"]].to_string(index=False))
        else:
            lines.append(macro_df.to_string(index=False))
        lines.append("")
    lines.append("VolSpikeFlags: " + (", ".join(vol_spike) if vol_spike else "(none)"))
    lines.append("RecentAbnormalFlags: " + (", ".join(recent_abn) if recent_abn else "(none)"))

    # Optional mini snapshot (safe if columns exist)
    try:
        if summary_df is not None and not summary_df.empty:
            cols = [c for c in ["symbol", "ret_d", "ret_w", "ret_m", "ratio_ewma_vs_21", "ratio_ewma_vs_spy", "z_last_21"]
                    if c in summary_df.columns]
            if cols and "ratio_ewma_vs_21" in summary_df.columns:
                top = summary_df.sort_values("ratio_ewma_vs_21", ascending=False).head(10)
                lines.append("\nTop 10 by ratio_ewma_vs_21:")
                lines.append(top[cols].to_string(index=False))
    except Exception:
        # don’t break the job for formatting
        pass

    p = _write_text_report(f"equity_flags_{ts}.txt", "\n".join(lines))
    _append_text(textlist, p)

    return EquityResult(summary_df=summary_df, vol_spike=vol_spike, recent_abn=recent_abn)


def pipeline_reddit_mongo() -> None:
    """
    Runs your friend's reddit->mongo ingestion + VADER sentiment push.
    No artifacts here (those come from the next step that reads Mongo).
    """
    from src.reddit.reddittomongo import (
        SUBREDDIT_NAME,
        POST_FETCH_LIMIT,
        COMMENT_LIMIT,
        load_whitelist,
        fetch_and_store_reddit_posts,
        run_vader_sentiment,
    )

    whitelist = load_whitelist()

    print("\n=== [Reddit] Ingest to Mongo ===")
    fetch_and_store_reddit_posts(SUBREDDIT_NAME, POST_FETCH_LIMIT, COMMENT_LIMIT)

    print("\n=== [Reddit] VADER sentiment ===")
    run_vader_sentiment(SUBREDDIT_NAME, whitelist)


def pipeline_reddit_graph_and_summary(textlist: List[str], jpblist: List[str]):
    """
    Reads Mongo sentiment records, generates a heatmap image, and appends:
      - a text snapshot file to textlist
      - the newest heatmap image to jpblist (if found)
    """
    from src.reddit.sentimentgraph import fetch_df, plot_sentiment_heatmap

    print("\n=== [Reddit] Fetch sentiment DF ===")
    df = fetch_df()
    n = len(df)
    uniq = df["ticker"].nunique() if (n and "ticker" in df.columns) else 0
    print(f"[Reddit] sentiment rows={n}, unique_tickers={uniq}")

    ts = _now_tag()

    if df is None or df.empty:
        p = _write_text_report(f"reddit_snapshot_{ts}.txt", "(no reddit sentiment records available)")
        _append_text(textlist, p)
        return

    # Write a compact “top mentions” snapshot
    try:
        grp = (df.groupby("ticker")
               .agg(mentions=("sentiment_score", "count"),
                    avg_sentiment=("sentiment_score", "mean"))
               .reset_index())
        grp["avg_sentiment"] = grp["avg_sentiment"].round(2)
        grp = grp.sort_values(["mentions", "avg_sentiment"], ascending=[False, False]).head(20)
        snap = grp.to_string(index=False)
    except Exception:
        snap = df.head(50).to_string(index=False)

    p = _write_text_report(f"reddit_snapshot_{ts}.txt", snap)
    _append_text(textlist, p)

    print("\n=== [Reddit] Plot heatmap ===")
    plot_sentiment_heatmap(df)

    # Try to locate latest image (sentimentgraph saves it)
    img = _find_latest_image()
    if img:
        print(f"[Reddit] found heatmap image: {img}")
        _append_img(jpblist, img)
    else:
        print("[Reddit] no heatmap image found to attach.")


# ---------------------------
# Final processor + email
# ---------------------------

def process_and_send(textlist: List[str], jpblist: List[str], send_email: bool = True):
    """
    Calls processor.process(...) to generate an email payload (and/or PDF),
    prints the returned body, then sends it via emailer.
    """
    print("\n=== [Processor] Building email payload ===")
    from src.processor.process import process  # your new processor entrypoint
    payload = process(textlist, jpblist)

    # Expect payload like: {"subject": str, "body": str, "attachments": [...]}
    body = payload.get("body", "")
    subject = payload.get("subject", "(no subject)")

    print("\n" + "=" * 70)
    print(f"EMAIL PREVIEW — {subject}")
    print("=" * 70)
    print(body)
    print("=" * 70 + "\n")

    if not send_email:
        print("[email-skip] send_email=False")
        return payload

    print("\n=== [Email] Sending ===")
    try:
        from src.utility.emailer import send_report
        send_report(payload)
        print("[email-ok] sent premarket report")
    except Exception as e:
        print(f"[email-error] {e}")
        traceback.print_exc()

    return payload


def _write_aggregate_pdf(filename: str, textlist: List[str], jpblist: List[str]) -> Path:
    out_dir = get_txt_dir()
    out_path = out_dir / filename

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter
    x = 50
    y = height - 60

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Premarket Temporary Report")
    y -= 28
    c.setFont("Helvetica", 10)

    for t in textlist:
        path = Path(t)
        if not path.exists():
            continue
        c.setFont("Helvetica-Bold", 12)
        if y < 80:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)
        c.drawString(x, y, f"===== FILE: {path.name} =====")
        y -= 16
        c.setFont("Helvetica", 10)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if y < 60:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 10)
            c.drawString(x, y, line[:140])
            y -= 13
        y -= 10

    for j in jpblist:
        path = Path(j)
        if not path.exists():
            continue
        c.showPage()
        y = height - 60
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, f"IMAGE: {path.name}")
        y -= 20
        try:
            c.drawImage(str(path), x, 80, width=width - 100, height=height - 160, preserveAspectRatio=True, anchor="sw")
        except Exception:
            c.setFont("Helvetica", 10)
            c.drawString(x, y, "[image failed to render]")
            y -= 13

    c.save()
    return out_path


def temporary_send(textlist: List[str], jpblist: List[str], send_email: bool = True):
    """
    Combine all text and image artifacts into one temporary PDF and send via email.
    """
    print("\n=== [Temporary] Building combined artifact ===")
    ts = _now_tag()
    pdf_path = _write_aggregate_pdf(f"temporary_report_{ts}.pdf", textlist, jpblist)
    _append_text(textlist, pdf_path)

    payload = {
        "subject": f"Premarket Temporary Report {ts}",
        "body": f"Temporary premarket report generated at {ts}. See attached PDF and image artifacts.",
        "attachments": [str(pdf_path)] + [str(Path(j)) for j in jpblist if Path(j).exists()],
    }

    if not send_email:
        print("[email-skip] send_email=False")
        return payload

    print("\n=== [Temporary] Sending email ===")
    try:
        from src.utility.emailer import send_report
        send_report(payload)
        print("[temporary-email-ok] sent temporary premarket report")
    except Exception as e:
        print(f"[temporary-email-error] {e}")
        traceback.print_exc()

    return payload


# ---------------------------
# Main
# ---------------------------

def main(send_email: bool = True, market_date=None):
    load_env()
    from src.utility.date import last_market_date

    market_date = market_date or last_market_date()
    print(f"[premarket] market_date={market_date}")

    # Artifact bundles:
    # - textlist: paths to txt/md/json outputs
    # - jpblist: paths to jpg/png heatmaps or other images
    textlist: List[str] = []
    jpblist: List[str] = []

    # 1) Equity pipeline (prices + flags)
    # Comment this block out if you want to skip equity.
    try:
        equity = pipeline_equity(textlist, jpblist, market_date=market_date)
    except Exception:
        print("\n[ERROR] Equity pipeline failed.")
        traceback.print_exc()

    # # 2) Reddit → Mongo ingestion + sentiment
    # # Comment this block out if you want to skip reddit ingestion.
    # try:
    #     pipeline_reddit_mongo()
    # except Exception:
    #     print("\n[ERROR] Reddit mongo/sentiment pipeline failed.")
    #     traceback.print_exc()

    # # 3) Reddit heatmap + top mentions snapshot
    # # Comment this block out if you want to skip reddit plotting.
    # try:
    #     pipeline_reddit_graph_and_summary(textlist, jpblist)
    # except Exception:
    #     print("\n[ERROR] Reddit graph/snapshot pipeline failed.")
    #     traceback.print_exc()

    # # 4) Processor → email payload → send
    # # Comment out to stop at artifacts only.
    # try:
    #     process_and_send(textlist, jpblist, send_email=send_email)
    # except Exception:
    #     print("\n[ERROR] Processor/email step failed.")
    #     traceback.print_exc()

    # 5) Temporary send fallback
    # Use this instead of process_and_send if OpenAI is unavailable.
    # Uncomment the lines below to run temporary_send instead of process_and_send.
    try:
        temporary_send(textlist, jpblist, send_email=send_email)
    except Exception:
        print("\n[ERROR] Temporary send step failed.")
        traceback.print_exc()


if __name__ == "__main__":
    # For now, hard default to sending. You can add argparse later.
    main(send_email=False)