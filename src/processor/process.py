from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


from openai import OpenAI

from src.common.env import getenv_required


def _read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _b64_data_url_jpg(p: Path) -> str:
    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _ensure_reports_dir() -> Path:
    out_dir = Path("data") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_pdf(out_path: Path, title: str, body: str) -> None:
    """
    Very simple PDF writer (monospace-ish layout).
    """
    c = canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter

    x = 50
    y = height - 60

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 25

    c.setFont("Helvetica", 10)

    # naive word wrap by lines
    lines = body.splitlines()
    for line in lines:
        if y < 60:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)
        # trim very long lines
        if len(line) > 140:
            line = line[:140] + "…"
        c.drawString(x, y, line)
        y -= 13

    c.save()


def _write_text_report(out_path: Path, title: str, body: str) -> None:
    out_path.write_text(f"{title}\n\n{body}\n", encoding="utf-8")


def process(textlist: List[str], jpblist: List[str]) -> Dict[str, Any]:
    """
    textlist: list of TEXT FILE PATHS (txt/md/json), each will be read and included
    jpblist: list of JPG FILE PATHS, each will be attached as image input

    Returns an "email payload" dict:
      {
        "subject": "...",
        "body": "...",
        "attachments": ["/path/to/report.pdf", "/path/to/heatmap.jpg", ...]
      }
    """
    # --- env / client ---
    api_key = getenv_required("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    client = OpenAI(api_key=api_key)

    # --- load inputs ---
    text_blobs = []
    for t in textlist:
        p = Path(t)
        if not p.exists():
            continue
        text_blobs.append(f"\n\n===== FILE: {p.name} =====\n" + _read_text_file(p))

    image_urls = []
    for j in jpblist:
        p = Path(j)
        if not p.exists():
            continue
        image_urls.append(_b64_data_url_jpg(p))

    # --- build prompt ---
    system_instructions = (
        "You are a market premarket report writer. "
        "Given raw outputs (tables, flags, notes) and a heatmap image, "
        "produce:\n"
        "1) A SHORT email (<= 12 lines) and\n"
        "2) A LONG email (detailed but still skimmable).\n"
        "Use crisp bullets. Avoid fluff. Highlight actionable tickers/themes first.\n"
        "If data is missing, say so explicitly."
    )

    # Responses API supports text + image inputs. :contentReference[oaicite:3]{index=3}
    content = [{"type": "input_text", "text": system_instructions}]
    if text_blobs:
        content.append({"type": "input_text", "text": "\n".join(text_blobs)})

    for u in image_urls:
        content.append({"type": "input_image", "image_url": u})

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
    )

    out_text = resp.output_text.strip()

    # --- write a report artifact ---
    out_dir = _ensure_reports_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    report_path = out_dir / f"premarket_report_{ts}.pdf"
    _write_pdf(report_path, title=f"Premarket Report {ts}", body=out_text)





    # --- payload for emailer ---
    payload = {
        "subject": f"Premarket Report {ts}",
        "body": out_text,
        "attachments": [str(report_path)] + [str(Path(x)) for x in jpblist if Path(x).exists()],
    }
    return payload