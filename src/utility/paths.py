from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from .constant import REPORTS_DIR


def _normalize_run_date(run_date: Optional[date | str] = None) -> str:
    if run_date is None:
        return date.today().strftime("%Y-%m-%d")
    if isinstance(run_date, date):
        return run_date.strftime("%Y-%m-%d")
    return str(run_date)


def get_report_date_str(run_date: Optional[date | str] = None) -> str:
    return _normalize_run_date(run_date)


def get_run_dir(run_date: Optional[date | str] = None, session: str = "premarket") -> Path:
    run_date_str = _normalize_run_date(run_date)
    run_dir = Path(REPORTS_DIR) / run_date_str / session
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_txt_dir(run_date: Optional[date | str] = None, session: str = "premarket") -> Path:
    txt_dir = get_run_dir(run_date=run_date, session=session) / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    return txt_dir


def get_img_dir(run_date: Optional[date | str] = None, session: str = "premarket") -> Path:
    img_dir = get_run_dir(run_date=run_date, session=session) / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    return img_dir


def get_json_dir(run_date: Optional[date | str] = None, session: str = "premarket") -> Path:
    json_dir = get_run_dir(run_date=run_date, session=session) / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    return json_dir
