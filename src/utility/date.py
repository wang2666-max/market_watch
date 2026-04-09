# src/utilities/date.py
from __future__ import annotations

from datetime import date as pydate
from typing import Optional

try:
    import QuantLib as ql
except Exception as e:
    ql = None


def _to_ql_date(d: pydate) -> "ql.Date":
    return ql.Date(d.day, d.month, d.year)


def _to_py_date(d: "ql.Date") -> pydate:
    return pydate(d.year(), d.month(), d.dayOfMonth())


def _us_calendar() -> "ql.Calendar":
    # US settlement calendar is a reasonable default for equities.
    # If you want NYSE specifically, QuantLib supports it depending on build.
    return ql.UnitedStates(ql.UnitedStates.Settlement)


def last_market_date(
    ref_date: Optional[pydate] = None,
    calendar: str = "US",
) -> pydate:
    """
    Return the most recent *trading* date strictly before ref_date (premarket-safe).
    Example: if ref_date is Sunday, returns Friday (assuming not holiday).

    calendar:
      - "US" uses QuantLib UnitedStates Settlement calendar.
    """
    if ql is None:
        raise ImportError(
            "QuantLib is not installed. Run: python -m pip install QuantLib "
            "or ask for the pure-Python fallback."
        )

    ref_date = ref_date or pydate.today()

    if calendar.upper() != "US":
        raise ValueError("Only calendar='US' is supported right now.")

    cal = _us_calendar()

    # Premarket convention: use the last completed trading day (strictly before ref_date)
    d = ref_date
    # step back at least one day
    d = d.fromordinal(d.toordinal() - 1)

    qd = _to_ql_date(d)
    while not cal.isBusinessDay(qd):
        qd = qd - 1

    return _to_py_date(qd)