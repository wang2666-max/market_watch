# options_client.py
# Build/append a daily series of *constant-maturity 30D ATM close IV* per ticker.
# - Uses your equity CSVs in data/<TICKER>.csv for the underlying close (S).
# - Lists option contracts/expiries *as of each historical date* with tight filters:
#     expiration_date in [as_of+20d, as_of+50d]   (brackets ~30D)
#     strike_price  in [S*(1-b), S*(1+b)]        (b = 20% → auto-tighten if paging)
# - Pulls the option's *daily close* and solves IV via Black–Scholes.
# - Within expiry: ATM by strike (bracket S; linear interpolate IV in strike).
# - Across expiries: 30D constant-maturity via *variance* interpolation (linear in T on σ²).
#
# Output per ticker: data/<TICKER>_atm_iv.csv with:
#   date, ticker, iv_cm_30d_close, method, expiry1, dte1, iv1, expiry2, dte2, iv2, w, ...
#
# Notes:
# - Keeps public function names stable.
# - Historical greeks/IV are not provided by Polygon REST; we compute from prices.
# - Paging control: we *prefer tightening filters* over increasing limit.

from __future__ import annotations
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Iterable, List, Dict, Tuple

import httpx
import numpy as np
import pandas as pd

from src.utility.constant import POLYGON_API_KEY, SMF_TICKERS, DATA_DIR
from polygon_client import DEFAULT_INIT_START  # "2024-01-01"

BASE = "https://api.polygon.io"

# Target maturity + guards
TARGET_CM_DAYS = 30         # constant-maturity target (calendar)
EXP_LO_BUMP = 20            # as_of + 20d  → lower bound for expiries
EXP_HI_BUMP = 50            # as_of + 50d  → upper bound for expiries
MIN_DTE = 2                 # avoid expiry-day noise

# Strike band strategy (around spot): try 20%, then 10%, then 5%
STRIKE_BANDS = [0.20, 0.10, 0.05]

# Minimal liquidity guard for historical daily bars
MIN_BAR_VOLUME = 10

# ------------------------ HTTP helpers ------------------------

def _client() -> httpx.Client:
    return httpx.Client(timeout=30.0)

def _get(path: str, params: dict) -> dict:
    if not POLYGON_API_KEY or "PUT_YOUR_KEY_HERE" in POLYGON_API_KEY:
        raise RuntimeError("Set POLYGON_API_KEY in cfg.py")
    qp = dict(params or {})
    qp["apiKey"] = POLYGON_API_KEY
    with _client() as c:
        r = c.get(f"{BASE}{path}", params=qp)
        r.raise_for_status()
        return r.json()

def _get_absolute(next_url: str) -> dict:
    # Follow Polygon's next_url directly (already includes query params & key if provided).
    # We add key defensively in case it's omitted.
    if "apiKey=" not in next_url:
        sep = "&" if ("?" in next_url) else "?"
        next_url = f"{next_url}{sep}apiKey={POLYGON_API_KEY}"
    with _client() as c:
        r = c.get(next_url)
        r.raise_for_status()
        return r.json()

# ------------------------ File paths ------------------------

def _spot_path(ticker: str) -> Path:
    return Path(DATA_DIR) / f"{ticker.upper()}.csv"

def _iv_path(ticker: str) -> Path:
    return Path(DATA_DIR) / f"{ticker.upper()}_atm_iv.csv"

def _ensure_data_dir():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# ------------------------ Spot close loader ------------------------

def _load_spot_series(ticker: str) -> pd.DataFrame:
    """
    Load per-ticker OHLC CSV and return df with ['date','spot_close'] sorted ascending.
    """
    p = _spot_path(ticker)
    if not p.exists():
        raise FileNotFoundError(f"Missing spot CSV for {ticker}: {p}")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df[["date", "close"]].rename(columns={"close": "spot_close"})

def _iv_series_last_date(ticker: str) -> Optional[date]:
    """
    Return most recent saved IV date from data/<TICKER>_atm_iv.csv, else None.
    """
    p = _iv_path(ticker)
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["date"])
    if df.empty:
        return None
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    if d.empty:
        return None
    return d.max().date()

# ------------------------ Contracts & bars (with tight filters) ------------------------

def _list_expiries(ticker: str, asof: date) -> List[date]:
    """
    Return expiration dates for the underlying *as of that date*, filtered to [as_of+20d, as_of+50d].
    This drastically reduces catalog size and naturally brackets ~30D.
    """
    exp_lo = (asof + timedelta(days=EXP_LO_BUMP)).isoformat()
    exp_hi = (asof + timedelta(days=EXP_HI_BUMP)).isoformat()

    j = _get("/v3/reference/options/contracts", {
        "underlying_ticker": ticker.upper(),
        "as_of": asof.isoformat(),
        "expiration_date.gte": exp_lo,
        "expiration_date.lte": exp_hi,
        "limit": 200,  # modest cap; expiries are few even on liquid names
    })

    exps = set()
    for item in (j.get("results") or []):
        exp = item.get("expiration_date")
        if not exp:
            continue
        ed = pd.to_datetime(exp, errors="coerce")
        if pd.isna(ed):
            continue
        ed = ed.date()
        if ed >= asof:
            exps.add(ed)
    return sorted(exps)

def _contracts_for_expiry(
    ticker: str,
    expiry: date,
    asof: Optional[date] = None,
    strike_gte: Optional[float] = None,
    strike_lte: Optional[float] = None,
    limit: int = 200,
    tighten_if_paged: bool = True,
) -> pd.DataFrame:
    """
    Contracts metadata for a given expiry (point-in-time), with *strike band* filters.
    If Polygon still paginates (returns next_url), we optionally *tighten the band* rather than paging.
    Columns: option_symbol, strike, type ('call'|'put'), expiration (date)
    """
    def query(band_low: Optional[float], band_high: Optional[float]) -> Tuple[pd.DataFrame, bool]:
        params = {
            "underlying_ticker": ticker.upper(),
            "expiration_date": expiry.isoformat(),
            "limit": limit,
        }
        if asof:
            params["as_of"] = asof.isoformat()
        if band_low is not None:
            params["strike_price.gte"] = band_low
        if band_high is not None:
            params["strike_price.lte"] = band_high

        j = _get("/v3/reference/options/contracts", params)
        paged = bool(j.get("next_url"))
        rows = []
        for it in (j.get("results") or []):
            try:
                rows.append({
                    "option_symbol": it["ticker"],
                    "strike": float(it["strike_price"]),
                    "type": it["contract_type"],     # 'call' or 'put'
                    "expiration": pd.to_datetime(it["expiration_date"]).date(),
                })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[df["expiration"] == expiry]
        return df, paged

    df, paged = query(strike_gte, strike_lte)

    if tighten_if_paged and paged:
        # If we still page, try progressively tighter bands around the *midpoint* of [gte, lte] or use spot-based band upstream.
        # Here we just halve width if bounds were provided; if not, we can't tighten meaningfully.
        for _ in range(2):  # try 2 more tighten attempts
            if strike_gte is None or strike_lte is None:
                break
            mid = 0.5 * (strike_gte + strike_lte)
            width = 0.5 * (strike_lte - strike_gte)
            if width <= 0:
                break
            strike_gte, strike_lte = mid - 0.5 * width, mid + 0.5 * width
            df2, paged2 = query(strike_gte, strike_lte)
            df, paged = (df2, paged2) if not df2.empty else (df, paged2)
            if not paged:
                break

    # As a last resort, if still big, keep only the 40 strikes closest to ATM-ish region (caller decides ATM).
    if not df.empty and len(df["strike"].unique()) > 40:
        ks = sorted(df["strike"].unique())
        # closeness to median strike (proxy; the real ATM proximity is enforced in caller with spot)
        med = ks[len(ks)//2]
        ks_sorted = sorted(ks, key=lambda k: abs(k - med))[:40]
        df = df[df["strike"].isin(ks_sorted)]

    return df

def _option_daily_close(option_symbol: str, asof: date) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch the option's *daily bar* close price and volume for 'asof' date.
    Returns (close_price, volume). None if not available.
    """
    path = f"/v2/aggs/ticker/{option_symbol}/range/1/day/{asof.isoformat()}/{asof.isoformat()}"
    try:
        j = _get(path, {"adjusted": "true", "sort": "asc", "limit": 500})
    except httpx.HTTPError:
        return (None, None)
    results = j.get("results") or []
    if not results:
        return (None, None)
    r = results[0]
    close = float(r.get("c")) if r.get("c") is not None else None
    vol = float(r.get("v")) if r.get("v") is not None else None
    return (close, vol)

# Kept for API surface compatibility; not used in historical flow
def _snapshot_quote(option_symbol: str) -> Optional[dict]:
    return None

# ------------------------ Black–Scholes helpers ------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_price(is_call: bool, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)

def _implied_vol_bisection(is_call: bool, S: float, K: float, T: float, r: float, q: float, price: float,
                           tol: float = 1e-6, max_iter: int = 100, low: float = 1e-4, high: float = 5.0) -> Optional[float]:
    if price is None or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
    upper_bound = S if is_call else K
    if price < intrinsic - 1e-8 or price > upper_bound:
        return None

    lo, hi = low, high
    f_lo = _bs_price(is_call, S, K, T, r, q, lo) - price
    f_hi = _bs_price(is_call, S, K, T, r, q, hi) - price

    tries = 0
    while f_lo * f_hi > 0 and tries < 10:
        hi *= 2.0
        f_hi = _bs_price(is_call, S, K, T, r, q, hi) - price
        tries += 1
        if hi > 100.0:
            break

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _bs_price(is_call, S, K, T, r, q, mid) - price
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))

def _risk_free_rate(asof: date, T_years: float) -> float:
    # v1: flat proxy; swap to actual curve later if needed
    return 0.03

# ------------------------ ATM-by-strike within one expiry ------------------------

def _best_iv_at_strike(meta: pd.DataFrame, strike: float, spot: float, asof: date, expiry: date) -> Tuple[Optional[float], Optional[dict]]:
    """
    For a given strike, fetch CALL and PUT daily close prices for 'asof',
    invert BS for each, and return the cleaner IV (or average if both exist).
    """
    rows = meta.loc[meta["strike"] == strike]
    chosen = []

    for side in ("call", "put"):
        sym = rows.loc[rows["type"] == side, "option_symbol"]
        if sym.empty:
            continue
        symbol = sym.iloc[0]
        px, vol = _option_daily_close(symbol, asof)
        if px is None or (vol is not None and vol < MIN_BAR_VOLUME):
            continue

        is_call = (side == "call")
        T = max((expiry - asof).days, MIN_DTE) / 365.0
        r = _risk_free_rate(asof, T)
        q = 0.0

        iv = _implied_vol_bisection(is_call, S=float(spot), K=float(strike), T=T, r=r, q=q, price=float(px))
        if iv is None or not (0 < iv < 5.0):
            continue

        chosen.append((iv, {
            f"{side}_symbol": symbol,
            "side": side,
            "price": px,
            "bar_vol": vol,
            "T": T,
            "r": r
        }))

    if not chosen:
        return None, None
    if len(chosen) == 1:
        return chosen[0][0], chosen[0][1]

    iv_avg = 0.5 * (chosen[0][0] + chosen[1][0])
    meta_out = {"call_put_avg": True}
    for _, info in chosen:
        meta_out.update(info)
    return iv_avg, meta_out

def _atm_iv_for_expiry(ticker: str, expiry: date, spot: float, asof: date) -> Optional[Tuple[float, dict]]:
    """
    Compute ATM IV at a specific expiry (as-of a date) with *strike-band filtered* contracts.
    Returns (iv_atm, trace) or None.
    """
    # choose strike band attempts around spot
    meta = None
    for b in STRIKE_BANDS:
        lo = spot * (1.0 - b)
        hi = spot * (1.0 + b)
        meta = _contracts_for_expiry(
            ticker, expiry, asof=asof,
            strike_gte=lo, strike_lte=hi,
            limit=200, tighten_if_paged=True
        )
        if not meta.empty:
            break
    if meta is None or meta.empty:
        return None

    strikes = sorted(meta["strike"].unique())
    # bracket spot
    lo_strike = max([k for k in strikes if k <= spot], default=None)
    hi_strike = min([k for k in strikes if k >= spot], default=None)

    if lo_strike is None and hi_strike is None:
        return None
    if lo_strike is None:
        iv_hi, info_hi = _best_iv_at_strike(meta, hi_strike, spot, asof, expiry)
        if iv_hi is None:
            return None
        return iv_hi, {"method": "single_hi", "strike_hi": hi_strike, "iv_hi": iv_hi, **(info_hi or {})}
    if hi_strike is None:
        iv_lo, info_lo = _best_iv_at_strike(meta, lo_strike, spot, asof, expiry)
        if iv_lo is None:
            return None
        return iv_lo, {"method": "single_lo", "strike_lo": lo_strike, "iv_lo": iv_lo, **(info_lo or {})}

    iv_lo, info_lo = _best_iv_at_strike(meta, lo_strike, spot, asof, expiry)
    iv_hi, info_hi = _best_iv_at_strike(meta, hi_strike, spot, asof, expiry)

    if iv_lo is None and iv_hi is None:
        return None
    if iv_lo is None:
        return iv_hi, {"method": "single_hi", "strike_hi": hi_strike, "iv_hi": iv_hi, **(info_hi or {})}
    if iv_hi is None:
        return iv_lo, {"method": "single_lo", "strike_lo": lo_strike, "iv_lo": iv_lo, **(info_lo or {})}

    w = (float(spot) - lo_strike) / (hi_strike - lo_strike) if hi_strike > lo_strike else 0.5
    iv_atm = (1 - w) * iv_lo + w * iv_hi
    trace = {
        "method": "interp",
        "strike_lo": lo_strike, "iv_lo": iv_lo,
        "strike_hi": hi_strike, "iv_hi": iv_hi,
        "w": float(w),
    }
    trace.update({f"lo_{k}": v for k, v in (info_lo or {}).items()})
    trace.update({f"hi_{k}": v for k, v in (info_hi or {}).items()})
    return iv_atm, trace

# ------------------------ Constant-maturity (variance interpolation) ------------------------

def _constant_maturity_atm_iv(ticker: str, asof: date, spot: float) -> Optional[Tuple[float, dict]]:
    """
    Compute 30D constant-maturity ATM IV via variance interpolation between two expiries.
    Returns (iv_cm_30d, trace) or None.
    """
    expiries = _list_expiries(ticker, asof)
    if not expiries:
        return None

    def dte(d): return (d - asof).days
    valid = [d for d in expiries if dte(d) >= MIN_DTE]
    if not valid:
        return None

    # pick lower/upper around target
    lower = None
    upper = None
    for d in valid:
        if dte(d) <= TARGET_CM_DAYS:
            lower = d
        if dte(d) >= TARGET_CM_DAYS and upper is None:
            upper = d
    if lower is None:
        lower = min(valid, key=lambda x: abs(dte(x) - TARGET_CM_DAYS))
    if upper is None:
        upper = min(valid, key=lambda x: abs(dte(x) - TARGET_CM_DAYS))

    if lower == upper:
        iv_e, tr_e = _atm_iv_for_expiry(ticker, lower, spot, asof)
        if iv_e is None:
            return None
        return iv_e, {"method": "single_expiry", "expiry1": lower.isoformat(), "dte1": dte(lower), "iv1": iv_e}

    iv_l, tr_l = _atm_iv_for_expiry(ticker, lower, spot, asof)
    iv_u, tr_u = _atm_iv_for_expiry(ticker, upper, spot, asof)

    if iv_l is None and iv_u is None:
        return None
    if iv_l is None:
        return iv_u, {
            "method": "single_expiry_upper",
            "expiry2": upper.isoformat(), "dte2": dte(upper), "iv2": iv_u,
            **({f"e2_{k}": v for k, v in (tr_u or {}).items()})
        }
    if iv_u is None:
        return iv_l, {
            "method": "single_expiry_lower",
            "expiry1": lower.isoformat(), "dte1": dte(lower), "iv1": iv_l,
            **({f"e1_{k}": v for k, v in (tr_l or {}).items()})
        }

    T1 = max(dte(lower), MIN_DTE) / 365.0
    T2 = max(dte(upper), MIN_DTE) / 365.0
    T_star = TARGET_CM_DAYS / 365.0
    w = (T_star - T1) / (T2 - T1) if T2 > T1 else 0.5
    v1 = iv_l ** 2
    v2 = iv_u ** 2
    v_star = (1 - w) * v1 + w * v2
    iv_star = float(np.sqrt(v_star))

    trace = {
        "method": "var_interp",
        "expiry1": lower.isoformat(), "dte1": dte(lower), "iv1": iv_l,
        "expiry2": upper.isoformat(), "dte2": dte(upper), "iv2": iv_u,
        "w": float(w)
    }
    trace.update({f"e1_{k}": v for k, v in (tr_l or {}).items()})
    trace.update({f"e2_{k}": v for k, v in (tr_u or {}).items()})
    return iv_star, trace

# ------------------------ Public entrypoint ------------------------

def update_atm_iv_series(tickers: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """
    For each ticker:
      - Determine start date: last saved iv date + 1, else DEFAULT_INIT_START.
      - Iterate spot close dates >= start date (from data/<TICKER>.csv).
      - For each date, compute 30D CM ATM close IV and append to data/<TICKER>_atm_iv.csv.
    Returns {ticker: rows_appended}.
    """
    tickers = list(tickers) if tickers is not None else list(SMF_TICKERS)
    _ensure_data_dir()
    appended: Dict[str, int] = {}

    for t in map(str.upper, tickers):
        try:
            spot_df = _load_spot_series(t)
        except FileNotFoundError as e:
            print(f"[spot-missing] {t}: {e}")
            appended[t] = 0
            continue

        last_iv_date = _iv_series_last_date(t)
        start_date = pd.to_datetime(last_iv_date or DEFAULT_INIT_START).date()

        dates = [d.date() for d in spot_df["date"] if d.date() > start_date]
        if not dates:
            appended[t] = 0
            continue

        out_rows = []
        for d in dates:
            row = spot_df.loc[spot_df["date"].dt.date == d]
            if row.empty:
                continue
            S = float(row["spot_close"].iloc[0])

            cm = _constant_maturity_atm_iv(t, asof=d, spot=S)
            if cm is None:
                continue
            iv_val, trace = cm
            out_rows.append({
                "date": d.isoformat(),
                "ticker": t,
                "iv_cm_30d_close": iv_val,
                **trace
            })

        if out_rows:
            p = _iv_path(t)
            df_out = pd.DataFrame(out_rows).sort_values("date")
            if p.exists():
                old = pd.read_csv(p)
                df_out = pd.concat([old, df_out], ignore_index=True)
                df_out = df_out.drop_duplicates(subset=["date"]).sort_values("date")
            df_out.to_csv(p, index=False)
            appended[t] = len(out_rows)
            print(f"[iv-ok] {t}: appended {len(out_rows)} rows to {p.name}")
        else:
            appended[t] = 0

    return appended
