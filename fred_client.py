"""State unemployment from FRED via public CSV (no API key required)."""

from __future__ import annotations

import io
import time
from datetime import date, timedelta

import pandas as pd
import requests

FRED_GRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Short, stable UA; overly “browser-like” strings have been flaky on some hosts.
_HEADERS = {
    "User-Agent": "housing-tracker/1.0 (local dashboard; educational)",
    "Accept": "text/csv,application/csv;q=0.9,*/*;q=0.8",
    "Connection": "close",
}


def _cosd_param() -> str:
    """Chart start trims the CSV (smaller download — fewer timeouts on slow / shared egress)."""
    return (date.today() - timedelta(days=365 * 15)).isoformat()


def unemployment_series_id(state_postal: str) -> str:
    """FRED convention: e.g. UT -> UTUR, DC -> DCUR."""
    p = state_postal.strip().upper()
    if p == "DC":
        return "DCUR"
    return f"{p}UR"


def _request_graph_csv(params: dict[str, str]) -> str | None:
    """GET fredgraph.csv; small manual retries (clearer than urllib3 adapter + long reads)."""
    for attempt in range(3):
        try:
            r = requests.get(
                FRED_GRAPH_CSV,
                params=params,
                timeout=(12, 35),
                headers=_HEADERS,
            )
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(0.5 * (2**attempt))
                continue
            r.raise_for_status()
            text = (r.text or "").strip()
            if not text or "observation_date" not in text[:800].lower():
                return None
            return text
        except (requests.RequestException, OSError):
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
    return None


def _fetch_graph_csv(series_id: str) -> str | None:
    # Prefer date-trimmed export; fall back to full series.
    for params in (
        {"id": series_id, "cosd": _cosd_param()},
        {"id": series_id},
    ):
        text = _request_graph_csv(params)
        if text:
            return text
    return None


def latest_state_unemployment_rate(
    state_postal: str,
) -> tuple[float | None, str | None, str | None]:
    """
    Returns (rate_percent, observation_date, series_id).
    Uses https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES — no FRED API key.
    """
    if not (state_postal or "").strip():
        return None, None, None
    sid = unemployment_series_id(state_postal)
    try:
        raw = _fetch_graph_csv(sid)
        if not raw:
            return None, None, sid
        df = pd.read_csv(io.StringIO(raw))
        df.columns = [str(c).strip() for c in df.columns]
        if df.empty or len(df.columns) < 2:
            return None, None, sid
        date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
        rest = [c for c in df.columns if c != date_col]
        val_col = sid if sid in df.columns else (rest[0] if rest else None)
        if val_col is None:
            return None, None, sid
        for i in range(len(df) - 1, -1, -1):
            cell = df.iloc[i][val_col]
            if isinstance(cell, str) and cell.strip().upper() in (".", "NAN", ""):
                continue
            try:
                rate = float(cell)
            except (TypeError, ValueError):
                continue
            if pd.isna(rate):
                continue
            d = df.iloc[i][date_col]
            return rate, str(d) if d is not None and str(d) != "nan" else None, sid
    except (ValueError, IndexError, KeyError):
        pass
    return None, None, sid
