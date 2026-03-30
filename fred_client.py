"""State unemployment from FRED via public CSV (no API key required)."""

from __future__ import annotations

import io

import pandas as pd
import requests

FRED_GRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Conservative UA; FRED may throttle bare clients.
_HEADERS = {"User-Agent": "housing-tracker/1.0 (local dashboard; educational)"}


def unemployment_series_id(state_postal: str) -> str:
    """FRED convention: e.g. UT -> UTUR, DC -> DCUR."""
    p = state_postal.strip().upper()
    if p == "DC":
        return "DCUR"
    return f"{p}UR"


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
        r = requests.get(
            FRED_GRAPH_CSV,
            params={"id": sid},
            timeout=60,
            headers=_HEADERS,
        )
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [str(c).strip() for c in df.columns]
        if df.empty or len(df.columns) < 2:
            return None, None, sid
        date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
        rest = [c for c in df.columns if c != date_col]
        val_col = sid if sid in df.columns else (rest[0] if rest else None)
        if val_col is None:
            return None, None, sid
        for i in range(len(df) - 1, -1, -1):
            raw = df.iloc[i][val_col]
            try:
                rate = float(raw)
            except (TypeError, ValueError):
                continue
            if pd.isna(rate):
                continue
            d = df.iloc[i][date_col]
            return rate, str(d) if d is not None and str(d) != "nan" else None, sid
    except (requests.RequestException, ValueError, IndexError, KeyError):
        pass
    return None, None, sid
