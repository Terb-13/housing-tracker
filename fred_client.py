"""FRED (Federal Reserve Economic Data) — state unemployment rate (monthly, %)."""

from __future__ import annotations

from typing import Any

import requests

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


def unemployment_series_id(state_postal: str) -> str:
    """FRED convention: e.g. UT -> UTUR, DC -> DCUR."""
    p = state_postal.strip().upper()
    if p == "DC":
        return "DCUR"
    return f"{p}UR"


def latest_state_unemployment_rate(
    state_postal: str, api_key: str
) -> tuple[float | None, str | None, str | None]:
    """
    Returns (rate_percent, observation_date, series_id) or (None, None, sid) on failure.
    Rate is a percent (e.g. 3.6 for 3.6%), not decimal.
    """
    key = (api_key or "").strip()
    if not key:
        return None, None, None
    sid = unemployment_series_id(state_postal)
    try:
        r = requests.get(
            FRED_OBSERVATIONS_URL,
            params={
                "series_id": sid,
                "api_key": key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 12,
            },
            timeout=45,
        )
        r.raise_for_status()
        payload: dict[str, Any] = r.json()
        obs = payload.get("observations") or []
        for row in obs:
            v = row.get("value")
            if v is None or str(v) in (".", "", "nan"):
                continue
            try:
                rate = float(v)
            except ValueError:
                continue
            d = row.get("date")
            return rate, str(d) if d else None, sid
    except (requests.RequestException, KeyError, ValueError, TypeError):
        pass
    return None, None, sid
