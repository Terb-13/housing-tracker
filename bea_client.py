from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

BEA_DATA_URL = "https://apps.bea.gov/api/data"

# State postal → 2-digit FIPS (used to build 5-digit state GeoFIPS: ss000)
POSTAL_TO_STATE_FIPS: dict[str, str] = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}


def state_postal_to_geofips(postal: str) -> str:
    s = postal.strip().upper()
    if s not in POSTAL_TO_STATE_FIPS:
        raise ValueError(f"Unknown state postal code: {postal!r}")
    return f"{POSTAL_TO_STATE_FIPS[s]}000"


@dataclass
class BeaSeriesResult:
    label: str
    latest_period: str | None
    prior_year_period: str | None
    latest_value: float | None
    prior_value: float | None
    yoy_pct: float | None
    geo_name: str | None
    table_used: str
    error: str | None = None


def _bea_request(user_id: str, params: dict[str, str]) -> dict[str, Any]:
    q = {"UserID": user_id, "method": "GetData", "datasetname": "Regional", "ResultFormat": "JSON"}
    q.update(params)
    r = requests.get(BEA_DATA_URL, params=q, timeout=120)
    r.raise_for_status()
    return r.json()


def _bea_error(payload: dict[str, Any]) -> str | None:
    try:
        err = payload["BEAAPI"]["Results"].get("Error")
        if err is None:
            return None
        return err.get("APIErrorDescription") or str(err)
    except (KeyError, TypeError):
        return None


def _parse_number(raw: str) -> float | None:
    if raw is None or str(raw).strip() in {"", "(NA)", "NA", "NAN"}:
        return None
    s = str(raw).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _rows_to_df(payload: dict[str, Any]) -> pd.DataFrame:
    err = _bea_error(payload)
    if err:
        raise RuntimeError(err)
    data = payload["BEAAPI"]["Results"].get("Data") or []
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def _quarter_yr_q(tp: str) -> tuple[int, int] | None:
    tp = str(tp).strip()
    for pat in (
        r"^(\d{4})Q([1-4])$",
        r"^(\d{4}):Q([1-4])$",
        r"^(\d{4})-Q([1-4])$",
    ):
        m = re.match(pat, tp)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _yoy_from_time_series(df: pd.DataFrame) -> tuple[str | None, float | None, str | None]:
    if df.empty or "TimePeriod" not in df.columns:
        return None, None, None
    d = df.copy()
    d["val"] = d["DataValue"].map(_parse_number)
    d = d[d["val"].notna()].copy()
    if d.empty:
        return None, None, None

    def sort_key(tp: str) -> tuple[int, int]:
        tp = str(tp).strip()
        yq = _quarter_yr_q(tp)
        if yq:
            return yq[0], yq[1]
        m = re.match(r"^(\d{4})$", tp)
        if m:
            return int(m.group(1)), 0
        return 0, 0

    d["_k"] = d["TimePeriod"].map(sort_key)
    d = d.sort_values("_k")
    latest_row = d.iloc[-1]
    latest_tp = str(latest_row["TimePeriod"]).strip()
    latest_v = float(latest_row["val"])
    geo = latest_row.get("GeoName")
    geo_s = str(geo) if geo is not None and str(geo) != "nan" else None

    yq = _quarter_yr_q(latest_tp)
    if yq:
        y, q = yq[0] - 1, yq[1]
        candidates = {f"{y}Q{q}", f"{y}:Q{q}", f"{y}-Q{q}"}
        past = d[d["TimePeriod"].astype(str).isin(candidates)]
    else:
        y_match2 = re.match(r"^(\d{4})$", latest_tp)
        if y_match2:
            target = str(int(y_match2.group(1)) - 1)
            past = d[d["TimePeriod"].astype(str) == target]
        else:
            return latest_tp, None, geo_s

    if past.empty:
        return latest_tp, None, geo_s
    prev_v = float(past.iloc[-1]["val"])
    if prev_v == 0:
        return latest_tp, None, geo_s
    return latest_tp, (latest_v - prev_v) / prev_v, geo_s


def fetch_regional_series(
    user_id: str,
    *,
    table_name: str,
    line_code: str,
    geo_fips: str,
    year_list: str,
) -> pd.DataFrame:
    payload = _bea_request(
        user_id,
        {
            "TableName": table_name,
            "LineCode": line_code,
            "GeoFIPS": geo_fips,
            "Year": year_list,
        },
    )
    return _rows_to_df(payload)


def fetch_state_quarterly_income_yoy(
    user_id: str, state_postal: str, years_back: int = 8
) -> BeaSeriesResult:
    """SQINC1: state quarterly personal income (Table 1 line 1 = personal income)."""
    g = state_postal_to_geofips(state_postal)
    y_end = pd.Timestamp.utcnow().year + 1
    y_start = y_end - years_back
    years = ",".join(str(y) for y in range(y_start, y_end + 1))
    label = f"State {state_postal} personal income (QoQ base, YoY %)"
    try:
        df = fetch_regional_series(
            user_id,
            table_name="SQINC1",
            line_code="1",
            geo_fips=g,
            year_list=years,
        )
    except Exception as e:
        return BeaSeriesResult(
            label=label,
            latest_period=None,
            prior_year_period=None,
            latest_value=None,
            prior_value=None,
            yoy_pct=None,
            geo_name=None,
            table_used="SQINC1",
            error=str(e),
        )
    if df.empty:
        return BeaSeriesResult(
            label=label,
            latest_period=None,
            prior_year_period=None,
            latest_value=None,
            prior_value=None,
            yoy_pct=None,
            geo_name=None,
            table_used="SQINC1",
            error="BEA returned no rows (check table/GeoFIPS).",
        )
    tp, yoy, geo = _yoy_from_time_series(df)
    return BeaSeriesResult(
        label=label,
        latest_period=tp,
        prior_year_period=None,
        latest_value=None,
        prior_value=None,
        yoy_pct=yoy,
        geo_name=geo,
        table_used="YOY from SQINC1 quarterly",
        error=None,
    )


def fetch_state_annual_pce_yoy(
    user_id: str, state_postal: str, years_back: int = 6
) -> BeaSeriesResult:
    """SAPCE1: annual personal consumption expenditures by state (line 1 = total PCE)."""
    g = state_postal_to_geofips(state_postal)
    y_end = pd.Timestamp.utcnow().year
    y_start = y_end - years_back
    years = ",".join(str(y) for y in range(y_start, y_end + 1))
    label = f"State {state_postal} total PCE (annual YoY %)"
    try:
        df = fetch_regional_series(
            user_id,
            table_name="SAPCE1",
            line_code="1",
            geo_fips=g,
            year_list=years,
        )
    except Exception as e:
        return BeaSeriesResult(
            label=label,
            latest_period=None,
            prior_year_period=None,
            latest_value=None,
            prior_value=None,
            yoy_pct=None,
            geo_name=None,
            table_used="SAPCE1",
            error=str(e),
        )
    if df.empty:
        return BeaSeriesResult(
            label=label,
            latest_period=None,
            prior_year_period=None,
            latest_value=None,
            prior_value=None,
            yoy_pct=None,
            geo_name=None,
            table_used="SAPCE1",
            error="BEA returned no rows for SAPCE1 (annual PCE may use a different LineCode in your BEA vintage).",
        )
    tp, yoy, geo = _yoy_from_time_series(df)
    return BeaSeriesResult(
        label=label,
        latest_period=tp,
        prior_year_period=None,
        latest_value=None,
        prior_value=None,
        yoy_pct=yoy,
        geo_name=geo,
        table_used="YoY from SAPCE1 annual",
        error=None,
    )


def fetch_metro_annual_income_yoy(
    user_id: str, cbsa_geofips: str, years_back: int = 10
) -> BeaSeriesResult:
    """MAINC1: metropolitan area personal income, millions; line 1."""
    y_end = pd.Timestamp.utcnow().year
    y_start = y_end - years_back
    years = ",".join(str(y) for y in range(y_start, y_end + 1))
    g = str(cbsa_geofips).strip()
    if len(g) != 5 or not g.isdigit():
        return BeaSeriesResult(
            label=f"Metro CBSA {g}",
            latest_period=None,
            prior_year_period=None,
            latest_value=None,
            prior_value=None,
            yoy_pct=None,
            geo_name=None,
            table_used="MAINC1",
            error="Invalid CBSA GeoFIPS (expected 5 digits). Re-ingest metro/city Redfin data to store metro_code.",
        )
    label = f"Metro (CBSA {g}) personal income (annual YoY %)"
    try:
        df = fetch_regional_series(
            user_id,
            table_name="MAINC1",
            line_code="1",
            geo_fips=g,
            year_list=years,
        )
    except Exception as e:
        return BeaSeriesResult(
            label=label,
            latest_period=None,
            prior_year_period=None,
            latest_value=None,
            prior_value=None,
            yoy_pct=None,
            geo_name=None,
            table_used="MAINC1",
            error=str(e),
        )
    if df.empty:
        return BeaSeriesResult(
            label=label,
            latest_period=None,
            prior_year_period=None,
            latest_value=None,
            prior_value=None,
            yoy_pct=None,
            geo_name=None,
            table_used="MAINC1",
            error="BEA returned no rows for this CBSA.",
        )
    tp, yoy, geo = _yoy_from_time_series(df)
    return BeaSeriesResult(
        label=label,
        latest_period=tp,
        prior_year_period=None,
        latest_value=None,
        prior_value=None,
        yoy_pct=yoy,
        geo_name=geo,
        table_used="YoY from MAINC1 annual",
        error=None,
    )


def growth_to_score(g: float | None, *, low: float = -0.02, high: float = 0.06) -> float:
    """Map growth rate to 0–100 (50 = neutral at 0 growth)."""
    if g is None or (isinstance(g, float) and pd.isna(g)):
        return 50.0
    x = (float(g) - low) / (high - low)
    return float(max(0.0, min(100.0, x * 100.0)))


def fear_barometer_score(
    income_yoy: float | None,
    spending_yoy: float | None,
) -> tuple[float, str]:
    """
    Returns (fear_index 0–100, interpretation).
    Higher = more “fear” / weaker household momentum (inverse of Unusual Whales greed).
    """
    s_inc = growth_to_score(income_yoy)
    s_pce = growth_to_score(spending_yoy)
    confidence = (s_inc + s_pce) / 2.0
    fear = 100.0 - confidence
    if fear >= 66:
        tone = "Elevated pressure — income or spending growth is weak versus typical ranges."
    elif fear <= 33:
        tone = "Mild — income and spending growth look firm versus typical ranges."
    else:
        tone = "Mixed — one of income or spending is soft relative to the other."
    return round(fear, 1), tone


def gauge_figure(
    fear_score: float,
    title: str,
    subtitle: str,
) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fear_score,
            number={"suffix": "/100"},
            title={"text": subtitle, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#5c6bc0"},
                "steps": [
                    {"range": [0, 33], "color": "#e8f5e9"},
                    {"range": [33, 66], "color": "#fff9c4"},
                    {"range": [66, 100], "color": "#ffebee"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.8,
                    "value": fear_score,
                },
            },
        )
    )
    fig.update_layout(
        title={"text": title, "xanchor": "left", "x": 0, "font": {"size": 18}},
        height=320,
        margin=dict(t=64, b=32, l=24, r=24),
    )
    return fig
