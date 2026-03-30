"""
Composite Consumer Financial Pressure Index (0–100).
Higher = more pressure on households (tighter budgets, softer demand, weaker labor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Weights sum to 1.0
W_DTI = 0.35
W_GAP = 0.18
W_HOUSING = 0.22
W_INCOME = 0.13
W_UNEMP = 0.12

NEUTRAL_SUBSCORE = 50.0


def _clamp100(x: float) -> float:
    return max(0.0, min(100.0, x))


def _dti_pressure_score(dti_mid: float | None) -> float:
    if dti_mid is None or (isinstance(dti_mid, float) and pd.isna(dti_mid)):
        return NEUTRAL_SUBSCORE
    # ~1.0 low pressure → 0; ~2.2+ very elevated → cap 100
    return _clamp100((float(dti_mid) - 1.0) / (2.2 - 1.0) * 100.0)


def _gap_pressure_score(pce_yoy: float | None, income_yoy: float | None) -> float:
    if pce_yoy is None or income_yoy is None:
        return NEUTRAL_SUBSCORE
    if pd.isna(pce_yoy) or pd.isna(income_yoy):
        return NEUTRAL_SUBSCORE
    gap = float(pce_yoy) - float(income_yoy)
    # PCE YoY > income YoY → more pressure (neutral pivot at 50 when gap = 0)
    return _clamp100(50.0 + gap * 320.0)


def _housing_pressure_score(
    median_dom: float | None,
    inventory_mom: float | None,
    median_dom_mom: float | None,
) -> float:
    parts: list[float] = []
    if median_dom is not None and pd.notna(median_dom):
        dom = float(median_dom)
        parts.append(min(50.0, (dom / 72.0) * 45.0))
    else:
        parts.append(20.0)

    if inventory_mom is not None and pd.notna(inventory_mom):
        im = float(inventory_mom)
        if im > 0:
            parts.append(min(35.0, im * 160.0))
        else:
            parts.append(max(0.0, min(15.0, 15.0 + im * 40.0)))
    else:
        parts.append(15.0)

    if median_dom_mom is not None and pd.notna(median_dom_mom):
        dm = float(median_dom_mom)
        if dm > 0:
            parts.append(min(35.0, dm / 12.0 * 28.0))
        else:
            parts.append(max(0.0, 12.0 + dm * 0.4))
    else:
        parts.append(12.0)

    return _clamp100(sum(parts))


def _income_pressure_score(income_yoy: float | None) -> float:
    if income_yoy is None or (isinstance(income_yoy, float) and pd.isna(income_yoy)):
        return NEUTRAL_SUBSCORE
    y = float(income_yoy)
    return _clamp100(48.0 - y * 130.0)


def _unemployment_pressure_rate(unemp_pct: float | None) -> float:
    if unemp_pct is None or (isinstance(unemp_pct, float) and pd.isna(unemp_pct)):
        return NEUTRAL_SUBSCORE
    u = float(unemp_pct)
    return _clamp100((u - 3.0) / 5.5 * 100.0)


def pressure_zone(score: float) -> str:
    if score <= 30:
        return "Very secure"
    if score <= 60:
        return "Moderate pressure"
    return "High pressure"


@dataclass
class CompositePressureResult:
    index: float
    zone: str
    dti_mid: float | None
    detail_rows: list[dict[str, Any]] = field(default_factory=list)


def compute_composite_pressure(
    *,
    dti_mid: float | None,
    income_yoy: float | None,
    pce_yoy: float | None,
    median_dom: float | None,
    inventory_mom: float | None,
    median_dom_mom: float | None,
    unemployment_pct: float | None,
) -> CompositePressureResult:
    s_dti = _dti_pressure_score(dti_mid)
    s_gap = _gap_pressure_score(pce_yoy, income_yoy)
    s_h = _housing_pressure_score(median_dom, inventory_mom, median_dom_mom)
    s_inc = _income_pressure_score(income_yoy)
    s_u = _unemployment_pressure_rate(unemployment_pct)

    idx = (
        W_DTI * s_dti
        + W_GAP * s_gap
        + W_HOUSING * s_h
        + W_INCOME * s_inc
        + W_UNEMP * s_u
    )
    idx = round(_clamp100(idx), 1)

    gap_note = "—"
    if pce_yoy is not None and income_yoy is not None:
        if not pd.isna(pce_yoy) and not pd.isna(income_yoy):
            gap_note = f"{(float(pce_yoy) - float(income_yoy)) * 100:+.1f} pp (PCE YoY − income YoY)"

    rows = [
        {
            "Factor": "Debt-to-income (state)",
            "Weight": f"{W_DTI:.0%}",
            "Pressure score": round(s_dti, 1),
            "Note": f"Midpoint {dti_mid:.2f}" if dti_mid is not None else "—",
        },
        {
            "Factor": "Spending vs income growth",
            "Weight": f"{W_GAP:.0%}",
            "Pressure score": round(s_gap, 1),
            "Note": gap_note,
        },
        {
            "Factor": "Housing softness (DOM, inventory)",
            "Weight": f"{W_HOUSING:.0%}",
            "Pressure score": round(s_h, 1),
            "Note": "Redfin latest month vs priors",
        },
        {
            "Factor": "Income growth (state)",
            "Weight": f"{W_INCOME:.0%}",
            "Pressure score": round(s_inc, 1),
            "Note": "Weaker YoY → more pressure",
        },
        {
            "Factor": "Unemployment (state, FRED)",
            "Weight": f"{W_UNEMP:.0%}",
            "Pressure score": round(s_u, 1),
            "Note": (
                f"{unemployment_pct:.2f}%"
                if unemployment_pct is not None
                else "FRED public CSV"
            ),
        },
    ]

    return CompositePressureResult(
        index=idx,
        zone=pressure_zone(idx),
        dti_mid=dti_mid,
        detail_rows=rows,
    )


def composite_pressure_gauge_figure(score: float, subtitle: str):
    import plotly.graph_objects as go

    if score <= 30:
        bar = "#1b5e20"
    elif score <= 60:
        bar = "#f57f17"
    else:
        bar = "#b71c1c"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 34}},
            title={"text": subtitle, "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar, "thickness": 0.35},
                "steps": [
                    {"range": [0, 30], "color": "#c8e6c9"},
                    {"range": [30, 60], "color": "#fff9c4"},
                    {"range": [60, 100], "color": "#ffcdd2"},
                ],
                "threshold": {
                    "line": {"color": "#212121", "width": 3},
                    "thickness": 0.85,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        title={
            "text": "Consumer Financial Pressure Index",
            "xanchor": "left",
            "x": 0,
            "font": {"size": 17},
        },
        height=360,
        margin=dict(t=68, b=52, l=24, r=24),
        annotations=[
            {
                "x": 0.17,
                "y": -0.08,
                "xref": "paper",
                "yref": "paper",
                "text": "<b>Very secure</b>",
                "showarrow": False,
                "font": {"size": 11, "color": "#1b5e20"},
            },
            {
                "x": 0.5,
                "y": -0.08,
                "xref": "paper",
                "yref": "paper",
                "text": "<b>Moderate</b>",
                "showarrow": False,
                "font": {"size": 11, "color": "#f57f17"},
            },
            {
                "x": 0.83,
                "y": -0.08,
                "xref": "paper",
                "yref": "paper",
                "text": "<b>High pressure</b>",
                "showarrow": False,
                "font": {"size": 11, "color": "#b71c1c"},
            },
        ],
    )
    return fig
