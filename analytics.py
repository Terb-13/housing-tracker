from __future__ import annotations

import pandas as pd


def trend_from_change(
    change: float | None,
    *,
    stable_band: float = 0.005,
) -> str:
    if change is None or (isinstance(change, float) and pd.isna(change)):
        return "unknown"
    if change > stable_band:
        return "increasing"
    if change < -stable_band:
        return "decreasing"
    return "stable"


def latest_row(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    return df.sort_values("period_begin").iloc[-1]


def format_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x * 100:+.1f}%"


def _fmt_num(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    x = float(v)
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    return f"{x:,.1f}"


def _fmt_money(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"${float(v):,.0f}"


def summary_markdown(latest: pd.Series, label: str) -> str:
    if latest is None or latest.empty:
        return "_No data for this selection._"

    price_m = latest.get("median_sale_price_mom")
    price_y = latest.get("median_sale_price_yoy")
    inv_m = latest.get("inventory_mom")
    dom_m = latest.get("median_dom_mom")

    try:
        pm = float(price_m) if price_m is not None and pd.notna(price_m) else None
    except (TypeError, ValueError):
        pm = None
    try:
        dm = float(dom_m) if dom_m is not None and pd.notna(dom_m) else None
    except (TypeError, ValueError):
        dm = None

    ptrend = trend_from_change(pm)
    itrend = trend_from_change(
        float(inv_m) if inv_m is not None and pd.notna(inv_m) else None,
        stable_band=0.02,
    )
    if dm is not None:
        if dm < -0.5:
            dom_trend = "faster"
        elif dm > 0.5:
            dom_trend = "slower"
        else:
            dom_trend = "about the same"
    else:
        dom_trend = "unknown"

    dom = latest.get("median_dom")
    dom_suffix = ""
    if dm is not None:
        dom_suffix = f" — selling **{dom_trend}** than prior month ({dm:+.1f} days MoM)."

    lines = [
        f"### {label}",
        "",
        f"- **Median sale price:** {_fmt_money(latest.get('median_sale_price'))} — MoM {format_pct(latest.get('median_sale_price_mom'))} ({ptrend}), YoY {format_pct(latest.get('median_sale_price_yoy'))}.",
        f"- **Inventory:** {_fmt_num(latest.get('inventory'))} — MoM {format_pct(latest.get('inventory_mom'))} ({itrend} vs prior month).",
        f"- **New listings:** {_fmt_num(latest.get('new_listings'))} — MoM {format_pct(latest.get('new_listings_mom'))}.",
        f"- **Median days on market:** {_fmt_num(dom)}{dom_suffix}",
        f"- **Months of supply:** {_fmt_num(latest.get('months_of_supply'))}.",
        f"- **Homes sold (period):** {_fmt_num(latest.get('homes_sold'))}.",
        "",
    ]
    return "\n".join(lines)
