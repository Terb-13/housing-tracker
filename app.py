from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analytics import latest_row, summary_markdown, trend_from_change
from bea_client import (
    BeaSeriesResult,
    fear_barometer_score,
    fetch_metro_annual_income_yoy,
    fetch_state_annual_pce_yoy,
    fetch_state_quarterly_income_yoy,
    gauge_figure,
)
from config import BEA_API_KEY, DB_PATH, DEFAULT_PROPERTY_TYPE, STATE_NAME_TO_POSTAL
from db import connect, fetch_series, init_db, list_regions_for_state, list_states
from fed_dti import (
    format_dti_period,
    format_dti_range,
    latest_msa_dti_row,
    latest_state_dti_row,
    load_dti_zip,
    MSA_ZIP_URL,
    STATE_ZIP_URL,
)
from ingest import (
    ingest_cities_for_state,
    ingest_metros,
    ingest_states,
    normalize_city_query,
    state_name_to_code,
)


def _conn():
    c = connect(DB_PATH)
    init_db(c)
    return c


def _figure_timeseries(df: pd.DataFrame, title: str) -> go.Figure:
    d = df.copy()
    d["period_begin"] = pd.to_datetime(d["period_begin"])
    d = d.sort_values("period_begin")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.38, 0.32, 0.30],
        subplot_titles=(
            "Median sale price",
            "Inventory & new listings",
            "Median days on market",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=d["period_begin"],
            y=d["median_sale_price"],
            name="Median sale price",
            mode="lines",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=d["period_begin"], y=d["inventory"], name="Inventory", opacity=0.75),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=d["period_begin"], y=d["new_listings"], name="New listings", opacity=0.65
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=d["period_begin"],
            y=d["median_dom"],
            name="Median DOM",
            mode="lines+markers",
            line=dict(width=2),
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        title=title,
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=24, r=24, t=80, b=24),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Days", row=3, col=1)
    return fig


def _state_postal_from_row(latest: pd.Series | None, sel_state: str) -> str:
    if latest is not None and pd.notna(latest.get("state_code")):
        return str(latest["state_code"]).strip().upper()
    key = sel_state.strip().lower()
    if key not in STATE_NAME_TO_POSTAL:
        raise KeyError(f"No postal code mapping for state {sel_state!r}")
    return STATE_NAME_TO_POSTAL[key]


def _metro_code_from_frame(df: pd.DataFrame) -> str | None:
    if df.empty or "metro_code" not in df.columns:
        return None
    s = df["metro_code"].dropna()
    if s.empty:
        return None
    v = str(s.iloc[-1]).strip()
    return v or None


@st.cache_data(ttl=3600, show_spinner=False)
def _bea_barometer_cached(
    geography: str,
    state_postal: str,
    metro_code: str | None,
    api_key: str,
) -> dict:
    pce = fetch_state_annual_pce_yoy(api_key, state_postal)
    if geography == "State":
        inc = fetch_state_quarterly_income_yoy(api_key, state_postal)
        scope_note = (
            "**Income:** state quarterly personal income (latest quarter vs same quarter a year earlier). "
            "**Spending:** state annual total personal consumption expenditures (PCE), latest year vs prior year."
        )
    else:
        if metro_code:
            inc = fetch_metro_annual_income_yoy(api_key, metro_code)
        else:
            inc = BeaSeriesResult(
                label="Metro personal income",
                latest_period=None,
                prior_year_period=None,
                latest_value=None,
                prior_value=None,
                yoy_pct=None,
                geo_name=None,
                table_used="MAINC1",
                error="No CBSA `metro_code` on file — re-run **Load metros** or **Load cities** to refresh Redfin metadata.",
            )
        scope_note = (
            "**Income:** metropolitan statistical area (CBSA) annual personal income (latest year vs prior). "
            "**Spending:** still **state-level** total PCE — BEA’s published PCE-by-state series does not map 1:1 to this metro."
        )
    fear, tone = fear_barometer_score(
        inc.yoy_pct if inc.error is None else None,
        pce.yoy_pct if pce.error is None else None,
    )
    return {
        "income": inc,
        "pce": pce,
        "fear": fear,
        "tone": tone,
        "scope_note": scope_note,
    }


@st.cache_data(ttl=86400 * 7, show_spinner=False)
def _fed_dti_frames():
    return load_dti_zip(STATE_ZIP_URL), load_dti_zip(MSA_ZIP_URL)


st.set_page_config(
    page_title="Housing market tracker",
    page_icon="",
    layout="wide",
)

st.title("Local housing market tracker")
st.caption(
    "Uses [Redfin Data Center](https://www.redfin.com/news/data-center/) public monthly metrics "
    "(median sale price, inventory, new listings, days on market, months of supply). "
    "**Not** live MLS; updates monthly."
)

with st.sidebar:
    st.header("Data refresh")
    if st.button("Refresh statewide data (fast ~9 MB)", use_container_width=True):
        with st.spinner("Downloading state-level file…"):
            try:
                n = ingest_states(DB_PATH)
                st.success(f"Upserted {n:,} state-month rows.")
            except Exception as e:
                st.error(f"State refresh failed: {e}")

    _st_conn = _conn()
    _states_sidebar = list_states(_st_conn)
    _st_conn.close()
    sel_state = st.selectbox(
        "State (full name, e.g. Utah)",
        options=_states_sidebar or ["— load state data first —"],
        index=0,
        disabled=not _states_sidebar,
    )
    if not _states_sidebar:
        sel_state = ""

    if sel_state and st.button(
        "Load metros for this state (~100 MB total file)",
        use_container_width=True,
    ):
        code = state_name_to_code(sel_state, DB_PATH)
        if not code:
            st.error("Could not resolve state code; refresh state data first.")
        else:
            with st.spinner(f"Downloading metros in {code}…"):
                try:
                    n = ingest_metros(DB_PATH, state_code=code)
                    st.success(f"Upserted {n:,} metro-month rows for {code}.")
                except Exception as e:
                    st.error(f"Metro load failed: {e}")

    if sel_state and st.button(
        "Load cities for this state (streams ~1 GB national file)",
        use_container_width=True,
    ):
        st.warning(
            "Redfin’s city file is large; this streams the full compressed file and keeps "
            f"only **{sel_state}**. It can take several minutes."
        )
        prog = st.progress(0.0, text="Starting…")

        def on_prog(rows_seen: int, rows_kept: int):
            prog.progress(
                min(0.99, rows_seen / 6_000_000.0),
                text=f"Scanned ~{rows_seen:,} source rows · kept {rows_kept:,} for {sel_state}",
            )

        try:
            n = ingest_cities_for_state(DB_PATH, sel_state, progress=on_prog)
            prog.progress(1.0, text="Done")
            st.success(f"Upserted {n:,} city-month rows for {sel_state}.")
        except Exception as e:
            st.error(f"City load failed: {e}")

    st.divider()
    level = st.radio(
        "Geography",
        options=["State", "Metro", "City"],
        horizontal=False,
    )

conn = _conn()
states = list_states(conn)
if not states:
    st.info(
        'Use the sidebar button **"Refresh statewide data"** to download Redfin state metrics, '
        "then pick a state."
    )
    conn.close()
    st.stop()

if not sel_state or sel_state.startswith("—"):
    sel_state = states[0]

metro_options: list[str] = []
city_options: list[str] = []
if level == "Metro":
    metro_options = list_regions_for_state(conn, "metro", sel_state)
elif level == "City":
    city_options = list_regions_for_state(conn, "place", sel_state)

region: str | None = None
if level == "State":
    region_type = "state"
    region = sel_state
    label = f"{sel_state} (statewide)"
elif level == "Metro":
    region_type = "metro"
    if not metro_options:
        st.warning(
            f"No metro rows in the database for **{sel_state}** yet. "
            "Click **Load metros for this state** in the sidebar."
        )
        conn.close()
        st.stop()
    region = st.selectbox("Metro area", options=metro_options, key="metro_pick")
    label = region or sel_state
else:
    region_type = "place"
    if not city_options:
        st.warning(
            f"No city rows for **{sel_state}** yet. "
            "Click **Load cities for this state** (large download) in the sidebar."
        )
        conn.close()
        st.stop()
    filter_q = st.text_input("Filter cities (optional)", "")
    opts = city_options
    if filter_q.strip():
        qn = normalize_city_query(filter_q)
        opts = [c for c in city_options if qn in c.lower()]
    region = st.selectbox("City", options=opts if opts else city_options, key="city_pick")
    label = region or sel_state

df = fetch_series(
    conn,
    region_type=region_type,
    state=sel_state,
    region=None if level == "State" else region,
    property_type=DEFAULT_PROPERTY_TYPE,
)
conn.close()

if df.empty:
    st.error("No series returned for this selection. Try refreshing the relevant dataset.")
    st.stop()

latest = latest_row(df)
mom = latest["median_sale_price_mom"] if latest is not None else None
try:
    mom_f = float(mom) if mom is not None and pd.notna(mom) else None
except (TypeError, ValueError):
    mom_f = None
price_trend = trend_from_change(mom_f)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Price trend (MoM)", price_trend.replace("unknown", "n/a").title())
with c2:
    v = latest["median_sale_price"] if latest is not None else None
    st.metric(
        "Median sale (latest month)",
        f"${float(v):,.0f}" if v is not None and pd.notna(v) else "—",
    )
with c3:
    v = latest["median_dom"] if latest is not None else None
    st.metric(
        "Median days on market",
        f"{int(v)}" if v is not None and pd.notna(v) else "—",
    )
with c4:
    v = latest["months_of_supply"] if latest is not None else None
    st.metric(
        "Months of supply",
        f"{float(v):.1f}" if v is not None and pd.notna(v) else "—",
    )

st.divider()
st.subheader("Household earning & spending barometer")
st.caption(
    "A simple macro “pressure” gauge from **Bureau of Economic Analysis** regional accounts — "
    "not options flow like Unusual Whales. **Higher reading = more pressure** (weaker growth in income or spending vs typical)."
)

if not BEA_API_KEY:
    st.info("Add `BEA_API_KEY` to your `.env` file to load BEA income & PCE data here.")
else:
    try:
        postal = _state_postal_from_row(latest, sel_state)
    except KeyError:
        postal = ""
        st.warning("Could not resolve a state postal code for BEA; check your housing row `state_code`.")

    if postal:
        mc = _metro_code_from_frame(df) if level != "State" else None
        pack = _bea_barometer_cached(level, postal, mc, BEA_API_KEY)
        inc, pce = pack["income"], pack["pce"]
        if inc.error:
            st.warning(f"BEA income series: {inc.error}")
        if pce.error:
            st.warning(f"BEA spending series: {pce.error}")

        m1, m2 = st.columns(2)
        with m1:
            ip = inc.yoy_pct
            st.metric(
                "Earnings momentum (BEA)",
                f"{ip * 100:+.1f}% YoY" if ip is not None and pd.notna(ip) else "—",
                help=inc.label,
            )
        with m2:
            pp = pce.yoy_pct
            st.metric(
                "Spending momentum (PCE, BEA)",
                f"{pp * 100:+.1f}% YoY" if pp is not None and pd.notna(pp) else "—",
                help=pce.label,
            )

        st.caption(pack["scope_note"])

        g1, g2 = st.columns([1, 2])
        with g1:
            st.plotly_chart(
                gauge_figure(
                    pack["fear"],
                    income_period=inc.latest_period or "—",
                    pce_period=pce.latest_period or "—",
                ),
                use_container_width=True,
            )
        with g2:
            st.markdown(
                "**How to read it:** Green **Confident**, yellow **Uncertain**, red **Fear** — based on BEA "
                "income and state PCE growth vs simple benchmarks (not a trading signal). "
                "Metro/city income uses CBSA when Redfin provides `metro_code`."
            )

st.divider()
st.subheader("Debt-to-income ratio (budget tightness)")
st.caption(
    "[Federal Reserve Z.1 Data Visualization](https://www.federalreserve.gov/releases/z1/default.htm) "
    "— household debt-to-income **ranges** (low–high) by state and by MSA. Quarterly; higher mid-range ≈ tighter budgets."
)

try:
    state_dti_df, msa_dti_df = _fed_dti_frames()
    dti_left, dti_right = st.columns(2)
    st_dti = latest_state_dti_row(state_dti_df, sel_state)
    with dti_left:
        if st_dti is not None:
            st.markdown(f"**{sel_state}** (state)")
            st.metric(
                "Debt-to-income (range)",
                format_dti_range(st_dti),
                help="Fed low–high band for the state.",
            )
            st.caption(format_dti_period(st_dti))
        else:
            st.warning(f"No state DTI for **{sel_state}**.")

    mc_dti = _metro_code_from_frame(df) if level != "State" else None
    with dti_right:
        if level != "State" and mc_dti:
            m_row = latest_msa_dti_row(msa_dti_df, mc_dti)
            if m_row is not None:
                st.markdown(f"**MSA** `{mc_dti}`")
                st.metric(
                    "Debt-to-income (range)",
                    format_dti_range(m_row),
                    help="Fed low–high band for this metropolitan area.",
                )
                st.caption(format_dti_period(m_row))
            else:
                st.info(f"No MSA row for CBSA **{mc_dti}** in the Fed extract.")
        elif level != "State":
            st.caption("Re-ingest **metro/city** so `metro_code` is present for MSA DTI.")

    st.info(
        "**Rough read of the range (midpoint):** ~1.0–1.4 often **more comfortable**, ~1.5–1.8 **moderate pressure**, "
        "above ~**1.8** **tighter** budgets — interpretation is qualitative; see Fed documentation for definitions."
    )
except Exception as e:
    st.error(f"Could not load Federal Reserve DTI files: {e}")

st.markdown(summary_markdown(latest, label))

try:
    fig = _figure_timeseries(df, label)
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not build chart: {e}")

st.subheader("Recent months (table)")
show = df.sort_values("period_begin", ascending=False).head(24)
cols = [
    "period_begin",
    "median_sale_price",
    "median_sale_price_mom",
    "median_sale_price_yoy",
    "inventory",
    "inventory_mom",
    "new_listings",
    "new_listings_mom",
    "median_dom",
    "median_dom_mom",
    "median_dom_yoy",
    "months_of_supply",
    "homes_sold",
]
show = show[[c for c in cols if c in show.columns]]
display = show.copy()
if "median_sale_price" in display.columns:
    display["median_sale_price"] = display["median_sale_price"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
    )
pct_exclude = {"median_dom_mom", "median_dom_yoy"}
for c in display.columns:
    if (c.endswith("_mom") or c.endswith("_yoy")) and c not in pct_exclude:
        display[c] = display[c].apply(
            lambda x: f"{float(x) * 100:+.1f}%" if pd.notna(x) else "—"
        )
for c in (
    "inventory",
    "new_listings",
    "median_dom",
    "homes_sold",
):
    if c in display.columns:
        display[c] = display[c].apply(
            lambda x: f"{int(round(float(x)))}" if pd.notna(x) else "—"
        )
for c in ("median_dom_mom", "median_dom_yoy"):
    if c in display.columns:
        display[c] = show[c].apply(
            lambda x: f"{float(x):+.1f} d" if pd.notna(x) else "—"
        )
if "months_of_supply" in display.columns:
    display["months_of_supply"] = display["months_of_supply"].apply(
        lambda x: f"{float(x):.1f}" if pd.notna(x) else "—"
    )
st.dataframe(display, use_container_width=True, hide_index=True)

st.caption(
    "**Sources:** Housing — [Redfin Data Center](https://www.redfin.com/news/data-center/) · "
    "Economic pressure — BEA Regional API · "
    "Debt-to-income — [Federal Reserve Z.1 data download](https://www.federalreserve.gov/releases/z1/default.htm)."
)
