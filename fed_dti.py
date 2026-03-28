"""Federal Reserve Z.1 Data Visualization — household debt-to-income by state and MSA."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from bea_client import POSTAL_TO_STATE_FIPS
from config import STATE_NAME_TO_POSTAL

STATE_ZIP_URL = (
    "https://www.federalreserve.gov/releases/z1/dataviz/download/zips/household-debt-by-state.zip"
)
MSA_ZIP_URL = (
    "https://www.federalreserve.gov/releases/z1/dataviz/download/zips/household-debt-by-msa.zip"
)


def load_dti_zip(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = [f for f in z.namelist() if f.lower().endswith(".csv")]
        if not names:
            raise ValueError("No CSV found in zip")
        with z.open(names[0]) as f:
            return pd.read_csv(f)


def state_name_to_fips_int(state_name: str) -> int:
    postal = STATE_NAME_TO_POSTAL[state_name.strip().lower()]
    return int(POSTAL_TO_STATE_FIPS[postal.upper()])


def latest_state_dti_row(df: pd.DataFrame, state_name: str) -> pd.Series | None:
    try:
        fid = state_name_to_fips_int(state_name)
    except KeyError:
        return None
    sub = df.loc[df["state_fips"].astype(int) == fid].copy()
    if sub.empty:
        return None
    sub = sub.assign(_y=sub["year"].astype(int), _q=sub["qtr"].astype(int))
    sub = sub.sort_values(["_y", "_q"], ascending=False)
    return sub.iloc[0]


def latest_msa_dti_row(df: pd.DataFrame, cbsa_code: str | None) -> pd.Series | None:
    if not cbsa_code:
        return None
    try:
        cid = int(str(cbsa_code).strip().split(".")[0])
    except ValueError:
        return None
    sub = df.loc[df["cbsa"].astype(int) == cid].copy()
    if sub.empty:
        return None
    sub = sub.assign(_y=sub["year"].astype(int), _q=sub["qtr"].astype(int))
    sub = sub.sort_values(["_y", "_q"], ascending=False)
    return sub.iloc[0]


def format_dti_period(row: pd.Series) -> str:
    return f"{int(row['year'])} Q{int(row['qtr'])}"


def format_dti_range(row: pd.Series) -> str:
    low = float(row["low"])
    high = float(row["high"])
    return f"{low:.2f} – {high:.2f}"
