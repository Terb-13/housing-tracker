from __future__ import annotations

import re
from typing import Callable

import pandas as pd
import requests

from config import DEFAULT_PROPERTY_TYPE, REDFIN_URLS, STATE_CODE_TO_NAME
from db import connect, init_db, upsert_dataframe


def _fill_state_name_from_code(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "state_code" not in df.columns:
        return df
    df["state"] = df["state"].astype("object")
    empty = df["state"].isna() | (df["state"].astype(str).str.strip() == "")
    codes = df["state_code"].astype(str).str.strip().str.upper()
    df.loc[empty, "state"] = codes[empty].map(STATE_CODE_TO_NAME)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "median_sale_price",
        "median_sale_price_mom",
        "median_sale_price_yoy",
        "median_list_price",
        "median_list_price_mom",
        "median_list_price_yoy",
        "new_listings",
        "new_listings_mom",
        "new_listings_yoy",
        "inventory",
        "inventory_mom",
        "inventory_yoy",
        "months_of_supply",
        "months_of_supply_mom",
        "median_dom",
        "median_dom_mom",
        "median_dom_yoy",
        "homes_sold",
        "homes_sold_mom",
        "pending_sales",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    df = _coerce_numeric(df)
    if "property_type" in df.columns:
        df = df[df["property_type"] == DEFAULT_PROPERTY_TYPE]
    keep = [
        "period_begin",
        "period_end",
        "region_type",
        "region",
        "city",
        "state",
        "state_code",
        "property_type",
        "median_sale_price",
        "median_sale_price_mom",
        "median_sale_price_yoy",
        "median_list_price",
        "median_list_price_mom",
        "median_list_price_yoy",
        "new_listings",
        "new_listings_mom",
        "new_listings_yoy",
        "inventory",
        "inventory_mom",
        "inventory_yoy",
        "months_of_supply",
        "months_of_supply_mom",
        "median_dom",
        "median_dom_mom",
        "median_dom_yoy",
        "homes_sold",
        "homes_sold_mom",
        "pending_sales",
        "parent_metro_region",
        "parent_metro_region_metro_code",
        "last_updated",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = None

    def _metro_code_cell(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip()
        if not s or s.upper() in ("NA", "N/A"):
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        if f != f:
            return None
        return str(int(f))

    df["metro_code"] = df["parent_metro_region_metro_code"].map(_metro_code_cell)
    base = [c for c in keep if c != "parent_metro_region_metro_code"]
    li = base.index("last_updated")
    out_cols = base[:li] + ["metro_code"] + base[li:]
    return df[out_cols]


def download_gz_tsv(url: str, session: requests.Session | None = None) -> pd.DataFrame:
    sess = session or requests.Session()
    r = sess.get(url, timeout=600, stream=True)
    r.raise_for_status()
    return pd.read_csv(r.raw, sep="\t", compression="gzip", low_memory=False)


def ingest_states(
    db_path,
    session: requests.Session | None = None,
) -> int:
    df = download_gz_tsv(REDFIN_URLS["state"], session)
    df = _prepare_frame(df)
    df = df[df["region_type"].astype(str).str.lower() == "state"]
    conn = connect(db_path)
    init_db(conn)
    n = upsert_dataframe(conn, df)
    conn.close()
    return n


def ingest_metros(
    db_path,
    state_code: str | None = None,
    session: requests.Session | None = None,
) -> int:
    df = download_gz_tsv(REDFIN_URLS["metro"], session)
    df = _prepare_frame(df)
    df = df[df["region_type"].astype(str).str.lower() == "metro"]
    df = _fill_state_name_from_code(df)
    conn = connect(db_path)
    init_db(conn)
    if state_code:
        sc = state_code.strip().upper()
        conn.execute(
            """
            DELETE FROM market_data
            WHERE region_type = 'metro'
              AND upper(trim(coalesce(state_code, ''))) = ?
            """,
            (sc,),
        )
        conn.commit()
        df = df[df["state_code"].astype(str).str.upper() == sc]
    n = upsert_dataframe(conn, df)
    conn.close()
    return n


def ingest_cities_for_state(
    db_path,
    state_name: str,
    chunk_rows: int = 150_000,
    progress: Callable[[int, int], None] | None = None,
    session: requests.Session | None = None,
) -> int:
    """Stream the national city file; keep rows for one state (large download)."""
    sess = session or requests.Session()
    url = REDFIN_URLS["city"]
    r = sess.get(url, timeout=600, stream=True)
    r.raise_for_status()

    total_written = 0
    chunks: list[pd.DataFrame] = []
    seen = 0
    target = state_name.strip().lower()

    for chunk in pd.read_csv(
        r.raw,
        sep="\t",
        compression="gzip",
        low_memory=False,
        chunksize=chunk_rows,
    ):
        chunk = _normalize_columns(chunk)
        if "state" not in chunk.columns:
            seen += chunk.shape[0]
            if progress:
                progress(seen, total_written)
            continue
        st = chunk["state"].fillna("").astype(str).str.lower()
        sub = chunk[st == target]
        if not sub.empty:
            sub = _prepare_frame(sub)
            sub = sub[sub["region_type"].astype(str).str.lower() == "place"]
            if not sub.empty:
                chunks.append(sub)
                total_written += len(sub)
        seen += chunk.shape[0]
        if progress:
            progress(seen, total_written)

    if not chunks:
        conn = connect(db_path)
        init_db(conn)
        conn.close()
        return 0

    merged = pd.concat(chunks, ignore_index=True)
    conn = connect(db_path)
    init_db(conn)
    conn.execute(
        """
        DELETE FROM market_data
        WHERE region_type = 'place'
          AND lower(trim(state)) = lower(trim(?))
        """,
        (state_name.strip(),),
    )
    conn.commit()
    n = upsert_dataframe(conn, merged)
    conn.close()
    return n


def state_name_to_code(state_name: str, db_path) -> str | None:
    conn = connect(db_path)
    init_db(conn)
    row = conn.execute(
        """
        SELECT state_code FROM market_data
        WHERE region_type = 'state' AND lower(state) = lower(?)
        LIMIT 1
        """,
        (state_name.strip(),),
    ).fetchone()
    conn.close()
    if row and row[0]:
        return str(row[0]).upper()
    return None


def normalize_city_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
