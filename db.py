import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_data (
    period_begin TEXT NOT NULL,
    period_end TEXT,
    region_type TEXT NOT NULL,
    region TEXT NOT NULL,
    city TEXT,
    state TEXT NOT NULL,
    state_code TEXT,
    property_type TEXT NOT NULL,
    median_sale_price REAL,
    median_sale_price_mom REAL,
    median_sale_price_yoy REAL,
    median_list_price REAL,
    median_list_price_mom REAL,
    median_list_price_yoy REAL,
    new_listings REAL,
    new_listings_mom REAL,
    new_listings_yoy REAL,
    inventory REAL,
    inventory_mom REAL,
    inventory_yoy REAL,
    months_of_supply REAL,
    months_of_supply_mom REAL,
    median_dom REAL,
    median_dom_mom REAL,
    median_dom_yoy REAL,
    homes_sold REAL,
    homes_sold_mom REAL,
    pending_sales REAL,
    parent_metro_region TEXT,
    last_updated TEXT,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (period_begin, region_type, region, state, property_type)
);
CREATE INDEX IF NOT EXISTS idx_market_state_period
    ON market_data (state, period_begin);
CREATE INDEX IF NOT EXISTS idx_market_region_type
    ON market_data (region_type, state, region);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open SQLite; if the path is corrupt or not a DB file, delete and recreate."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    def _open() -> sqlite3.Connection:
        c = sqlite3.connect(db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    conn = _open()
    try:
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
    except sqlite3.DatabaseError:
        conn.close()
        try:
            db_path.unlink(missing_ok=True)
        except OSError:
            pass
        conn = _open()
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(market_data)").fetchall()}
    if "metro_code" not in cols:
        conn.execute("ALTER TABLE market_data ADD COLUMN metro_code TEXT")
    conn.commit()


def _df_to_rows(df: pd.DataFrame) -> list[tuple[Any, ...]]:
    cols = [
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
        "metro_code",
        "last_updated",
        "ingested_at",
    ]
    now = datetime.now(timezone.utc).isoformat()
    out: list[tuple[Any, ...]] = []
    for _, row in df.iterrows():
        vals = [row.get(c) for c in cols[:-1]]
        vals.append(now)
        cleaned = tuple(
            None if (isinstance(v, float) and pd.isna(v)) or pd.isna(v) else v
            for v in vals
        )
        out.append(cleaned)
    return out


def upsert_dataframe(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = _df_to_rows(df)
    conn.executemany(
        """
        INSERT INTO market_data (
            period_begin, period_end, region_type, region, city, state, state_code,
            property_type, median_sale_price, median_sale_price_mom, median_sale_price_yoy,
            median_list_price, median_list_price_mom, median_list_price_yoy,
            new_listings, new_listings_mom, new_listings_yoy,
            inventory, inventory_mom, inventory_yoy,
            months_of_supply, months_of_supply_mom,
            median_dom, median_dom_mom, median_dom_yoy,
            homes_sold, homes_sold_mom, pending_sales,
            parent_metro_region, metro_code, last_updated, ingested_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(period_begin, region_type, region, state, property_type) DO UPDATE SET
            period_end=excluded.period_end,
            city=excluded.city,
            state_code=excluded.state_code,
            median_sale_price=excluded.median_sale_price,
            median_sale_price_mom=excluded.median_sale_price_mom,
            median_sale_price_yoy=excluded.median_sale_price_yoy,
            median_list_price=excluded.median_list_price,
            median_list_price_mom=excluded.median_list_price_mom,
            median_list_price_yoy=excluded.median_list_price_yoy,
            new_listings=excluded.new_listings,
            new_listings_mom=excluded.new_listings_mom,
            new_listings_yoy=excluded.new_listings_yoy,
            inventory=excluded.inventory,
            inventory_mom=excluded.inventory_mom,
            inventory_yoy=excluded.inventory_yoy,
            months_of_supply=excluded.months_of_supply,
            months_of_supply_mom=excluded.months_of_supply_mom,
            median_dom=excluded.median_dom,
            median_dom_mom=excluded.median_dom_mom,
            median_dom_yoy=excluded.median_dom_yoy,
            homes_sold=excluded.homes_sold,
            homes_sold_mom=excluded.homes_sold_mom,
            pending_sales=excluded.pending_sales,
            parent_metro_region=excluded.parent_metro_region,
            metro_code=excluded.metro_code,
            last_updated=excluded.last_updated,
            ingested_at=excluded.ingested_at
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def fetch_series(
    conn: sqlite3.Connection,
    *,
    region_type: str,
    state: str,
    region: str | None = None,
    property_type: str,
) -> pd.DataFrame:
    q = """
        SELECT * FROM market_data
        WHERE region_type = ? AND state = ? AND property_type = ?
    """
    params: list[Any] = [region_type, state, property_type]
    if region is not None:
        q += " AND region = ?"
        params.append(region)
    q += " ORDER BY period_begin ASC"
    return pd.read_sql_query(q, conn, params=params)


def list_states(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        """
        SELECT DISTINCT state FROM market_data
        WHERE region_type = 'state'
        ORDER BY state
        """
    )
    return [r[0] for r in cur.fetchall() if r[0]]


def list_regions_for_state(
    conn: sqlite3.Connection, region_type: str, state: str
) -> list[str]:
    cur = conn.execute(
        """
        SELECT DISTINCT region FROM market_data
        WHERE region_type = ? AND state = ?
        ORDER BY region
        """,
        (region_type, state),
    )
    return [r[0] for r in cur.fetchall() if r[0]]
