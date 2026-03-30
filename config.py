import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
# Always load `.env` next to this package (Streamlit’s cwd may not be the project root).
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = Path(os.getenv("HOUSING_DATA_DIR", PROJECT_ROOT / "data"))
DB_PATH = Path(os.getenv("HOUSING_DB_PATH", PROJECT_ROOT / "housing_market.sqlite"))
BEA_API_KEY = (os.getenv("BEA_API_KEY") or "").strip()

REDFIN_BASE = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker"
REDFIN_URLS = {
    "state": f"{REDFIN_BASE}/state_market_tracker.tsv000.gz",
    "metro": f"{REDFIN_BASE}/redfin_metro_market_tracker.tsv000.gz",
    "city": f"{REDFIN_BASE}/city_market_tracker.tsv000.gz",
}

DEFAULT_PROPERTY_TYPE = "All Residential"

STATE_CODE_TO_NAME: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

STATE_NAME_TO_POSTAL: dict[str, str] = {
    name.lower(): code for code, name in STATE_CODE_TO_NAME.items()
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
