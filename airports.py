"""Airport database lookup (IATA/ICAO codes → coordinates)."""
import csv
import os
import logging
from functools import lru_cache
from typing import Optional
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# OurAirports.com publishes a free, comprehensive airport database
AIRPORTS_CSV_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
DATA_DIR = Path(__file__).parent / "data"
AIRPORTS_FILE = DATA_DIR / "airports.csv"


def _ensure_airports_data() -> None:
    """Download the airport database if missing."""
    if AIRPORTS_FILE.exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    logger.info(f"Downloading airports database from {AIRPORTS_CSV_URL}")
    response = requests.get(AIRPORTS_CSV_URL, timeout=60)
    response.raise_for_status()
    AIRPORTS_FILE.write_bytes(response.content)
    logger.info(f"Saved to {AIRPORTS_FILE}")


@lru_cache(maxsize=1)
def _load_airports() -> dict[str, dict]:
    """Load airports into two lookup dicts (by IATA and ICAO)."""
    _ensure_airports_data()
    
    iata_index: dict[str, dict] = {}
    icao_index: dict[str, dict] = {}
    
    with open(AIRPORTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter out heliports, closed airports, small fields
            if row.get("type") not in {"large_airport", "medium_airport"}:
                continue
            if row.get("scheduled_service") != "yes":
                continue
            
            record = {
                "name": row.get("name", ""),
                "iata": row.get("iata_code", "").upper(),
                "icao": row.get("ident", "").upper(),
                "city": row.get("municipality", ""),
                "country": row.get("iso_country", ""),
                "latitude": float(row["latitude_deg"]) if row.get("latitude_deg") else None,
                "longitude": float(row["longitude_deg"]) if row.get("longitude_deg") else None,
                "elevation_ft": int(row["elevation_ft"]) if row.get("elevation_ft") else None,
            }
            
            if record["iata"]:
                iata_index[record["iata"]] = record
            if record["icao"]:
                icao_index[record["icao"]] = record
    
    logger.info(f"Loaded {len(icao_index)} airports ({len(iata_index)} with IATA codes)")
    return {"iata": iata_index, "icao": icao_index}


def lookup_airport(code: str) -> Optional[dict]:
    """Look up an airport by IATA (3 chars) or ICAO (4 chars) code."""
    code = code.strip().upper()
    db = _load_airports()
    
    if len(code) == 3:
        return db["iata"].get(code)
    elif len(code) == 4:
        return db["icao"].get(code)
    return None


def search_airport(query: str, limit: int = 5) -> list[dict]:
    """Fuzzy search airports by name or city."""
    query = query.strip().lower()
    db = _load_airports()
    matches = []
    seen = set()
    
    for airport in db["icao"].values():
        if airport["icao"] in seen:
            continue
        haystack = f"{airport['name']} {airport['city']}".lower()
        if query in haystack:
            matches.append(airport)
            seen.add(airport["icao"])
            if len(matches) >= limit:
                break
    
    return matches
