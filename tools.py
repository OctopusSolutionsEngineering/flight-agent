"""Flight tracking tools using OpenSky + optional Aviationstack."""
import time
import logging
from typing import Optional
import requests
from langchain_core.tools import tool

from config import get_settings
from cache import get_cache, make_cache_key
from airports import lookup_airport, search_airport

logger = logging.getLogger(__name__)

OPENSKY_BASE = "https://opensky-network.org/api"


def _opensky_auth() -> Optional[tuple]:
    """Return (user, pass) tuple if credentials are configured, else None."""
    s = get_settings()
    if s.opensky_username and s.opensky_password:
        return (s.opensky_username, s.opensky_password)
    return None


def _cached_call(namespace: str, ttl: int, fn, *args, **kwargs):
    settings = get_settings()
    if not settings.feature_tool_cache:
        return fn(*args, **kwargs)
    cache = get_cache()
    key = make_cache_key(namespace, *args, **kwargs)
    cached = cache.get(key)
    if cached is not None:
        logger.info(f"🎯 Cache HIT: {namespace}")
        return cached
    logger.info(f"💨 Cache MISS: {namespace}")
    result = fn(*args, **kwargs)
    if isinstance(result, dict) and "error" not in result:
        cache.set(key, result, ttl)
    return result


# ============================================================
# Airport lookup tools
# ============================================================

@tool
def find_airport(query: str) -> dict:
    """Find an airport by IATA code (LHR), ICAO code (EGLL), or name/city.
    
    Args:
        query: An IATA code (3 letters), ICAO code (4 letters), or city/name.
    
    Returns:
        Airport details including coordinates, or a list of matches for fuzzy searches.
    """
    # Try direct lookup first
    code = query.strip()
    if len(code) in (3, 4) and code.isalpha():
        airport = lookup_airport(code)
        if airport:
            return airport
    
    # Fuzzy search
    matches = search_airport(query, limit=5)
    if not matches:
        return {"error": f"No airport found for '{query}'"}
    if len(matches) == 1:
        return matches[0]
    return {"matches": matches, "count": len(matches)}


# ============================================================
# Live aircraft state (OpenSky)
# ============================================================

# OpenSky's /states/all returns this column order. See API docs.
STATE_FIELDS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source",
]


def _parse_state_vector(vec: list) -> dict:
    """Turn OpenSky's raw array into a dict with readable fields."""
    result = dict(zip(STATE_FIELDS, vec))
    # Cleanups
    if result.get("callsign"):
        result["callsign"] = result["callsign"].strip()
    # Convert m/s to km/h and knots
    if result.get("velocity"):
        result["velocity_kmh"] = round(result["velocity"] * 3.6, 1)
        result["velocity_knots"] = round(result["velocity"] * 1.94384, 1)
    # Convert m to ft
    if result.get("baro_altitude"):
        result["altitude_ft"] = round(result["baro_altitude"] * 3.28084)
    return result


def _fetch_aircraft_by_callsign(callsign: str) -> dict:
    """Find a specific aircraft by callsign across all states."""
    try:
        response = requests.get(
            f"{OPENSKY_BASE}/states/all",
            auth=_opensky_auth(),
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        callsign = callsign.strip().upper()
        for vec in (data.get("states") or []):
            if vec[1] and vec[1].strip().upper() == callsign:
                return {
                    "found": True,
                    "timestamp": data["time"],
                    "aircraft": _parse_state_vector(vec),
                }
        return {"found": False, "callsign": callsign}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return {"error": "Rate limited — try again in a minute"}
        return {"error": f"OpenSky error: {e}"}
    except Exception as e:
        return {"error": f"Failed to fetch state: {e}"}


@tool
def get_aircraft_by_callsign(callsign: str) -> dict:
    """Find a specific aircraft in flight right now by its callsign.
    
    Args:
        callsign: The flight callsign (e.g., 'BAW123', 'UAL456').
    
    Returns:
        Current position, altitude, speed, and heading — or 'not found' if not airborne.
    """
    ttl = get_settings().cache_ttl_live_state
    return _cached_call("aircraft_callsign", ttl, _fetch_aircraft_by_callsign, callsign)


def _fetch_aircraft_in_bbox(
    min_lat: float, max_lat: float, min_lon: float, max_lon: float
) -> dict:
    """Fetch all aircraft in a geographic bounding box."""
    try:
        params = {
            "lamin": min_lat,
            "lamax": max_lat,
            "lomin": min_lon,
            "lomax": max_lon,
        }
        response = requests.get(
            f"{OPENSKY_BASE}/states/all",
            params=params,
            auth=_opensky_auth(),
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        states = data.get("states") or []
        aircraft = [_parse_state_vector(v) for v in states[:50]]  # cap at 50
        return {
            "timestamp": data["time"],
            "count": len(aircraft),
            "total_in_area": len(states),
            "aircraft": aircraft,
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return {"error": "Rate limited — try again in a minute"}
        return {"error": f"OpenSky error: {e}"}
    except Exception as e:
        return {"error": f"Failed to fetch states: {e}"}


@tool
def get_aircraft_near_location(
    latitude: float,
    longitude: float,
    radius_km: float = 50,
) -> dict:
    """Get all aircraft currently flying within a radius of a location.
    
    Args:
        latitude: Center latitude.
        longitude: Center longitude.
        radius_km: Radius in kilometers (max 200 recommended).
    
    Returns:
        List of aircraft with positions and callsigns.
    """
    radius_km = min(radius_km, 200)
    # ~111 km per degree of latitude (longitude varies but close enough for small radii)
    delta = radius_km / 111.0
    
    ttl = get_settings().cache_ttl_live_state
    return _cached_call(
        "aircraft_bbox", ttl, _fetch_aircraft_in_bbox,
        round(latitude - delta, 3),
        round(latitude + delta, 3),
        round(longitude - delta, 3),
        round(longitude + delta, 3),
    )


# ============================================================
# Airport activity (OpenSky)
# ============================================================

def _fetch_airport_flights(
    icao: str, kind: str, hours_ago: int
) -> dict:
    """Get arrivals or departures for an airport in the recent past."""
    now = int(time.time())
    end = now
    begin = now - (hours_ago * 3600)
    
    # OpenSky limits time range to 7 days
    if hours_ago > 24 * 7:
        return {"error": "hours_ago cannot exceed 168 (7 days)"}
    
    endpoint = "arrival" if kind == "arrival" else "departure"
    try:
        response = requests.get(
            f"{OPENSKY_BASE}/flights/{endpoint}",
            params={"airport": icao.upper(), "begin": begin, "end": end},
            auth=_opensky_auth(),
            timeout=20,
        )
        if response.status_code == 404:
            return {"flights": [], "count": 0, "note": "No flights found in window"}
        response.raise_for_status()
        flights = response.json() or []
        
        # Trim verbose fields and format times
        simplified = []
        for f in flights[:30]:  # cap to keep response small
            simplified.append({
                "callsign": (f.get("callsign") or "").strip(),
                "icao24": f.get("icao24"),
                "from": f.get("estDepartureAirport"),
                "to": f.get("estArrivalAirport"),
                "departure_time": f.get("firstSeen"),
                "arrival_time": f.get("lastSeen"),
            })
        return {"flights": simplified, "count": len(simplified), "kind": kind}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return {"error": "Rate limited"}
        return {"error": f"OpenSky error: {e}"}
    except Exception as e:
        return {"error": f"Failed: {e}"}


@tool
def get_airport_arrivals(airport_code: str, hours_ago: int = 2) -> dict:
    """Get aircraft that recently arrived at an airport.
    
    Args:
        airport_code: IATA (e.g., 'LHR') or ICAO (e.g., 'EGLL') code.
        hours_ago: How many hours back to look (max 168 = 7 days).
    
    Returns:
        List of arrival flights with timing and origin airports.
    """
    airport = lookup_airport(airport_code)
    if not airport:
        return {"error": f"Unknown airport: {airport_code}"}
    
    ttl = get_settings().cache_ttl_flights_by_airport
    return _cached_call(
        "airport_arrivals", ttl,
        _fetch_airport_flights, airport["icao"], "arrival", hours_ago,
    )


@tool
def get_airport_departures(airport_code: str, hours_ago: int = 2) -> dict:
    """Get aircraft that recently departed from an airport.
    
    Args:
        airport_code: IATA (e.g., 'JFK') or ICAO (e.g., 'KJFK') code.
        hours_ago: How many hours back to look (max 168 = 7 days).
    """
    airport = lookup_airport(airport_code)
    if not airport:
        return {"error": f"Unknown airport: {airport_code}"}
    
    ttl = get_settings().cache_ttl_flights_by_airport
    return _cached_call(
        "airport_departures", ttl,
        _fetch_airport_flights, airport["icao"], "departure", hours_ago,
    )


# ============================================================
# Flight track history (OpenSky)
# ============================================================

def _fetch_track(icao24: str) -> dict:
    try:
        response = requests.get(
            f"{OPENSKY_BASE}/tracks/all",
            params={"icao24": icao24.lower(), "time": 0},  # 0 = current/live track
            auth=_opensky_auth(),
            timeout=20,
        )
        if response.status_code == 404:
            return {"error": "No track available for this aircraft"}
        response.raise_for_status()
        data = response.json()
        # Trim path to ~20 waypoints
        path = data.get("path", [])
        if len(path) > 20:
            step = len(path) // 20
            path = path[::step][:20]
        waypoints = [
            {"time": p[0], "latitude": p[1], "longitude": p[2],
             "altitude_m": p[3], "heading": p[4], "on_ground": p[5]}
            for p in path
        ]
        return {
            "icao24": data.get("icao24"),
            "callsign": (data.get("callsign") or "").strip(),
            "start_time": data.get("startTime"),
            "end_time": data.get("endTime"),
            "waypoints": waypoints,
            "waypoint_count": len(waypoints),
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return {"error": "Rate limited"}
        return {"error": f"OpenSky error: {e}"}
    except Exception as e:
        return {"error": f"Failed: {e}"}


@tool
def get_flight_track(icao24: str) -> dict:
    """Get the recent flight path of an aircraft by its ICAO24 transponder ID.
    
    Args:
        icao24: The 24-bit ICAO transponder address (6 hex chars, e.g., 'a8c456').
    
    Returns:
        A series of waypoints showing the aircraft's path.
    """
    ttl = get_settings().cache_ttl_flight_track
    return _cached_call("flight_track", ttl, _fetch_track, icao24)


# ============================================================
# Aviationstack (optional — for schedules and delays)
# ============================================================

def _fetch_flight_schedule(flight_iata: str) -> dict:
    settings = get_settings()
    if not settings.feature_aviationstack or not settings.aviationstack_api_key:
        return {"error": "Aviationstack is not enabled. Use OpenSky tools for live data."}
    
    try:
        response = requests.get(
            "http://api.aviationstack.com/v1/flights",
            params={
                "access_key": settings.aviationstack_api_key,
                "flight_iata": flight_iata.upper(),
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            return {"error": data["error"].get("message", "Aviationstack error")}
        
        flights = data.get("data", [])
        if not flights:
            return {"error": f"No schedule found for {flight_iata}"}
        
        # Return the first matching flight (simplified)
        f = flights[0]
        return {
            "flight_number": f.get("flight", {}).get("iata"),
            "airline": f.get("airline", {}).get("name"),
            "status": f.get("flight_status"),
            "departure": {
                "airport": f.get("departure", {}).get("airport"),
                "iata": f.get("departure", {}).get("iata"),
                "scheduled": f.get("departure", {}).get("scheduled"),
                "estimated": f.get("departure", {}).get("estimated"),
                "delay_minutes": f.get("departure", {}).get("delay"),
                "terminal": f.get("departure", {}).get("terminal"),
                "gate": f.get("departure", {}).get("gate"),
            },
            "arrival": {
                "airport": f.get("arrival", {}).get("airport"),
                "iata": f.get("arrival", {}).get("iata"),
                "scheduled": f.get("arrival", {}).get("scheduled"),
                "estimated": f.get("arrival", {}).get("estimated"),
                "delay_minutes": f.get("arrival", {}).get("delay"),
                "terminal": f.get("arrival", {}).get("terminal"),
                "gate": f.get("arrival", {}).get("gate"),
            },
        }
    except Exception as e:
        return {"error": f"Aviationstack failed: {e}"}


@tool
def get_flight_schedule(flight_iata: str) -> dict:
    """Get scheduled departure/arrival times and delay info for a flight.
    
    Only available if Aviationstack is configured. Use for booking-style data
    (gate, terminal, scheduled vs actual times).
    
    Args:
        flight_iata: IATA flight code (e.g., 'BA123', 'UA456').
    """
    ttl = 300  # 5 min — schedules update gradually
    return _cached_call("flight_schedule", ttl, _fetch_flight_schedule, flight_iata)


# ============================================================
# Tool registry
# ============================================================

FLIGHT_TOOLS = [
    find_airport,
    get_aircraft_by_callsign,
    get_aircraft_near_location,
    get_airport_arrivals,
    get_airport_departures,
    get_flight_track,
    get_flight_schedule,
]
