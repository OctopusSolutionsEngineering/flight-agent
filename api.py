"""FastAPI service for the flight tracking agent."""
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import get_agent
from cache import get_cache, make_cache_key
from config import get_settings, refresh_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting flight-agent (model={settings.openai_model})")
    logger.info(f"OpenSky auth: {'authenticated' if settings.opensky_username else 'anonymous'}")
    logger.info(f"Aviationstack: {'enabled' if settings.feature_aviationstack else 'disabled'}")
    
    # Warm up airport database
    from airports import _load_airports
    _load_airports()
    
    get_cache()
    get_agent()
    logger.info("Agent ready ✓")
    yield


app = FastAPI(title="Flight Tracking Agent", version="1.0.0", lifespan=lifespan)
logger = logging.getLogger(__name__)


class FlightQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500,
                       examples=["Is BA123 airborne right now?"])
    bypass_cache: bool = False


class FlightResponse(BaseModel):
    answer: str
    latency_ms: int
    cached: bool = False
    model_used: str


@app.get("/")
def root():
    return {"service": "flight-agent", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    try:
        settings = get_settings()
        return {
            "status": "ready",
            "model": settings.openai_model,
            "cache_backend": settings.cache_backend,
            "opensky_authenticated": bool(settings.opensky_username),
            "features": {
                "aviationstack": settings.feature_aviationstack,
                "response_cache": settings.feature_response_cache,
                "tool_cache": settings.feature_tool_cache,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ask", response_model=FlightResponse)
def ask_flight(payload: FlightQuery):
    start = time.time()
    settings = get_settings()
    cache = get_cache()
    
    # Response cache (short TTL — flight data is dynamic!)
    cache_key = None
    if settings.feature_response_cache and not payload.bypass_cache:
        normalized = payload.query.strip().lower()
        cache_key = make_cache_key("response", normalized, settings.openai_model)
        cached = cache.get(cache_key)
        if cached:
            return FlightResponse(
                answer=cached,
                latency_ms=int((time.time() - start) * 1000),
                cached=True,
                model_used=settings.openai_model,
            )
    
    try:
        agent = get_agent()
        result = agent.invoke({"input": payload.query})
        answer = result["output"]
        latency = int((time.time() - start) * 1000)
        
        if cache_key:
            # Use a short TTL for flight responses — even 60s is risky for live data
            cache.set(cache_key, answer, 30)
        
        return FlightResponse(
            answer=answer,
            latency_ms=latency,
            cached=False,
            model_used=settings.openai_model,
        )
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
def get_config():
    settings = get_settings()
    data = settings.model_dump()
    for key in list(data.keys()):
        if any(s in key.lower() for s in ("key", "secret", "password", "token")):
            if data[key]:
                data[key] = f"***{str(data[key])[-4:]}"
    return data


@app.post("/config/refresh")
def trigger_refresh():
    try:
        s = refresh_settings()
        return {"status": "refreshed", "model": s.openai_model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
def cache_stats():
    return get_cache().stats()


@app.post("/cache/clear")
def cache_clear():
    get_cache().clear()
    return {"status": "cleared"}
