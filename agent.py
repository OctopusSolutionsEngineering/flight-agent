"""Flight tracking agent."""
import logging
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache as LCInMemoryCache
from langchain_community.cache import RedisCache as LCRedisCache

from tools import FLIGHT_TOOLS
from config import get_settings

logger = logging.getLogger(__name__)


def _configure_llm_cache() -> None:
    settings = get_settings()
    if settings.cache_backend == "redis" and settings.redis_url:
        try:
            import redis
            client = redis.Redis.from_url(settings.redis_url)
            set_llm_cache(LCRedisCache(redis_=client))
            logger.info("LangChain LLM cache → Redis")
            return
        except Exception as e:
            logger.warning(f"Redis LLM cache failed: {e}")
    set_llm_cache(LCInMemoryCache())


SYSTEM_PROMPT = """You are a knowledgeable flight tracking assistant.

You have tools that can:
1. `find_airport` — Look up airports by IATA/ICAO code or name
2. `get_aircraft_by_callsign` — Find a specific flight in the air right now
3. `get_aircraft_near_location` — See planes flying near a location
4. `get_airport_arrivals` / `get_airport_departures` — Recent traffic at an airport
5. `get_flight_track` — Recent path of a specific aircraft
6. `get_flight_schedule` — Scheduled times, delays, gates (if Aviationstack enabled)

## Guidelines

- When the user mentions an airport by city/name, use `find_airport` first to get its code
- IATA codes are 3 letters (LHR, JFK, NRT), ICAO are 4 (EGLL, KJFK, RJAA)
- Flight callsigns are usually airline + number (e.g., 'BAW123' = British Airways 123)
- `icao24` is the unique transponder ID, not the airport code — looks like 'a8c456'
- Times from OpenSky are Unix timestamps — convert them to human-readable UTC
- If a search returns multiple airport matches, list a few and ask which one
- Live data has ~5–30 second latency from real-time

## Response Style

- Lead with the most important fact (e.g., "Yes, it's currently airborne over Iceland")
- Format altitudes in feet, speeds in knots AND km/h
- Include emojis sparingly: ✈️ 🛫 🛬 🌍
- Acknowledge data limitations (e.g., "OpenSky shows ADS-B receivers, so coverage varies")

If a tool returns an error like 'Rate limited', explain it and suggest waiting a minute.
Be specific about what's live vs scheduled data.
"""


def build_agent() -> AgentExecutor:
    settings = get_settings()
    _configure_llm_cache()
    
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
        cache=True,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, FLIGHT_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=FLIGHT_TOOLS,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )


_agent_executor = None

def get_agent() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = build_agent()
    return _agent_executor
