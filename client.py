"""Client for the Flight Tracking Agent."""
import os
import sys
import argparse
import requests


def ask(base_url: str, query: str, timeout: int = 60) -> None:
    response = requests.post(
        f"{base_url}/ask",
        json={"query": query},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    print(f"\n✈️  Agent: {data['answer']}")
    print(f"⏱️  Latency: {data['latency_ms']} ms"
          f"{' (cached)' if data.get('cached') else ''}\n")


def main():
    parser = argparse.ArgumentParser(description="Flight Tracking Agent client")
    parser.add_argument("query", nargs="*")
    parser.add_argument("--url", default=os.getenv("FLIGHT_AGENT_URL", "http://localhost:8000"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("-i", "--interactive", action="store_true")
    args = parser.parse_args()
    
    base = args.url.rstrip("/")
    
    try:
        requests.get(f"{base}/health", timeout=5).raise_for_status()
    except requests.exceptions.RequestException:
        print(f"❌ Could not connect to {base}")
        sys.exit(1)
    
    if args.interactive or not args.query:
        print("✈️  Flight Tracker — type 'quit' to exit\n")
        while True:
            try:
                q = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋")
                break
            if not q:
                continue
            if q.lower() in {"quit", "exit", "q"}:
                break
            try:
                ask(base, q, args.timeout)
            except requests.exceptions.HTTPError as e:
                print(f"❌ {e.response.text}\n")
    else:
        query = " ".join(args.query)
        print(f"✈️  You: {query}")
        ask(base, query, args.timeout)


if __name__ == "__main__":
    main()
