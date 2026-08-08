"""Wait until the production Advisor reports the exact rebuilt index."""

import argparse
import json
import time
from urllib.request import Request, urlopen


DEFAULT_HEALTH_URL = (
    "https://us-central1-ai-discovery-platform.cloudfunctions.net/"
    "proxyAdvisorRequest"
)


def read_health(url):
    request = Request(
        url,
        data=json.dumps({"data": {"path": "/health"}}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=90) as response:
        envelope = json.load(response)
    return envelope.get("result") or envelope.get("data") or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="index/index_manifest.json")
    parser.add_argument("--url", default=DEFAULT_HEALTH_URL)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--initial-delay", type=int, default=60)
    args = parser.parse_args()
    with open(args.manifest, encoding="utf-8") as handle:
        expected = json.load(handle)
    expected_built_at = expected.get("built_at")
    expected_rows = expected.get("rows")
    if not expected_built_at:
        raise ValueError("Index manifest has no built_at version.")

    time.sleep(max(0, args.initial_delay))
    deadline = time.monotonic() + args.timeout
    last = {}
    while time.monotonic() < deadline:
        try:
            last = read_health(args.url)
            live_manifest = last.get("index_manifest") or {}
            if (
                last.get("ok") is True
                and last.get("vectors_loaded") is True
                and live_manifest.get("built_at") == expected_built_at
                and live_manifest.get("rows") == expected_rows
            ):
                print(
                    "Production Advisor is serving index "
                    f"{expected_built_at} with {expected_rows} tools."
                )
                return
        except Exception as error:
            last = {"error": str(error)}
        print("Waiting for production Advisor deployment...")
        time.sleep(30)
    raise TimeoutError(
        "Production Advisor did not serve the rebuilt manifest in time: "
        + json.dumps(last, default=str)[:500]
    )


if __name__ == "__main__":
    main()
