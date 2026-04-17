import requests


API = "https://comai-recommender-1.onrender.com"

# Verifies that the system is running and has at least 1 tool


def test_health():
    r = requests.get(f"{API}/health", timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert data["items"] > 0

# It verifies that the system returns exactly 5 results


def test_recommend_returns_five_results():
    r = requests.post(f"{API}/recommend",
                      json={"q": "I need a tool to transcribe audio", "retrieve_k": 30, "final_k": 5},
                      timeout=60)
    assert r.status_code == 200
    hits = r.json().get("hits", [])
    assert len(hits) == 5

# Verifies that each tool has got all fields with info and a geenrated reason


def test_recommend_results_have_required_fields():
    r = requests.post(f"{API}/recommend",
                      json={"q": "I need a free image generator", "retrieve_k": 30, "final_k": 5},
                      timeout=60)
    hits = r.json().get("hits", [])
    for hit in hits:
        assert "meta" in hit
        assert "why" in hit
        assert hit["meta"].get("Name") != ""

# Verifies the clarify endpoint returns a valid action


def test_clarify_returns_valid_action():
    r = requests.post(f"{API}/clarify",
                      json={"q": "I need a tool"},
                      timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert data["action"] in ("clarify", "search")

# Verifies that system returns HTTP 400 for empty query


def test_recommend_empty_query_returns_error():
    r = requests.post(f"{API}/recommend",
                      json={"q": "", "retrieve_k": 30, "final_k": 5},
                      timeout=30)
    assert r.status_code == 400

# Verifies a recommendation request returns within 7 seconds


def test_recommend_latency():
    import time
    start = time.time()
    requests.post(f"{API}/recommend",
                  json={"q": "I need a coding assistant", "retrieve_k": 30, "final_k": 5}, timeout=60)
    elapsed = time.time() - start
    assert elapsed < 7, f"Response took {elapsed:.2f}s, exceeding 7s NFR1 target"
