import requests
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

API = os.getenv("API_BASE_URL", "http://localhost:10000").rstrip("/")
client = OpenAI()

QUERIES = [
    "I need a tool to transcribe audio meetings",
    "I need a free AI tool to help me write essays",
    "I need a tool to generate images from text",
    "I need an AI tool to help me write and debug Python code",
    "I need a tool to create presentations automatically",
    "I need a free AI tool to help me learn a new language",
    "I need a tool to edit and enhance videos",
    "I need an AI tool to manage my social media posts",
    "I need a free AI tool to help me with university assignments",
    "I need a tool to generate a logo for my startup",
    "I need a free AI tool to build a website without coding"
]

GPT_SYSTEM = (
    "Recommend exactly 5 AI tools for the user's need. "
    "Do not ask clarifying questions. "
    "Return ONLY valid JSON: "
    "{\"recommendations\": [{\"name\": \"\", \"category\": \"\",\"description\": \"\", \"price\": \"\", \"reason\": \"\"}]}"
)

advisor_results = {}
gpt_results = {}
latencies = {}

for q in QUERIES:
    print(f"\nQuery: {q[:60]}")

    # AI Advisor
    try:
        start = time.time()
        r = requests.post(
            f"{API}/recommend",
            json={"q": q, "retrieve_k": 30, "final_k": 5},
            timeout=60
        )
        advisor_time = round(time.time() - start, 2)
        hits = r.json().get("hits", [])
        advisor_results[q] = [
            {
                "name": h["meta"].get("Name", ""),
                "categories": h["meta"].get("Categories", ""),
                "description": h["meta"].get("Description", ""),
                "price": h["meta"].get("Price", ""),
                "why": h.get("why", "")
            }
            for h in hits
        ]
        print(f"  AI Advisor: {len(hits)} results in {advisor_time}s")
    except Exception as e:
        print(f"  AI Advisor error: {e}")
        advisor_results[q] = []
        advisor_time = None

    time.sleep(2)

    # ── GPT baseline ──
    try:
        start = time.time()
        resp = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": GPT_SYSTEM},
                {"role": "user", "content": q}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        gpt_time = round(time.time() - start, 2)
        gpt_results[q] = json.loads(resp.choices[0].message.content).get("recommendations", [])
        print(f"  GPT:   {len(gpt_results[q])} results in {gpt_time}s")
    except Exception as e:
        print(f"  GPT error: {e}")
        gpt_results[q] = []
        gpt_time = None

    latencies[q] = {"advisor": advisor_time, "gpt": gpt_time}
    time.sleep(1)

json.dump(advisor_results, open("advisor_results.json", "w"), indent=2)
json.dump(gpt_results, open("gpt_results.json", "w"), indent=2)
json.dump(latencies, open("latencies.json", "w"), indent=2)
print("\nDone. Saved advisor_results.json, gpt_results.json, latencies.json")
