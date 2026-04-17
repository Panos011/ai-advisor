import requests
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

API = "https://comai-recommender-1.onrender.com"
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

comai_results = {}
gpt_results = {}
latencies = {}

for q in QUERIES:
    print(f"\nQuery: {q[:60]}")

    # ── ComAI ──
    try:
        start = time.time()
        r = requests.post(
            f"{API}/recommend",
            json={"q": q, "retrieve_k": 30, "final_k": 5},
            timeout=60
        )
        comai_time = round(time.time() - start, 2)
        hits = r.json().get("hits", [])
        comai_results[q] = [
            {
                "name": h["meta"].get("Name", ""),
                "categories": h["meta"].get("Categories", ""),
                "description": h["meta"].get("Description", ""),
                "price": h["meta"].get("Price", ""),
                "why": h.get("why", "")
            }
            for h in hits
        ]
        print(f"  ComAI: {len(hits)} results in {comai_time}s")
    except Exception as e:
        print(f"  ComAI error: {e}")
        comai_results[q] = []
        comai_time = None

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

    latencies[q] = {"comai": comai_time, "gpt": gpt_time}
    time.sleep(1)

json.dump(comai_results, open("comai_results.json", "w"), indent=2)
json.dump(gpt_results, open("gpt_results.json", "w"), indent=2)
json.dump(latencies, open("latencies.json", "w"), indent=2)
print("\nDone. Saved comai_results.json, gpt_results.json, latencies.json")
