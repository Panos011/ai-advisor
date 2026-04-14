import json
import numpy as np
from collections import Counter

comai = json.load(open("comai_results.json"))
gpt = json.load(open("gpt_results.json"))
lat = json.load(open("latencies.json"))


def precision(labels):
    return round(sum(labels) / len(labels), 3) if labels else 0


def diversity(tools, cat_key="categories"):
    cats = []
    for t in tools:
        raw = t.get(cat_key) or t.get("category", "")
        for c in str(raw).replace("|", ",").split(","):
            c = c.strip().lower()
            if c:
                cats.append(c)
    return round(len(set(cats)) / len(cats), 3) if cats else 0


def gini(names):
    counts = sorted(Counter(n for n in names if n).values())
    n, s = len(counts), sum(counts)
    if n == 0 or s == 0:
        return 0.0
    return round((2 * sum((i+1)*v for i, v in enumerate(counts)) - (n+1)*s) / (n*s), 3)

# ── Annotation ──


annotations = {}
print("="*60)
print("For each tool enter 1 if relevant to the query, 0 if not")
print("="*60)

for q in comai:
    print(f"\nQUERY: {q}")
    annotations[q] = {"comai": [], "gpt": []}

    print("-- ComAI --")
    for t in comai[q]:
        print(f"  {t['name']} | {t['categories'][:50]} | {t['description'][:1200]} | {t['why'][:1000]}")
        while True:
            v = input("  Relevant? 1/0: ").strip()
            if v in ("0", "1"):
                annotations[q]["comai"].append(int(v))
                break

    print("-- GPT --")
    for t in gpt.get(q, []):
        print(f"  {t['name']} | {t.get('category', '')} | {t.get('description', '')[:1000]} | {t.get('reason', '')[:1000]}")
        while True:
            v = input("  Relevant? 1/0: ").strip()
            if v in ("0", "1"):
                annotations[q]["gpt"].append(int(v))
                break

json.dump(annotations, open("annotations.json", "w"), indent=2)

# ── Metrics ──
cp, gp, cd, gd, cl, gl = [], [], [], [], [], []

print("\n" + "="*60)
print("PER QUERY RESULTS")
print("="*60)

for q in annotations:
    c_prec = precision(annotations[q]["comai"])
    g_prec = precision(annotations[q]["gpt"])
    c_div = diversity(comai.get(q, []))
    g_div = diversity(gpt.get(q, []))
    c_lat = lat[q]["comai"]
    g_lat = lat[q]["gpt"]

    cp.append(c_prec); gp.append(g_prec)
    cd.append(c_div);  gd.append(g_div)
    if c_lat: cl.append(c_lat)
    if g_lat: gl.append(g_lat)

    print(f"\nQ: {q[:55]}")
    print(f"  ComAI  P@5:{c_prec:.2f}  Div:{c_div:.3f}  Time:{c_lat}s")
    print(f"  GPT    P@5:{g_prec:.2f}  Div:{g_div:.3f}  Time:{g_lat}s")

comai_tools = [t["name"].lower() for q in comai for t in comai[q]]
gpt_tools = [t["name"].lower() for q in gpt for t in gpt[q]]

print("\n" + "="*60)
print("AGGREGATE RESULTS")
print("="*60)
print(f"              ComAI       GPT")
print(f"Precision@5:  {np.mean(cp):.3f}       {np.mean(gp):.3f}")
print(f"Diversity:    {np.mean(cd):.3f}       {np.mean(gd):.3f}")
print(f"Gini:         {gini(comai_tools):.3f}       {gini(gpt_tools):.3f}")
print(f"Avg latency:  {np.mean(cl):.2f}s      {np.mean(gl):.2f}s")

json.dump({
    "comai": {"avg_precision_at_5": round(float(np.mean(cp)), 3), "avg_diversity": round(float(np.mean(cd)), 3), "gini": gini(comai_tools), "avg_latency_s": round(float(np.mean(cl)), 2)},
    "gpt":   {"avg_precision_at_5": round(float(np.mean(gp)), 3), "avg_diversity": round(float(np.mean(gd)), 3), "gini": gini(gpt_tools),   "avg_latency_s": round(float(np.mean(gl)), 2)}
}, open("metrics_results.json", "w"), indent=2)
print("\nSaved metrics_results.json")
