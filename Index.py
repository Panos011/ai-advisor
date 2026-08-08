import os
import json
import pandas as pd
import faiss
import numpy as np
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI

CSV_IN = "Embeddings_AI_tools.csv"
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)
df = pd.read_csv(CSV_IN, dtype=str).fillna("")
# Keep rows with content
df = df[df["doc"].str.len() > 5].reset_index(drop=True)

# Lightweight metadata for display

meta = []
for _, r in df.iterrows():
    meta.append({
        "Tool_ID": r.get("Tool_ID", ""),
        "Name": r.get("Name", ""),
        "Logo_URL": r.get("Logo_URL", ""),
        "Logo_File": r.get("Logo_File", ""),
        "Source_URL": r.get("Source_URL", ""),
        "Rating": r.get("Rating", ""),
        "Tool_link": r.get("Tool_link", ""),
        "Categories": r.get("Categories", ""),
        "Price": r.get("Price", ""),
        "Description": r.get("Description", ""),
        "Features": r.get("Features", ""),
        "Pros": r.get("Pros", ""),
        "Cons": r.get("Cons", ""),
        "Use_cases": r.get("Use_cases", ""),
        "Unique_Value": r.get("Unique_Value", ""),
    })
client = OpenAI()
EMB_MODEL = "text-embedding-3-small"

texts = df["doc"].tolist()


def batched(seq, n=256):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


vectors = []
for chunk in batched(texts, 256):
    resp = client.embeddings.create(model=EMB_MODEL, input=chunk)
    vectors.extend([d.embedding for d in resp.data])
embs = np.array(vectors, dtype="float32")
# FAISS (cosine via IP on normalized vectors)
faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(embs.shape[1])
index = faiss.IndexIDMap(index)
ids = (pd.Series(range(len(df)))).astype("int64").to_numpy()
index.add_with_ids(embs, ids)

# Save index + metadata
faiss.write_index(index, os.path.join(INDEX_DIR, "tools.faiss"))
np.save(os.path.join(INDEX_DIR, "tool_vectors.npy"), embs)
with open(os.path.join(INDEX_DIR, "meta.jsonl"), "w", encoding="utf-8") as f:
    for m in meta:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

# Version-stamp the artifact set so the API can detect drift at startup:
# index, vectors, metadata, and embedding model must stay in lockstep or
# retrieval is silently wrong. api.load_tool_store() verifies this manifest.
from datetime import datetime, timezone

manifest = {
    "schema": 1,
    "emb_model": EMB_MODEL,
    "rows": len(meta),
    "dim": int(embs.shape[1]),
    "source_csv": CSV_IN,
    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
catalog_state_path = os.path.join(INDEX_DIR, "catalog_publication.json")
if os.path.exists(catalog_state_path):
    with open(catalog_state_path, encoding="utf-8") as state_file:
        catalog_state = json.load(state_file)
    if catalog_state.get("catalogVersion"):
        manifest["catalog_version"] = catalog_state["catalogVersion"]
    if catalog_state.get("catalogGeneratedAt"):
        manifest["catalog_generated_at"] = catalog_state["catalogGeneratedAt"]
with open(os.path.join(INDEX_DIR, "index_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Indexed {len(meta)} tools → {INDEX_DIR}/tools.faiss")
print(f"Saved vector matrix → {INDEX_DIR}/tool_vectors.npy")
print(f"Wrote manifest → {INDEX_DIR}/index_manifest.json")

# Keep the claim index in lockstep with every catalogue refresh. It is optional
# at runtime, so an older deployment continues to serve whole-tool retrieval
# until these artifacts are available.
from Build_Claim_Index import main as build_claim_index

build_claim_index()
