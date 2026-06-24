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

print(f"Indexed {len(meta)} tools → {INDEX_DIR}/tools.faiss")
print(f"Saved vector matrix → {INDEX_DIR}/tool_vectors.npy")
