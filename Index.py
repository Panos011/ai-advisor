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
        "Tool_link": r.get("Tool_link", ""),
        "Categories": r.get("Categories", ""),
        "Price": r.get("Price", ""),
        "Description": r.get("Description", ""),
        "Pros": r.get("Pros", ""),
        "Cons": r.get("Cons", ""),
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
index = faiss.IndexFlatIP(embs.shape[1])
index = faiss.IndexIDMap(index)
ids = (pd.Series(range(len(df)))).astype("int64").to_numpy()
index.add_with_ids(embs, ids)

# Save index + metadata
faiss.write_index(index, os.path.join(INDEX_DIR, "tools.faiss"))
with open(os.path.join(INDEX_DIR, "meta.jsonl"), "w", encoding="utf-8") as f:
    for m in meta:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print(f"Indexed {len(meta)} tools → {INDEX_DIR}/tools.faiss")
