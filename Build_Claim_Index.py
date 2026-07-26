"""Build the optional open-vocabulary capability-claim FAISS index.

The index is separate from the existing whole-tool index. A query can therefore
retrieve a tool because one specific feature entails the requested job even
when that feature is diluted inside a long product record.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from claims import claim_embedding_text, extract_evidence_claims


load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "index")
TOOL_META_PATH = os.getenv("META_PATH", os.path.join(INDEX_DIR, "meta.jsonl"))
CLAIM_INDEX_PATH = os.getenv("CLAIM_INDEX_PATH", os.path.join(INDEX_DIR, "claims.faiss"))
CLAIM_META_PATH = os.getenv("CLAIM_META_PATH", os.path.join(INDEX_DIR, "claims_meta.jsonl"))
CLAIM_MANIFEST_PATH = os.getenv(
    "CLAIM_MANIFEST_PATH", os.path.join(INDEX_DIR, "claims_manifest.json")
)
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")
MAX_CLAIMS_PER_TOOL = int(os.getenv("MAX_CLAIMS_PER_TOOL", "6"))
BATCH_SIZE = int(os.getenv("CLAIM_EMBED_BATCH_SIZE", "256"))


def batched(values: list[str], size: int):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def main() -> None:
    with open(TOOL_META_PATH, "r", encoding="utf-8") as handle:
        tools = [json.loads(line) for line in handle if line.strip()]

    claims: list[dict] = []
    texts: list[str] = []
    for tool_id, meta in enumerate(tools):
        extracted = extract_evidence_claims(meta, tool_id, MAX_CLAIMS_PER_TOOL)
        categories = str(meta.get("Categories") or "")
        claims.extend(extracted)
        texts.extend(claim_embedding_text(claim, categories) for claim in extracted)

    if not claims:
        raise RuntimeError("No evidence claims were extracted from the catalogue.")

    client = OpenAI()
    vectors: list[list[float]] = []
    for chunk in batched(texts, BATCH_SIZE):
        response = client.embeddings.create(model=EMB_MODEL, input=chunk)
        vectors.extend(item.embedding for item in response.data)

    embeddings = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, CLAIM_INDEX_PATH)
    with open(CLAIM_META_PATH, "w", encoding="utf-8") as handle:
        for claim in claims:
            handle.write(json.dumps(claim, ensure_ascii=False) + "\n")

    manifest = {
        "schema": 1,
        "emb_model": EMB_MODEL,
        "claims": len(claims),
        "tools": len(tools),
        "dim": int(embeddings.shape[1]),
        "max_claims_per_tool": MAX_CLAIMS_PER_TOOL,
        "source_meta": TOOL_META_PATH,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(CLAIM_MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Indexed {len(claims)} evidence claims from {len(tools)} tools.")
    print(f"Wrote {CLAIM_INDEX_PATH}, {CLAIM_META_PATH}, and {CLAIM_MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
