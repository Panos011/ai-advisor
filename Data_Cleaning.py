import re
import ast
import pandas as pd

IN_CSV = "AI_tools.csv"
OUT_CLEAN = "Clean_AI_tools.csv"
OUT_EMB = "Embeddings_AI_tools.csv"


def norm_ws(x) -> str:
    return re.sub(r"\s+", " ", str(x)).strip()


def clean_list_remove_brackets(x) -> str:
    s = norm_ws(x)
    if not (s.startswith("[") and s.endswith("]")):
        return s

    try:
        v = ast.literal_eval(s)
        if not isinstance(v, list):
            return s

        out = []
        seen = set()
        for item in v:
            item_s = norm_ws(item)
            if item_s and item_s not in seen:
                seen.add(item_s)
                out.append(item_s)

        #  no [ ] brackets:
        return " | ".join(out)

    except Exception:
        return s

# Read


df = pd.read_csv(IN_CSV, dtype=str, na_filter=False).fillna("")
original_cols = list(df.columns)

# Ensure expected columns exist (optional)
expected = ["Name", "Description", "Price", "Tool_link", "Source_URL", "Unique_Value",
            "Features", "Pros", "Cons", "Use_cases", "Categories"]
for c in expected:
    if c not in df.columns:
        df[c] = ""

# Clean scalars
for c in ["Name", "Description", "Price", "Tool_link", "Source_URL", "Unique_Value"]:
    df[c] = df[c].map(norm_ws)

# Fix Name: convert URL slug to proper title case


def slug_to_name(s: str) -> str:
    s = norm_ws(s)
    if not s:
        return s
    return s.replace("-", " ").title()


df["Name"] = df["Name"].map(slug_to_name)
# Clean list-like columns (remove [ ])
for c in ["Features", "Pros", "Cons", "Use_cases", "Categories"]:
    df[c] = df[c].map(clean_list_remove_brackets)

# --- Output 1: Clean ---
df_clean = df[original_cols]
df_clean.to_csv(OUT_CLEAN, index=False, encoding="utf-8")

# Output 2: Embeddings-friendly


def flatten_to_pipes(x) -> str:
    s = norm_ws(x)
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return " | ".join(norm_ws(i) for i in v if norm_ws(i))
        except Exception:
            pass
    return s


df_emb = df.copy()
for c in ["Features", "Pros", "Cons", "Use_cases", "Categories"]:
    df_emb[c] = df_emb[c].map(flatten_to_pipes)

df_emb["doc"] = (
    "Name: " + df_emb["Name"] + "\n" +
    "Categories: " + df_emb["Categories"] + "\n" +
    "Features: " + df_emb["Features"] + "\n" +
    "Use cases: " + df_emb["Use_cases"] + "\n" +
    "Pros: " + df_emb["Pros"] + "\n" +
    "Cons: " + df_emb["Cons"] + "\n" +
    "Price: " + df_emb["Price"] + "\n" +
    "Description: " + df_emb["Description"]
)

df_emb.to_csv(OUT_EMB, index=False, encoding="utf-8")

print("Saved:", OUT_CLEAN, "and", OUT_EMB)
