"""Normalise a privately stored API key without exposing it in logs."""

import os


def normalise_secret(value):
    text = str(value or "").strip()
    if text.startswith("OPENAI_API_KEY="):
        text = text.split("=", 1)[1].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1]
    return "".join(text.split())


def main():
    cleaned = normalise_secret(os.environ.get("OPENAI_API_KEY"))
    if not cleaned.startswith("sk-") or len(cleaned) < 20:
        raise ValueError("The private OpenAI workflow key has an invalid shape.")
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        raise RuntimeError("GITHUB_ENV is required inside the publication workflow.")
    # Register the cleaned value as masked before exporting it to later steps.
    # GitHub consumes this command; the workflow never prints the key as text.
    print(f"::add-mask::{cleaned}")
    with open(github_env, "a", encoding="utf-8") as handle:
        handle.write(f"OPENAI_API_KEY={cleaned}\n")
    print("Private OpenAI workflow key normalised.")


if __name__ == "__main__":
    main()
