import os
import re
import time

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "https://comai-recommender-1.onrender.com")

def warm_up():
    url = f"{API_BASE}/health"
    for attempt in range(2):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code in (429, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            return
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return

if "api_warmed" not in st.session_state:
    warm_up()
    st.session_state.api_warmed = True

def parse_categories(raw):
    return [c.strip() for c in re.split(r"[|,/]", str(raw)) if c.strip()]


@st.cache_data(ttl=3600)
def get_toolcount():
    url = f"{API_BASE}/health"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code ==429:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json().get("items", 0)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None

if "toolcount" not in st.session_state:
    st.session_state.toolcount = get_toolcount ()

count = st.session_state.toolcount
count_label = "many" if count is None else str(count)

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
st.title("ComAI Recommender", text_alignment="left", width="stretch")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What tool do you need?")

if "messages" not in st.session_state:
    st.session_state.messages = []

def search_api(q, k=5):
    url = f"{API_BASE}/search"
    payload = {"q": q, "k": k }
    for attempt in range(5):
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue

        r.raise_for_status()
        return r.json().get("hits", [])
    return []


if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        st.badge(prompt)
    with st.status("Thinking for the most compatible tools...", expanded=True) as status:
        st.write(f"Searching through {count_label} tools")
        try:
            hits = search_api(prompt, k=5)
            if not hits:
                ai_text = "No results"
            else:
                parts = []
                for h in hits:
                    m = h.get("meta", {})
                    categ = parse_categories(m.get("Categories", " "))
                    badges = "".join(f":blue-badge[{c}]" for c in categ)
                    parts.append(
                        f"**{m.get('Name', '(no name)')}** — score `{h.get('score', 0):.3f}`"
                        f"[Visit tool official website]({m.get('Tool_link', '#')})   \n"
                        f"\n {badges}  \n"
                        f"\n {m.get('Description', '')}\n"
                        f"\nPrice: {m.get('Price', '')} \n"
                    )
                ai_text = "\n\n---\n\n".join(parts)
                status.update(label="Compatible tools have been found", state="complete", expanded=True)
        except Exception as e:
            ai_text = f"Error calling API: {e}"
        with st.chat_message("assistant"):
            st.markdown(ai_text)
        st.session_state.messages.append({"role": "assistant", "content": ai_text})
