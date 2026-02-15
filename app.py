import os
import re
import time

import requests
import streamlit as st


def parse_categories(raw):
    return [c.strip() for c in re.split(r"[|,/]", str(raw)) if c.strip()]


API_BASE = os.getenv("API_BASE_URL", "https://comai-recommender-1.onrender.com")


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


count = get_toolcount()
if count is None:
    count_label = "many"
else:
    count_label = str(count)

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
    r = requests.post(f"{API_BASE}/search", json={"q": q, "k": k}, timeout=30)
    r.raise_for_status()
    return r.json().get("hits", [])


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
