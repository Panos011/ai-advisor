import os
import re
import time
import requests
import streamlit as st


API_BASE = os.getenv("API_BASE_URL", "https://comai-recommender-1.onrender.com")

st.set_page_config(
    page_title="ComAI Recommender",
    page_icon="",
    layout="wide"
)

# Session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "saved" not in st.session_state:
    st.session_state.saved = {}
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "pending_clarify" not in st.session_state:
    st.session_state.pending_clarify = False
if "clarify_base_query" not in st.session_state:
    st.session_state.clarify_base_query = ""
if "clarify_question" not in st.session_state:
    st.session_state.clarify_question = ""
if "clarify_cache" not in st.session_state:
    st.session_state.clarify_cache = {}
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0.0
if "results_history" not in st.session_state:
    st.session_state.results_history = []

MIN_REQUEST_GAP = 1
def parse_categories(raw):
    return [c.strip() for c in re.split(r"[|,/]", str(raw)) if c.strip()]
def is_free(price_text: str) -> bool:
    return "free" in (price_text or "").lower()
def tool_id_from_meta(m:dict) -> str:
    return (m.get("Tool_link") or m.get("Source_URL") or m.get("Name") or "").strip()
def throttle():
    gap = time.time() - st.session_state.last_request_time
    if gap < MIN_REQUEST_GAP:
        time.sleep(MIN_REQUEST_GAP - gap)
    st.session_state.last_request_time = time.time()
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
def render_results(hits):
    for idx, h in enumerate(hits):
        m = h.get("meta", {}) or {}
        score = float(h.get("score", 0.0))
        why = h.get("why")  # reason from LLM
        name = m.get("Name", "(no name)")
        link = m.get("Tool_link", "#")
        desc = m.get("Description", "")
        price = m.get("Price", "")
        cats = parse_categories(m.get("Categories", ""))
        tid = tool_id_from_meta(m)

        with st.container(border=True):
            top = st.columns([6, 2])
            with top[0]:
                st.markdown(f"**{name}**")
            with top[1]:
                saved = tid in st.session_state.saved
                label = "✅ Saved" if saved else "⭐ Save"
                if st.button(label, key=f"save_{tid}_{idx}"):
                    if not saved:
                        st.session_state.saved[tid] = m
                    else:
                        st.session_state.saved.pop(tid, None)
                    st.rerun()
            if cats:
                st.markdown(" ".join(f":blue-badge[{c}]" for c in cats[:8]))
            if why:
                st.info(why)  # show the LLM's reason
            if desc:
                st.write(desc)
            bottom = st.columns([2, 2, 3])
            with bottom[0]:
                st.markdown(f"**Price:** {price if price else '—'}")
            with bottom[1]:
                if link and link != "#":
                    st.link_button("Visit official site", link)

RETRIEVAL_K = 30
FINAL_K = 5
def clarify_api(q):
    try:
        for attempt in range(3):
            throttle()
            r = requests.post(f"{API_BASE}/clarify", json={"q": q}, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
    except Exception:
        pass
    return {"action": "search", "refined_query": q}

def recommend_api(q, retrieve_k=30, final_k=5):
    throttle()
    r = requests.post(
        f"{API_BASE}/recommend",
        json={"q": q, "retrieve_k": retrieve_k, "final_k": final_k},
        timeout=60
    )
    r.raise_for_status()
    return r.json().get("hits", [])
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
# Sidebar controls

st.sidebar.header("Filters")
free_only = st.sidebar.toggle("Free only", value=False)
category_filter = st.sidebar.text_input("Category contains (optional)", value="").strip().lower()
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.last_error = None
    st.session_state.last_results = []
    st.session_state.results_history = []
    st.rerun()

if st.sidebar.button("Clear Saved Tools"):
    st.session_state.saved = {}
    st.rerun()


# Main Layout
st.title("ComAI Recommender")
prompt = st.chat_input("What tool do you need?")
st.caption("Find the right AI tool in seconds")
left = st.container()

with left:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        if message.get("type") == "results":
            idx = message.get("result_index", -1)
            if 0 <= idx < len(st.session_state.results_history):
                with st.chat_message("assistant"):
                    render_results(st.session_state.results_history[idx])


    def passes_filters(meta: dict) -> bool:
        # free-only filter
        if free_only and not is_free(meta.get("Price", "")):
            return False

        # category filter (simple contains)
        if category_filter:
            cats = " ".join(parse_categories(meta.get("Categories", ""))).lower()
            if category_filter not in cats:
                return False

        return True


    if prompt:
        st.session_state.last_prompt = prompt
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.status("Thinking for the most compatible tools...", expanded=True) as status:
            st.write(f"Searching through {count_label} tools")

            if st.session_state.pending_clarify:
                refined = f"{st.session_state.clarify_base_query}. {prompt}"
                st.session_state.pending_clarify = False
                st.session_state.clarify_base_query = ""
                st.session_state.clarify_question = ""
            else:
                # build context from previous search so the LLM understands follow-ups
                context = ""
                if st.session_state.last_prompt:
                    context = f"Previous search was about: {st.session_state.last_prompt}. "
                contextual_prompt = context + prompt

                if contextual_prompt in st.session_state.clarify_cache:
                    decision = st.session_state.clarify_cache[contextual_prompt]
                else:
                    try:
                        decision = clarify_api(contextual_prompt)
                    except Exception:
                        decision = {"action": "search", "refined_query": contextual_prompt}
                    st.session_state.clarify_cache[contextual_prompt] = decision

                if decision.get("action") == "clarify":
                    ai_text = decision.get("question", "Can you clarify what you need?")
                    st.session_state.pending_clarify = True
                    st.session_state.clarify_base_query = prompt
                    st.session_state.clarify_question = ai_text
                    status.update(label="Need clarification", state="complete", expanded=True)
                    with st.chat_message("assistant"):
                        st.markdown(ai_text)
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                    st.stop()

                refined = decision.get("refined_query") or prompt

            err = None
            try:
                hits = recommend_api(refined, retrieve_k=RETRIEVAL_K, final_k=FINAL_K)
            except Exception as e:
                hits = []
                err = str(e)

            if err:
                st.session_state.last_error = err
                status.update(label="API request failed", state="error", expanded=True)
                with st.chat_message("assistant"):
                    st.markdown(f"Error calling API: {err}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error calling API: {err}"})
            else:
                st.session_state.last_error = None

                filtered = []
                for h in hits:
                    m = h.get("meta", {}) or {}
                    if passes_filters(m):
                        filtered.append(h)

                st.session_state.last_results = filtered
                st.session_state.results_history.append(filtered)

                if not filtered:
                    status.update(label="No results found", state="complete", expanded=True)
                    with st.chat_message("assistant"):
                        st.markdown("No results (try removing filters or changing the query).")
                    st.session_state.messages.append({"role": "assistant", "content": "No results."})
                    st.session_state.last_results = []
                else:
                    status.update(label="Compatible tools have been found", state="complete", expanded=True)
                    with st.chat_message("assistant"):
                        render_results(filtered)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Returned {len(filtered)} tools."})

                    st.session_state.results_history.append(filtered)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Returned {len(filtered)} tools",
                        "type": "results",
                        "result_index": len(st.session_state.results_history) - 1
                    })

# Pop-over window
with st.popover("Saved tools"):
    st.subheader("Saved tools")

    saved_items = list(st.session_state.saved.items())

    if not saved_items:
        st.info("No saved tools yet. Click Save on a result.")
    else:
        # list saved with remove buttons
        for tid, m in saved_items:
            cols = st.columns([6, 2])
            cols[0].write(m.get("Name", tid))
            if cols[1].button("Remove", key=f"rm_{tid}"):
                st.session_state.saved.pop(tid, None)
                st.rerun()


        st.divider()

        st.subheader("Compare (2–3 tools)")
        name_map = {m.get("Name", tid): tid for tid, m in saved_items}
        selected_names = st.multiselect(
            "Pick tools",
            options=list(name_map.keys()),
            default=list(name_map.keys())[:2] if len(name_map) >= 2 else list(name_map.keys())
        )

        if st.button("Compare selected", type="primary",
                        disabled=(len(selected_names) < 2 or len(selected_names) > 3)):
            selected_meta = [st.session_state.saved[name_map[n]] for n in selected_names]
            st.divider()
            cols = st.columns(len(selected_meta))
            for col, meta in zip(cols, selected_meta):
                with col:
                    st.markdown(f"### {meta.get('Name', '(no name)')}")
                    cats = parse_categories(meta.get("Categories", ""))
                    if cats:
                        st.markdown(" ".join(f":blue-badge[{c}]" for c in cats[:6]))
                    st.markdown(f"**Price:** {meta.get('Price', '—')}")
                    desc = meta.get("Description", "")
                    if desc:
                        st.write(desc[:300] + ("..." if len(desc) > 300 else ""))
                    link = meta.get("Tool_link", "")
                    if link:
                        st.link_button("Visit site", link)

