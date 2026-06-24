import os
import re
import time
import requests
import streamlit as st


API_BASE = os.getenv("API_BASE_URL", "").rstrip("/")

st.set_page_config(
    page_title="AI Tool Advisor",
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
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

MIN_REQUEST_GAP = 1


def parse_categories(raw):
    return [c.strip() for c in re.split(r"[|,/]", str(raw)) if c.strip()]


def render_logo(meta, size=44):
    """Show the tool's logo, or its initials if there's no logo."""
    logo_url = (meta.get("Logo_URL") or "").strip()
    if logo_url:
        st.image(logo_url, width=size)
        return
    # fallback: initials badge
    name = meta.get("Name", "?")
    parts = name.split()
    initials = (parts[0][0] + parts[1][0]).upper() if len(parts) >= 2 else name[:2].upper()
    st.markdown(
        f"<div style='width:{size}px;height:{size}px;border-radius:10px;"
        f"background:#e0e7ff;display:flex;align-items:center;justify-content:center;"
        f"font-weight:700;color:#3730a3;'>{initials}</div>",
        unsafe_allow_html=True,
    )


def is_free(price_text: str) -> bool:
    return "free" in (price_text or "").lower()


def tool_id_from_meta(m: dict) -> str:
    return (m.get("Tool_link") or m.get("Source_URL") or m.get("Name") or "").strip()


def throttle():
    gap = time.time() - st.session_state.last_request_time
    if gap < MIN_REQUEST_GAP:
        time.sleep(MIN_REQUEST_GAP - gap)
    st.session_state.last_request_time = time.time()


def warm_up():
    if not API_BASE:
        return
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


def render_results(hits, history_idx=0):
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
            top = st.columns([1, 5, 2], vertical_alignment="center")
            with top[0]:
                render_logo(m)
            with top[1]:
                st.markdown(f"### {name}")          # ← the tool name, restored
            with top[2]:
                saved = tid in st.session_state.saved
                label = "✅ Saved" if saved else "⭐ Save"
                if st.button(label, key=f"save_{history_idx}_{tid}_{idx}"):
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
    if not API_BASE:
        return {"action": "search", "refined_query": q}
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
    if not API_BASE:
        raise RuntimeError("API_BASE_URL is not configured")
    throttle()
    r = requests.post(
        f"{API_BASE}/recommend",
        json={"q": q, "retrieve_k": retrieve_k, "final_k": final_k},
        timeout=60
    )
    r.raise_for_status()
    data = r.json()
    return data.get("hits", []), data.get("message")


def detect_intent(prompt, last_query):
    if not last_query or not API_BASE:
        return "new"
    try:
        throttle()
        r = requests.post(
            f"{API_BASE}/detect_intent",
            json={"prompt": prompt, "last_query": last_query},
            timeout=30
        )
        r.raise_for_status()
        return r.json().get("intent", "new")
    except Exception:
        return "new"


def is_explanation_prompt(text: str) -> bool:
    return bool(re.search(
        r"\b(why\s+(these|this|those)|why\s+did|explain|reason|reasons|why\s+recommended|why\s+recommend)\b",
        (text or "").lower(),
    ))


def explain_results(hits):
    if not hits:
        return "I need a current shortlist before I can explain the recommendations."

    names = [((h.get("meta") or {}).get("Name") or "").strip() for h in hits]
    names = [name for name in names if name]
    if not names:
        return "I picked these tools because their descriptions were the closest match to your request."

    first = hits[0]
    first_meta = first.get("meta", {}) or {}
    first_name = first_meta.get("Name", names[0])
    first_reason = (first.get("why") or "").strip()
    if first_reason:
        first_reason = re.sub(r"\s+", " ", first_reason)
        return f"I put {first_name} first because {first_reason[0].lower() + first_reason[1:]}"

    if len(names) == 1:
        return f"I picked {first_name} because it was the closest match to your request."
    return f"I picked {first_name} first because it looked like the strongest match, with {names[1]} as another close option."


@st.cache_data(ttl=3600)
def get_toolcount():
    if not API_BASE:
        return None
    url = f"{API_BASE}/health"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json().get("items", 0)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


if "toolcount" not in st.session_state:
    st.session_state.toolcount = get_toolcount()

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
st.title("AI Tool Advisor")
prompt = st.chat_input("What tool do you need?")
st.caption(f"Find the right AI tool in seconds — searching across {count_label} tools")
left = st.container()

with left:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        if message.get("type") == "results":
            idx = message.get("result_index", -1)
            if 0 <= idx < len(st.session_state.results_history):
                with st.chat_message("assistant"):
                    render_results(st.session_state.results_history[idx], history_idx=idx)


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
                intent = "explain" if is_explanation_prompt(prompt) else detect_intent(prompt, st.session_state.last_query)
                if intent == "explain":
                    ai_text = explain_results(st.session_state.last_results)
                    status.update(label="Explained current shortlist", state="complete", expanded=True)
                    with st.chat_message("assistant"):
                        st.markdown(ai_text)
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                    st.stop()

                if intent == "refine":
                    combined = f" {st.session_state.last_query}.{prompt} "
                else:
                    combined = prompt

                if combined in st.session_state.clarify_cache:
                    decision = st.session_state.clarify_cache[combined]
                else:
                    try:
                        decision = clarify_api(combined)
                    except Exception:
                        decision = {"action": "search", "refined_query": combined}
                    st.session_state.clarify_cache[combined] = decision

                if decision.get("action") == "clarify":
                    ai_text = decision.get("question", "Can you clarify what you need?")
                    st.session_state.pending_clarify = True
                    st.session_state.clarify_base_query = combined
                    st.session_state.clarify_question = ai_text
                    status.update(label="Need clarification", state="complete", expanded=True)
                    with st.chat_message("assistant"):
                        st.markdown(ai_text)
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                    st.stop()
                if decision.get("action") == "explain":
                    ai_text = explain_results(st.session_state.last_results)
                    status.update(label="Explained current shortlist", state="complete", expanded=True)
                    with st.chat_message("assistant"):
                        st.markdown(ai_text)
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                    st.stop()

                refined = decision.get("refined_query") or combined

            err = None
            result_message = None
            try:
                hits, result_message = recommend_api(refined, retrieve_k=RETRIEVAL_K, final_k=FINAL_K)
            except Exception as e:
                hits = []
                err = str(e)

            if err:
                st.session_state.last_error = err
                status.update(label="API request failed", state="error", expanded=True)
                with st.chat_message("assistant"):
                    st.markdown(f"Error calling API: {err}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error calling API: {err}"})
                st.rerun()
            else:
                st.session_state.last_error = None

                filtered = []
                for h in hits:
                    m = h.get("meta", {}) or {}
                    if passes_filters(m):
                        filtered.append(h)

                st.session_state.last_results = filtered
                st.session_state.last_query = refined

                if not filtered:
                    status.update(label="No results found", state="complete", expanded=True)
                    with st.chat_message("assistant"):
                        st.markdown("No results (try removing filters or changing the query).")
                    st.session_state.messages.append({"role": "assistant", "content": "No results."})
                    st.session_state.last_results = []
                else:
                    status.update(label="Compatible tools have been found", state="complete", expanded=True)
                    st.session_state.results_history.append(filtered)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result_message or f"Returned {len(filtered)} tools",
                        "type": "results",
                        "result_index": len(st.session_state.results_history) - 1
                    })
                    st.rerun()

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
