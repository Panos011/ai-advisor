Pipeline:
1. Scrape Futurepedia tools -> 'AI_tools.csv'
2. Clean and build doc field -> 'Clean_AI_tools.csv', 'Embeddings_AI_tools.csv'
3. Build FAISS index with OpenAI embeddings -> 'index/tools.faiss', 'index/tool_vectors.npy'
4. Query: Retrieve K results and ask GPT to rank

## Setup
Python 3.12+
'pip install -r requirements.txt'
Copy '.env.example' to '.env' and set 'OPENAI_API_KEY'.

## Backend API
Run the API:
'uvicorn api:app --host 0.0.0.0 --port 10000'

Useful endpoints:
'GET /health' checks index, metadata, OpenAI configuration and vector loading.
  It also returns 'degradation_counters' (non-zero means requests were served on
  fallback paths) and 'index_manifest' (with a 'verified' flag). Pass '?strict=1'
  to make it return 503 when the advisor cannot serve fully-intelligent responses
  (no OpenAI key or index not ready) — use this for deploy/uptime checks.
'GET /metrics' returns request counters, stage latency averages and cache sizes.
'POST /chat' is the main GPT-wrapper endpoint for the app.
'POST /chat/stream' is the SSE variant of '/chat': it streams 'status' events at
  each pipeline stage (routing, planning, retrieving, ranking, replying), then one
  'result' event whose payload is identical to '/chat', then 'done'. Use it to show
  live progress and cut perceived latency. '/chat' is unchanged; adopt at your pace.
'POST /search', 'POST /recommend', 'POST /clarify', and 'POST /detect_intent' remain lower-level building blocks.

### Degradation visibility
'/chat', '/recommend' and '/search' responses now carry two optional fields:
'degraded' (bool) and 'degradation' (list of reasons) whenever a request fell back
to a non-LLM path — embedding outage -> keyword search, LLM rank failure ->
retrieval-order results, planner failure -> rule-based routing, etc. The fields
default to false/[] so the existing wire format is unchanged. Surface 'degraded'
in the client (or alert on '/metrics' 'degraded_requests') so the advisor silently
becoming a keyword-search engine is visible instead of hidden.

## React Native Client Contract
The backend is API-only. The mobile app should treat 'POST /chat' like a GPT-wrapper call: send one user message and render the returned assistant bubble plus any structured tool cards.

Recommended chat flow:
1. Call 'POST /chat' for every user message.
2. Send a stable 'conversation_id' on every turn so the backend can remember the current shortlist.
3. Send the recent 'history' and current 'visible_tools' when available. This lets follow-ups like "why these?", "which is best?", "is it free?", and "show me another one" answer from the cards already on screen.
3a. Optionally send 'shown_tools' — the names of every tool the app has surfaced this session (not just the currently visible cards). The server keeps its shortlist/shown memory in-process, so a deploy or a second instance would otherwise forget what "show me another" already showed and could repeat a tool. Passing 'shown_tools' makes that memory client-owned, so alternatives keep advancing across deploys. Absent it, behavior is unchanged.
3b. Alternative requests ("show me another", "any other tool?") never return a tool the server knows was already displayed — on '/chat' and '/recommend' alike. When every stored shortlist entry has been shown, the server searches the catalogue for a fresh distinct option (excluding 'shown_tools') before saying there is nothing left.
4. Render 'message' as the assistant bubble.
5. Render 'hits' as tool cards. If 'hits' is empty, keep the current cards unless the UI intentionally clears them.
6. Use 'action' to decide UI behavior: 'chat_only' and 'clarify' do not replace cards; 'recommend' and 'refine' replace the shortlist; 'explain', 'pick_best', and 'show_alternative' can highlight the returned card(s).

Behind 'POST /chat', the recommender uses:
'FAISS' for vector retrieval, 'MMR' for candidate diversity, and a RAG ranking step where the chat model chooses from retrieved catalogue records only.

Each response includes a 'contract' object that names this pipeline and the expected tool-card fields. Each recommendation hit contains:
'score', 'why', 'tradeoff', 'best_for', 'fit_label', 'cost_summary', and 'meta'. The mobile card can use 'meta.Name', 'meta.Categories', 'meta.Price', 'meta.Description', 'meta.Tool_link', 'meta.Logo_URL', and 'meta.Logo_File'.

'cost_summary' is a short, deterministic cost line for the card — e.g. 'Free',
'Free tier, then from $12/mo', 'From $15/mo', 'Paid', or 'Pricing not public yet'.
It is derived only from the catalog price (never from the LLM, so it cannot drift
or hallucinate a number) and is null/absent when the price is genuinely unknown,
so the client should hide the line rather than show a misleading one. This is the
"what it'll cost you" half of an explainable recommendation.

## Scripts 
'Data_Collection.py'
'Data_Cleaning.py'
'Index.py' — builds the FAISS index and also writes 'index/index_manifest.json'
  (embedding model, row count, vector dim, source CSV, build time). The API
  verifies this manifest at startup and logs loud errors if the artifacts drift
  from each other or from 'EMB_MODEL', so a mismatched index fails visibly
  instead of returning silently wrong results. A missing manifest only warns.

### Catalog freshness ('Freshness.py')
An advisor is only as good as its catalog, so 'Freshness.py' checks every tool's
'Tool_link' for liveness and tracks the result across runs in 'freshness_state.json'.
It writes 'freshness_report.json' (dead + unreachable tools) each run.

Safety: a single failed request never prunes a tool. A tool must fail on
'--fail-threshold' consecutive runs (default 2) before it is eligible for removal,
so a transient outage or a bot block cannot delete a good listing. Bot-blocked
responses (401/403/405/429) count as alive; only 404/410 are 'dead', and
timeouts / DNS / 5xx are 'unreachable' (ambiguous, gated by the threshold).

With '--emit-deletions' it appends prune-eligible tools to 'deleted_tools.json',
reusing the existing deletion path ('Apply_Deletions.py' / 'Filter_Blocklist.py')
rather than editing the CSVs directly. Run it on a schedule (e.g. before the
biweekly refresh):
  'python Freshness.py --workers 16'                 # audit + report only
  'python Freshness.py --emit-deletions'             # also queue dead tools for removal

## Platform sync (CommAI)
Deleted-tool lifecycle scripts keep the dataset aligned with the CommAI platform:
- 'Apply_Deletions.py' — consumes 'advisor-tool-deleted' / 'advisor-tool-restored'
  repository_dispatch payloads from Firebase: strips the tool from all CSVs and
  maintains the persistent 'deleted_tools.json' blocklist
  ('.github/workflows/tool_deletion.yml' runs it and rebuilds the index).
- 'Filter_Blocklist.py' — runs after the scraper in the biweekly refresh; removes
  previously deleted tools from the fresh scrape and records them in
  'flagged_tools.csv' for inspection.
- 'Sync_To_Firebase.py' — 'export-blocklist' reconciles the blocklist from the
  'advisorDatasetDeletions' Firestore collection; 'sync' pushes newly collected
  tools to the platform as pending 'toolSubmissions' (re-collected deleted tools
  are flagged 'flaggedPreviouslyDeleted'). Requires the 'FIREBASE_SERVICE_ACCOUNT'
  GitHub Actions secret (service-account JSON); needs 'pip install firebase-admin'.

## Tests
Local unit tests do not require an OpenAI key:
'python -m unittest discover tests'

Live API checks:
'python -m unittest Testing.py'

### Routing eval
Chat routing (search / refine / explain / alternative / chat) is the part that
breaks most when guards are added. 'routing_golden.jsonl' captures that contract
as replayable cases; 'eval_routing.py' replays them through 'chat()':
- 'python eval_routing.py' — offline, deterministic stub client, no key needed.
  Locks down the gate contract and the two known UX-bug guardrails (compare must
  not re-recommend; alternatives must not re-show already-shown tools). Also runs
  inside the unittest suite.
- 'python eval_routing.py --live' — routes the same cases through the real OpenAI
  planner (needs 'OPENAI_API_KEY'). This is the number to watch when editing the
  planner prompt, and where 'chat_decision_invalid' surfaces.
Add a case whenever you fix a routing bug: it turns a one-off regex patch into a
permanent regression check, which is the safe way to migrate routing off the
regex gates and onto the planner over time.

### Planner shadow mode (regex→planner migration, Stage 1)
Set 'PLANNER_SHADOW=1' to make every '/chat' request also ask the planner what it
would route to and log whether that matches what the gate logic shipped
('planner_shadow_agree' / 'planner_shadow_disagree' counters in '/metrics', plus a
'PLANNER SHADOW disagree' log line per mismatch). This is how you measure — with
real traffic and zero user impact — which gates the planner already agrees with
before retiring them. It never changes the response, and it restores the request's
degradation state so the extra planner call can't flip the 'degraded' flag. It is
off by default because it adds a second LLM round-trip; enable it for a measurement
window, not in steady state.

## Notes
'.env' is git-ignored.
