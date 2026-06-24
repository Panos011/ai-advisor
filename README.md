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
'GET /metrics' returns request counters, stage latency averages and cache sizes.
'POST /search', 'POST /recommend', 'POST /clarify', 'POST /detect_intent' power the app.

## React Native Client Contract
The backend is API-only. The old Streamlit UI has been removed; the mobile app should call these endpoints directly.

Recommended chat flow:
1. For every user message after a previous search, call 'POST /detect_intent' with '{ "prompt": "...", "last_query": "..." }'.
2. If intent is 'explain', answer from the current visible shortlist using each card's 'why' text. Do not call '/recommend' for explanation questions like "Why these tools?".
3. Otherwise call 'POST /clarify' with the new or refined query.
4. If clarify returns '{ "action": "clarify" }', show the returned 'question' as the assistant bubble.
5. If clarify returns '{ "action": "search" }', call 'POST /recommend' with 'refined_query'.
6. Render '/recommend' response 'message' as the assistant bubble and 'hits' as cards. If 'hits' is empty, still show the backend 'message' instead of inventing cards.

Each recommendation hit contains:
'score', 'why', and 'meta'. The mobile card can use 'meta.Name', 'meta.Categories', 'meta.Price', 'meta.Description', 'meta.Tool_link', 'meta.Logo_URL', and 'meta.Logo_File'.

## Scripts 
'Data_Collection.py'
'Data_Cleaning.py'
'Index.py'

## Tests
Local unit tests do not require an OpenAI key:
'python -m unittest discover tests'

Live API checks:
'python -m unittest Testing.py'

## Notes
'.env' is git-ignored.
