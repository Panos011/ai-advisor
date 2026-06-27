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
'POST /chat' is the main GPT-wrapper endpoint for the app.
'POST /search', 'POST /recommend', 'POST /clarify', and 'POST /detect_intent' remain lower-level building blocks.

## React Native Client Contract
The backend is API-only. The mobile app should treat 'POST /chat' like a GPT-wrapper call: send one user message and render the returned assistant bubble plus any structured tool cards.

Recommended chat flow:
1. Call 'POST /chat' for every user message.
2. Send a stable 'conversation_id' on every turn so the backend can remember the current shortlist.
3. Send the recent 'history' and current 'visible_tools' when available. This lets follow-ups like "why these?", "which is best?", "is it free?", and "show me another one" answer from the cards already on screen.
4. Render 'message' as the assistant bubble.
5. Render 'hits' as tool cards. If 'hits' is empty, keep the current cards unless the UI intentionally clears them.
6. Use 'action' to decide UI behavior: 'chat_only' and 'clarify' do not replace cards; 'recommend' and 'refine' replace the shortlist; 'explain', 'pick_best', and 'show_alternative' can highlight the returned card(s).

Behind 'POST /chat', the recommender uses:
'FAISS' for vector retrieval, 'MMR' for candidate diversity, and a RAG ranking step where the chat model chooses from retrieved catalogue records only.

Each response includes a 'contract' object that names this pipeline and the expected tool-card fields. Each recommendation hit contains:
'score', 'why', 'tradeoff', 'best_for', 'fit_label', and 'meta'. The mobile card can use 'meta.Name', 'meta.Categories', 'meta.Price', 'meta.Description', 'meta.Tool_link', 'meta.Logo_URL', and 'meta.Logo_File'.

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
