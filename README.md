Pipeline:
1. Scrape Futurepedia tools -> 'AI_tools.csv'
2. Clean and build doc field -> 'Clean_AI_tools.csv', 'Embeddings_AI_tools.csv'
3. Build FAISS index with OpenAI embeddings -> 'index/tools.faiss', 'index/tool_vectors.npy'
4. Query: Retrieve K results and ask GPT to rank

## Setup
Python 3.12+
'pip install -r requirements.txt'
Copy '.env.example' to '.env' and set 'OPENAI_API_KEY'.

## Backend
Run the API:
'uvicorn api:app --host 0.0.0.0 --port 10000'

Useful endpoints:
'GET /health' checks index, metadata, OpenAI configuration and vector loading.
'GET /metrics' returns request counters, stage latency averages and cache sizes.
'POST /search', 'POST /recommend', 'POST /clarify', 'POST /detect_intent' power the app.

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
