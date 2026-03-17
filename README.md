Pipeline:
1. Scrape Futurepedia tools -> 'AI_tools.csv'
2. Clean and build doc field -> 'Clean_AI_tools.csv', 'Embeddings_AI_tools.csv'
3. Build FAISS index with OpenAI embeddings -> 'index/tools.faiss'
4. Query: Retrieve K results and ask GPT to rank

## Setup
Python 3.12+
'pip install -r requirements.txt'
Copy '.env.example' to '.env' and set 'OPENAI_API_KEY'.

## Scripts 
'Data_Collection.py'
'Data_Cleaning.py'
'Index.py'
'Query.py'

## Notes
'.env', 'index/', 'logos/' are git-ignored.
