# SHL Assessment Recommender -- Setup

## 1. Install dependencies

```bash
cd shl-recommender
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

(You're on conda with a `(base)` env already active, so `python3 -m venv venv`
+ activating it keeps this project's packages separate from your base env --
recommended, but you can skip the venv steps and just `pip install -r
requirements.txt` directly into base if you'd rather.)

## 2. Add your Groq API key

```bash
cp .env.example .env
```

Open `.env` and paste your key:
```
GROQ_API_KEY=gsk_...your key...
```

## 3. (Optional but recommended) Build dense embeddings

This step needs internet access to download the embedding model from
Hugging Face (one-time, ~90MB). If you skip it, the app still works using
TF-IDF-only retrieval -- just weaker on synonym matching.

```bash
python app/ingestion/clean.py          # regenerates cleaned_catalog.json (already done, safe to re-run)
python app/ingestion/embed_catalog.py  # builds data/vector_store/catalog_dense.npy
```

## 4. Run the eval harness against the 10 provided traces

This replays each trace's real user turns against our agent and reports
recall + a couple of behavior checks, without needing to deploy anything:

```bash
export GROQ_API_KEY=$(grep GROQ_API_KEY .env | cut -d= -f2)
python eval/run_eval.py
```

(Don't use `export $(cat .env | xargs)` -- the comment lines in `.env` break it, as you saw.)

Paste me the full output -- that's what we'll use to find and fix whatever
the agent gets wrong.

## 5. Run the API locally

```bash
export GROQ_API_KEY=$(grep GROQ_API_KEY .env | cut -d= -f2)
uvicorn app.main:app --reload --port 8000
```

Then in another terminal:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need an assessment for hiring a Java developer"}]}'
```

## Project layout

```
app/
  main.py                 FastAPI app: GET /health, POST /chat
  models.py                Pydantic request/response schemas
  catalog/
    raw_catalog.json       original scrape (377 items)
    cleaned_catalog.json   normalized (test_type codes derived, durations parsed)
  ingestion/
    clean.py                raw -> cleaned
    embed_catalog.py         cleaned -> dense embeddings (offline build step)
  retrieval/
    retriever.py             hybrid TF-IDF + dense + name-match retrieval
  agent/
    prompts.py                system prompt + candidate formatting
    guard.py                  fast prompt-injection backstop
    llm_client.py             Groq API wrapper
    agent.py                  orchestration: retrieve -> LLM -> validate/ground
data/
  vector_store/              dense embeddings live here after step 3
eval/
  traces/                    the 10 provided conversation traces
  parse_trace.py              markdown trace -> structured turns
  run_eval.py                  replays traces against our agent, reports recall
```
