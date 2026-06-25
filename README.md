# Apple Watch Health Assistant

A fully local RAG system that lets you have a conversation with your Apple Watch data. Ask natural language questions about your health trends, sleep patterns, HRV, steps, and more — all without sending any data to an external API.

## What It Does

Your Apple Health export is a massive XML file that no LLM can fit in its context window. This project solves that by building a retrieval pipeline on top of it:

1. **Parse** — streams the XML with `iterparse` to avoid memory spikes, extracts resting HR, HRV, steps, active calories, VO2 max, and sleep data into DataFrames
2. **Chunk** — converts raw numbers into natural-language prose (raw numbers embed poorly), generating three chunk types: daily summaries, per-sleep-session chunks, and weekly rollups
3. **Index** — embeds all chunks into a persistent ChromaDB vector store using `BAAI/bge-base-en-v1.5` running locally
4. **Query** — a custom `RecencyAwareRetriever` handles retrieval, anchoring chunks from both ends of the timeline so the model always knows the full date range of available data
5. **Answer** — LlamaIndex routes the query to Ollama (Mistral 7B Q4), with two system prompt modes: **Health Analyst** for general trends and **Sleep Coach** for sleep-focused analysis

Aggregation queries ("how many days was my HRV above 50?") are handled by a separate pandas-based router rather than the RAG layer, since vector search can't do exact arithmetic over the full history.

## Stack

| Component | Tool |
|---|---|
| LLM inference | Ollama (Mistral 7B Q4) |
| Embeddings | BAAI/bge-base-en-v1.5 (local) |
| Vector store | ChromaDB (persistent) |
| Orchestration | LlamaIndex |
| UI | Gradio |

Runs entirely on CPU/local GPU. No API keys required.

## Setup

**Prerequisites:** Ollama installed and running, Python 3.10+

```bash
# Install dependencies
pip install -r requirements.txt

# Pull the model
ollama pull mistral

# Export your Apple Health data:
# Health app → profile picture → Export All Health Data → unzip → get export.xml

# Parse the XML into DataFrames
python parse_health_data.py

# Build natural-language chunks
python build_chunks.py

# Embed and index into ChromaDB
python index_health_data.py

# Launch the app
python app.py
```

The app runs at `http://localhost:7860` by default.

## Usage

Once running, you can ask questions like:

- *"How has my resting heart rate trended over the past year?"*
- *"What are my typical deep sleep and REM percentages?"*
- *"Were there any weeks in 2023 where my HRV dropped significantly?"*
- *"How many days did I hit 10,000 steps last month?"*

Switch between **Health Analyst** and **Sleep Coach** modes in the UI to adjust the system prompt focus.

## Project Structure

```
RAG_LLM/
├── parse_health_data.py   # XML → DataFrames
├── build_chunks.py        # DataFrames → natural-language chunks
├── index_health_data.py   # Chunks → ChromaDB
├── query_router.py        # Aggregation query handler
├── app.py                 # Gradio UI + retrieval logic
├── requirements.txt
├── chunks/                # Generated chunk JSON files
└── health_db/             # Persistent ChromaDB directory
```

## Design Notes

**Why prose chunks instead of raw numbers?** Embedding models treat numbers as tokens with no inherent semantic meaning. "Resting heart rate: 58 bpm" embeds more usefully than a bare `58`.

**Why the recency-aware retriever?** `bge-small-en` embeds daily summaries with cosine similarity ~0.95+ regardless of content, collapsing retrieval onto one time period. Anchoring chunks from the start and end of the timeline ensures the model knows the full span of available data.

**Why a separate aggregation router?** RAG can't count. Questions like "how many days" require iterating over the full dataset — a pandas query is the right tool.
