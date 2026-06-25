"""
app.py
------
Gradio chat UI with two modes:
  • Health Analyst  — general trends, patterns, anomalies
  • Sleep Coach     — sleep-focused analysis and suggestions

Requires:
  - Ollama running locally with Mistral pulled: `ollama pull mistral`
  - ChromaDB index built: `python index_health_data.py`

Usage:
    python app.py
    python app.py --db ./health_db --model mistral --port 7860
"""

import argparse
import calendar
import datetime
import json
import re
import urllib.error
import urllib.request

import gradio as gr
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

from index_health_data import load_index
from query_router import try_aggregation_answer


# ── Ollama status check ─────────────────────────────────────────────────────────

OLLAMA_HOST = "http://localhost:11434"


def check_ollama_status(model_name: str) -> str:
    """Ping the local Ollama server and report whether it's running and the model is pulled."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError):
        return f"🔴 **Model:** `{model_name}` — Ollama not running. Start it with `ollama serve`."

    names = [m.get("name", "") for m in data.get("models", [])]
    if any(n == model_name or n.split(":")[0] == model_name for n in names):
        return f"🟢 **Model:** `{model_name}` — Ollama running"
    return f"🟡 **Model:** `{model_name}` — Ollama running, but not pulled. Run `ollama pull {model_name}`."


# ── System prompts ─────────────────────────────────────────────────────────────

ANALYST_SYSTEM = """You are a personal health analyst with access to the user's
Apple Watch data spanning several years. Your job is to answer questions about
trends, patterns, and anomalies in their health data.

Guidelines:
- Always cite the specific dates or time ranges your answer draws from.
- Be precise — this is personal health data, not general advice.
- If the retrieved context doesn't contain data relevant to the question,
  say "I don't have data for that" rather than guessing.
- When spotting a notable pattern, briefly explain what it might mean.
- Keep answers clear and direct. Avoid unnecessary medical disclaimers."""

SLEEP_SYSTEM = """You are a sleep coach analyzing the user's Apple Watch sleep data.
You focus on sleep duration, stages (deep, REM, core/light), consistency,
and how sleep connects to next-day metrics like HRV and resting heart rate.

Guidelines:
- Always reference the specific dates you're drawing from.
- Highlight concrete patterns: is deep sleep consistently low? Are there nights
  with notably poor recovery (low HRV the next morning)?
- Suggest evidence-based improvements grounded in the user's actual data,
  not generic sleep hygiene tips.
- If stage data is unavailable for a date range (older data), say so and work
  with what's available (total duration, in-bed time).
- Keep answers actionable and specific to this user's patterns."""

QA_TEMPLATE = PromptTemplate(
    "Context from your Apple Watch health data:\n"
    "─────────────────────────────────────────\n"
    "{context_str}\n"
    "─────────────────────────────────────────\n\n"
    "{system_prompt}\n\n"
    "User question: {query_str}\n\n"
    "Answer:"
)


# ── Date-range extraction ───────────────────────────────────────────────────────

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_QUARTER_RE = re.compile(r"\bQ([1-4])\s+(\d{4})\b", re.IGNORECASE)
_MONTH_NAMES = {}
for _i, _name in enumerate(calendar.month_name):
    if _name:
        _MONTH_NAMES[_name.lower()] = _i
for _i, _abbr in enumerate(calendar.month_abbr):
    if _abbr:
        _MONTH_NAMES[_abbr.lower()] = _i
_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")\.?\s+(\d{4})\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def extract_date_range(query: str) -> tuple[str, str] | None:
    """Detect a specific date, month, quarter, or year mentioned in `query`
    and return (start_date, end_date) as ISO "YYYY-MM-DD" strings, or None
    if the query doesn't mention a date range."""
    m = _ISO_DATE_RE.search(query)
    if m:
        return m.group(0), m.group(0)

    m = _QUARTER_RE.search(query)
    if m:
        quarter, year = int(m.group(1)), int(m.group(2))
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        last_day = calendar.monthrange(year, end_month)[1]
        return f"{year}-{start_month:02d}-01", f"{year}-{end_month:02d}-{last_day:02d}"

    m = _MONTH_YEAR_RE.search(query)
    if m:
        month, year = _MONTH_NAMES[m.group(1).lower()], int(m.group(2))
        last_day = calendar.monthrange(year, month)[1]
        return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"

    m = _YEAR_RE.search(query)
    if m:
        year = int(m.group(0))
        return f"{year}-01-01", f"{year}-12-31"

    return None


# ── Retrieval ───────────────────────────────────────────────────────────────────

class RecencyAwareRetriever(BaseRetriever):
    """Vector similarity retrieval, plus a handful of the earliest and most
    recent daily/weekly chunks as fixed "anchor" context.

    The daily/weekly chunks are short, template-like prose ("Health summary
    for 2020-03-06: Total steps: ..."), and bge-small-en embeds them with
    cosine similarity 0.95+ to *each other* regardless of date or content.
    That collapse means pure vector top-k can be dominated by a handful of
    generic-looking chunks from one period (often 2020) no matter what's
    asked — including questions about the most recent data or the overall
    date range. Anchoring a few chunks from each end of the timeline ensures
    the model always sees what data is actually available.
    """

    def __init__(self, index, similarity_top_k: int = 4, anchor_k: int = 2):
        self._index = index
        self._similarity_top_k = similarity_top_k
        self._vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
        self._collection_size = len(index.vector_store.get_nodes(node_ids=None))
        self._anchors = self._build_anchors(index.vector_store, anchor_k)
        super().__init__()

    @staticmethod
    def _build_anchors(vector_store, anchor_k: int) -> list[NodeWithScore]:
        anchors = []
        for chunk_type in ("daily", "weekly"):
            filters = MetadataFilters(filters=[MetadataFilter(key="type", value=chunk_type)])
            nodes = sorted(
                vector_store.get_nodes(node_ids=None, filters=filters),
                key=lambda n: n.metadata["date"],
            )
            for n in nodes[:anchor_k] + nodes[-anchor_k:]:
                anchors.append(NodeWithScore(node=n, score=None))
        return anchors

    def _retrieve(self, query_bundle) -> list[NodeWithScore]:
        date_range = extract_date_range(query_bundle.query_str)
        if date_range:
            start, end = date_range
            # ChromaDB's $gte/$lte filters require numeric metadata, but our
            # "date" field is stored as an ISO string ("YYYY-MM-DD"), so we
            # can't push a range filter down to the vector store. Instead,
            # rank the whole collection by similarity and restrict to chunks
            # whose date falls in range before taking the top-k.
            all_retriever = VectorIndexRetriever(index=self._index, similarity_top_k=self._collection_size)
            ranked = all_retriever.retrieve(query_bundle)
            in_range = [n for n in ranked if start <= n.node.metadata.get("date", "") <= end]
            nodes = (in_range or ranked)[: self._similarity_top_k]
        else:
            nodes = self._vector_retriever.retrieve(query_bundle)

        seen_ids = {n.node.node_id for n in nodes}
        for anchor in self._anchors:
            if anchor.node.node_id not in seen_ids:
                nodes.append(anchor)
                seen_ids.add(anchor.node.node_id)
        return nodes


# ── Source formatting ────────────────────────────────────────────────────────────

_SOURCE_TYPE_LABELS = {
    "daily": "Daily summary",
    "sleep": "Sleep session",
    "weekly": "Weekly rollup",
}


def _format_source(node) -> str:
    """Render one retrieved node as a '- <label> · <date>' bullet, using the
    ISO week ('2024-W11') for weekly rollups rather than their start date."""
    chunk_type = node.metadata.get("type", "unknown")
    label = _SOURCE_TYPE_LABELS.get(chunk_type, chunk_type)
    date_str = node.metadata.get("date", "")

    if chunk_type == "weekly" and date_str:
        try:
            d = datetime.date.fromisoformat(date_str)
            iso = d.isocalendar()
            date_str = f"{iso.year}-W{iso.week:02d}"
        except ValueError:
            pass

    return f"- {label} · {date_str}"


def _format_sources_section(source_nodes) -> str:
    if not source_nodes:
        return ""
    lines = [_format_source(n.node) for n in source_nodes]
    count = len(source_nodes)
    return (
        "\n\n---\n"
        f"**Sources used:** {count} chunk{'s' if count != 1 else ''}\n"
        + "\n".join(lines)
    )


# ── Query engine factory ───────────────────────────────────────────────────────

def make_query_engine(index, llm, mode: str, top_k: int = 4):
    """Build a query engine with the appropriate system prompt injected."""
    system = ANALYST_SYSTEM if mode == "analyst" else SLEEP_SYSTEM

    retriever = RecencyAwareRetriever(index=index, similarity_top_k=top_k)

    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import get_response_synthesizer

    qa_prompt = PromptTemplate(
        "Context from your Apple Watch health data:\n"
        "─────────────────────────────────────────\n"
        "{context_str}\n"
        "─────────────────────────────────────────\n\n"
        + system + "\n\n"
        "User question: {query_str}\n\n"
        "Answer:"
    )

    synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=qa_prompt,
        response_mode="compact",
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def build_ui(db_path: str, model_name: str, top_k: int):
    print(f"Loading index from {db_path}…")
    index = load_index(db_path)

    print(f"Connecting to Ollama ({model_name})…")
    llm = Ollama(model=model_name, request_timeout=180.0)
    Settings.llm = llm

    analyst_engine = make_query_engine(index, llm, "analyst", top_k)
    sleep_engine   = make_query_engine(index, llm, "sleep",   top_k)

    def respond(message: str, history: list, mode: str):
        engine = analyst_engine if mode == "Health Analyst" else sleep_engine

        # Aggregation/counting/statistics questions ("how many days was my
        # HRV above 50?") are answered directly from the parsed data —
        # RAG can't do exact arithmetic over the full history.
        try:
            agg_answer = try_aggregation_answer(message, llm)
        except Exception:
            agg_answer = None
        if agg_answer is not None:
            return agg_answer

        # Prepend recent conversation turns so the engine has context for
        # follow-up questions (e.g. "what about the week before that?").
        recent_turns = [
            f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
            for turn in history[-6:]
            if isinstance(turn.get("content"), str)
        ]
        query = message
        if recent_turns:
            query = (
                "Previous conversation:\n"
                + "\n".join(recent_turns)
                + f"\n\nCurrent question: {message}"
            )

        try:
            response = engine.query(query)
            answer = str(response) + _format_sources_section(response.source_nodes)
        except Exception as e:
            answer = f"Error: {e}\n\nMake sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull {model_name}`)."
        return answer

    with gr.Blocks(title="Health Assistant") as demo:
        gr.Markdown(
            "# 🫀 Personal Health Assistant\n"
            "Ask questions about your Apple Watch data. "
            "Switch modes to focus on sleep or general health trends."
        )

        with gr.Row():
            mode = gr.Radio(
                choices=["Health Analyst", "Sleep Coach"],
                value="Health Analyst",
                label="Mode",
                interactive=True,
            )
            status_box = gr.Markdown(check_ollama_status(model_name))

        gr.Timer(10).tick(lambda: check_ollama_status(model_name), outputs=status_box)

        chatbot = gr.ChatInterface(
            fn=respond,
            additional_inputs=[mode],
            examples=[
                ["How has my resting heart rate trended over the past year?", "Health Analyst"],
                ["What are my typical deep sleep and REM percentages?", "Sleep Coach"],
                ["Are there patterns in my sleep quality — any consistently bad nights?", "Sleep Coach"],
            ],
            cache_examples=False,
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",    default="./health_db", help="Path to ChromaDB directory")
    parser.add_argument("--model", default="mistral",     help="Ollama model name")
    parser.add_argument("--top_k", default=4, type=int,   help="Number of chunks retrieved per query")
    parser.add_argument("--port",  default=7860, type=int, help="Gradio port")
    args = parser.parse_args()

    demo = build_ui(db_path=args.db, model_name=args.model, top_k=args.top_k)
    demo.launch(server_port=args.port, share=False, theme=gr.themes.Soft())
