"""
query_router.py
----------------
Lightweight aggregation query router for the Health Assistant.

RAG (vector retrieval) is good at "what does my data look like" questions,
but bad at exact counting and arithmetic ("how many days was my HRV above
50?", "what was my average step count in Q1 2024?"). This module detects
those aggregation-style questions and answers them directly from the parsed
Parquet data in ./parsed/ using pandas, bypassing the vector retriever
entirely.

Usage:
    from query_router import try_aggregation_answer
    answer = try_aggregation_answer(message, llm)
    if answer is not None:
        # use `answer` directly — skip the RAG query engine
    else:
        # not an aggregation question — fall back to RAG
"""

import datetime
import json
import re
from pathlib import Path

import pandas as pd


PARSED_DIR = Path("./parsed")

# metric key -> (display label, unit, daily aggregation: "mean" or "sum")
METRICS = {
    "resting_hr": ("resting heart rate", "bpm", "mean"),
    "hrv": ("HRV (heart rate variability)", "ms", "mean"),
    "steps": ("step count", "steps", "sum"),
    "active_cal": ("active calories", "kcal", "sum"),
    "vo2max": ("VO2 max", "mL/kg/min", "mean"),
    "walking_hr": ("walking heart rate", "bpm", "mean"),
}

# Fast pre-filter — only pay for the LLM extraction call if the query looks
# like it's asking for a count, total, average, or extreme.
_AGGREGATION_KEYWORDS = re.compile(
    r"\b(how many|average|avg|total|sum|highest|lowest|maximum|minimum|"
    r"max|min|most|least|count|which (month|week|day|year)|"
    r"above|below|over|under|greater than|less than|more than|fewer than)\b",
    re.IGNORECASE,
)

_EXTRACTION_PROMPT = """You convert health-data questions into a JSON aggregation spec.

Today's date is {today}.

Available metrics (use these exact keys):
- resting_hr: resting heart rate (bpm)
- hrv: heart rate variability (ms)
- steps: step count (steps)
- active_cal: active calories burned (kcal)
- vo2max: VO2 max estimate (mL/kg/min)
- walking_hr: walking heart rate (bpm)

Available operations:
- "count_above": count days where the daily value is above a threshold
- "count_below": count days where the daily value is below a threshold
- "average": average daily value over a period
- "total": sum of daily values over a period
- "max_day": the single day with the highest daily value
- "min_day": the single day with the lowest daily value
- "max_period": the month (or week) with the highest average/total
- "min_period": the month (or week) with the lowest average/total

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"is_aggregation": true or false, "metric": "<metric key>" or null, "operation": "<operation>" or null, "threshold": <number> or null, "start_date": "YYYY-MM-DD" or null, "end_date": "YYYY-MM-DD" or null, "group_by": "month" or "week" or null}}

If the question can't be answered with a single metric and operation from the
lists above, set "is_aggregation" to false and leave the other fields null.

Only set "start_date"/"end_date" if the question explicitly names a date,
month, quarter, or year (e.g. "in March 2024", "in 2023", "Q1 2024"). If the
question doesn't mention a time period, set both to null — do NOT default to
the current year.

Examples:
Q: "How many days was my HRV above 50?"
{{"is_aggregation": true, "metric": "hrv", "operation": "count_above", "threshold": 50, "start_date": null, "end_date": null, "group_by": null}}

Q: "What was my average step count in Q1 2024?"
{{"is_aggregation": true, "metric": "steps", "operation": "average", "threshold": null, "start_date": "2024-01-01", "end_date": "2024-03-31", "group_by": null}}

Q: "Which month had my highest resting heart rate?"
{{"is_aggregation": true, "metric": "resting_hr", "operation": "max_period", "threshold": null, "start_date": null, "end_date": null, "group_by": "month"}}

Q: "How has my sleep quality changed over time?"
{{"is_aggregation": false, "metric": null, "operation": null, "threshold": null, "start_date": null, "end_date": null, "group_by": null}}

Q: "{query}"
"""


def _looks_like_aggregation(query: str) -> bool:
    """Cheap keyword check so we don't pay for an LLM call on every query."""
    return bool(_AGGREGATION_KEYWORDS.search(query))


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _daily_series(df: pd.DataFrame, agg: str) -> pd.Series:
    if agg == "sum":
        return df.groupby("date")["value"].sum()
    return df.groupby("date")["value"].mean()


def execute_spec(spec: dict, parsed_dir: str | Path = PARSED_DIR) -> str | None:
    """Compute the answer for a parsed aggregation spec, or return None if
    the spec is incomplete/invalid/has no matching data."""
    metric = spec.get("metric")
    op = spec.get("operation")
    if metric not in METRICS or op is None:
        return None

    label, unit, agg = METRICS[metric]
    path = Path(parsed_dir) / f"{metric}.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)

    start, end = spec.get("start_date"), spec.get("end_date")
    period_desc = ""
    if start:
        df = df[df["date"] >= datetime.date.fromisoformat(start)]
    if end:
        df = df[df["date"] <= datetime.date.fromisoformat(end)]
    if start or end:
        period_desc = f" between {start or 'the start of your data'} and {end or 'now'}"

    if df.empty:
        return f"I don't have any {label} data{period_desc}."

    daily = _daily_series(df, agg)

    if op in ("count_above", "count_below"):
        threshold = spec.get("threshold")
        if threshold is None:
            return None
        if op == "count_above":
            count = int((daily > threshold).sum())
            comparison = "above"
        else:
            count = int((daily < threshold).sum())
            comparison = "below"
        return (
            f"Your {label} was {comparison} {threshold} {unit} on "
            f"{count} of {len(daily)} day(s) with data{period_desc}."
        )

    if op == "average":
        return f"Your average {label}{period_desc} was {daily.mean():,.1f} {unit}/day."

    if op == "total":
        return f"Your total {label}{period_desc} was {daily.sum():,.0f} {unit}."

    if op == "max_day":
        d = daily.idxmax()
        return f"Your highest {label} was {daily.loc[d]:,.1f} {unit}, on {d}."

    if op == "min_day":
        d = daily.idxmin()
        return f"Your lowest {label} was {daily.loc[d]:,.1f} {unit}, on {d}."

    if op in ("max_period", "min_period"):
        group_by = spec.get("group_by") or "month"
        if group_by == "week":
            period_keys = [f"{d.isocalendar().year}-W{d.isocalendar().week:02d}" for d in daily.index]
        else:
            period_keys = [f"{d.year}-{d.month:02d}" for d in daily.index]

        grouped = daily.groupby(period_keys)
        grouped = grouped.mean() if agg == "mean" else grouped.sum()
        if grouped.empty:
            return None

        key = grouped.idxmax() if op == "max_period" else grouped.idxmin()
        val = grouped.loc[key]
        which = "highest" if op == "max_period" else "lowest"
        how = "average" if agg == "mean" else "total"
        return f"{key} had your {which} {label}: {val:,.1f} {unit} ({how})."

    return None


def try_aggregation_answer(query: str, llm, parsed_dir: str | Path = PARSED_DIR) -> str | None:
    """If `query` is an aggregation/counting/statistics question, compute and
    return the answer directly from ./parsed/. Otherwise return None so the
    caller can fall back to the RAG query engine."""
    if not _looks_like_aggregation(query):
        return None

    prompt = _EXTRACTION_PROMPT.format(today=datetime.date.today().isoformat(), query=query)
    try:
        response = llm.complete(prompt)
        spec = _extract_json(str(response))
    except Exception:
        return None

    if not spec or not spec.get("is_aggregation"):
        return None

    return execute_spec(spec, parsed_dir=parsed_dir)
