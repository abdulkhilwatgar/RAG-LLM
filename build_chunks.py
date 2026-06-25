"""
build_chunks.py
---------------
Builds three natural-language chunk types from parsed DataFrames:
  1. Daily summary   — one doc per calendar day
  2. Sleep session   — one doc per night's sleep
  3. Weekly rollup   — one doc per calendar week

Raw numbers embed poorly in vector stores; every metric is expressed as
a prose sentence so the embedding model has semantic signal to work with.

Usage:
    python build_chunks.py
    python build_chunks.py --parsed ./parsed --out ./chunks
"""

import pandas as pd
import json
from pathlib import Path
from datetime import timedelta
import argparse

from parse_health_data import load as load_dataframes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(val, unit="", precision=1, fallback="N/A"):
    if pd.isna(val):
        return fallback
    return f"{val:.{precision}f}{(' ' + unit) if unit else ''}"


def _safe_mean(series):
    s = series.dropna()
    return s.mean() if len(s) else float("nan")


def _safe_sum(series):
    s = series.dropna()
    return s.sum() if len(s) else float("nan")


# ── 1. Daily summary chunks ───────────────────────────────────────────────────

def build_daily_chunks(dfs: dict) -> list[dict]:
    """One natural-language document per calendar day."""
    # Collect all dates that appear across any metric
    all_dates = set()
    for key in ["resting_hr", "hrv", "steps", "active_cal", "vo2max"]:
        if key in dfs:
            all_dates.update(dfs[key]["date"].unique())

    chunks = []
    for date in sorted(all_dates):
        parts = [f"Health summary for {date}:"]

        if "resting_hr" in dfs:
            day_hr = dfs["resting_hr"][dfs["resting_hr"]["date"] == date]["value"]
            if len(day_hr):
                parts.append(f"Resting heart rate: {_fmt(_safe_mean(day_hr), 'bpm', 0)}.")

        if "hrv" in dfs:
            day_hrv = dfs["hrv"][dfs["hrv"]["date"] == date]["value"]
            if len(day_hrv):
                parts.append(f"Heart rate variability (HRV): {_fmt(_safe_mean(day_hrv), 'ms')}.")

        if "steps" in dfs:
            day_steps = dfs["steps"][dfs["steps"]["date"] == date]["value"]
            if len(day_steps):
                parts.append(f"Total steps: {_fmt(_safe_sum(day_steps), precision=0)}.")

        if "active_cal" in dfs:
            day_cal = dfs["active_cal"][dfs["active_cal"]["date"] == date]["value"]
            if len(day_cal):
                parts.append(f"Active calories burned: {_fmt(_safe_sum(day_cal), 'kcal', 0)}.")

        if "vo2max" in dfs:
            day_vo2 = dfs["vo2max"][dfs["vo2max"]["date"] == date]["value"]
            if len(day_vo2):
                parts.append(f"VO2 max estimate: {_fmt(_safe_mean(day_vo2), 'mL/kg/min')}.")

        if "walking_hr" in dfs:
            day_whr = dfs["walking_hr"][dfs["walking_hr"]["date"] == date]["value"]
            if len(day_whr):
                parts.append(f"Average walking heart rate: {_fmt(_safe_mean(day_whr), 'bpm', 0)}.")

        # Only emit chunk if we have at least 2 metrics beyond the header
        if len(parts) >= 3:
            chunks.append({
                "text": " ".join(parts),
                "type": "daily",
                "date": str(date),
            })

    return chunks


# ── 2. Sleep session chunks ───────────────────────────────────────────────────

def _group_sleep_sessions(sleep_df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Group individual sleep segments into discrete overnight sessions.
    Strategy: Apple Watch segments within the same session all share the same
    creationDate batch. We instead group by contiguous time blocks — any gap
    > 90 minutes between segment end and next segment start is a session break.
    """
    # Focus on actual sleep stages and in-bed (not awake micro-segments)
    relevant = sleep_df[sleep_df["stage"].isin(
        ["deep", "rem", "core", "unspecified", "asleep", "in_bed", "awake"]
    )].copy().sort_values("start")

    if relevant.empty:
        return []

    sessions = []
    session_rows = [relevant.iloc[0]]

    for i in range(1, len(relevant)):
        prev_end = session_rows[-1]["end"]
        curr_start = relevant.iloc[i]["start"]
        gap = (curr_start - prev_end).total_seconds() / 60  # minutes

        if gap > 90:
            sessions.append(pd.DataFrame(session_rows))
            session_rows = []

        session_rows.append(relevant.iloc[i])

    if session_rows:
        sessions.append(pd.DataFrame(session_rows))

    return sessions


def build_sleep_chunks(dfs: dict) -> list[dict]:
    """One natural-language document per sleep session."""
    if "sleep" not in dfs:
        return []

    sleep_df = dfs["sleep"]
    sessions = _group_sleep_sessions(sleep_df)

    chunks = []
    for session in sessions:
        start = session["start"].min()
        end = session["end"].max()
        total_min = (end - start).total_seconds() / 60
        total_hrs = total_min / 60

        # Skip sessions shorter than 2 hours (likely naps or noise)
        if total_hrs < 2:
            continue

        session_date = end.tz_convert("America/New_York").date()
        has_stages = session["stage"].isin(["deep", "rem", "core"]).any()

        if has_stages:
            stage_min = session[session["stage"].isin(["deep", "rem", "core", "asleep", "unspecified"])]["duration_min"].sum()
            deep_min  = session[session["stage"] == "deep"]["duration_min"].sum()
            rem_min   = session[session["stage"] == "rem"]["duration_min"].sum()
            core_min  = session[session["stage"] == "core"]["duration_min"].sum()
            unspecified_min = session[session["stage"] == "unspecified"]["duration_min"].sum()

            deep_pct = (deep_min / stage_min * 100) if stage_min else 0
            rem_pct  = (rem_min  / stage_min * 100) if stage_min else 0
            core_pct = (core_min / stage_min * 100) if stage_min else 0
            unspecified_pct = (unspecified_min / stage_min * 100) if stage_min else 0
            awake_min = session[session["stage"] == "awake"]["duration_min"].sum()
            wake_events = len(session[session["stage"] == "awake"])

            text = (
                f"Sleep session on the night of {session_date}: "
                f"Went to sleep around {start.tz_convert('America/New_York').strftime('%I:%M %p')}, "
                f"woke up around {end.tz_convert('America/New_York').strftime('%I:%M %p')}. "
                f"Total time in bed: {total_hrs:.1f} hours. "
                f"Sleep stages — deep sleep: {deep_pct:.0f}% ({deep_min:.0f} min), "
                f"REM sleep: {rem_pct:.0f}% ({rem_min:.0f} min), "
                f"core/light sleep: {core_pct:.0f}% ({core_min:.0f} min)"
                + (f", unclassified sleep: {unspecified_pct:.0f}% ({unspecified_min:.0f} min)" if unspecified_min else "")
                + ". "
                f"Wake events: {wake_events} ({awake_min:.0f} min awake)."
            )
        else:
            # Either legacy data (only has in_bed / asleep) or a session where the
            # Watch recorded sleep but couldn't classify it into stages.
            has_unspecified = session["stage"].isin(["unspecified"]).any()
            note = (
                "(Detailed sleep stages were not recorded for this session.)"
                if has_unspecified else
                "(Sleep stage detail not available for this date — older data.)"
            )
            text = (
                f"Sleep session on the night of {session_date}: "
                f"Went to sleep around {start.tz_convert('America/New_York').strftime('%I:%M %p')}, "
                f"woke up around {end.tz_convert('America/New_York').strftime('%I:%M %p')}. "
                f"Total time in bed: {total_hrs:.1f} hours. "
                f"{note}"
            )

        chunks.append({
            "text": text,
            "type": "sleep",
            "date": str(session_date),
        })

    return chunks


# ── 3. Weekly rollup chunks ───────────────────────────────────────────────────

def build_weekly_chunks(dfs: dict) -> list[dict]:
    """One natural-language document per ISO calendar week, aggregated
    directly from the raw DataFrames (same pattern as build_daily_chunks)."""
    # Collect all dates that appear across any metric
    all_dates = set()
    for key in ["resting_hr", "hrv", "steps", "active_cal", "vo2max"]:
        if key in dfs:
            all_dates.update(dfs[key]["date"].unique())

    if not all_dates:
        return []

    # Group dates into ISO calendar weeks
    weeks: dict[tuple[int, int], list] = {}
    for date in all_dates:
        iso = date.isocalendar()
        weeks.setdefault((iso.year, iso.week), []).append(date)

    chunks = []
    for (year, week), week_dates in sorted(weeks.items()):
        week_dates = sorted(week_dates)
        num_days = len(week_dates)

        def metric_values(key):
            if key not in dfs:
                return pd.Series(dtype=float)
            df = dfs[key]
            return df[df["date"].isin(week_dates)]["value"]

        hr_vals    = metric_values("resting_hr")
        hrv_vals   = metric_values("hrv")
        steps_vals = metric_values("steps")
        cal_vals   = metric_values("active_cal")
        vo2_vals   = metric_values("vo2max")

        parts = [
            f"Weekly health summary for {year}-W{week:02d} "
            f"({week_dates[0]} to {week_dates[-1]}), covering {num_days} days:"
        ]

        if hr_vals.notna().any():
            parts.append(f"Average resting heart rate: {_safe_mean(hr_vals):.0f} bpm.")
        if hrv_vals.notna().any():
            parts.append(f"Average HRV: {_safe_mean(hrv_vals):.1f} ms.")
        if steps_vals.notna().any():
            total_steps = _safe_sum(steps_vals)
            parts.append(f"Total steps: {total_steps:,.0f} (avg {total_steps / num_days:,.0f}/day).")
        if cal_vals.notna().any():
            total_cal = _safe_sum(cal_vals)
            parts.append(f"Total active calories: {total_cal:,.0f} kcal (avg {total_cal / num_days:,.0f}/day).")
        if vo2_vals.notna().any():
            parts.append(f"Average VO2 max estimate: {_safe_mean(vo2_vals):.1f} mL/kg/min.")

        if len(parts) >= 2:
            chunks.append({
                "text": " ".join(parts),
                "type": "weekly",
                "date": str(week_dates[0]),   # week start date
            })

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def build_all(parsed_dir: str = "./parsed") -> dict[str, list[dict]]:
    print("Loading parsed DataFrames…")
    dfs = load_dataframes(parsed_dir)

    if not dfs:
        raise RuntimeError(f"No Parquet files found in {parsed_dir}. Run parse_health_data.py first.")

    print("Building daily chunks…")
    daily = build_daily_chunks(dfs)
    print(f"  {len(daily):,} daily chunks")

    print("Building sleep session chunks…")
    sleep = build_sleep_chunks(dfs)
    print(f"  {len(sleep):,} sleep session chunks")

    print("Building weekly rollup chunks…")
    weekly = build_weekly_chunks(dfs)
    print(f"  {len(weekly):,} weekly chunks")

    return {"daily": daily, "sleep": sleep, "weekly": weekly}


def save_chunks(chunks: dict[str, list[dict]], out_dir: str = "./chunks"):
    Path(out_dir).mkdir(exist_ok=True)
    for chunk_type, data in chunks.items():
        path = Path(out_dir) / f"{chunk_type}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved {len(data):,} chunks → {path}")


def load_chunks(chunks_dir: str = "./chunks") -> list[dict]:
    """Load all chunk types and return a flat list."""
    all_chunks = []
    for name in ["daily", "sleep", "weekly"]:
        path = Path(chunks_dir) / f"{name}.json"
        if path.exists():
            with open(path) as f:
                all_chunks.extend(json.load(f))
    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", default="./parsed", help="Directory of Parquet files from parse_health_data.py")
    parser.add_argument("--out",    default="./chunks",  help="Output directory for JSON chunk files")
    args = parser.parse_args()

    chunks = build_all(args.parsed)
    save_chunks(chunks, args.out)

    total = sum(len(v) for v in chunks.values())
    print(f"\nTotal chunks built: {total:,}")

    # Show a sample of each type
    print("\n── Sample chunks ─────────────────────────────────────────────────")
    for chunk_type, data in chunks.items():
        if data:
            print(f"\n[{chunk_type}]\n{data[0]['text'][:300]}…")
