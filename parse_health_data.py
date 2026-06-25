"""
parse_health_data.py
--------------------
Streams Apple Health export.xml using iterparse (memory-safe) and builds
flat DataFrames for each metric type. Run this once; outputs are saved
to ./parsed/ as Parquet files for fast reuse.

Usage:
    python parse_health_data.py
    python parse_health_data.py --xml path/to/export.xml
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import argparse
import sys

# Windows consoles default to cp1252, which can't encode the arrow/box-drawing
# characters used in the progress and summary output below.
sys.stdout.reconfigure(encoding="utf-8")

# ── Metric type strings ────────────────────────────────────────────────────────
TYPES = {
    "resting_hr":   "HKQuantityTypeIdentifierRestingHeartRate",
    "hrv":          "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    "steps":        "HKQuantityTypeIdentifierStepCount",
    "active_cal":   "HKQuantityTypeIdentifierActiveEnergyBurned",
    "sleep":        "HKCategoryTypeIdentifierSleepAnalysis",
    "vo2max":       "HKQuantityTypeIdentifierVO2Max",
    "walking_hr":   "HKQuantityTypeIdentifierWalkingHeartRateAverage",
}

# Reverse map: type string → friendly name
TYPE_TO_KEY = {v: k for k, v in TYPES.items()}

# Sleep stage values
SLEEP_STAGES = {
    "HKCategoryValueSleepAnalysisAsleepDeep":         "deep",
    "HKCategoryValueSleepAnalysisAsleepREM":          "rem",
    "HKCategoryValueSleepAnalysisAsleepCore":         "core",
    "HKCategoryValueSleepAnalysisAsleepUnspecified":  "unspecified",  # Watch sleep that couldn't be stage-classified
    "HKCategoryValueSleepAnalysisAsleep":      "asleep",   # legacy, no stage
    "HKCategoryValueSleepAnalysisInBed":       "in_bed",
    "HKCategoryValueSleepAnalysisAwake":       "awake",
}

TARGET_TYPES = set(TYPES.values())


def stream_records(filepath: str):
    """Yield Record attrib dicts one at a time — never loads the full file."""
    for event, elem in ET.iterparse(filepath, events=["end"]):
        if elem.tag == "Record":
            yield elem.attrib
            elem.clear()


def parse(xml_path: str) -> dict[str, pd.DataFrame]:
    """
    Stream export.xml and return a dict of DataFrames keyed by metric name.
    Date/time columns are parsed as timezone-aware datetimes.
    """
    rows: dict[str, list] = {k: [] for k in TYPES}
    total = 0

    print(f"Streaming {xml_path} …", flush=True)
    for attrib in stream_records(xml_path):
        rec_type = attrib.get("type", "")
        if rec_type not in TARGET_TYPES:
            continue

        key = TYPE_TO_KEY[rec_type]
        total += 1
        if total % 50_000 == 0:
            print(f"  {total:,} records processed…", flush=True)

        row = {
            "start":  attrib.get("startDate"),
            "end":    attrib.get("endDate"),
            "value":  attrib.get("value"),
            "source": attrib.get("sourceName"),
        }
        rows[key].append(row)

    print(f"Done. {total:,} matching records found.\n")

    dfs = {}
    for key, data in rows.items():
        if not data:
            print(f"  WARNING: no records found for '{key}'")
            continue

        df = pd.DataFrame(data)

        # Parse datetimes (Apple Health uses "yyyy-MM-dd HH:mm:ss ±HHMM" format)
        df["start"] = pd.to_datetime(df["start"], format="%Y-%m-%d %H:%M:%S %z", utc=True)
        df["end"]   = pd.to_datetime(df["end"],   format="%Y-%m-%d %H:%M:%S %z", utc=True)

        # Derive a local-time date column (Eastern assumed; adjust if needed)
        df["date"] = df["start"].dt.tz_convert("America/New_York").dt.date

        if key == "sleep":
            # Map stage strings to friendly names
            df["stage"] = df["value"].map(SLEEP_STAGES).fillna("unknown")
            df["duration_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60
        else:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        dfs[key] = df

    return dfs


def save(dfs: dict[str, pd.DataFrame], out_dir: str = "./parsed"):
    Path(out_dir).mkdir(exist_ok=True)
    for key, df in dfs.items():
        path = f"{out_dir}/{key}.parquet"
        df.to_parquet(path, index=False)
        print(f"  Saved {len(df):,} rows → {path}")


def load(out_dir: str = "./parsed") -> dict[str, pd.DataFrame]:
    """Load previously parsed Parquet files."""
    dfs = {}
    for key in TYPES:
        path = Path(out_dir) / f"{key}.parquet"
        if path.exists():
            dfs[key] = pd.read_parquet(path)
    return dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="export.xml", help="Path to export.xml")
    parser.add_argument("--out", default="./parsed",   help="Output directory for Parquet files")
    args = parser.parse_args()

    if not Path(args.xml).exists():
        print(f"ERROR: {args.xml} not found.", file=sys.stderr)
        sys.exit(1)

    dfs = parse(args.xml)
    save(dfs, args.out)

    # Quick sanity-check summary
    print("\n── Summary ───────────────────────────────")
    for key, df in dfs.items():
        date_range = f"{df['date'].min()} → {df['date'].max()}"
        print(f"  {key:12s}  {len(df):>8,} rows   {date_range}")
