"""Extract and use the official quarterly HDB Resale Price Index locally.

The supplied PDF lists quarters in reverse chronological order. PDFKit exposes
the table rows before the separately positioned year labels, so the parser
validates the complete expected quarter sequence rather than pairing text by
visual line position.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "data" / "raw" / "reference" / "rpi-table.pdf"
OUTPUT_PATH = ROOT / "data" / "processed" / "hdb_rpi.csv"
ROW_PATTERN = re.compile(r"(?m)^([1-4])Q\s+(\d+(?:\.\d+)?)\s*(?:([-+]?\d+(?:\.\d+)?)%)?\s*$")


def extract_pdf_text(path: Path) -> str:
    """Extract PDF text with built-in macOS PDFKit; no package install needed."""
    swift = r'''
import Foundation
import PDFKit
let path = CommandLine.arguments.last!
guard let document = PDFDocument(url: URL(fileURLWithPath: path)) else {
    fatalError("Unable to open PDF")
}
for index in 0..<document.pageCount {
    print(document.page(at: index)?.string ?? "")
}
'''
    cache = Path(tempfile.gettempdir()) / "singapore-property-swift-cache"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["swift", "-module-cache-path", str(cache), "-e", swift, str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            "Could not extract RPI PDF with macOS PDFKit. Use the already generated "
            "data/processed/hdb_rpi.csv or run this command on macOS with Swift available."
        ) from error
    return result.stdout


def previous_quarter(year: int, quarter: int) -> tuple[int, int]:
    return (year - 1, 4) if quarter == 1 else (year, quarter - 1)


def parse_rpi_text(text: str) -> pd.DataFrame:
    rows = ROW_PATTERN.findall(text)
    if len(rows) != 146:
        raise ValueError(f"Expected 146 quarterly rows, extracted {len(rows)}")
    expected_year, expected_quarter = 2026, 2
    records: list[dict[str, object]] = []
    for quarter_text, rpi_text, change_text in rows:
        quarter = int(quarter_text)
        if quarter != expected_quarter:
            raise ValueError(
                f"Unexpected quarter order: expected {expected_year}-Q{expected_quarter}, found Q{quarter}"
            )
        records.append({
            "year": expected_year,
            "quarter": quarter,
            "rpi": float(rpi_text),
            "qoq_percent": float(change_text) if change_text else pd.NA,
        })
        expected_year, expected_quarter = previous_quarter(expected_year, expected_quarter)
    frame = pd.DataFrame(records).sort_values(["year", "quarter"]).reset_index(drop=True)
    frame["quarter_start"] = pd.PeriodIndex(
        frame["year"].astype(str) + "Q" + frame["quarter"].astype(str), freq="Q"
    ).start_time
    frame["source"] = "HDB rpi-table.pdf (2009-Q1=100)"
    validate_rpi(frame)
    return frame


def validate_rpi(frame: pd.DataFrame) -> None:
    if len(frame) != 146 or frame[["year", "quarter"]].duplicated().any():
        raise ValueError("RPI row count/quarter uniqueness failed")
    period = pd.PeriodIndex(
        frame["year"].astype(str) + "Q" + frame["quarter"].astype(str), freq="Q"
    )
    expected = pd.period_range("1990Q1", "2026Q2", freq="Q")
    if not period.equals(expected):
        raise ValueError("RPI quarters are not a complete 1990-Q1–2026-Q2 sequence")
    checks = {(1990, 1): 24.3, (2009, 1): 100.0, (2026, 2): 202.8}
    lookup = frame.set_index(["year", "quarter"])["rpi"]
    for key, expected_value in checks.items():
        if float(lookup.loc[key]) != expected_value:
            raise ValueError(f"RPI spot check failed for {key}: {lookup.loc[key]}")
    calculated = frame["rpi"].pct_change(fill_method=None).mul(100).round(1)
    supplied = pd.to_numeric(frame["qoq_percent"], errors="coerce")
    mismatch = supplied.notna() & calculated.ne(supplied)
    if mismatch.any():
        bad = frame.loc[mismatch, ["year", "quarter", "rpi", "qoq_percent"]]
        raise ValueError(f"RPI quarter-over-quarter validation failed:\n{bad.to_string(index=False)}")


def load_rpi(path: Path = OUTPUT_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["quarter_start"])
    validate_rpi(frame)
    return frame


def add_hdb_rpi_features(transactions: pd.DataFrame, rpi: pd.DataFrame) -> pd.DataFrame:
    """Add diagnostic contemporaneous and inference-safe prior-quarter RPI."""
    result = transactions.copy()
    period = pd.PeriodIndex(
        result["transaction_year"].astype(int).astype(str)
        + "Q"
        + result["transaction_quarter"].astype(int).astype(str),
        freq="Q",
    )
    lookup = rpi.set_index(pd.PeriodIndex(rpi["quarter_start"], freq="Q"))["rpi"]
    result["contemporaneous_rpi"] = period.map(lookup).astype(float)
    result["lagged_rpi"] = (period - 1).map(lookup).astype(float)
    result["months_since_1990"] = (
        (result["transaction_year"].astype(int) - 1990) * 12
        + result["transaction_month"].astype(int) - 1
    )
    return result


def rpi_for_valuation_month(value: str | pd.Timestamp, rpi: pd.DataFrame) -> dict[str, object]:
    """Return the latest official RPI strictly before the valuation quarter."""
    timestamp = pd.Timestamp(value)
    valuation_period = timestamp.to_period("Q")
    periods = pd.PeriodIndex(rpi["quarter_start"], freq="Q")
    eligible = rpi.loc[periods < valuation_period]
    if eligible.empty:
        raise ValueError("No prior-quarter RPI exists for the requested valuation month")
    row = eligible.iloc[-1]
    source_period = pd.Period(row["quarter_start"], freq="Q")
    return {
        "lagged_rpi": float(row["rpi"]),
        "rpi_source_quarter": f"{source_period.year}-Q{source_period.quarter}",
        "is_latest_available_fallback": source_period != valuation_period - 1,
    }


def main() -> None:
    text = extract_pdf_text(SOURCE_PATH)
    frame = parse_rpi_text(text)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Wrote {len(frame)} RPI quarters: "
        f"{frame.iloc[0]['year']}-Q{frame.iloc[0]['quarter']} through "
        f"{frame.iloc[-1]['year']}-Q{frame.iloc[-1]['quarter']}"
    )


if __name__ == "__main__":
    main()
