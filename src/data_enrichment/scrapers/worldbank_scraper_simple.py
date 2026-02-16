"""
Lightweight World Bank Data Scraper (no pandas/wbdata dependency)

Fetches a few key indicators directly via the World Bank REST API and
saves them as raw CSVs under `data/external/raw/`.

This is a fallback for environments where `wbdata`/`pandas` are problematic.
"""

import csv
import os
from typing import Dict, List, Any

import requests


WB_API_BASE = "https://api.worldbank.org/v2"

# World Bank indicator codes we care about
LPI_INDICATORS = {
    "LP.LPI.OVRL.XQ": "LPI_Overall",
    "LP.LPI.CUST.XQ": "LPI_Customs",
    "LP.LPI.INFR.XQ": "LPI_Infrastructure",
    "LP.LPI.LOGS.XQ": "LPI_Logistics",
    "LP.LPI.TRAC.XQ": "LPI_Tracking",
    "LP.LPI.TIME.XQ": "LPI_Timeliness",
}

TRADE_FACILITATION_INDICATORS = {
    "IC.CUS.DURS": "Customs_Clearance_Days",
    "IC.TRD.DURS": "Trade_Duration_Days",
}


def _fetch_indicator(indicator_code: str) -> List[Dict[str, Any]]:
    """
    Fetch raw observations for a single World Bank indicator.

    Returns a list of dicts with keys:
      - country_id
      - country_name
      - year
      - value
    """
    results: List[Dict[str, Any]] = []

    page = 1
    while True:
        url = (
            f"{WB_API_BASE}/country/all/indicator/"
            f"{indicator_code}?format=json&per_page=20000&page={page}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # data[0] is metadata, data[1] is the list of observations
        if not isinstance(data, list) or len(data) < 2:
            break

        meta, observations = data
        if not observations:
            break

        for obs in observations:
            country = obs.get("country") or {}
            date = obs.get("date")
            value = obs.get("value")

            # Skip aggregates like "World", "High income", etc.
            if country.get("id", "").startswith("X") or country.get("id") in {
                "WLD",
                "HIC",
                "LIC",
                "LMC",
                "UMC",
            }:
                continue

            if value is None:
                continue

            results.append(
                {
                    "country_id": country.get("id"),
                    "country_name": country.get("value"),
                    "year": int(date) if date and date.isdigit() else None,
                    "value": float(value),
                }
            )

        total_pages = int(meta.get("pages", 1))
        if page >= total_pages:
            break
        page += 1

    return results


def _latest_by_country(raw: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Collapse observations to the latest year per country.

    Returns mapping: country_id -> {country_name, year, value}
    """
    latest: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        cid = row["country_id"]
        year = row["year"]
        if year is None:
            continue

        if cid not in latest or year > latest[cid]["year"]:
            latest[cid] = {
                "country_name": row["country_name"],
                "year": year,
                "value": row["value"],
            }
    return latest


def fetch_lpi_data_simple() -> None:
    """
    Fetch multiple LPI indicators and save a combined CSV.
    """
    os.makedirs("data/external/raw", exist_ok=True)

    # country_id -> row data with multiple indicators
    combined: Dict[str, Dict[str, Any]] = {}

    for code, col_name in LPI_INDICATORS.items():
        raw = _fetch_indicator(code)
        latest = _latest_by_country(raw)
        for cid, info in latest.items():
            if cid not in combined:
                combined[cid] = {
                    "country_id": cid,
                    "country_name": info["country_name"],
                    "year": info["year"],
                }
            combined[cid][col_name] = info["value"]

    fieldnames = ["country_id", "country_name", "year"] + list(LPI_INDICATORS.values())

    out_path = "data/external/raw/worldbank_lpi_simple.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined.values():
            writer.writerow(row)


def fetch_trade_facilitation_simple() -> None:
    """
    Fetch trade facilitation indicators and save a CSV.
    """
    os.makedirs("data/external/raw", exist_ok=True)

    combined: Dict[str, Dict[str, Any]] = {}

    for code, col_name in TRADE_FACILITATION_INDICATORS.items():
        raw = _fetch_indicator(code)
        latest = _latest_by_country(raw)
        for cid, info in latest.items():
            if cid not in combined:
                combined[cid] = {
                    "country_id": cid,
                    "country_name": info["country_name"],
                    "year": info["year"],
                }
            combined[cid][col_name] = info["value"]

    fieldnames = [
        "country_id",
        "country_name",
        "year",
    ] + list(TRADE_FACILITATION_INDICATORS.values())

    out_path = "data/external/raw/worldbank_trade_facilitation_simple.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined.values():
            writer.writerow(row)


def main() -> None:
    fetch_lpi_data_simple()
    fetch_trade_facilitation_simple()


if __name__ == "__main__":
    main()


