"""
Enrich the DataCo Supply Chain dataset with World Bank LPI features.

Source country  -> `Customer Country`
Destination     -> `Order Country`

Outputs:
  - new_data/processed/worldbank_lpi_latest.csv (scraped via World Bank API)
  - new_data/processed/DataCoSupplyChainDataset_enriched.csv
"""

from __future__ import annotations

import csv
import os
import re
import sys
import unicodedata
from typing import Dict, Optional, Tuple

sys.path.append(os.path.abspath("."))  # allow `src.*` imports when run as a script
from src.data_enrichment.scrapers.worldbank_scraper_simple import fetch_lpi_data_simple


RAW_DATASET_PATH = "new_data/raw/DataCoSupplyChainDataset.csv"
LPI_OUT_PATH = "new_data/processed/worldbank_lpi_latest.csv"
ENRICHED_OUT_PATH = "new_data/processed/DataCoSupplyChainDataset_enriched.csv"

SOURCE_COUNTRY_COL = "Customer Country"
DST_COUNTRY_COL = "Order Country"


LPI_COLS = [
    "LPI_Overall",
    "LPI_Customs",
    "LPI_Infrastructure",
    "LPI_Logistics",
    "LPI_Tracking",
    "LPI_Timeliness",
]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _normalize_country_name(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    s = _strip_accents(s)
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[.\u00b7]", " ", s)  # dots / middle dot
    s = re.sub(r"[^a-z0-9'\\s,-]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def _compact_key(s: str) -> str:
    # Helps match mojibake like "m xico" -> "mexico"
    return re.sub(r"[^a-z0-9]", "", s)


def _build_aliases() -> Dict[str, str]:
    """
    Map common dataset variants (often Spanish) -> World Bank country name.
    Keys and values are stored normalized.
    """
    raw_aliases = {
        "ee uu": "United States",
        "ee uu ": "United States",
        "ee. uu.": "United States",
        "estados unidos": "United States",
        "usa": "United States",
        "u s a": "United States",
        "puerto rico": "United States",
        "reino unido": "United Kingdom",
        "u k": "United Kingdom",
        "uk": "United Kingdom",
        "emiratos arabes unidos": "United Arab Emirates",
        "eau": "United Arab Emirates",
        "singapur": "Singapore",
        "francia": "France",
        "alemania": "Germany",
        "brasil": "Brazil",
        "italia": "Italy",
        "mexico": "Mexico",
        "m xico": "Mexico",
        "espana": "Spain",
        "espa a": "Spain",
        "turquia": "Turkiye",
        "turqu a": "Turkiye",
        "filipinas": "Philippines",
        "panama": "Panama",
        "panam": "Panama",
        "paises bajos": "Netherlands",
        "pa ses bajos": "Netherlands",
        "nueva zelanda": "New Zealand",
        "iran": "Iran, Islamic Rep.",
        "ir n": "Iran, Islamic Rep.",
        "egipto": "Egypt, Arab Rep.",
        "marruecos": "Morocco",
        "sudafrica": "South Africa",
        "rusia": "Russian Federation",
        "rep blica dominicana": "Dominican Republic",
        "republica dominicana": "Dominican Republic",
        "rep blica democr tica del congo": "Congo, Dem. Rep.",
        "republica democratica del congo": "Congo, Dem. Rep.",
        "irak": "Iraq",
        "ucrania": "Ukraine",
        "suecia": "Sweden",
        "tailandia": "Thailand",
        "corea del sur": "Korea, Rep.",
        "corea": "Korea, Rep.",
        "rusia": "Russian Federation",
        "venezuela": "Venezuela, RB",
        "bolivia": "Bolivia",
        "tanzania": "Tanzania",
        "vietnam": "Viet Nam",
        "laos": "Lao PDR",
        "siria": "Syrian Arab Republic",
        "moldavia": "Moldova",
        "hong kong": "Hong Kong SAR, China",
        "taiwan": "Taiwan, China",
    }
    out = {}
    for k, v in raw_aliases.items():
        nk = _normalize_country_name(k)
        nv = _normalize_country_name(v)
        out[nk] = nv
        out[_compact_key(nk)] = nv
    return out


def _lookup_lpi(
    country_value: str,
    lpi_lookup: Dict[str, Dict[str, float]],
    lpi_lookup_keys: Dict[str, str],
    aliases: Dict[str, str],
) -> Optional[Dict[str, float]]:
    key = _normalize_country_name(str(country_value or ""))
    if not key:
        return None

    # Alias map first (dataset variants -> WB name)
    alias_key = aliases.get(key) or aliases.get(_compact_key(key))
    if alias_key:
        return lpi_lookup.get(alias_key)

    # Direct match against WB names (normalized)
    if key in lpi_lookup:
        return lpi_lookup[key]
    compact = _compact_key(key)
    if compact in lpi_lookup:
        return lpi_lookup[compact]

    # Fallback: try to match by removing commas / suffixes
    simplified = key.replace(",", "")
    if simplified in lpi_lookup:
        return lpi_lookup[simplified]

    # Fallback: if we precomputed a "contains" key map, try it
    contained = lpi_lookup_keys.get(key)
    if contained:
        return lpi_lookup.get(contained)

    return None


def _load_lpi_lookup_from_csv(path: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """
    Returns:
      - lpi_lookup: normalized WB country name -> {LPI_*: float}
      - year_counts: counts by year (int)
    """
    lpi_lookup: Dict[str, Dict[str, float]] = {}
    year_counts: Dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            country_name = str(row.get("country_name", "") or "")
            key = _normalize_country_name(country_name)
            if not key:
                continue

            year = str(row.get("year", "") or "").strip()
            if year:
                year_counts[year] = year_counts.get(year, 0) + 1

            vals: Dict[str, float] = {}
            for c in LPI_COLS:
                v = row.get(c)
                if v is None or v == "":
                    continue
                try:
                    vals[c] = float(v)
                except ValueError:
                    continue

            if vals:
                lpi_lookup[key] = vals
                lpi_lookup[_compact_key(key)] = vals

    return lpi_lookup, {int(k): v for k, v in year_counts.items() if k.isdigit()}


def enrich() -> None:
    if not os.path.exists(RAW_DATASET_PATH):
        raise FileNotFoundError(f"Raw dataset not found at `{RAW_DATASET_PATH}`")

    os.makedirs(os.path.dirname(ENRICHED_OUT_PATH) or ".", exist_ok=True)

    # 1) "Web scrape" (World Bank REST API) -> LPI CSV saved under new_data/processed
    # In locked-down environments without outbound access, fall back to the vendored CSV.
    try:
        if os.environ.get("SKIP_WB_FETCH") == "1":
            raise RuntimeError("Skipping WB fetch due to SKIP_WB_FETCH=1")
        fetch_lpi_data_simple(out_path=LPI_OUT_PATH)
    except Exception as e:
        fallback = "data/external/raw/worldbank_lpi_simple.csv"
        if not os.path.exists(fallback):
            raise RuntimeError(
                f"World Bank fetch failed ({e}) and fallback file not found at `{fallback}`"
            ) from e
        os.makedirs(os.path.dirname(LPI_OUT_PATH) or ".", exist_ok=True)
        with open(fallback, "r", encoding="utf-8") as src, open(
            LPI_OUT_PATH, "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())
        print(f"World Bank fetch unavailable; used cached `{fallback}` instead.")
    lpi_lookup, year_counts = _load_lpi_lookup_from_csv(LPI_OUT_PATH)
    aliases = _build_aliases()

    # Optional helper: precompute a weak "contains" match map for odd names
    # key(dataset_normalized) -> key(wb_normalized)
    lpi_lookup_keys: Dict[str, str] = {}
    wb_keys = list(lpi_lookup.keys())
    for wb in wb_keys:
        # map "united states" -> itself etc. (cheap and helpful)
        lpi_lookup_keys[wb] = wb

    if 2022 in year_counts:
        print(f"LPI dataset contains year 2022 rows: {year_counts[2022]}")

    # Stream-enrich the raw dataset (keeps memory usage low)
    total_rows = 0
    src_missing = 0
    dst_missing = 0

    lpi_cache: Dict[str, Optional[Dict[str, float]]] = {}

    with open(RAW_DATASET_PATH, "r", encoding="utf-8", errors="replace") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError("Input dataset appears to have no header row.")

        missing_cols = [c for c in [SOURCE_COUNTRY_COL, DST_COUNTRY_COL] if c not in reader.fieldnames]
        if missing_cols:
            raise KeyError(f"Missing expected columns in dataset: {missing_cols}")

        out_fieldnames = list(reader.fieldnames)
        out_fieldnames += [f"Source_{c}" for c in LPI_COLS]
        out_fieldnames += [f"Destination_{c}" for c in LPI_COLS]
        out_fieldnames += ["Route_LPI_Average", "Route_LPI_Difference"]

        with open(ENRICHED_OUT_PATH, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1

                src = row.get(SOURCE_COUNTRY_COL, "") or ""
                dst = row.get(DST_COUNTRY_COL, "") or ""

                if src not in lpi_cache:
                    lpi_cache[src] = _lookup_lpi(src, lpi_lookup, lpi_lookup_keys, aliases)
                if dst not in lpi_cache:
                    lpi_cache[dst] = _lookup_lpi(dst, lpi_lookup, lpi_lookup_keys, aliases)

                src_lpi = lpi_cache.get(src) or {}
                dst_lpi = lpi_cache.get(dst) or {}

                for c in LPI_COLS:
                    row[f"Source_{c}"] = src_lpi.get(c, "")
                    row[f"Destination_{c}"] = dst_lpi.get(c, "")

                if not row["Source_LPI_Overall"]:
                    src_missing += 1
                if not row["Destination_LPI_Overall"]:
                    dst_missing += 1

                try:
                    s_overall = float(row["Source_LPI_Overall"]) if row["Source_LPI_Overall"] != "" else None
                    d_overall = float(row["Destination_LPI_Overall"]) if row["Destination_LPI_Overall"] != "" else None
                except ValueError:
                    s_overall, d_overall = None, None

                if s_overall is not None and d_overall is not None:
                    row["Route_LPI_Average"] = (s_overall + d_overall) / 2
                    row["Route_LPI_Difference"] = d_overall - s_overall
                else:
                    row["Route_LPI_Average"] = ""
                    row["Route_LPI_Difference"] = ""

                writer.writerow(row)

    print(f"Rows: {total_rows}")
    print(f"Missing Source_LPI_Overall: {src_missing}")
    print(f"Missing Destination_LPI_Overall: {dst_missing}")
    print(f"Wrote enriched dataset to `{ENRICHED_OUT_PATH}`")


if __name__ == "__main__":
    enrich()

