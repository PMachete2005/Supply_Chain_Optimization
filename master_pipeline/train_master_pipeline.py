#!/usr/bin/env python3
"""
train_master_pipeline.py
========================
End-to-end ML pipeline for DataCo Supply Chain dataset.
  - Task 1  Classification: Predict Late_delivery_risk   (Accuracy ~76%)
  - Task 2  Regression:     Predict Days for shipping (real) (R² ~0.53)

Leak-free feature sets: no target leakers are used in either task.
"""

import json
import os
import re
import unicodedata
import warnings
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATACO = os.path.join(BASE_DIR, "new_data", "raw", "DataCoSupplyChainDataset.csv")
LPI_CSV = os.path.join(BASE_DIR, "new_data", "processed", "worldbank_lpi_latest.csv")
SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

# ── LPI World Bank API config ────────────────────────────────────────────
LPI_INDICATORS = {
    "LP.LPI.OVRL.XQ": "LPI_Overall",
    "LP.LPI.CUST.XQ": "LPI_Customs",
    "LP.LPI.INFR.XQ": "LPI_Infrastructure",
    "LP.LPI.LOGS.XQ": "LPI_Logistics",
    "LP.LPI.TRAK.XQ": "LPI_Tracking",
    "LP.LPI.TIME.XQ": "LPI_Timeliness",
}


# ══════════════════════════════════════════════════════════════════════════
#  HELPER: LPI loading with LFS-pointer fallback
# ══════════════════════════════════════════════════════════════════════════
def _is_lfs_pointer(path: str) -> bool:
    """Return True when the file is missing or is a Git LFS stub."""
    if not os.path.exists(path):
        return True
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    return first_line.startswith("version https://git-lfs")


def _fetch_lpi_from_worldbank() -> pd.DataFrame:
    """Fetch latest LPI scores from the World Bank Indicators REST API."""
    print("[INFO] Fetching LPI data from World Bank API …")
    records: dict[str, dict] = {}  # keyed by country_name
    for indicator_code, col_name in LPI_INDICATORS.items():
        for year in ("2023", "2022", "2018"):
            url = (
                f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
                f"?date={year}&format=json&per_page=500"
            )
            try:
                with urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                if len(data) >= 2 and data[1]:
                    break
            except Exception:
                data = [None, None]
        if not data or len(data) < 2 or data[1] is None:
            print(f"  [WARN] No data for {indicator_code}")
            continue
        for entry in data[1]:
            if entry.get("value") is None:
                continue
            cname = entry["country"]["value"]
            cid = entry.get("countryiso3code") or entry["country"]["id"]
            yr = entry["date"]
            rec = records.setdefault(cname, {"country_id": cid, "country_name": cname, "year": yr})
            rec[col_name] = entry["value"]
    df = pd.DataFrame(list(records.values()))
    for col in LPI_INDICATORS.values():
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_lpi() -> pd.DataFrame | None:
    """Load LPI CSV; if it is a Git LFS pointer, fetch from the World Bank API."""
    if not _is_lfs_pointer(LPI_CSV):
        print(f"[INFO] Loading LPI from local CSV ({LPI_CSV})")
        return pd.read_csv(LPI_CSV)
    print("[INFO] Local LPI CSV is missing / LFS pointer → fetching from World Bank API …")
    try:
        df = _fetch_lpi_from_worldbank()
        os.makedirs(os.path.dirname(LPI_CSV), exist_ok=True)
        df.to_csv(LPI_CSV, index=False)
        print(f"[INFO] Saved fetched LPI data → {LPI_CSV}")
        return df
    except Exception as exc:
        print(f"[WARN] World Bank API fetch failed: {exc}")
        print("[WARN] Proceeding without LPI features.")
        return None


# ══════════════════════════════════════════════════════════════════════════
#  HELPER: Country-name normalisation & alias map for LPI merge
# ══════════════════════════════════════════════════════════════════════════
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _normalise(s: str) -> str:
    s = str(s).strip()
    if not s:
        return ""
    s = _strip_accents(s).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9'\s,-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_COMPACT = re.compile(r"[^a-z0-9]")

COUNTRY_ALIASES: dict[str, str] = {
    "ee uu": "united states", "ee. uu.": "united states",
    "estados unidos": "united states", "usa": "united states",
    "puerto rico": "united states",
    "reino unido": "united kingdom", "uk": "united kingdom",
    "emiratos arabes unidos": "united arab emirates",
    "singapur": "singapore", "francia": "france",
    "alemania": "germany", "brasil": "brazil", "italia": "italy",
    "mexico": "mexico", "m xico": "mexico",
    "espana": "spain", "espa a": "spain",
    "turquia": "turkiye", "turqu a": "turkiye",
    "filipinas": "philippines", "panama": "panama", "panam": "panama",
    "paises bajos": "netherlands", "pa ses bajos": "netherlands",
    "nueva zelanda": "new zealand",
    "iran": "iran, islamic rep", "ir n": "iran, islamic rep",
    "egipto": "egypt, arab rep", "marruecos": "morocco",
    "sudafrica": "south africa", "rusia": "russian federation",
    "rep blica dominicana": "dominican republic",
    "republica dominicana": "dominican republic",
    "irak": "iraq", "ucrania": "ukraine", "suecia": "sweden",
    "tailandia": "thailand", "corea del sur": "korea, rep",
    "corea": "korea, rep", "venezuela": "venezuela, rb",
    "bolivia": "bolivia", "tanzania": "tanzania",
    "vietnam": "viet nam", "laos": "lao pdr",
    "siria": "syrian arab republic", "moldavia": "moldova",
    "hong kong": "hong kong sar, china", "taiwan": "taiwan, china",
}


def _resolve_country(name: str, lpi_norm_map: dict[str, str]) -> str | None:
    """Map a DataCo Order Country value to a World Bank country_name."""
    n = _normalise(name)
    if not n:
        return None
    # 1) alias table
    if n in COUNTRY_ALIASES:
        n = COUNTRY_ALIASES[n]
    # 2) direct normalised match
    if n in lpi_norm_map:
        return lpi_norm_map[n]
    # 3) compact (letters+digits only)
    compact = _COMPACT.sub("", n)
    for k, v in lpi_norm_map.items():
        if _COMPACT.sub("", k) == compact:
            return v
    # 4) substring containment
    for k, v in lpi_norm_map.items():
        if n in k or k in n:
            return v
    return None


# ══════════════════════════════════════════════════════════════════════════
#  HELPER: Encode categoricals + impute
# ══════════════════════════════════════════════════════════════════════════
def prepare_features(X: pd.DataFrame, *, label_encoders=None, imputer=None, fit=True):
    """LabelEncode object columns → int, then median-impute all NaNs."""
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if label_encoders is None:
        label_encoders = {}

    for col in cat_cols:
        X[col] = X[col].astype(str)
        if fit:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is not None:
                classes_set = set(le.classes_)
                X[col] = X[col].map(
                    lambda v, _le=le, _cs=classes_set: (
                        _le.transform([v])[0] if v in _cs else -1
                    )
                )
            else:
                X[col] = -1

    if fit:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        arr = imputer.fit_transform(X)
    else:
        arr = imputer.transform(X)

    return pd.DataFrame(arr, columns=X.columns, index=X.index), label_encoders, imputer


# ══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 70)
    print("  MASTER ML PIPELINE — DataCo Supply Chain")
    print("=" * 70)

    # ── 1. Data Loading & Merging ─────────────────────────────────────────
    print("\n[STEP 1] Loading DataCo dataset …")
    df = pd.read_csv(RAW_DATACO, encoding="latin1")
    print(f"  Shape: {df.shape}")

    print("\n[STEP 1b] Loading LPI dataset …")
    lpi_df = load_lpi()

    if lpi_df is not None and not lpi_df.empty:
        lpi_norm_map: dict[str, str] = {}
        for name in lpi_df["country_name"].dropna().unique():
            lpi_norm_map[_normalise(name)] = name

        df["_lpi_key"] = df["Order Country"].apply(lambda x: _resolve_country(x, lpi_norm_map))

        lpi_cols = ["country_name"] + [c for c in lpi_df.columns if c.startswith("LPI_")]
        lpi_dedup = lpi_df[lpi_cols].drop_duplicates(subset=["country_name"])

        n_before = df.shape[1]
        df = df.merge(lpi_dedup, left_on="_lpi_key", right_on="country_name", how="left")
        df.drop(columns=["_lpi_key", "country_name"], inplace=True, errors="ignore")
        matched = df["LPI_Overall"].notna().sum()
        print(f"  Merged {df.shape[1] - n_before + 2} LPI columns  |  "
              f"match rate: {matched}/{len(df)} ({100 * matched / len(df):.1f}%)")
    else:
        print("  ⚠ Skipping LPI merge (data unavailable).")

    # ── 2. Feature Engineering & ID Drops ─────────────────────────────────
    print("\n[STEP 2] Feature engineering …")

    date_col = "order date (DateOrders)"
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["Order_Month"] = df[date_col].dt.month
        df["Order_DayOfWeek"] = df[date_col].dt.dayofweek
        df["Order_Hour"] = df[date_col].dt.hour
        df.drop(columns=[date_col], inplace=True)
        print("  Created: Order_Month, Order_DayOfWeek, Order_Hour")

    id_pii_cols = [
        "Customer Email", "Customer Password", "Product Image",
        "Order Id", "Customer Id", "Order Customer Id",
        "Product Card Id", "Category Id", "Department Id",
        "Order Item Cardprod Id", "Order Item Id",
    ]
    to_drop = [c for c in id_pii_cols if c in df.columns]
    df.drop(columns=to_drop, inplace=True)
    print(f"  Dropped {len(to_drop)} ID / PII columns → shape {df.shape}")

    # ── 3. Classification: Late_delivery_risk ─────────────────────────────
    print("\n" + "=" * 70)
    print("  TASK 1 — Classification: Late_delivery_risk")
    print("=" * 70)

    cls_target = "Late_delivery_risk"
    cls_leak = [
        cls_target,
        "Days for shipping (real)",
        "Delivery Status",
        "shipping date (DateOrders)",
    ]
    cls_leak += [c for c in df.columns if "Order Status" in c]
    cls_leak = list({c for c in cls_leak if c in df.columns})

    y_cls = df[cls_target].copy()
    X_cls = df.drop(columns=cls_leak).copy()
    print(f"  Features: {X_cls.shape[1]}  |  Classes: {sorted(y_cls.unique())}")

    X_cls_enc, cls_encoders, cls_imputer = prepare_features(X_cls, fit=True)
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_cls_enc, y_cls, test_size=0.2, random_state=42, stratify=y_cls,
    )

    print("  Training RandomForestClassifier (n=200, depth=20) …")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr_c, y_tr_c)
    y_pred_c = clf.predict(X_te_c)

    acc = accuracy_score(y_te_c, y_pred_c)
    f1 = f1_score(y_te_c, y_pred_c, average="macro")
    print(f"\n  ┌─── Classification Results ───┐")
    print(f"  │  Accuracy  : {acc:.4f}         │")
    print(f"  │  Macro F1  : {f1:.4f}         │")
    print(f"  └───────────────────────────────┘")

    imp_c = pd.Series(clf.feature_importances_, index=X_cls_enc.columns).nlargest(5)
    print("  Top 5 Feature Importances:")
    for feat, val in imp_c.items():
        print(f"    {feat:40s} {val:.4f}")

    # ── 4. Regression: Days for shipping (real) ──────────────────────────
    print("\n" + "=" * 70)
    print("  TASK 2 — Regression: Days for shipping (real)")
    print("=" * 70)

    reg_target = "Days for shipping (real)"
    reg_leak = [
        reg_target,
        "Late_delivery_risk",
        "Delivery Status",
        "shipping date (DateOrders)",
    ]
    reg_leak += [c for c in df.columns if "Order Status" in c]
    reg_leak = list({c for c in reg_leak if c in df.columns})

    y_reg = df[reg_target].copy()
    X_reg = df.drop(columns=reg_leak).copy()
    sched_kept = "Days for shipment (scheduled)" in X_reg.columns
    print(f"  Features: {X_reg.shape[1]}  |  'Days for shipment (scheduled)' kept: {sched_kept}")

    X_reg_enc, reg_encoders, reg_imputer = prepare_features(X_reg, fit=True)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg_enc, y_reg, test_size=0.2, random_state=42,
    )

    print("  Training RandomForestRegressor (n=200, depth=20) …")
    rfr = RandomForestRegressor(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
    )
    rfr.fit(X_tr_r, y_tr_r)
    y_pred_r = rfr.predict(X_te_r)

    mae = mean_absolute_error(y_te_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_te_r, y_pred_r))
    r2 = r2_score(y_te_r, y_pred_r)
    print(f"\n  ┌─── Regression Results ────────┐")
    print(f"  │  MAE   : {mae:.4f}              │")
    print(f"  │  RMSE  : {rmse:.4f}              │")
    print(f"  │  R²    : {r2:.4f}              │")
    print(f"  └────────────────────────────────┘")

    imp_r = pd.Series(rfr.feature_importances_, index=X_reg_enc.columns).nlargest(5)
    print("  Top 5 Feature Importances:")
    for feat, val in imp_r.items():
        print(f"    {feat:40s} {val:.4f}")

    # ── 5. Save Artifacts ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Saving Artifacts")
    print("=" * 70)

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    joblib.dump(clf, os.path.join(SAVED_MODELS_DIR, "classification_model.joblib"))
    joblib.dump(
        {"encoders": cls_encoders, "imputer": cls_imputer},
        os.path.join(SAVED_MODELS_DIR, "classification_preprocessor.joblib"),
    )
    joblib.dump(rfr, os.path.join(SAVED_MODELS_DIR, "regression_model.joblib"))
    joblib.dump(
        {"encoders": reg_encoders, "imputer": reg_imputer},
        os.path.join(SAVED_MODELS_DIR, "regression_preprocessor.joblib"),
    )

    for f in os.listdir(SAVED_MODELS_DIR):
        size_mb = os.path.getsize(os.path.join(SAVED_MODELS_DIR, f)) / 1024 / 1024
        print(f"  {f:45s} {size_mb:.1f} MB")

    print("\n✅  Pipeline complete.")


if __name__ == "__main__":
    main()
