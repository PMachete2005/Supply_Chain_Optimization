import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def _series_equal_with_tolerance(a: pd.Series, b: pd.Series) -> float:
    """Return row-wise equality ratio with numeric tolerance when possible."""
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        return float(np.isclose(a_num, b_num, rtol=1e-6, atol=1e-8, equal_nan=True).mean())

    a_str = a.astype(str).fillna("<NA>")
    b_str = b.astype(str).fillna("<NA>")
    return float(a_str.eq(b_str).mean())


def can_align_by_index(
    raw_df: pd.DataFrame,
    target_df: pd.DataFrame,
    key_cols: list[str],
    anchor_cols: list[str] | None = None,
) -> bool:
    """
    Return True when index-based alignment appears safe.

    We verify:
    1) Same row count
    2) Key columns are present in both
    3) Row-wise key values match exactly (with numeric tolerance)
    """
    if len(raw_df) != len(target_df):
        return False

    # First, try alignment checks on stable "anchor" columns that should remain untransformed.
    # This is necessary because key columns may be standardized in preprocessed datasets.
    if anchor_cols is None:
        anchor_cols = ["HS_Code", "Customs_Delay_Days", "Planned_Transit_Days", "Actual_Transit_Days"]

    available_anchors = [c for c in anchor_cols if c in raw_df.columns and c in target_df.columns]
    if available_anchors:
        anchor_ratios = {
            c: _series_equal_with_tolerance(
                raw_df[c].reset_index(drop=True),
                target_df[c].reset_index(drop=True),
            )
            for c in available_anchors
        }
        strong_anchors = [c for c, r in anchor_ratios.items() if r >= 0.999]
        if strong_anchors:
            print(f"Index alignment evidence from anchors: {anchor_ratios}")
            return True

    # Fallback to key-column check when anchors are unavailable.
    if not set(key_cols).issubset(raw_df.columns) or not set(key_cols).issubset(target_df.columns):
        return False

    raw_keys = raw_df[key_cols].reset_index(drop=True)
    tgt_keys = target_df[key_cols].reset_index(drop=True)

    for col in key_cols:
        r = raw_keys[col]
        t = tgt_keys[col]

        same = _series_equal_with_tolerance(r, t) >= 0.999999

        if not same:
            return False

    return True


def build_engineered_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    # LPI / Infrastructure columns
    lpi_cols = [
        c for c in raw_df.columns if ("LPI" in c) or ("Infrastructure_Gap" in c)
    ]
    if not lpi_cols:
        raise ValueError(
            "No columns containing 'LPI' or 'Infrastructure_Gap' were found in raw dataset."
        )

    lpi_data = raw_df[lpi_cols].apply(pd.to_numeric, errors="coerce")
    lpi_data = lpi_data.fillna(lpi_data.median(numeric_only=True))
    lpi_data = lpi_data.fillna(0.0)

    scaler = StandardScaler()
    lpi_scaled = scaler.fit_transform(lpi_data)
    lpi_scaled_df = pd.DataFrame(
        lpi_scaled,
        columns=[f"scaled_{c}" for c in lpi_cols],
        index=raw_df.index,
    )

    # Delay reason TF-IDF
    if "Delay_Reason" not in raw_df.columns:
        raise ValueError("'Delay_Reason' column not found in raw dataset.")

    delay_text = raw_df["Delay_Reason"].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=10)
    delay_tfidf = tfidf.fit_transform(delay_text)
    delay_tfidf_df = pd.DataFrame(
        delay_tfidf.toarray(),
        columns=[f"delay_tfidf_{f}" for f in tfidf.get_feature_names_out()],
        index=raw_df.index,
    )

    return pd.concat([lpi_scaled_df, delay_tfidf_df], axis=1)


def normalize_merge_keys(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Normalize composite key columns for safer matching across preprocessing artifacts."""
    out = df.copy()
    for c in key_cols:
        if c not in out.columns:
            continue

        if c == "HS_Code":
            out[c] = (
                out[c]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
    return out


def merge_features(
    raw_df: pd.DataFrame,
    target_df: pd.DataFrame,
    engineered_df: pd.DataFrame,
    key_cols: list[str],
    target_name: str,
) -> pd.DataFrame:
    """Merge engineered features into target dataframe using safest strategy."""
    if can_align_by_index(raw_df, target_df, key_cols):
        print(f"[{target_name}] Index alignment is safe. Merging by index.")
        merged = pd.concat(
            [target_df.reset_index(drop=True), engineered_df.reset_index(drop=True)], axis=1
        )
    else:
        print(
            f"[{target_name}] Index alignment is NOT safe. Merging via composite key: {key_cols}"
        )

        missing_keys = [c for c in key_cols if c not in target_df.columns or c not in raw_df.columns]
        if missing_keys:
            raise KeyError(
                f"Cannot perform key-based merge for {target_name}. Missing keys: {missing_keys}"
            )

        raw_keys_norm = normalize_merge_keys(raw_df[key_cols].copy(), key_cols)
        tgt_keys_norm = normalize_merge_keys(target_df[key_cols].copy(), key_cols)

        raw_feature_map = pd.concat([raw_keys_norm, engineered_df.copy()], axis=1)

        dup_mask = raw_feature_map.duplicated(subset=key_cols, keep=False)
        if dup_mask.any():
            dup_count = int(dup_mask.sum())
            print(
                f"[{target_name}] Warning: {dup_count} raw rows share duplicate composite keys. "
                "Keeping first occurrence per key to avoid row explosion."
            )
            raw_feature_map = raw_feature_map.drop_duplicates(subset=key_cols, keep="first")

        merged = tgt_keys_norm.merge(raw_feature_map, how="left", on=key_cols)
        merged = pd.concat([target_df.reset_index(drop=True), merged[engineered_df.columns]], axis=1)

        hit_rate = float(merged[engineered_df.columns[0]].notna().mean()) if len(engineered_df.columns) else 0.0
        print(f"[{target_name}] Key-merge non-null hit rate: {hit_rate:.4f}")

    # Drop duplicate column names if any
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def main(data_dir: Path) -> None:
    raw_path = data_dir / "trade_customs_dataset(2).csv"
    reg_path = data_dir / "customs_regression_dataset(3).csv"
    cls_path = data_dir / "customs_classification_dataset(3).csv"

    print("Loading datasets...")
    raw_df = pd.read_csv(raw_path)
    reg_df = pd.read_csv(reg_path)
    cls_df = pd.read_csv(cls_path)

    print(f"Raw shape: {raw_df.shape}")
    print(f"Regression shape: {reg_df.shape}")
    print(f"Classification shape: {cls_df.shape}")

    key_cols = ["Declared_Value_USD", "Weight_kg", "HS_Code"]

    print("Building engineered features from raw dataset...")
    engineered_df = build_engineered_features(raw_df)
    print(f"Engineered feature shape: {engineered_df.shape}")

    final_reg = merge_features(raw_df, reg_df, engineered_df, key_cols, "regression")
    final_cls = merge_features(raw_df, cls_df, engineered_df, key_cols, "classification")

    out_reg = data_dir / "final_regression_data.csv"
    out_cls = data_dir / "final_classification_data.csv"

    final_reg.to_csv(out_reg, index=False)
    final_cls.to_csv(out_cls, index=False)

    print(f"Saved: {out_reg}")
    print(f"Saved: {out_cls}")
    print(f"Final regression shape: {final_reg.shape}")
    print(f"Final classification shape: {final_cls.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge missing LPI/Infrastructure and Delay_Reason-derived features into preprocessed customs datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the three source CSV files.",
    )
    args = parser.parse_args()

    main(args.data_dir.resolve())
