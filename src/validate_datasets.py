import pandas as pd
import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

# known leakage indicators (original column names or substrings from one-hot encoding)
LEAKAGE_KEYWORDS = [
    'Delivery Status', 'Order Status', 'shipping date',
    'Order_Status', 'Delivery_Status'
]

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def validate_dataset(df, name, target_col):
    section(f"VALIDATING: {name}")
    all_passed = True

    # ---- 1. Missing values ----
    total_missing = df.isnull().sum().sum()
    status = PASS if total_missing == 0 else FAIL
    if total_missing > 0:
        all_passed = False
    print(f"\n1. Missing values: {status}  (total = {total_missing})")
    if total_missing > 0:
        missing_cols = df.isnull().sum()
        print(missing_cols[missing_cols > 0])

    # ---- 2. No object/string columns ----
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    status = PASS if len(obj_cols) == 0 else FAIL
    if len(obj_cols) > 0:
        all_passed = False
    print(f"2. All numeric columns: {status}  (object cols = {len(obj_cols)})")
    if obj_cols:
        print(f"   Offending columns: {obj_cols}")

    # ---- 3. Duplicate rows ----
    n_dups = df.duplicated().sum()
    status = PASS if n_dups == 0 else WARN
    print(f"3. No duplicate rows: {status}  (duplicates = {n_dups})")

    # ---- 4. Shape and feature count ----
    n_rows, n_cols = df.shape
    n_features = n_cols - 1  # exclude target
    print(f"4. Dataset shape: {n_rows} rows x {n_cols} cols  ({n_features} features + 1 target)")

    # ---- 5. Target variable validation ----
    if target_col not in df.columns:
        print(f"5. Target column '{target_col}': {FAIL}  (NOT FOUND in dataset)")
        all_passed = False
    else:
        target = df[target_col]
        if name == "Classification":
            unique_vals = sorted(target.unique())
            is_binary = set(unique_vals).issubset({0, 1})
            status = PASS if is_binary else FAIL
            if not is_binary:
                all_passed = False
            print(f"5. Target '{target_col}': {status}  (unique = {unique_vals}, binary = {is_binary})")
            print(f"   Class distribution:\n{target.value_counts().to_string()}")
        else:  # Regression
            is_numeric = pd.api.types.is_numeric_dtype(target)
            status = PASS if is_numeric else FAIL
            if not is_numeric:
                all_passed = False
            print(f"5. Target '{target_col}': {status}  (numeric = {is_numeric})")
            print(f"   min={target.min():.2f}, max={target.max():.2f}, "
                  f"mean={target.mean():.2f}, std={target.std():.2f}")

    # ---- 6. Data leakage check ----
    leaked = []
    for col in df.columns:
        for keyword in LEAKAGE_KEYWORDS:
            if keyword.lower() in col.lower():
                leaked.append(col)
                break
    status = PASS if len(leaked) == 0 else FAIL
    if len(leaked) > 0:
        all_passed = False
    print(f"6. No data leakage columns: {status}")
    if leaked:
        print(f"   Leaked columns found: {leaked}")

    # ---- 7. Extreme correlations with target ----
    if target_col in df.columns:
        numeric_df = df.select_dtypes(include=[np.number])
        corr_with_target = numeric_df.corr()[target_col].drop(target_col).abs()
        high_corr = corr_with_target[corr_with_target > 0.9].sort_values(ascending=False)
        status = PASS if len(high_corr) == 0 else WARN
        print(f"7. High correlation check (>0.9 with target): {status}  ({len(high_corr)} found)")
        if len(high_corr) > 0:
            for col_name, corr_val in high_corr.items():
                print(f"   {col_name}: {corr_val:.4f}")

    # ---- 8. Summary statistics ----
    print(f"\n8. Summary statistics (selected numeric columns):")
    desc = df.describe().T[['mean', 'min', 'max']]
    # show first 15 and last 5 if too many columns
    if len(desc) > 20:
        print(desc.head(15).to_string())
        print(f"   ... ({len(desc) - 20} more columns) ...")
        print(desc.tail(5).to_string())
    else:
        print(desc.to_string())

    # ---- 9. Final verdict ----
    section(f"{'READY' if all_passed else 'ISSUES FOUND'}: {name}")
    if all_passed:
        print(f"{PASS} {name} dataset is ready for ML models.\n")
    else:
        print(f"{FAIL} {name} dataset has issues — review above.\n")

    return all_passed


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    clf_path = "../src/data/classification_dataset.csv"
    reg_path = "../src/data/regression_dataset.csv"

    print("Loading datasets...")
    clf_df = pd.read_csv(clf_path)
    reg_df = pd.read_csv(reg_path)

    clf_ok = validate_dataset(clf_df, "Classification", "Late_delivery_risk")
    reg_ok = validate_dataset(reg_df, "Regression", "Days for shipping (real)")

    section("OVERALL RESULT")
    if clf_ok and reg_ok:
        print(f"{PASS} Both datasets passed all checks and are ready for ML.\n")
    else:
        print(f"{FAIL} One or more datasets have issues. See details above.\n")
