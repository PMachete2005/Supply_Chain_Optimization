import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Safe output directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
def load_data():
    reg_df = pd.read_csv(
        "data/regression_dataset.csv",
        low_memory=True
    )

    clf_df = pd.read_csv(
        "data/classification_dataset.csv",
        low_memory=True
    )

    return reg_df, clf_df


# --------------------------------------------------
# 2. SHIPMENT DURATION DISTRIBUTION
# --------------------------------------------------
def shipment_duration_distribution(reg_df):
    plt.figure(figsize=(8, 5))
    sns.histplot(reg_df["Days for shipment (scheduled)"], bins=30, kde=True)
    plt.title("Shipment Duration Distribution")

    plt.savefig(os.path.join(OUTPUT_DIR, "shipment_duration.png"), dpi=100)
    plt.close()

    stats = reg_df["Days for shipment (scheduled)"].describe()

    print("\n--- Shipment Duration Distribution ---")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Median: {stats['50%']:.2f}")
    print(f"Std Dev: {stats['std']:.2f}")
    print(f"Range: {stats['min']:.0f} - {stats['max']:.0f}")


# --------------------------------------------------
# 3. LATE DELIVERY RISK
# --------------------------------------------------
def late_delivery_risk_distribution(clf_df):
    plt.figure(figsize=(6, 4))

    sns.countplot(
        x="Late_delivery_risk",
        hue="Late_delivery_risk",
        data=clf_df,
        legend=False
    )

    plt.title("Late Delivery Risk Distribution")

    plt.savefig(os.path.join(OUTPUT_DIR, "late_delivery_risk.png"), dpi=100)
    plt.close()

    risk_pct = clf_df["Late_delivery_risk"].value_counts(normalize=True) * 100

    print("\n--- Late Delivery Risk Distribution ---")
    print(f"On Time: {risk_pct.get(0, 0):.2f}%")
    print(f"Late: {risk_pct.get(1, 0):.2f}%")


# --------------------------------------------------
# 4. SHIPPING MODE IMPACT
# --------------------------------------------------
def shipping_mode_analysis(reg_df, clf_df):
    modes = [
        "Shipping Mode_Same Day",
        "Shipping Mode_Second Class",
        "Shipping Mode_Standard Class"
    ]

    labels = ["Same Day", "Second Class", "Standard Class"]

    duration = []
    risk = []

    for col in modes:
        if col in reg_df.columns:
            duration.append(reg_df[reg_df[col] == 1]["Days for shipment (scheduled)"].mean())
        if col in clf_df.columns:
            risk.append(clf_df[clf_df[col] == 1]["Late_delivery_risk"].mean())

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(labels, duration)
    plt.title("Shipment Duration by Mode")

    plt.subplot(1, 2, 2)
    plt.bar(labels, risk)
    plt.title("Late Risk by Mode")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shipping_mode_analysis.png"), dpi=100)
    plt.close()

    print("\n--- Shipping Mode Impact ---")
    for l, d, r in zip(labels, duration, risk):
        print(f"{l}: {d:.2f} days, {r:.2%} risk")


# --------------------------------------------------
# 5. LPI ANALYSIS (LIMITED FEATURES)
# --------------------------------------------------
def lpi_analysis(df, target):
    lpi_cols = [col for col in df.columns if "LPI" in col]

    correlations = []

    for col in lpi_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            corr = df[col].corr(df[target])
            if not pd.isna(corr):
                correlations.append((col, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    top = correlations[:10]

    cols, vals = zip(*top)

    plt.figure(figsize=(8, 5))
    plt.barh(cols, vals)

    plt.title(f"LPI Impact on {target}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"lpi_{target}.png"), dpi=100)
    plt.close()

    print(f"\n--- LPI Correlation with {target} ---")
    for c, v in top:
        print(f"{c}: {v:.4f}")


# --------------------------------------------------
# 6. MARKET ANALYSIS (FIXED MEMORY ISSUE)
# --------------------------------------------------
def market_analysis(reg_df, clf_df):
    markets = [
        "Market_Europe",
        "Market_LATAM",
        "Market_Pacific Asia",
        "Market_USCA"
    ]

    labels = ["Europe", "LATAM", "Pacific Asia", "USCA"]

    duration = []
    risk = []

    for m in markets:
        if m in reg_df.columns:
            duration.append(reg_df[reg_df[m] == 1]["Days for shipment (scheduled)"].mean())
        if m in clf_df.columns:
            risk.append(clf_df[clf_df[m] == 1]["Late_delivery_risk"].mean())

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(labels, duration)

    plt.subplot(1, 2, 2)
    plt.bar(labels, risk)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "market_analysis.png"), dpi=100)
    plt.close()

    print("\n--- Market Impact ---")
    for l, d, r in zip(labels, duration, risk):
        print(f"{l}: {d:.2f} days, {r:.2%} risk")


# --------------------------------------------------
# 7. TOP FEATURE CORRELATION (SAFE)
# --------------------------------------------------
def top_feature_analysis(df, target):
    
    print(f"\n--- Top Features for {target} ---")

    correlations = []

    # Iterate column-wise instead of full matrix
    for col in df.columns:
        if col != target and pd.api.types.is_numeric_dtype(df[col]):
            try:
                corr = df[col].corr(df[target])
                if not pd.isna(corr):
                    correlations.append((col, corr))
            except:
                continue

    # Sort
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    top = correlations[:10]

    # Split for plotting
    cols = [c[0] for c in top]
    vals = [c[1] for c in top]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(cols[::-1], vals[::-1])  # reverse for readability
    plt.title(f"Top Features for {target}")

    plt.savefig(os.path.join(OUTPUT_DIR, f"top_features_{target}.png"), dpi=100)
    plt.close()

    # Print results
    for c, v in top:
        print(f"{c}: {v:.4f}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print("Loading datasets...")
    reg_df, clf_df = load_data()

    print("\nGenerating visualizations...")

    shipment_duration_distribution(reg_df)
    late_delivery_risk_distribution(clf_df)
    shipping_mode_analysis(reg_df, clf_df)

    lpi_analysis(clf_df, "Late_delivery_risk")
    lpi_analysis(reg_df, "Days for shipment (scheduled)")

    market_analysis(reg_df, clf_df)

    top_feature_analysis(reg_df, "Days for shipment (scheduled)")
    top_feature_analysis(clf_df, "Late_delivery_risk")

    print("\n✅ All plots saved in outputs/plots/")


if __name__ == "__main__":
    main()