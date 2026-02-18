import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
os.makedirs("outputs/plots", exist_ok=True)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
def load_data():
    reg_df = pd.read_csv("src/processed/new_customs_regression_dataset.csv")
    clf_df = pd.read_csv("src/processed/new_customs_classification_dataset.csv")
    return reg_df, clf_df


# --------------------------------------------------
# 2. DELAY DISTRIBUTION ANALYSIS
# --------------------------------------------------
def delay_distribution(reg_df):
    plt.figure(figsize=(8, 5))
    sns.histplot(reg_df["Arrival_Delay_Days"], bins=30, kde=True)
    plt.title("Distribution of Arrival Delay Days")
    plt.savefig("outputs/plots/delay_distribution.png")
    plt.close()

    skewness = reg_df["Arrival_Delay_Days"].skew()
    print("\n--- Delay Distribution ---")
    print(f"Skewness: {skewness:.3f}")


# --------------------------------------------------
# 3. RISK DISTRIBUTION
# --------------------------------------------------
def risk_distribution(clf_df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Route_Risk_Level", data=clf_df)
    plt.title("Risk Level Distribution")
    plt.savefig("outputs/plots/risk_distribution.png")
    plt.close()

    print("\n--- Risk Distribution ---")
    print(clf_df["Route_Risk_Level"].value_counts(normalize=True))


# --------------------------------------------------
# 4. TRANSIT EFFICIENCY IMPACT
# --------------------------------------------------
def transit_analysis(reg_df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="Actual_Transit_Days",
        y="Arrival_Delay_Days",
        data=reg_df,
        alpha=0.4
    )
    plt.title("Actual Transit Days vs Arrival Delay")
    plt.savefig("outputs/plots/transit_vs_delay.png")
    plt.close()

    print("\n--- Transit Impact ---")
    print("Correlation:",
          reg_df["Actual_Transit_Days"].corr(
              reg_df["Arrival_Delay_Days"]
          ))


# --------------------------------------------------
# 5. COMPLIANCE ANALYSIS
# --------------------------------------------------
def compliance_analysis(reg_df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x="Has_Prior_Offense",
        y="Arrival_Delay_Days",
        data=reg_df
    )
    plt.title("Prior Offense vs Delay")
    plt.savefig("outputs/plots/prior_offense_vs_delay.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="Compliance_Risk_Score",
        y="Arrival_Delay_Days",
        data=reg_df,
        alpha=0.4
    )
    plt.title("Compliance Risk Score vs Delay")
    plt.savefig("outputs/plots/compliance_score_vs_delay.png")
    plt.close()

    print("\n--- Compliance Impact ---")
    print("Compliance Correlation:",
          reg_df["Compliance_Risk_Score"].corr(
              reg_df["Arrival_Delay_Days"]
          ))


# --------------------------------------------------
# 6. DOCUMENT & INSPECTION IMPACT
# --------------------------------------------------
def inspection_analysis(clf_df):
    plt.figure(figsize=(10, 5))
    sns.countplot(
        x="Inspection_Type",
        hue="Route_Risk_Level",
        data=clf_df
    )
    plt.xticks(rotation=45)
    plt.title("Inspection Type vs Risk Level")
    plt.savefig("outputs/plots/inspection_vs_risk.png")
    plt.close()


# --------------------------------------------------
# 7. ROUTE OPTIMIZATION ANALYSIS
# --------------------------------------------------
def route_optimization(reg_df, clf_df):

    route_delay = reg_df.groupby("Route_Code")["Arrival_Delay_Days"].mean()
    route_risk = clf_df.groupby("Route_Code")["Route_Risk_Level"].mean()

    route_summary = pd.DataFrame({
        "avg_delay": route_delay,
        "risk_rate": route_risk
    })

    # Normalization
    route_summary["delay_norm"] = (
        route_summary["avg_delay"] - route_summary["avg_delay"].min()
    ) / (route_summary["avg_delay"].max() - route_summary["avg_delay"].min())

    route_summary["risk_norm"] = (
        route_summary["risk_rate"] - route_summary["risk_rate"].min()
    ) / (route_summary["risk_rate"].max() - route_summary["risk_rate"].min())

    route_summary["optimization_score"] = (
        route_summary["delay_norm"] + route_summary["risk_norm"]
    )

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(route_summary["avg_delay"],
                route_summary["risk_rate"],
                alpha=0.6)
    plt.xlabel("Average Delay")
    plt.ylabel("Average Risk")
    plt.title("Trade Route Optimization View")
    plt.savefig("outputs/plots/route_optimization.png")
    plt.close()

    print("\n--- Top 10 Optimal Routes ---")
    print(route_summary.sort_values("optimization_score").head(10))


# --------------------------------------------------
# 8. CORRELATION MATRIX
# --------------------------------------------------
def correlation_analysis(reg_df):
    plt.figure(figsize=(12, 10))
    corr = reg_df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.savefig("outputs/plots/correlation_matrix.png")
    plt.close()

    print("\n--- Strongest Predictors of Delay ---")
    print(corr["Arrival_Delay_Days"].sort_values(ascending=False).head(10))


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    reg_df, clf_df = load_data()

    delay_distribution(reg_df)
    risk_distribution(clf_df)
    transit_analysis(reg_df)
    compliance_analysis(reg_df)
    inspection_analysis(clf_df)
    correlation_analysis(reg_df)
    route_optimization(reg_df, clf_df)


if __name__ == "__main__":
    main()
