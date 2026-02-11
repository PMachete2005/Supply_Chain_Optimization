import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
os.makedirs("outputs/plots", exist_ok=True)
sns.set_style("whitegrid")


def load_data():
    reg_df = pd.read_csv("data/processed/customs_regression_dataset.csv")
    clf_df = pd.read_csv("data/processed/customs_classification_dataset.csv")
    return reg_df, clf_df


def delay_distribution(reg_df):
    plt.figure(figsize=(8, 5))
    sns.histplot(reg_df["Arrival_Delay_Days"], bins=30, kde=True)
    plt.title("Distribution of Arrival Delay Days")
    plt.savefig("outputs/plots/delay_distribution.png")
    plt.close()

    skewness = reg_df["Arrival_Delay_Days"].skew()
    print(f"Delay Skewness: {skewness:.3f}")


def risk_distribution(clf_df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Route_Risk_Level", data=clf_df)
    plt.title("Shipment Risk Distribution")
    plt.savefig("outputs/plots/risk_distribution.png")
    plt.close()

    print("Risk Proportions:")
    print(clf_df["Route_Risk_Level"].value_counts(normalize=True))


def correlation_analysis(reg_df):
    plt.figure(figsize=(12, 10))
    corr = reg_df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.savefig("outputs/plots/correlation_matrix.png")
    plt.close()

    print("\nTop Correlations with Arrival Delay:")
    print(corr["Arrival_Delay_Days"].sort_values(ascending=False))


def route_optimization(reg_df, clf_df):
    route_delay = reg_df.groupby("Route_Code")["Arrival_Delay_Days"].mean()
    route_risk = clf_df.groupby("Route_Code")["Route_Risk_Level"].mean()

    route_summary = pd.DataFrame({
        "avg_delay": route_delay,
        "risk_rate": route_risk
    })

    plt.figure(figsize=(8, 6))
    plt.scatter(route_summary["avg_delay"],
                route_summary["risk_rate"],
                alpha=0.6)

    plt.xlabel("Average Delay")
    plt.ylabel("Average Risk Level")
    plt.title("Trade Route Optimization View")
    plt.savefig("outputs/plots/route_optimization.png")
    plt.close()

    print("\nBest Routes (Low Delay + Low Risk):")
    route_summary["score"] = (
        (route_summary["avg_delay"] - route_summary["avg_delay"].min()) /
        (route_summary["avg_delay"].max() - route_summary["avg_delay"].min())
    ) + (
        (route_summary["risk_rate"] - route_summary["risk_rate"].min()) /
        (route_summary["risk_rate"].max() - route_summary["risk_rate"].min())
    )

    print(route_summary.sort_values("score").head(10))


def main():
    reg_df, clf_df = load_data()
    delay_distribution(reg_df)
    risk_distribution(clf_df)
    correlation_analysis(reg_df)
    route_optimization(reg_df, clf_df)


if __name__ == "__main__":
    main()
