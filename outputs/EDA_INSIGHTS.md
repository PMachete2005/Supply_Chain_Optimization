Exploratory Data Analysis & Key Findings

Project: Cross-Border Customs Delay & Risk Optimization

1. Delay Distribution Analysis

Skewness: 1.031

The delay distribution is positively skewed.

Interpretation

Most shipments experience low arrival delay, while a small proportion face significantly higher delays. This indicates the presence of extreme delay cases (right tail).

Implication

Linear models may struggle due to skewness.

Tree-based models (Random Forest, Gradient Boosting) are likely more suitable.

Log transformation of delay could improve regression performance.

2. Risk Distribution Analysis
Risk Level	Proportion
0	33.42%
1	33.68%
2	32.45%
3	0.45%
Interpretation

Risk levels 0, 1, and 2 are nearly perfectly balanced.

Risk level 3 is extremely rare.

Implication

Classification task is well-balanced for main categories.

Risk level 3 may require special handling or class weighting.

Model bias toward majority classes is unlikely.

3. Transit Impact on Delay

Correlation (Actual Transit Days vs Arrival Delay): 0.278

Interpretation

Transit duration has a moderate positive relationship with arrival delay.

This suggests that logistical inefficiencies in transport directly contribute to overall delay, though it is not the sole determining factor.

Implication

Transit performance is an important but not dominant predictor of delay.

4. Compliance Impact on Delay

Compliance Risk Score Correlation: 0.0085

Interpretation

Compliance-related features show negligible linear relationship with arrival delay.

This suggests:

Customs delay is not strongly driven by compliance risk in this dataset.

Operational factors likely dominate over regulatory risk factors.

Implication

Compliance variables may still contribute in non-linear models but are weak standalone predictors.

5. Strongest Predictors of Delay

Top correlations with Arrival_Delay_Days:

Customs_Delay_Days: 1.000

Actual_Transit_Days: 0.278

All other features: very weak correlations (< 0.02)

Critical Observation

Customs_Delay_Days shows perfect correlation (1.000) with Arrival_Delay_Days.

This indicates:

Arrival_Delay_Days likely includes Customs_Delay_Days in its calculation.

Implication

Customs_Delay_Days must be removed from regression modeling to avoid data leakage.

Without removal, model performance would be artificially inflated.

6. Route Optimization Analysis

A composite optimization score was created using normalized:

Average delay

Average risk rate

Lower score = better trade route.

Top 10 Optimal Routes (Lowest Delay + Lowest Risk)

Example:

Route_Code	Avg Delay	Risk Rate
718	0.20	0.00
111	0.83	0.33
396	0.67	0.44
220	0.78	0.56
762	1.00	0.50
Interpretation

Route 718 is the most efficient route with near-zero delay and zero risk.

Several routes maintain moderate delay with moderate risk.

High delay + high risk routes should be deprioritized.

Business Implication

Trade route selection should focus on minimizing both delay and risk simultaneously.

The optimization scoring framework provides a data-driven decision tool for route recommendation.

Overall Conclusions

Delay distribution is right-skewed; non-linear models are preferred.

Risk categories are well-balanced, making classification reliable.

Transit duration has measurable impact on delay.

Compliance features show weak linear impact.

Customs delay variable creates data leakage and must be excluded from regression training.

A multi-objective optimization approach successfully identifies efficient trade routes.