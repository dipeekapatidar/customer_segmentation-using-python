"""Generate a synthetic customer dataset (500 customers) for the project.

Run:  python generate_data.py   ->  creates data/customers.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

# share, (age mean, sd), (income mean, sd) in thousands, (spend score), (orders/year), (avg order value)
GROUPS = [
    (0.22, (28, 4), (85, 10), (82, 8), (28, 5), (120, 20)),  # young, affluent, big spenders
    (0.18, (45, 6), (88, 10), (20, 8), (6, 2), (60, 15)),    # high income, low spend
    (0.25, (35, 8), (50, 8), (50, 8), (14, 3), (55, 10)),    # mainstream
    (0.17, (24, 3), (30, 7), (78, 9), (24, 5), (35, 8)),     # low income, high spend
    (0.18, (52, 7), (28, 7), (22, 8), (5, 2), (30, 8)),      # budget conscious
]


def make_customers(n=500, seed=42):
    rng = np.random.default_rng(seed)
    counts = [int(n * g[0]) for g in GROUPS]
    counts[-1] = n - sum(counts[:-1])

    parts = []
    for count, (_, age, inc, spend, freq, aov) in zip(counts, GROUPS):
        parts.append(
            pd.DataFrame(
                {
                    "Age": rng.normal(*age, count),
                    "Annual_Income": rng.normal(*inc, count),
                    "Spending_Score": rng.normal(*spend, count),
                    "Purchase_Frequency": rng.normal(*freq, count),
                    "Avg_Order_Value": rng.normal(*aov, count),
                }
            )
        )
    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    df["Age"] = df["Age"].clip(18, 70).round().astype(int)
    df["Annual_Income"] = df["Annual_Income"].clip(10, None).round(1)
    df["Spending_Score"] = df["Spending_Score"].clip(1, 100).round().astype(int)
    df["Purchase_Frequency"] = df["Purchase_Frequency"].clip(1, None).round().astype(int)
    df["Avg_Order_Value"] = df["Avg_Order_Value"].clip(10, None).round(2)

    df.insert(0, "CustomerID", [f"C{i:04d}" for i in range(1, n + 1)])
    df.insert(2, "Gender", rng.choice(["Female", "Male"], n))

    # a few dirty values so the preprocessing step has something to do
    for col, how_many in [("Age", 4), ("Annual_Income", 4), ("Avg_Order_Value", 3)]:
        df[col] = df[col].astype(float)
        df.loc[rng.choice(n, how_many, replace=False), col] = np.nan
    df.loc[rng.choice(n, 4, replace=False), "Avg_Order_Value"] *= 4  # outliers
    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "customers.csv"
    out.parent.mkdir(exist_ok=True)
    make_customers().to_csv(out, index=False)
    print(f"Saved {out}")
