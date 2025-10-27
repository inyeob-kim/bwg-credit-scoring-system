import pandas as pd
import numpy as np

n = 1000
rng = np.random.default_rng(42)

df = pd.DataFrame({
    "income": rng.normal(5000, 1500, n).clip(1000, 15000),
    "debt_ratio": rng.uniform(0.1, 1.5, n),
    "utilization": rng.uniform(0, 1, n),
    "credit_history_months": rng.integers(6, 120, n),
    "n_late_30d": rng.integers(0, 5, n),
    "employment_type": rng.choice(["Permanent", "Contract", "SelfEmployed", "Unemployed"], n),
    "region": rng.choice(["North", "Central", "South"], n),
    "channel": rng.choice(["App", "Web", "Branch", "Partner"], n),
})

# 실제 연체(default_12m)를 생성 (소득 낮고 부채비율 높을수록 부도 확률↑)
prob_default = (
    0.2 * (df["debt_ratio"] > 1).astype(int)
    + 0.3 * (df["utilization"] > 0.8).astype(int)
    + 0.2 * (df["income"] < 3000).astype(int)
)
prob_default = np.clip(prob_default + rng.normal(0, 0.05, n), 0, 1)
df["default_12m"] = (prob_default > 0.5).astype(int)

df.to_csv("sample_credit_data.csv", index=False)
print("✅ sample_credit_data.csv saved")
print(df.head())
