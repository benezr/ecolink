from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless - no display needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
import re
import statsmodels.formula.api as smf

BASE = Path(__file__).parent

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_excel(BASE / "merged_final_new.xlsx")

# Drop non-EU
df = df[df["eu27"] != "Not country group"].copy()

# ── Treatment variables ────────────────────────────────────────────────────
df["employee_treatment"] = df["scr10"].map({
    "1-9": 0,
    "10-49": 0,
    "50-249": 0,
    "250-499": 1,
    "500+": 1
})

df["turnover_treatment"] = df["scr14"].map({
    "Less than 25,000 euro": 0,
    "More than 25,000 to 50,000 euro": 0,
    "More than 50,000 to 100,000 euro": 0,
    "More than 100,000 to 250,000 euro": 0,
    "More than 250,000 to 500,000 euro": 0,
    "More than 500,000 to 2 million euro": 0,
    "More than 2 to 10 million euro": 0,
    "Don't know/No Answer": 0,
    "More than 10 to 50 million euro": 1,
    "More than 50 million euro": 1,
})

# ── Merge transposition data ───────────────────────────────────────────────
transposition = pd.read_excel(BASE / "Transposition_data.xlsx")
transposition_subset = transposition[["ipscntry", "implementation_country"]]
df = df.merge(transposition_subset, on="ipscntry", how="left")

# ── Outcome variable ───────────────────────────────────────────────────────
df['target_engaged'] = df['q14'].isin([
    'Yes',
    'No, but you are planning to define a strategy',
    'You are already climate neutral'
]).astype(int)

df["target_engaged_ord"] = df["q14"].map({
    "Yes": 2,
    "No, but you are planning to define a strategy": 1,
    "You are already climate neutral": 3,
    "No, and you are not planning to do so": 0,
    "Don't know/No Answer": 0
})

# ── East/West classification ───────────────────────────────────────────────
east_west_map = {
    "Romania": 0, "Poland": 0, "France": 1, "Belgium": 1, "Netherlands": 1,
    "Sweden": 1, "Portugal": 1, "Italy": 1, "Greece": 1, "Czechia": 0,
    "Germany": 1, "Spain": 1, "Croatia": 0, "Slovenia": 0, "Hungary": 0,
    "Austria": 1, "Bulgaria": 0, "Latvia": 0, "Estonia": 0, "Ireland": 1,
    "Finland": 1, "Denmark": 1, "Lithuania": 0, "Slovakia": 0,
    "Luxembourg": 1, "Malta": 1, "Republic of Cyprus": 1
}
df["east_west"] = df["ipscntry"].map(east_west_map)

# ── Firm age ───────────────────────────────────────────────────────────────
age_map = {
    'Before 1 January 2014': 4,
    'Between 1 January 2014 and 31 December 2016': 3,
    'Between 1 January 2017 and 1 January 2021': 2,
    'After 1 January 2021': 1,
    "Don't know/No Answer": np.nan,
    'Before 1 January 2016': 4,
    'Between 1 January 2016 and 31 December 2018': 3,
    'Between 1 January 2019 and 1 January 2023': 2,
    'After 1 January 2023': 1,
    "Don't know/No Answer (DO NOT READ OUT)": np.nan
}
df['firm_age_ord'] = df['scr12'].map(age_map)
df["firm_age_ord"] = df["firm_age_ord"].fillna(df["firm_age_ord"].mode()[0])

age_map_d = {
    'Before 1 January 2014': 1,
    'Between 1 January 2014 and 31 December 2016': 1,
    'Between 1 January 2017 and 1 January 2021': 0,
    'After 1 January 2021': 0,
    "Don't know/No Answer": np.nan,
    'Before 1 January 2016': 1,
    'Between 1 January 2016 and 31 December 2018': 1,
    'Between 1 January 2019 and 1 January 2023': 0,
    'After 1 January 2023': 0,
    "Don't know/No Answer (DO NOT READ OUT)": np.nan
}
df['firm_age_ord_d'] = df['scr12'].map(age_map_d)
df["firm_age_ord_d"] = df["firm_age_ord_d"].fillna(df["firm_age_ord_d"].mode()[0])

# ── Sector and employee dummies ────────────────────────────────────────────
dummies_emp = pd.get_dummies(df["scr10"], prefix="scr10")
df = pd.concat([df, dummies_emp], axis=1)

dummies_sec = pd.get_dummies(df["nace_b"], prefix="nace_b")
df = pd.concat([df, dummies_sec], axis=1)

# ── Firm growth ────────────────────────────────────────────────────────────
growth_map = {'Decreased': -1, 'Remained unchanged': 0, 'Increased': 1, "Don't know/No Answer": np.nan}
df['firm_growth_ord'] = df['scr13a'].map(growth_map)
df["firm_growth_ord"] = df["firm_growth_ord"].fillna(df["firm_growth_ord"].mode()[0])

growth_map_d = {'Decreased': 0, 'Remained unchanged': 0, 'Increased': 1, "Don't know/No Answer": np.nan}
df['firm_growth_ord_d'] = df['scr13a'].map(growth_map_d)
df["firm_growth_ord_d"] = df["firm_growth_ord_d"].fillna(df["firm_growth_ord_d"].mode()[0])

# ── Resource efficiency action maturity ────────────────────────────────────
def to_binary_action(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) != 0).astype('int8')
    s = series.astype('string').str.strip()
    return ((s.notna()) & (s != "Not mentioned")).astype('int8')

domain_pairs = {
    'water': ('q1_1','q2_1'),
    'energy_saving': ('q1_2','q2_2'),
    'green_energy': ('q1_3','q2_3'),
    'materials': ('q1_4','q2_4'),
    'green_suppliers': ('q1_5','q2_5'),
    'waste_reduction': ('q1_6','q2_6'),
    'sell_residues': ('q1_7','q2_7'),
    'recycling': ('q1_8','q2_8'),
    'eco_design': ('q1_9','q2_9'),
    'other': ('q1_10','q2_10')
}

cols_to_convert = sorted({c for pair in domain_pairs.values() for c in pair})
for c in cols_to_convert:
    df[c] = to_binary_action(df[c])

for name, (cur, plan) in domain_pairs.items():
    df[f'{name}_maturity'] = np.select(
        [(df[cur] == 0) & (df[plan] == 0), (df[cur] == 0) & (df[plan] == 1), (df[cur] == 1)],
        [0, 1, 2], default=0
    ).astype('int8')

for name, (cur, plan) in domain_pairs.items():
    df[f'{name}_maturity_d'] = np.select(
        [(df[cur] == 0) & (df[plan] == 0), (df[cur] == 0) & (df[plan] == 1), (df[cur] == 1)],
        [0, 1, 1], default=0
    ).astype('int8')

# ── Green staff ────────────────────────────────────────────────────────────
s = df['dx5r'].astype(str).str.strip()
mapping_ord = {
    "0 employees": 0, "1-5 employees": 1, "6-9 employees": 2,
    "10-50 employees": 3, "51-100 employees": 4, "101+ employees": 5,
    "Don't know/No Answer (DO NOT READ OUT)": np.nan
}
df['green_staff_any'] = np.where(
    s == "0 employees", 0,
    np.where(s == "Don't know/No Answer (DO NOT READ OUT)", np.nan, 1)
).astype('float')
df['green_staff_ord'] = s.map(mapping_ord).astype('float')
df["green_staff_ord"] = df["green_staff_ord"].fillna(df["green_staff_ord"].mode()[0])

# ── Investment ─────────────────────────────────────────────────────────────
df["investment_dummy"] = (df["q4"] != "Nothing").astype(int)

# ── Barriers (Q7) ─────────────────────────────────────────────────────────
q7_cols = [c for c in df.columns if c.startswith('q7_')]
for col in q7_cols:
    df[col] = np.where(df[col].isna(), 0, np.where(df[col] == "Not mentioned", 0, 1)).astype("int8")

institutional_cols = ['q7_1','q7_2','q7_3','q7_9','q7_10']
capability_cols    = ['q7_4','q7_6']
market_cols        = ['q7_5','q7_7','q7_8']

df['barrier_institutional']   = np.minimum(df[institutional_cols].sum(axis=1), 2)
df['barrier_capability']      = np.minimum(df[capability_cols].sum(axis=1), 2)
df['barrier_market']          = np.minimum(df[market_cols].sum(axis=1), 2)
df['barrier_institutional_d'] = np.minimum(df[institutional_cols].sum(axis=1), 1)
df['barrier_capability_d']    = np.minimum(df[capability_cols].sum(axis=1), 1)
df['barrier_market_d']        = np.minimum(df[market_cols].sum(axis=1), 1)

# ── Support (Q8) ──────────────────────────────────────────────────────────
knowledge_cols = ['q8_1','q8_2','q8_5','q8_6']
financial_cols = ['q8_3','q8_4']
support_cols = knowledge_cols + financial_cols
for col in support_cols:
    df[col] = np.where(df[col].isna(), 0, np.where(df[col] == "Not mentioned", 0, 1)).astype("int8")

df['support_knowledge']   = np.minimum(df[knowledge_cols].sum(axis=1), 2)
df['support_financial']   = np.minimum(df[financial_cols].sum(axis=1), 2)
df['support_knowledge_d'] = np.minimum(df[knowledge_cols].sum(axis=1), 1)
df['support_financial_d'] = np.minimum(df[financial_cols].sum(axis=1), 1)

df["support_finance_internal"]  = (df["q5_1"] == "Its own financial resources").astype(int)
df["support_technical_internal"] = (df["q5_2"] == "Its own technical expertise").astype(int)
df["support_external"]          = (df["q5_3"] == "External support").astype(int)
df["support_internal"]          = ((df["support_finance_internal"] == 1) | (df["support_technical_internal"] == 1)).astype(int)

df["support_finance_internal_pr"]   = (df["q13_1"] == "Its own financial resources").astype(int)
df["support_technical_internal_pr"] = (df["q13_2"] == "Its own technical expertise").astype(int)
df["support_external_pr"]           = (df["q13_3"] == "External support").astype(int)
df["support_internal_pr"]           = ((df["support_finance_internal_pr"] == 1) | (df["support_technical_internal_pr"] == 1)).astype(int)

# ── Green market outcomes ──────────────────────────────────────────────────
df['green_offer_ord'] = df['q9'].map({
    "No and you are not planning to do so": 0,
    "No, but you are planning to do so in the next 2 years": 1,
    "Yes": 2
}).fillna(0)

df['green_offer_ord_d'] = df['q9'].map({
    "No and you are not planning to do so": 0,
    "No, but you are planning to do so in the next 2 years": 1,
    "Yes": 1
}).fillna(0)

mapping_q10 = {
    "Up to 5%": 2.5, "6-10%": 8, "11-30%": 20, "31-50%": 40,
    "51-75%": 63, "More than 75%": 87
}
mode_val = df['q10'].map(mapping_q10).mode()[0]
mapping_q10["Don't know/No Answer (DO NOT READ OUT)"] = mode_val
df['green_turnover_pct'] = df['q10'].map(mapping_q10).fillna(0)

mapping_q10_d = {
    "Up to 5%": 1, "6-10%": 1, "11-30%": 1, "31-50%": 1,
    "51-75%": 1, "More than 75%": 1
}
mode_val_d = df['q10'].map(mapping_q10_d).mode()[0]
mapping_q10_d["Don't know/No Answer (DO NOT READ OUT)"] = mode_val_d
df['green_turnover_pct_d'] = df['q10'].map(mapping_q10_d).fillna(0)

# ── DiD variables ──────────────────────────────────────────────────────────
df = df.copy()
df["post"] = (df["year"] == 2024).astype(int)
df["sample_main"] = (df["implementation_country"] == 1).astype(int)

df["treat"] = np.where(
    (df["employee_treatment"] == 1) | (df["turnover_treatment"] == 1),
    1,
    np.where(
        (df["employee_treatment"] == 0) & (df["turnover_treatment"] == 0),
        0,
        np.nan
    )
)

# ── Sample restriction and cleaning ───────────────────────────────────────
df_main = df[df["sample_main"] == 1]

df_clean = df_main.dropna(subset=[
    "target_engaged", "treat", "post", "scr12", "nace_b", "ipscntry"
])

print(f"Sample size: {df_clean.shape[0]} obs")

# ── Model 1: Basic DiD (no barriers) ──────────────────────────────────────
print("\n" + "="*70)
print("MODEL 1 — Basic DiD (sector + age + country FE)")
print("="*70)
model1_basic = smf.ols(
    "target_engaged ~ treat*post + C(nace_b) + C(firm_age_ord) + C(ipscntry)",
    data=df_clean
).fit(cov_type="cluster", cov_kwds={"groups": df_clean["ipscntry"]})
print(model1_basic.summary())

# Parallel-trends plot for model 1
means = df_clean.groupby(["treat","post"])["target_engaged"].mean()
raw_small = means.loc[0]
raw_big   = means.loc[1]

df_pred1 = pd.DataFrame({"treat": [0, 0, 1, 1], "post": [0, 1, 0, 1]})
df_pred1["nace_b"]       = df_clean["nace_b"].mode()[0]
df_pred1["firm_age_ord"] = df_clean["firm_age_ord"].mode()[0]
df_pred1["ipscntry"]     = df_clean["ipscntry"].mode()[0]
df_pred1["predicted"]    = model1_basic.predict(df_pred1)
pred_small1 = df_pred1[df_pred1["treat"] == 0]["predicted"].values
pred_big1   = df_pred1[df_pred1["treat"] == 1]["predicted"].values

plt.figure()
plt.plot(["Before","After"], raw_small,   linestyle="--", marker="o", label="Small (raw)")
plt.plot(["Before","After"], raw_big,     linestyle="--", marker="o", label="Large (raw)")
plt.plot(["Before","After"], pred_small1, marker="o", label="Small (model)")
plt.plot(["Before","After"], pred_big1,   marker="o", label="Large (model)")
plt.ylabel("Sustainability adoption")
plt.title("DiD: Raw vs Model (No barriers)")
plt.legend()
plt.savefig(BASE / "plot_model1_basic.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_model1_basic.png")

# ── Model 2: Full controls ─────────────────────────────────────────────────
print("\n" + "="*70)
print("MODEL 2 — Full controls")
print("="*70)
model2 = smf.ols(
    """
    target_engaged ~ treat * post
    + east_west + firm_growth_ord_d + green_staff_any + investment_dummy
    + energy_saving_maturity_d + materials_maturity_d + green_suppliers_maturity_d
    + waste_reduction_maturity_d + sell_residues_maturity_d + recycling_maturity_d
    + eco_design_maturity_d + barrier_institutional_d + barrier_capability_d
    + barrier_market_d + support_external + support_internal
    + green_offer_ord_d + green_turnover_pct_d + firm_age_ord_d + support_internal_pr
    """,
    data=df_clean
).fit(cov_type="cluster", cov_kwds={"groups": df_clean["ipscntry"]})
print(model2.summary())

base = df_clean.median(numeric_only=True).to_frame().T
df_pred2 = pd.concat([base]*4, ignore_index=True)
df_pred2["treat"]     = [0, 0, 1, 1]
df_pred2["post"]      = [0, 1, 0, 1]
df_pred2["predicted"] = model2.predict(df_pred2)
print(df_pred2[["treat","post","predicted"]])

control = df_pred2[df_pred2["treat"] == 0]
treated = df_pred2[df_pred2["treat"] == 1]
plt.figure()
plt.plot(control["post"], control["predicted"], marker="o", label="Control")
plt.plot(treated["post"], treated["predicted"], marker="o", label="Treated")
plt.xticks([0,1], ["Pre", "Post"])
plt.xlabel("Time")
plt.ylabel("Predicted target_engaged")
plt.title("Difference-in-Differences Plot (Model 2)")
plt.legend()
plt.savefig(BASE / "plot_model2_full.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_model2_full.png")

# ── Model 1 (with barriers) ────────────────────────────────────────────────
print("\n" + "="*70)
print("MODEL 1 (with barriers) — sector + age + country FE + barriers")
print("="*70)
model1 = smf.ols(
    "target_engaged ~ treat*post + C(nace_b) + C(firm_age_ord) + C(ipscntry) + barrier_institutional + barrier_capability",
    data=df_clean
).fit(cov_type="cluster", cov_kwds={"groups": df_clean["ipscntry"]})
print(model1.summary())

df_pred = pd.DataFrame({"treat": [0, 0, 1, 1], "post": [0, 1, 0, 1]})
df_pred["nace_b"]               = df_clean["nace_b"].mode()[0]
df_pred["firm_age_ord"]         = df_clean["firm_age_ord"].mode()[0]
df_pred["ipscntry"]             = df_clean["ipscntry"].mode()[0]
df_pred["barrier_institutional"] = df_clean["barrier_institutional"].mean()
df_pred["barrier_capability"]   = df_clean["barrier_capability"].mean()
df_pred["barrier_market"]       = df_clean["barrier_market"].mean()
df_pred["predicted"]            = model1.predict(df_pred)
pred_small = df_pred[df_pred["treat"] == 0]["predicted"].values
pred_big   = df_pred[df_pred["treat"] == 1]["predicted"].values

plt.figure()
plt.plot(["Before","After"], raw_small,  linestyle="--", marker="o", label="Small (raw)")
plt.plot(["Before","After"], raw_big,    linestyle="--", marker="o", label="Large (raw)")
plt.plot(["Before","After"], pred_small, marker="o", label="Small (model)")
plt.plot(["Before","After"], pred_big,   marker="o", label="Large (model)")
plt.ylabel("Sustainability adoption")
plt.title("DiD: Raw vs Controlled Effect (with barriers)")
plt.legend()
plt.savefig(BASE / "plot_model1_barriers.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_model1_barriers.png")

# ── Model 3: Simplified ────────────────────────────────────────────────────
print("\n" + "="*70)
print("MODEL 3 — Simplified")
print("="*70)
model3 = smf.ols(
    """
    target_engaged ~ treat * post
    + C(nace_b) + C(ipscntry)
    + firm_growth_ord_d + green_staff_any + investment_dummy
    + energy_saving_maturity_d + green_suppliers_maturity_d
    + support_external + support_internal
    + green_offer_ord_d + green_turnover_pct_d
    """,
    data=df_clean
).fit(cov_type="cluster", cov_kwds={"groups": df_clean["ipscntry"]})
print(model3.summary())

# ── Save all summaries to text file ───────────────────────────────────────
out = BASE / "model_summaries.txt"
with open(out, "w") as f:
    f.write("MODEL 1 — Basic DiD\n" + "="*70 + "\n")
    f.write(str(model1_basic.summary()) + "\n\n")
    f.write("MODEL 2 — Full controls\n" + "="*70 + "\n")
    f.write(str(model2.summary()) + "\n\n")
    f.write("MODEL 1 (with barriers)\n" + "="*70 + "\n")
    f.write(str(model1.summary()) + "\n\n")
    f.write("MODEL 3 — Simplified\n" + "="*70 + "\n")
    f.write(str(model3.summary()) + "\n\n")
print(f"\nAll summaries saved to: {out}")
