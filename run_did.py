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

# ── Model 1 Growth: DiD with firm growth as outcome ───────────────────────
print("\n" + "="*70)
print("MODEL 1 GROWTH — DiD with firm_growth_ord as outcome")
print("="*70)
df_growth = df_clean.dropna(subset=["firm_growth_ord"])
model1_growth = smf.ols(
    "firm_growth_ord ~ treat*post + C(nace_b) + C(firm_age_ord) + C(ipscntry)",
    data=df_growth
).fit(cov_type="cluster", cov_kwds={"groups": df_growth["ipscntry"]})
print(model1_growth.summary())

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
    f.write("MODEL 1 GROWTH — firm_growth_ord as outcome\n" + "="*70 + "\n")
    f.write(str(model1_growth.summary()) + "\n\n")
print(f"\nAll summaries saved to: {out}")

# ═══════════════════════════════════════════════════════════════════════════
# COMPARATIVE SPECIFICATIONS
# ───────────────────────────────────────────────────────────────────────────
# 1. Simple OLS — treatment effect of firm size on outcome, no DiD.
#    Estimated for target_engaged and firm_growth_ord_d, with and without
#    country fixed effects.
#
# 2. DiD — adds treat × post interaction (implementing-country sample only).
#    With and without country FEs.
#
# 3. DDD — extends DiD with impl as a third dimension, using never-treated
#    countries (implementation_country=0) as an additional control group.
#    treat × post × impl is the DDD coefficient.
#    Country FEs cannot be included (impl is collinear); sector + age FEs
#    are used instead for the "with FE" variant.
#
# 4. DDD with high-dimensional FEs (Callaway–Sant'Anna style):
#      μc × Eligible_i  → C(cntry_treat): country × eligibility FE
#      λc × t           → C(cntry_post):  country × time FE
#      δ Eligible_i × t → treat:post:     eligibility × time FE
#    The triple interaction treat:post:impl is the DDD coefficient after
#    absorbing all two-way interactions.
#
# 5. Comparison table — all key treatment coefficients side-by-side.
# ═══════════════════════════════════════════════════════════════════════════

from statsmodels.iolib.summary2 import summary_col

# ── Build DDD sample (impl=0 and impl=1 countries) ─────────────────────────
df_ddd = df.dropna(subset=[
    "target_engaged", "treat", "post",
    "scr12", "nace_b", "ipscntry", "implementation_country"
]).copy()
df_ddd["impl"] = df_ddd["implementation_country"]

# Analysis samples
df_s = df_clean.dropna(subset=["firm_growth_ord_d"]).copy()   # OLS / DiD (impl=1)
df_d = df_ddd.dropna(subset=["firm_growth_ord_d"]).copy()      # DDD (all countries)

# Extended FE dummies (spec 4): country × eligibility, country × time
df_d["cntry_treat"] = df_d["ipscntry"].astype(str) + "_" + df_d["treat"].astype(str)
df_d["cntry_post"]  = df_d["ipscntry"].astype(str) + "_" + df_d["post"].astype(str)

def cl(df_):
    return {"cov_type": "cluster", "cov_kwds": {"groups": df_["ipscntry"]}}

# ── 1. Simple OLS ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("1. SIMPLE OLS  (no DiD treatment)")
print("="*70)

s1a = smf.ols("target_engaged    ~ treat + C(nace_b) + C(firm_age_ord)",                data=df_s).fit(**cl(df_s))
s1b = smf.ols("target_engaged    ~ treat + C(nace_b) + C(firm_age_ord) + C(ipscntry)", data=df_s).fit(**cl(df_s))
s1c = smf.ols("firm_growth_ord_d ~ treat + C(nace_b) + C(firm_age_ord)",                data=df_s).fit(**cl(df_s))
s1d = smf.ols("firm_growth_ord_d ~ treat + C(nace_b) + C(firm_age_ord) + C(ipscntry)", data=df_s).fit(**cl(df_s))

for lbl, m in [("1a target_engaged  no FE", s1a), ("1b target_engaged  country FE", s1b),
               ("1c firm_growth_ord_d no FE", s1c), ("1d firm_growth_ord_d country FE", s1d)]:
    print(f"  {lbl}: treat = {m.params['treat']:.4f}  p = {m.pvalues['treat']:.3f}")

# ── 2. DiD ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("2. DIFFERENCE-IN-DIFFERENCES  (treat × post)")
print("="*70)

s2a = smf.ols("target_engaged    ~ treat*post + C(nace_b) + C(firm_age_ord)",                data=df_s).fit(**cl(df_s))
s2b = smf.ols("target_engaged    ~ treat*post + C(nace_b) + C(firm_age_ord) + C(ipscntry)", data=df_s).fit(**cl(df_s))
s2c = smf.ols("firm_growth_ord_d ~ treat*post + C(nace_b) + C(firm_age_ord)",                data=df_s).fit(**cl(df_s))
s2d = smf.ols("firm_growth_ord_d ~ treat*post + C(nace_b) + C(firm_age_ord) + C(ipscntry)", data=df_s).fit(**cl(df_s))

for lbl, m in [("2a target_engaged  no FE", s2a), ("2b target_engaged  country FE", s2b),
               ("2c firm_growth_ord_d no FE", s2c), ("2d firm_growth_ord_d country FE", s2d)]:
    print(f"  {lbl}: treat:post = {m.params['treat:post']:.4f}  p = {m.pvalues['treat:post']:.3f}")

# ── 3. DDD ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("3. DIFFERENCE-IN-DIFFERENCE-IN-DIFFERENCES  (treat × post × impl)")
print("   impl=0 countries are the never-treated control group.")
print("   Country FEs excluded — impl absorbs the country-group dimension.")
print("="*70)

s3a = smf.ols("target_engaged    ~ treat*post*impl + C(nace_b)",                  data=df_d).fit(**cl(df_d))
s3b = smf.ols("target_engaged    ~ treat*post*impl + C(nace_b) + C(firm_age_ord)", data=df_d).fit(**cl(df_d))
s3c = smf.ols("firm_growth_ord_d ~ treat*post*impl + C(nace_b)",                  data=df_d).fit(**cl(df_d))
s3d = smf.ols("firm_growth_ord_d ~ treat*post*impl + C(nace_b) + C(firm_age_ord)", data=df_d).fit(**cl(df_d))

for lbl, m in [("3a target_engaged  no FE", s3a), ("3b target_engaged  sector+age FE", s3b),
               ("3c firm_growth_ord_d no FE", s3c), ("3d firm_growth_ord_d sector+age FE", s3d)]:
    print(f"  {lbl}: treat:post:impl = {m.params['treat:post:impl']:.4f}  p = {m.pvalues['treat:post:impl']:.3f}")

# ── 4. DDD — High-dimensional FEs ──────────────────────────────────────────
print("\n" + "="*70)
print("4. DDD — HIGH-DIMENSIONAL FEs")
print("   mu_c x Eligible_i : C(cntry_treat) -- country x eligibility FE")
print("   lambda_c x t      : C(cntry_post)  -- country x time FE")
print("   delta Eligible x t: treat:post     -- eligibility x time FE")
print("   All lower-order terms of treat:post:impl are absorbed by the FEs.")
print("="*70)

s4a = smf.ols(
    """target_engaged ~ treat:post:impl
       + C(cntry_treat) + C(cntry_post) + treat:post
       + C(nace_b) + C(firm_age_ord)""",
    data=df_d
).fit(**cl(df_d))

s4b = smf.ols(
    """firm_growth_ord_d ~ treat:post:impl
       + C(cntry_treat) + C(cntry_post) + treat:post
       + C(nace_b) + C(firm_age_ord)""",
    data=df_d
).fit(**cl(df_d))

for lbl, m in [("4a target_engaged", s4a), ("4b firm_growth_ord_d", s4b)]:
    print(f"  {lbl}: treat:post:impl = {m.params['treat:post:impl']:.4f}  p = {m.pvalues['treat:post:impl']:.3f}")

# ── 5. Comparison table ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("5. COMPARISON TABLE")
print("   Rows show key treatment coefficients; *** p<0.01, ** p<0.05, * p<0.1")
print("   (1a-d) Simple OLS | (2a-d) DiD | (3a-d) DDD | (4a-b) DDD ext FE")
print("="*70)

comp_table = summary_col(
    [s1a, s1b, s1c, s1d, s2a, s2b, s2c, s2d, s3a, s3b, s3c, s3d, s4a, s4b],
    model_names=[
        "(1a)", "(1b)", "(1c)", "(1d)",
        "(2a)", "(2b)", "(2c)", "(2d)",
        "(3a)", "(3b)", "(3c)", "(3d)",
        "(4a)", "(4b)",
    ],
    stars=True,
    float_format="%0.4f",
    regressor_order=["treat", "treat:post", "treat:post:impl"],
    drop_omitted=True,
    info_dict={
        "N":          lambda x: f"{int(x.nobs)}",
        "R-squared":  lambda x: f"{x.rsquared:.3f}",
        "Outcome":    lambda x: x.model.endog_names,
        "Country FE": lambda x: "Yes" if any("ipscntry" in v for v in x.model.exog_names) else "No",
    },
)
print(comp_table)

with open(BASE / "comparison_table.txt", "w") as f:
    f.write(str(comp_table))
print(f"Saved: {BASE / 'comparison_table.txt'}")
