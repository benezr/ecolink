"""
preprocess.py — build analysis_data.parquet from the two raw Excel files.

Run once (or whenever the source data changes):
    python preprocess.py

Output: analysis_data.parquet  (~3 MB, loads in <1 s vs ~30 s for Excel)

What this script does:
  1. Load merged_final_new.xlsx, keep EU-27 rows
  2. Merge Transposition_data.xlsx (country-level policy variables)
  3. Build all analysis variables used in comparative_specifications.ipynb:
       - Treatment indicators (treat, post, sample_main)
       - Outcome variables (target_engaged, firm_growth_ord, firm_growth_ord_d)
       - Firm controls (age, east/west, resource-efficiency maturity, green staff,
                        investment, barriers, support, green market outcomes)
  4. Save to analysis_data.parquet

The modeling samples (df_main, df_clean, df_ddd, df_s, df_d) and any
decision-specific recodes (e.g. impl_cyp for Cyprus) are left to the notebook.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading Excel files …")
df = pd.read_excel(BASE / "merged_final_new.xlsx")
df = df[df["eu27"] != "Not country group"].copy()
print(f"  merged_final_new: {df.shape[0]:,} rows, {df.shape[1]} cols")

transposition = pd.read_excel(BASE / "Transposition_data.xlsx")
transposition["t_date"] = pd.to_datetime(transposition["t_date"])
df = df.merge(
    transposition[["ipscntry", "implementation_country", "t_date"]],
    on="ipscntry", how="left"
)
print(f"  after transposition merge: {df.shape[0]:,} rows, {df.shape[1]} cols")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Treatment variables
# ─────────────────────────────────────────────────────────────────────────────
df["employee_treatment"] = df["scr10"].map({
    "1-9": 0, "10-49": 0, "50-249": 0, "250-499": 1, "500+": 1
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
df["treat"] = np.where(
    (df["employee_treatment"] == 1) | (df["turnover_treatment"] == 1), 1,
    np.where(
        (df["employee_treatment"] == 0) & (df["turnover_treatment"] == 0), 0, np.nan
    ),
)

# Below-threshold placebo components (employees >= 50; turnover > €250k)
df["employee_50plus"] = df["scr10"].map({
    "1-9": 0, "10-49": 0, "50-249": 1, "250-499": 1, "500+": 1,
})
df["turnover_250k"] = df["scr14"].map({
    "Less than 25,000 euro": 0,
    "More than 25,000 to 50,000 euro": 0,
    "More than 50,000 to 100,000 euro": 0,
    "More than 100,000 to 250,000 euro": 0,
    "More than 250,000 to 500,000 euro": 1,
    "More than 500,000 to 2 million euro": 1,
    "More than 2 to 10 million euro": 1,
    "Don't know/No Answer": 0,
    "More than 10 to 50 million euro": 1,
    "More than 50 million euro": 1,
})

# Bandwidth sample flags: firms just around the CSRD threshold
df["employee_50to499"] = df["scr10"].map({
    "1-9": 0, "10-49": 0, "50-249": 1, "250-499": 1, "500+": 0,
})
df["turnover_2m_50m"] = df["scr14"].map({
    "Less than 25,000 euro": 0,
    "More than 25,000 to 50,000 euro": 0,
    "More than 50,000 to 100,000 euro": 0,
    "More than 100,000 to 250,000 euro": 0,
    "More than 250,000 to 500,000 euro": 0,
    "More than 500,000 to 2 million euro": 0,
    "More than 2 to 10 million euro": 1,
    "Don't know/No Answer": 0,
    "More than 10 to 50 million euro": 1,
    "More than 50 million euro": 0,
})

df["post"] = (df["year"] == 2024).astype(int)
df["sample_main"] = (df["implementation_country"] == 1).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Outcome variables
# ─────────────────────────────────────────────────────────────────────────────
df["target_engaged"] = df["q14"].isin([
    "Yes",
    "No, but you are planning to define a strategy",
    "You are already climate neutral",
]).astype(int)

df["target_engaged_ord"] = df["q14"].map({
    "Yes": 2,
    "No, but you are planning to define a strategy": 1,
    "You are already climate neutral": 3,
    "No, and you are not planning to do so": 0,
    "Don't know/No Answer": 0,
})

growth_map = {
    "Decreased": -1, "Remained unchanged": 0, "Increased": 1,
    "Don't know/No Answer": np.nan,
}
df["firm_growth_ord"] = df["scr13a"].map(growth_map)
df["firm_growth_ord"] = df["firm_growth_ord"].fillna(df["firm_growth_ord"].mode()[0])

growth_map_d = {
    "Decreased": 0, "Remained unchanged": 0, "Increased": 1,
    "Don't know/No Answer": np.nan,
}
df["firm_growth_ord_d"] = df["scr13a"].map(growth_map_d)
df["firm_growth_ord_d"] = df["firm_growth_ord_d"].fillna(df["firm_growth_ord_d"].mode()[0])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Firm-level controls
# ─────────────────────────────────────────────────────────────────────────────

# Firm age
age_map = {
    "Before 1 January 2014": 4,       "Before 1 January 2016": 4,
    "Between 1 January 2014 and 31 December 2016": 3,
    "Between 1 January 2016 and 31 December 2018": 3,
    "Between 1 January 2017 and 1 January 2021": 2,
    "Between 1 January 2019 and 1 January 2023": 2,
    "After 1 January 2021": 1,        "After 1 January 2023": 1,
    "Don't know/No Answer": np.nan,
    "Don't know/No Answer (DO NOT READ OUT)": np.nan,
}
df["firm_age_ord"] = df["scr12"].map(age_map)
df["firm_age_ord"] = df["firm_age_ord"].fillna(df["firm_age_ord"].mode()[0])

age_map_d = {k: (1 if v is not np.nan and v >= 3 else 0) for k, v in age_map.items()}
age_map_d["Don't know/No Answer"] = np.nan
age_map_d["Don't know/No Answer (DO NOT READ OUT)"] = np.nan
df["firm_age_ord_d"] = df["scr12"].map(age_map_d)
df["firm_age_ord_d"] = df["firm_age_ord_d"].fillna(df["firm_age_ord_d"].mode()[0])

# East / West
east_west_map = {
    "Romania": 0, "Poland": 0, "Czechia": 0, "Croatia": 0, "Slovenia": 0,
    "Hungary": 0, "Bulgaria": 0, "Latvia": 0, "Estonia": 0, "Lithuania": 0,
    "Slovakia": 0,
    "France": 1, "Belgium": 1, "Netherlands": 1, "Sweden": 1, "Portugal": 1,
    "Italy": 1, "Greece": 1, "Germany": 1, "Spain": 1, "Austria": 1,
    "Ireland": 1, "Finland": 1, "Denmark": 1, "Luxembourg": 1, "Malta": 1,
    "Republic of Cyprus": 1,
}
df["east_west"] = df["ipscntry"].map(east_west_map)

# Resource-efficiency action maturity (q1/q2 domain pairs)
def to_binary_action(series):
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) != 0).astype("int8")
    s = series.astype("string").str.strip()
    return ((s.notna()) & (s != "Not mentioned")).astype("int8")

domain_pairs = {
    "water":           ("q1_1",  "q2_1"),
    "energy_saving":   ("q1_2",  "q2_2"),
    "green_energy":    ("q1_3",  "q2_3"),
    "materials":       ("q1_4",  "q2_4"),
    "green_suppliers": ("q1_5",  "q2_5"),
    "waste_reduction": ("q1_6",  "q2_6"),
    "sell_residues":   ("q1_7",  "q2_7"),
    "recycling":       ("q1_8",  "q2_8"),
    "eco_design":      ("q1_9",  "q2_9"),
    "other":           ("q1_10", "q2_10"),
}
for col in sorted({c for pair in domain_pairs.values() for c in pair}):
    df[col] = to_binary_action(df[col])
for name, (cur, plan) in domain_pairs.items():
    df[f"{name}_maturity_d"] = np.select(
        [(df[cur] == 0) & (df[plan] == 0),
         (df[cur] == 0) & (df[plan] == 1),
         (df[cur] == 1)],
        [0, 1, 1], default=0,
    ).astype("int8")

# Green staff
s = df["dx5r"].astype(str).str.strip()
df["green_staff_any"] = np.where(
    s == "0 employees", 0,
    np.where(s == "Don't know/No Answer (DO NOT READ OUT)", np.nan, 1),
).astype("float")

# Investment dummy
df["investment_dummy"] = (df["q4"] != "Nothing").astype(int)

# Barriers (Q7)
q7_cols = [c for c in df.columns if c.startswith("q7_")]
for col in q7_cols:
    df[col] = np.where(df[col].isna(), 0, np.where(df[col] == "Not mentioned", 0, 1)).astype("int8")
for name, cols in [
    ("institutional", ["q7_1", "q7_2", "q7_3", "q7_9", "q7_10"]),
    ("capability",    ["q7_4", "q7_6"]),
    ("market",        ["q7_5", "q7_7", "q7_8"]),
]:
    df[f"barrier_{name}"]   = np.minimum(df[cols].sum(axis=1), 2)
    df[f"barrier_{name}_d"] = np.minimum(df[cols].sum(axis=1), 1)

# Support (Q8)
knowledge_cols = ["q8_1", "q8_2", "q8_5", "q8_6"]
financial_cols = ["q8_3", "q8_4"]
for col in knowledge_cols + financial_cols:
    df[col] = np.where(df[col].isna(), 0, np.where(df[col] == "Not mentioned", 0, 1)).astype("int8")
df["support_knowledge_d"] = np.minimum(df[knowledge_cols].sum(axis=1), 1)
df["support_financial_d"] = np.minimum(df[financial_cols].sum(axis=1), 1)
df["support_external"]    = (df["q5_3"] == "External support").astype(int)
df["support_internal"]    = (
    (df["q5_1"] == "Its own financial resources") |
    (df["q5_2"] == "Its own technical expertise")
).astype(int)
df["support_internal_pr"] = (
    (df["q13_1"] == "Its own financial resources") |
    (df["q13_2"] == "Its own technical expertise")
).astype(int)

# Green market outcomes
df["green_offer_ord_d"] = df["q9"].map({
    "No and you are not planning to do so": 0,
    "No, but you are planning to do so in the next 2 years": 1,
    "Yes": 1,
}).fillna(0)

mapping_q10_d = {
    "Up to 5%": 1, "6-10%": 1, "11-30%": 1,
    "31-50%": 1, "51-75%": 1, "More than 75%": 1,
}
mode_q10 = df["q10"].map(mapping_q10_d).mode()[0]
mapping_q10_d["Don't know/No Answer (DO NOT READ OUT)"] = mode_q10
df["green_turnover_pct_d"] = df["q10"].map(mapping_q10_d).fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Select analysis columns and save
# ─────────────────────────────────────────────────────────────────────────────
KEEP_COLS = [
    # Identifiers
    "ipscntry", "nace_b", "year",
    # Policy / exposure
    "implementation_country", "t_date",
    # Treatment
    "employee_treatment", "turnover_treatment", "treat", "post", "sample_main",
    # Placebo treatment components
    "employee_50plus", "turnover_250k",
    # Bandwidth sample flags
    "employee_50to499", "turnover_2m_50m",
    # Outcomes
    "target_engaged", "target_engaged_ord", "firm_growth_ord", "firm_growth_ord_d",
    # Firm-level controls
    "firm_age_ord", "firm_age_ord_d", "east_west",
    "green_staff_any", "investment_dummy",
    # Resource-efficiency maturity (one per domain)
    *[f"{name}_maturity_d" for name in domain_pairs],
    # Barriers
    "barrier_institutional", "barrier_capability", "barrier_market",
    "barrier_institutional_d", "barrier_capability_d", "barrier_market_d",
    # Support
    "support_knowledge_d", "support_financial_d",
    "support_external", "support_internal", "support_internal_pr",
    # Green market outcomes
    "green_offer_ord_d", "green_turnover_pct_d",
]

out = BASE / "analysis_data.parquet"
df[KEEP_COLS].to_parquet(out, index=False)

saved = pd.read_parquet(out)
size_mb = out.stat().st_size / 1_048_576
print(f"\nSaved: analysis_data.parquet  ({size_mb:.1f} MB, {saved.shape[0]:,} rows x {saved.shape[1]} cols)")
print(f"  year distribution: {df['year'].value_counts().to_dict()}")
print(f"  treat=1: {(df['treat']==1).sum():,}   treat=0: {(df['treat']==0).sum():,}   NaN: {df['treat'].isna().sum():,}")
print(f"  impl=1 countries: {df.loc[df['implementation_country']==1,'ipscntry'].nunique()}")
print(f"  impl=0 countries: {df.loc[df['implementation_country']==0,'ipscntry'].nunique()}")
print(f"  target_engaged mean: {df['target_engaged'].mean():.3f}")
print(f"  firm_growth_ord_d mean: {df['firm_growth_ord_d'].mean():.3f}")
