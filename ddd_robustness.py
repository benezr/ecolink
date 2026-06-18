"""
DDD robustness: treatment-timing recoding + wild cluster bootstrap inference.

Addresses three points from the review against Ortiz-Villavicencio & Sant'Anna
(2025), "Better Understanding Triple Differences Estimators":

  (1) Recode treatment by ACTUAL transposition date instead of the static
      `implementation_country` binary. Cyprus transposed 2025-07-29 -- after the
      entire 2024 post wave -- so it is NOT exposed and is moved into the
      comparison group. Countries that transposed late in 2024 are likewise
      treated as not-yet-enabled under stricter cutoffs.

  (2) The original binary 3WFE DDD is KEPT as a baseline (definition D0) but is
      explicitly labelled as relying on UNCONDITIONAL parallel trends and on
      ignoring transposition timing -- exactly the rigid specification the paper
      (sec. 3.1-3.2) flags as generally invalid once timing/covariates matter.
      The recoded definitions D1-D3 are robustness checks on exposure
      measurement; they remain within the rigid binary 3WFE form, so the
      doubly-robust DDD (paper eqs. 4.3-4.4) is still the principled next step.

  (3) Inference via WILD CLUSTER BOOTSTRAP (Cameron, Gelbach & Miller 2008),
      Rademacher weights, null imposed (WCR), clustered on country. With only
      ~12-27 country clusters -- and as few as 5 enabling clusters under the
      strict cutoff -- analytic cluster-robust SEs over-reject; WCB is the fix.

NOTE on data: the survey waves are 2021 (pre) and 2024 (post); there is no
within-wave fielding date in the data, so the exact 2024 cutoff is unknown.
The ladder of cutoffs makes the sensitivity to that unknown explicit.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices

BASE = Path(r"C:\Users\zolb\Files C\ClaudeFolder\EcoLink")
B = 999                       # bootstrap replications
SEED = 12345
DDD_TERM = "treat:post:impl"  # patsy name of the triple interaction

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load + feature engineering (mirrors run_did.py)
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_excel(BASE / "merged_final_new.xlsx")
df = df[df["eu27"] != "Not country group"].copy() if "eu27" in df.columns else df

df["employee_treatment"] = df["scr10"].map(
    {"1-9": 0, "10-49": 0, "50-249": 0, "250-499": 1, "500+": 1}
)
df["turnover_treatment"] = df["scr14"].map({
    "Less than 25,000 euro": 0, "More than 25,000 to 50,000 euro": 0,
    "More than 50,000 to 100,000 euro": 0, "More than 100,000 to 250,000 euro": 0,
    "More than 250,000 to 500,000 euro": 0, "More than 500,000 to 2 million euro": 0,
    "More than 2 to 10 million euro": 0, "Don't know/No Answer": 0,
    "More than 10 to 50 million euro": 1, "More than 50 million euro": 1,
})

df["post"] = (df["year"] == 2024).astype(int)
df["treat"] = np.where(
    (df["employee_treatment"] == 1) | (df["turnover_treatment"] == 1), 1,
    np.where((df["employee_treatment"] == 0) & (df["turnover_treatment"] == 0), 0, np.nan),
)

df["target_engaged"] = df["q14"].isin(
    ["Yes", "No, but you are planning to define a strategy", "You are already climate neutral"]
).astype(int)

age_map = {
    "Before 1 January 2014": 4, "Between 1 January 2014 and 31 December 2016": 3,
    "Between 1 January 2017 and 1 January 2021": 2, "After 1 January 2021": 1,
    "Don't know/No Answer": np.nan, "Before 1 January 2016": 4,
    "Between 1 January 2016 and 31 December 2018": 3,
    "Between 1 January 2019 and 1 January 2023": 2, "After 1 January 2023": 1,
    "Don't know/No Answer (DO NOT READ OUT)": np.nan,
}
df["firm_age_ord"] = df["scr12"].map(age_map)
df["firm_age_ord"] = df["firm_age_ord"].fillna(df["firm_age_ord"].mode()[0])

growth_map_d = {"Decreased": 0, "Remained unchanged": 0, "Increased": 1,
                "Don't know/No Answer": np.nan}
df["firm_growth_ord_d"] = df["scr13a"].map(growth_map_d)
df["firm_growth_ord_d"] = df["firm_growth_ord_d"].fillna(df["firm_growth_ord_d"].mode()[0])

# ─────────────────────────────────────────────────────────────────────────────
# 2. Merge transposition data WITH the actual dates (not just the binary flag)
# ─────────────────────────────────────────────────────────────────────────────
tr = pd.read_excel(BASE / "Transposition_data.xlsx")[
    ["ipscntry", "implementation_country", "t_date"]
]
tr["t_date"] = pd.to_datetime(tr["t_date"])
df = df.merge(tr, on="ipscntry", how="left")

# DDD sample (same row set as run_did.py's df_d: both outcomes share N)
df_d = df.dropna(subset=[
    "target_engaged", "treat", "post", "scr12", "nace_b",
    "ipscntry", "implementation_country", "firm_growth_ord_d",
]).copy()
df_d["treat"] = df_d["treat"].astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Treatment definitions (exposure = transposed by cutoff date)
#    D0 = original binary baseline (includes Cyprus; ignores timing).
# ─────────────────────────────────────────────────────────────────────────────
DEFS = [
    ("D0 baseline binary (orig, incl. Cyprus)", None),
    ("D1 exposed by 2024-12-31 (Cyprus -> control)", pd.Timestamp("2024-12-31")),
    ("D2 exposed by 2024-06-30 (strict)",            pd.Timestamp("2024-06-30")),
    ("D3 exposed by 2024-03-31 (very strict)",       pd.Timestamp("2024-03-31")),
]

def make_impl(frame, cutoff):
    if cutoff is None:
        return frame["implementation_country"].astype(int).to_numpy()
    exposed = (
        (frame["implementation_country"] == 1)
        & frame["t_date"].notna()
        & (frame["t_date"] <= cutoff)
    )
    return exposed.astype(int).to_numpy()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Wild cluster bootstrap (WCR: Rademacher weights, null imposed)
# ─────────────────────────────────────────────────────────────────────────────
def _crse_t(Xv, XtX_inv, clusters, G, N, K, j, beta, resid):
    """Cluster-robust t-stat for coefficient j (CGM small-sample correction)."""
    S = np.zeros((G, K))
    np.add.at(S, clusters, Xv * resid[:, None])
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    c = (G / (G - 1)) * ((N - 1) / (N - K))
    se = np.sqrt(c * V[j, j])
    return beta[j] / se

def wcb(formula, data, cluster_col, term=DDD_TERM, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    y, X = dmatrices(formula, data, return_type="dataframe")
    yv = y.to_numpy()[:, 0]
    Xv = X.to_numpy()
    names = list(X.columns)
    j = names.index(term)
    clusters = pd.Categorical(data[cluster_col]).codes
    G, (N, K) = clusters.max() + 1, Xv.shape

    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    beta = XtX_inv @ (Xv.T @ yv)
    t_obs = _crse_t(Xv, XtX_inv, clusters, G, N, K, j, beta, yv - Xv @ beta)

    # restricted fit: drop the DDD column, impose H0 beta_ddd = 0
    keep = [c for c in range(K) if c != j]
    Xr = Xv[:, keep]
    br = np.linalg.solve(Xr.T @ Xr, Xr.T @ yv)
    fitted_r = Xr @ br
    resid_r = yv - fitted_r

    count = 0
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], size=G)[clusters]
        ystar = fitted_r + resid_r * w
        bstar = XtX_inv @ (Xv.T @ ystar)
        tstar = _crse_t(Xv, XtX_inv, clusters, G, N, K, j, bstar, ystar - Xv @ bstar)
        if abs(tstar) >= abs(t_obs):
            count += 1
    p_wcb = (count + 1) / (B + 1)
    return t_obs, p_wcb, G

# ─────────────────────────────────────────────────────────────────────────────
# 5. Run the ladder for both outcomes
# ─────────────────────────────────────────────────────────────────────────────
OUTCOMES = {
    "target_engaged":    "target_engaged    ~ treat*post*impl + C(nace_b) + C(firm_age_ord)",
    "firm_growth_ord_d": "firm_growth_ord_d ~ treat*post*impl + C(nace_b) + C(firm_age_ord)",
}

def cl(frame):
    return {"cov_type": "cluster", "cov_kwds": {"groups": frame["ipscntry"]}}

rows = []
for def_label, cutoff in DEFS:
    work = df_d.copy()
    work["impl"] = make_impl(work, cutoff)
    n_enabling = work.loc[work["impl"] == 1, "ipscntry"].nunique()
    n_treated_cells = int(((work["treat"] == 1) & (work["impl"] == 1) & (work["post"] == 1)).sum())
    for out_name, formula in OUTCOMES.items():
        m = smf.ols(formula, data=work).fit(**cl(work))
        coef = m.params[DDD_TERM]
        p_analytic = m.pvalues[DDD_TERM]
        t_obs, p_wcb, G = wcb(formula, work, "ipscntry")
        rows.append({
            "definition": def_label, "outcome": out_name,
            "n_enabling_clusters": n_enabling, "n_total_clusters": G,
            "n_eff_treated_obs": n_treated_cells, "N": int(m.nobs),
            "ddd_coef": coef, "t_obs": t_obs,
            "p_analytic": p_analytic, "p_wcb": p_wcb,
        })

res = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Report
# ─────────────────────────────────────────────────────────────────────────────
def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

lines = []
lines.append("=" * 100)
lines.append("DDD ROBUSTNESS: treatment-timing recoding + wild cluster bootstrap")
lines.append("Outcome eq.: <Y> ~ treat*post*impl + C(nace_b) + C(firm_age_ord), clustered by country")
lines.append("DDD coefficient = treat:post:impl.  B=%d Rademacher draws, null imposed (WCR)." % B)
lines.append("")
lines.append("D0 is the ORIGINAL binary 3WFE DDD, kept as baseline. It relies on UNCONDITIONAL")
lines.append("parallel trends and ignores transposition timing (paper sec. 3.1-3.2): treat with caution.")
lines.append("D1 fixes the unambiguous Cyprus misclassification (transposed 2025-07-29, after the 2024 wave).")
lines.append("D2/D3 progressively require earlier exposure; late-2024 transposers become not-yet-enabled controls.")
lines.append("=" * 100)
header = (f"{'Definition':<46}{'Outcome':<18}{'enbl/tot cl':>12}"
          f"{'N':>8}{'DDD coef':>11}{'p(analytic)':>13}{'p(WCB)':>11}")
lines.append(header)
lines.append("-" * len(header))
for _, r in res.iterrows():
    lines.append(
        f"{r['definition']:<46}{r['outcome']:<18}"
        f"{str(r['n_enabling_clusters'])+'/'+str(r['n_total_clusters']):>12}"
        f"{r['N']:>8}{r['ddd_coef']:>11.4f}"
        f"{r['p_analytic']:>10.3f}{stars(r['p_analytic']):<3}"
        f"{r['p_wcb']:>8.3f}{stars(r['p_wcb']):<3}"
    )
lines.append("-" * len(header))
lines.append("Stars on each p-value: *** p<0.01, ** p<0.05, * p<0.10.")
lines.append("'enbl/tot cl' = enabling (impl=1) country clusters / total country clusters.")
report = "\n".join(lines)

print(report)
(BASE / "ddd_robustness.txt").write_text(report, encoding="utf-8")
res.to_csv(BASE / "ddd_robustness.csv", index=False)
print("\nSaved: ddd_robustness.txt and ddd_robustness.csv")
