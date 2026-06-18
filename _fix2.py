"""
_fix2.py  — patch comparative_specifications.ipynb:
  1. Delete cells 20-21 (Section 4: HDFE DDD)
  2. Delete cells 22-24 (mid-notebook comparison table)
  3. Remove all Cyprus recodes from Section 6 → renumber to Section 4
  4. Add overview table (all models + DR DDD) as new Section 5 at the end
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\zolb\Files C\ClaudeFolder\EcoLink\comparative_specifications.ipynb")

def src(lines):
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def code_cell(lines):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src(lines)}

def md_cell(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": src(lines)}

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
assert len(cells) == 38, f"Expected 38 cells, got {len(cells)}"

# ─── Sanity-check key cells before any changes ────────────────────────────────
assert "High-Dimensional" in "".join(cells[20]["source"])
assert "summary_col"      in "".join(cells[23]["source"])
assert "## 6. Doubly-Robust" in "".join(cells[25]["source"])
assert "Cyprus"           in "".join(cells[27]["source"])
assert "impl_cyp"         in "".join(cells[33]["source"])

# ─── 1. Delete cells 20-24 ────────────────────────────────────────────────────
del cells[20:25]
# After this: orig[25] → new[20], orig[26] → 21, orig[27] → 22 ...
# orig[28] → 23, orig[29] → 24, orig[30] → 25, orig[31] → 26
# orig[32] → 27, orig[33] → 28, orig[34] → 29, orig[35] → 30
# orig[36] → 31, orig[37] → 32

# ─── 2. new[20] (orig 25): rewrite Section 6→4 header, strip Cyprus paragraph ─
cells[20]["source"] = src([
    "---",
    "## 4. Doubly-Robust DDD",
    "",
    "Implements the **DR DDD estimator** from Ortiz-Villavicencio & Sant'Anna (2025),",
    '"Better Understanding Triple Differences Estimators" (OVS 2025), eq. 4.1.',
    "",
    "### Why we need it",
    "",
    "Section 3 uses the three-way fixed effects (3WFE) regression",
    "`Y ~ treat*post*impl + FEs`.  OVS (2025) show this estimator is **generally biased** when",
    "covariates are needed to justify the conditional DDD parallel-trends assumption (DDD-CPT).",
    "The bias arises because 3WFE integrates covariates over the *control* distribution instead",
    "of the *treated* distribution (paper sec. 3.1, Figure 1).",
    "",
    "### Identification structure",
    "",
    "In our setting a firm is **effectively treated** if it satisfies **two** criteria:",
    "",
    "| Paper symbol | Our variable | Meaning |",
    "|---|---|---|",
    "| Q = 1 (eligible) | `treat = 1` | Large firm above CSRD threshold |",
    "| S = g (enabling) | `impl = 1` | Country transposed the directive |",
    "",
    "The never-enabling countries (`impl = 0`) serve as the comparison group (`g_c = inf`).",
    "Standard errors are clustered by country (`ipscntry`, 27 clusters).",
    "",
    "### Estimator",
    "",
    "The DR DDD decomposes into **three DR DiD components** (OVS 2025, eq. 4.1):",
    "",
    r"$$\widehat{ATT}_{\text{dr-ddd}} = \widehat{ATT}_{k3} + \widehat{ATT}_{k2} - \widehat{ATT}_{k1}$$",
    "",
    "| Component | Treated group | Comparison group | Identifies |",
    "|---|---|---|---|",
    r"| $k_3$ | impl=1, treat=1 | impl=1, treat=0 | Within-country eligibility DiD |",
    r"| $k_2$ | impl=1, treat=1 | impl=0, treat=1 | Between-country, among eligible |",
    r"| $k_1$ | impl=1, treat=1 | impl=0, treat=0 | Between-country, between eligibility |",
    "",
    "Each component uses the **Sant'Anna & Zhao (2020) DR DiD** for repeated cross-sections:",
    "a logistic propensity score and linear outcome regressions for each (group × period) cell.",
    "",
    "The ATT is doubly robust: consistent if **either** the propensity score model **or**",
    "the outcome regression model is correctly specified (not necessarily both).",
    "",
    "Standard errors are computed from the combined **influence function**, clustered by country.",
])

# ─── 3. new[21] (orig 26): section title, no Cyprus ──────────────────────────
cells[21]["source"] = src([
    "### 4.1  Covariates and DR estimation setup",
])

# ─── 4. new[22] (orig 27): remove Cyprus recode ──────────────────────────────
cells[22]["source"] = src([
    "from sklearn.linear_model import LogisticRegression, LinearRegression",
    "from scipy import stats as scipy_stats",
    "import warnings",
    "",
    "# One-hot sector FEs + firm age (ordinal numeric) + east/west",
    "sector_dummies = pd.get_dummies(df_ddd['nace_b'], prefix='sec', drop_first=True)",
    "X_cols_dr = sector_dummies.columns.tolist() + ['firm_age_ord', 'east_west']",
    "",
    "df_dr_base = pd.concat([df_ddd.reset_index(drop=True),",
    "                         sector_dummies.reset_index(drop=True)], axis=1)",
    "",
    "df_dr = df_dr_base.dropna(subset=X_cols_dr + ['implementation_country']).copy()",
    "print(f'DR sample: {len(df_dr):,} obs, {df_dr[\"ipscntry\"].nunique()} countries')",
    "print(f'impl=1 countries: {df_dr.loc[df_dr[\"implementation_country\"]==1,\"ipscntry\"].nunique()}')",
    "print(f'impl=0 countries: {df_dr.loc[df_dr[\"implementation_country\"]==0,\"ipscntry\"].nunique()}')",
    "",
    "for name, cond in [('impl=1 treat=1', (df_dr['implementation_country']==1)&(df_dr['treat']==1)),",
    "                   ('impl=1 treat=0', (df_dr['implementation_country']==1)&(df_dr['treat']==0)),",
    "                   ('impl=0 treat=1', (df_dr['implementation_country']==0)&(df_dr['treat']==1)),",
    "                   ('impl=0 treat=0', (df_dr['implementation_country']==0)&(df_dr['treat']==0))]:",
    "    print(f'  {name}: {cond.sum():,}')",
])

# ─── 5. new[23] (orig 28): rename 6.2 → 4.2 ─────────────────────────────────
s = "".join(cells[23]["source"])
cells[23]["source"] = s.replace("### 6.2", "### 4.2").splitlines(keepends=True)

# ─── 6. new[25] (orig 30): rename 6.3 → 4.3 ─────────────────────────────────
s = "".join(cells[25]["source"])
cells[25]["source"] = s.replace("### 6.3", "### 4.3").splitlines(keepends=True)

# ─── 7. new[27] (orig 32): rename 6.4 → 4.4 ─────────────────────────────────
s = "".join(cells[27]["source"])
cells[27]["source"] = s.replace("### 6.4", "### 4.4").splitlines(keepends=True)

# ─── 8. new[28] (orig 33): impl_cyp → implementation_country ─────────────────
s = "".join(cells[28]["source"])
cells[28]["source"] = s.replace("impl_col    = 'impl_cyp'",
                                 "impl_col    = 'implementation_country'").splitlines(keepends=True)

# ─── 9. new[29] (orig 34): rename 6.5 → 4.5, update wording ─────────────────
cells[29]["source"] = src([
    "### 4.5  DR DDD vs 3WFE DDD — comparison",
    "",
    "The 3WFE coefficients come from the baseline spec in Section 3 (`treat*post*impl`",
    "+ sector + age FEs, analytic cluster-robust SE).",
])

# ─── 10. new[30] (orig 35): rename sec.6→sec.4, remove Cyprus line ────────────
cells[30]["source"] = src([
    "def stars(p):",
    "    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''",
    "",
    "# 3WFE DDD references from Section 3 (sector+age FE, analytic cluster-robust SE)",
    "ref_3wfe = {",
    "    'target_engaged':    (s3b.params['treat:post:impl'], s3b.pvalues['treat:post:impl']),",
    "    'firm_growth_ord_d': (s3d.params['treat:post:impl'], s3d.pvalues['treat:post:impl']),",
    "}",
    "",
    "print('=' * 78)",
    "print(f'{\"Outcome\":<22} {\"Estimator\":<20} {\"ATT\":>9} {\"SE\":>8} {\"p\":>8} {\"CI 95%\":>22}')",
    "print('-' * 78)",
    "",
    "for col, label in OUTCOMES_DR.items():",
    "    dr  = dr_results[col]",
    "    c3, p3 = ref_3wfe[col]",
    "    se3 = s3b.bse['treat:post:impl'] if col == 'target_engaged' else s3d.bse['treat:post:impl']",
    "    ci3_lo = c3 - 1.96 * se3;  ci3_hi = c3 + 1.96 * se3",
    "",
    "    print(f'{label:<22} {\"3WFE DDD (sec.3)\":<20} {c3:>+9.4f} {se3:>8.4f}',",
    "          f'{p3:>6.3f}{stars(p3):<2} [{ci3_lo:+.4f}, {ci3_hi:+.4f}]')",
    "    print(f'{\"\":<22} {\"DR DDD (sec.4)\":<20} {dr[\"att\"]:>+9.4f} {dr[\"se\"]:>8.4f}',",
    "          f'{dr[\"p\"]:>6.3f}{stars(dr[\"p\"]):<2} [{dr[\"ci_low\"]:+.4f}, {dr[\"ci_high\"]:+.4f}]')",
    "    print()",
    "",
    "print('=' * 78)",
    "print('SE: both estimators cluster by ipscntry (27 country clusters).')",
    "print('3WFE SE: analytic cluster-robust. DR DDD SE: influence-function cluster sandwich.')",
    "print('3WFE relies on UNCONDITIONAL DDD-CPT; DR DDD is valid under CONDITIONAL DDD-CPT.')",
])

# ─── 11. new[31] (orig 36): rename 6.6 → 4.6 ────────────────────────────────
s = "".join(cells[31]["source"])
cells[31]["source"] = s.replace("### 6.6", "### 4.6").splitlines(keepends=True)

# ─── 12. Append Section 5: Overview Table ────────────────────────────────────
cells.append(md_cell([
    "---",
    "## 5. Overview Table",
    "",
    "All parametric specifications (Sections 1–3) and the DR DDD estimates (Section 4).",
    "Standard errors clustered by country (`ipscntry`).",
    "Significance: \\* p<0.1, \\*\\* p<0.05, \\*\\*\\* p<0.01.",
    "",
    "| Section | Models | Key coefficient |",
    "|---|---|---|",
    "| 1 OLS  | 1a–1d | `treat` |",
    "| 2 DiD  | 2a–2d | `treat:post` |",
    "| 3 DDD  | 3a–3d | `treat:post:impl` |",
    "| 4 DR DDD | — | ATT (influence-function SE) |",
]))

cells.append(code_cell([
    "from statsmodels.iolib.summary2 import summary_col",
    "",
    "# ── Parametric models 1a–3d ─────────────────────────────────────────────",
    "comp_table = summary_col(",
    "    [s1a, s1b, s1c, s1d, s2a, s2b, s2c, s2d, s3a, s3b, s3c, s3d],",
    "    model_names=[",
    '        "(1a)","(1b)","(1c)","(1d)",',
    '        "(2a)","(2b)","(2c)","(2d)",',
    '        "(3a)","(3b)","(3c)","(3d)",',
    "    ],",
    "    stars=True,",
    '    float_format="%0.4f",',
    '    regressor_order=["treat", "treat:post", "treat:post:impl"],',
    "    drop_omitted=True,",
    "    info_dict={",
    '        "N":         lambda x: f"{int(x.nobs)}",',
    '        "R-squared": lambda x: f"{x.rsquared:.3f}",',
    '        "Outcome":   lambda x: x.model.endog_names,',
    "    },",
    ")",
    "print(comp_table)",
    "",
    "# ── DR DDD (Section 4) ──────────────────────────────────────────────────",
    "def _stars(p):",
    "    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''",
    "",
    "print()",
    "print('DR DDD estimates (Section 4)')",
    "print('=' * 72)",
    "print(f'{\"Outcome\":<30} {\"ATT\":>9} {\"SE\":>8} {\"p\":>8} {\"95% CI\":>22}')",
    "print('-' * 72)",
    "for col, label in OUTCOMES_DR.items():",
    "    r = dr_results[col]",
    "    print(f'{label:<30} {r[\"att\"]:>+9.4f} {r[\"se\"]:>8.4f} '",
    "          f'{r[\"p\"]:>6.3f}{_stars(r[\"p\"]):<2} [{r[\"ci_low\"]:+.4f}, {r[\"ci_high\"]:+.4f}]')",
    "print('=' * 72)",
    "print(f'N: {dr_results[\"target_engaged\"][\"n\"]:,}  |  G clusters: {dr_results[\"target_engaged\"][\"G\"]}')",
    "print('SE: influence-function cluster sandwich (ipscntry).')",
]))

cells.append(code_cell([
    "with open('overview_table.txt', 'w', encoding='utf-8') as f:",
    "    f.write(str(comp_table))",
    "    f.write('\\n\\nDR DDD estimates (Section 4)\\n')",
    "    f.write('=' * 72 + '\\n')",
    "    f.write(f'{\"Outcome\":<30} {\"ATT\":>9} {\"SE\":>8} {\"p\":>8} {\"95% CI\":>22}\\n')",
    "    f.write('-' * 72 + '\\n')",
    "    for col, label in OUTCOMES_DR.items():",
    "        r = dr_results[col]",
    "        f.write(f'{label:<30} {r[\"att\"]:>+9.4f} {r[\"se\"]:>8.4f} '",
    "                f'{r[\"p\"]:>6.3f}{_stars(r[\"p\"]):<2} [{r[\"ci_low\"]:+.4f}, {r[\"ci_high\"]:+.4f}]\\n')",
    "    f.write('=' * 72 + '\\n')",
    "print('Saved: overview_table.txt')",
]))

nb["cells"] = cells
print(f"Total cells: {len(cells)}")

# ─── Final assertions ─────────────────────────────────────────────────────────
full_src = "\n".join("".join(c["source"]) for c in cells)
assert "impl_cyp"        not in full_src, "impl_cyp still present!"
assert "Cyprus"          not in full_src, "Cyprus still present!"
assert "s4a"             not in full_src, "s4a still present!"
assert "s4b"             not in full_src, "s4b still present!"
assert "High-Dimensional" not in full_src, "HDFE section still present!"
assert "## 4. Doubly-Robust" in full_src, "Section 4 DR DDD header missing!"
assert "## 5. Overview Table" in full_src, "Section 5 Overview Table missing!"
assert "overview_table.txt"   in full_src, "overview_table.txt save missing!"
print("All assertions passed.")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Written.")
