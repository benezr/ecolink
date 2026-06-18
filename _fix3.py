"""
_fix3.py — append Section 6 (Robustness Checks) to comparative_specifications.ipynb
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
assert len(cells) == 36, f"Expected 36 cells, got {len(cells)}"
assert "## 5. Overview Table" in "".join(cells[33]["source"])

new = []

# ── Section header ─────────────────────────────────────────────────────────────
new.append(md_cell([
    "---",
    "## 6. Robustness Checks",
    "",
    "| Subsection | Check |",
    "|---|---|",
    "| 6.1 | Logistic regression for binary growth outcome |",
    "| 6.2 | Repeated cross-section composition checks |",
    "| 6.3 | Placebo treatment definitions |",
    "| 6.4 | DDD first stage and covariate overlap |",
    "| 6.5 | DiD with alternative outcome variables |",
    "| 6.6 | OLS / DiD / DDD with East\\u2013West control |",
]))

# ── 6.1 Logistic regression ────────────────────────────────────────────────────
new.append(md_cell([
    "### 6.1  Logistic Regression for Binary Growth Outcome",
    "",
    "`firm_growth_ord_d` is binary (0/1). The OLS linear probability model (LPM) used in",
    "Sections 1\\u20133 is consistent but may predict probabilities outside [0, 1].  These",
    "logit models replicate the OLS, DiD, and DDD structures.",
    "Coefficients are log-odds; the adjacent LPM column (s1b/s2b/s3d) enables comparison.",
]))

new.append(code_cell([
    "# 6.1  Logit models for firm_growth_ord_d",
    "lg_ols = smf.logit('firm_growth_ord_d ~ treat + C(nace_b) + C(firm_age_ord)',",
    "                    data=df_s).fit(**cl(df_s))",
    "lg_did = smf.logit('firm_growth_ord_d ~ treat*post + C(nace_b) + C(firm_age_ord)',",
    "                    data=df_s).fit(**cl(df_s))",
    "lg_ddd = smf.logit('firm_growth_ord_d ~ treat*post*impl + C(nace_b) + C(firm_age_ord)',",
    "                    data=df_d).fit(**cl(df_d))",
    "",
    "logit_specs = [",
    "    ('OLS analog', lg_ols, 'treat',           s1b, 'treat'),",
    "    ('DiD',        lg_did, 'treat:post',       s2b, 'treat:post'),",
    "    ('DDD',        lg_ddd, 'treat:post:impl',  s3d, 'treat:post:impl'),",
    "]",
    "print('%-14s %11s %8s %8s   %11s %8s %8s' % (",
    "      'Model', 'Logit coef', 'SE', 'p', 'LPM coef', 'SE', 'p'))",
    "print('-' * 76)",
    "for lbl, ml, kl, mo, ko in logit_specs:",
    "    cl_ = ml.params[kl]; sl_ = ml.bse[kl]; pl_ = ml.pvalues[kl]",
    "    co_ = mo.params[ko]; so_ = mo.bse[ko]; po_ = mo.pvalues[ko]",
    "    sg  = '***' if pl_<0.01 else '**' if pl_<0.05 else '*' if pl_<0.10 else ''",
    "    print('%-14s %+11.4f %8.4f %6.3f%-3s  %+11.4f %8.4f %6.3f' % (",
    "          lbl, cl_, sl_, pl_, sg, co_, so_, po_))",
    "print()",
    "print('Logit coefs = log-odds. LPM coefs from s1b/s2b/s3d. SE clustered by ipscntry.')",
]))

# ── 6.2 Composition checks ─────────────────────────────────────────────────────
new.append(md_cell([
    "### 6.2  Repeated Cross-Section Composition Checks",
    "",
    "The survey is not a true firm panel: each wave samples independently.",
    "Observed outcome changes could reflect shifts in sample composition (sector mix,",
    "size distribution, geography) rather than genuine firm-level responses.",
    "We test whether key observable characteristics differ across the 2022 and 2024 waves.",
]))

new.append(code_cell([
    "# 6.2  Composition checks",
    "from scipy.stats import chi2_contingency",
    "",
    "print('Chi-square tests: independence of composition variable and survey year (df_s)')",
    "print('%-30s %8s %4s %8s' % ('Variable', 'chi2', 'df', 'p-value'))",
    "print('-' * 55)",
    "for col in ['treat', 'nace_b', 'east_west', 'firm_age_ord']:",
    "    sub = df_s[['year', col]].dropna()",
    "    ch, pv, dof, _ = chi2_contingency(pd.crosstab(sub['year'], sub[col]))",
    "    sg = '***' if pv<0.01 else '**' if pv<0.05 else '*' if pv<0.10 else ''",
    "    print('%-30s %8.2f %4d %8.3f%s' % (col, ch, dof, pv, sg))",
    "",
    "print()",
    "print('Mean outcomes by year x treat group (df_s)')",
    "tbl = (df_s.groupby(['year', 'treat'])",
    "           [['target_engaged', 'firm_growth_ord_d', 'firm_age_ord']]",
    "           .mean())",
    "print(tbl.to_string(float_format='%.4f'))",
    "",
    "print()",
    "print('Sample size by year (df_s)')",
    "print(df_s.groupby('year').size().rename('n').to_string())",
]))

# ── 6.3 Placebo treatment definitions ──────────────────────────────────────────
new.append(md_cell([
    "### 6.3  Placebo Treatment Definitions",
    "",
    "The baseline `treat` indicator is 1 when a firm meets **either** the employee",
    "(\\u2265 250) **or** turnover (\\u2265 \\u20ac10 M) threshold (CSRD Article 3).",
    "We re-estimate the DiD and DDD using three alternative definitions built from the",
    "same pre-computed binary indicators in the parquet:",
    "",
    "| Definition | Rule |",
    "|---|---|",
    "| Baseline (any) | employee \\u2265 250 **OR** turnover \\u2265 \\u20ac10 M (main spec) |",
    "| Employees only | `employee_treatment = 1` regardless of turnover |",
    "| Turnover only  | `turnover_treatment = 1` regardless of employees |",
    "| Both (strict)  | `employee_treatment = 1` **AND** `turnover_treatment = 1` |",
]))

new.append(code_cell([
    "# 6.3  Placebo treatment definitions",
    "df_s_pl = df_s.copy()",
    "df_d_pl = df_d.copy()",
    "for df_ in [df_s_pl, df_d_pl]:",
    "    df_['treat_emp']    = df_['employee_treatment'].astype(float)",
    "    df_['treat_turn']   = df_['turnover_treatment'].astype(float)",
    "    df_['treat_strict'] = (df_['employee_treatment'] * df_['turnover_treatment']).astype(float)",
    "",
    "print('Treatment group size by definition (df_s)')",
    "for tcol, lbl in [('treat','Baseline'),('treat_emp','Employees only'),",
    "                   ('treat_turn','Turnover only'),('treat_strict','Both (strict)')]:",
    "    n = int(df_s_pl[tcol].fillna(0).sum())",
    "    print('  %-18s treat=1: %5d  (%.1f%%)' % (lbl, n, 100*n/len(df_s_pl)))",
    "",
    "print()",
    "print('%-20s %10s %8s %8s   %10s %8s %8s' % (",
    "      'Treatment def.', 'DiD coef', 'SE', 'p', 'DDD coef', 'SE', 'p'))",
    "print('-' * 80)",
    "for label, tcol in [('Baseline (any)',  'treat'),",
    "                     ('Employees only', 'treat_emp'),",
    "                     ('Turnover only',  'treat_turn'),",
    "                     ('Both (strict)',  'treat_strict')]:",
    "    if tcol == 'treat':",
    "        md_ = s2a;  m3_ = s3b",
    "    else:",
    "        ss = df_s_pl.dropna(subset=[tcol])",
    "        sd = df_d_pl.dropna(subset=[tcol])",
    "        md_ = smf.ols('target_engaged ~ %s*post + C(nace_b) + C(firm_age_ord)' % tcol,",
    "                       data=ss).fit(cov_type='cluster', cov_kwds={'groups': ss['ipscntry']})",
    "        m3_ = smf.ols('target_engaged ~ %s*post*impl + C(nace_b) + C(firm_age_ord)' % tcol,",
    "                       data=sd).fit(cov_type='cluster', cov_kwds={'groups': sd['ipscntry']})",
    "    kd = tcol + ':post';  k3 = tcol + ':post:impl'",
    "    cd = md_.params[kd]; sd_ = md_.bse[kd]; pd_ = md_.pvalues[kd]",
    "    c3 = m3_.params[k3]; s3_ = m3_.bse[k3]; p3_ = m3_.pvalues[k3]",
    "    sgd = '***' if pd_<0.01 else '**' if pd_<0.05 else '*' if pd_<0.10 else ''",
    "    sg3 = '***' if p3_<0.01 else '**' if p3_<0.05 else '*' if p3_<0.10 else ''",
    "    print('%-20s %+10.4f %8.4f %6.3f%-3s  %+10.4f %8.4f %6.3f%s' % (",
    "          label, cd, sd_, pd_, sgd, c3, s3_, p3_, sg3))",
]))

# ── 6.4 First stage / overlap ──────────────────────────────────────────────────
new.append(md_cell([
    "### 6.4  DDD First Stage and Covariate Overlap",
    "",
    "**Panel A** plots mean `target_engaged` by year for the four DDD cells (treat \\u00d7 impl).",
    "Non-parallel pre-trends between the two treatment-group pairs would undermine the",
    "identification assumption.",
    "",
    "**Panel B** shows the `firm_age_ord` distribution within each cell as a covariate",
    "overlap check.  The DR DDD estimator requires common support across all four groups.",
    "",
    "The group count table below confirms adequate observations in each cell.",
]))

new.append(code_cell([
    "# 6.4  DDD first stage and overlap",
    "import matplotlib.ticker as mticker",
    "",
    "print('Observation counts by (impl x treat x post)')",
    "cnt = df_d.groupby(['impl', 'treat', 'post']).size().reset_index(name='n')",
    "print(cnt.to_string(index=False))",
    "print()",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(13, 5))",
    "",
    "# Panel A: mean outcome trajectories for 4 groups",
    "style_map = {(1, 1): ('steelblue', '-',  'o'),",
    "             (1, 0): ('steelblue', '--', 's'),",
    "             (0, 1): ('tomato',    '-',  'o'),",
    "             (0, 0): ('tomato',    '--', 's')}",
    "for (impl_v, treat_v), (col, ls, mk) in style_map.items():",
    "    sub = df_d[(df_d['impl'] == impl_v) & (df_d['treat'] == treat_v)]",
    "    means = sub.groupby('year')['target_engaged'].mean()",
    "    axes[0].plot(means.index, means.values, color=col, linestyle=ls,",
    "                 marker=mk, linewidth=1.8,",
    "                 label='impl=%d, treat=%d' % (impl_v, treat_v))",
    "axes[0].set_title('Target engagement by group and year', fontsize=10)",
    "axes[0].set_xlabel('Survey year')",
    "axes[0].set_ylabel('Mean target_engaged')",
    "axes[0].legend(fontsize=8)",
    "axes[0].grid(alpha=0.3)",
    "",
    "# Panel B: firm_age_ord distribution — overlap check",
    "color_b = {(1, 1): 'steelblue', (1, 0): 'lightblue',",
    "           (0, 1): 'tomato',    (0, 0): 'lightsalmon'}",
    "for (impl_v, treat_v), col in color_b.items():",
    "    vals = df_d[(df_d['impl'] == impl_v) & (df_d['treat'] == treat_v)]['firm_age_ord'].dropna()",
    "    axes[1].hist(vals, bins=[0.5, 1.5, 2.5, 3.5, 4.5], alpha=0.55, density=True,",
    "                 color=col, label='impl=%d, treat=%d (n=%d)' % (impl_v, treat_v, len(vals)))",
    "axes[1].set_title('Firm age distribution by group (overlap)', fontsize=10)",
    "axes[1].set_xlabel('firm_age_ord  (1=youngest, 4=oldest)')",
    "axes[1].set_ylabel('Density')",
    "axes[1].legend(fontsize=7)",
    "axes[1].grid(alpha=0.3)",
    "axes[1].xaxis.set_major_locator(mticker.MultipleLocator(1))",
    "",
    "plt.tight_layout()",
    "plt.savefig('ddd_overlap.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
    "print('Saved: ddd_overlap.png')",
]))

# ── 6.5 DiD with alternative outcomes ─────────────────────────────────────────
new.append(md_cell([
    "### 6.5  DiD with Alternative Outcome Variables",
    "",
    "Running DiD on pre-determined firm characteristics (`firm_age_ord`) serves as a",
    "**falsification test**: a significant `treat:post` coefficient would indicate",
    "compositional shifts rather than genuine treatment effects.",
    "For forward-looking variables (`green_offer_ord_d`, `green_staff_any`, etc.) the DiD",
    "captures potential spillover mechanisms of the CSRD.",
]))

new.append(code_cell([
    "# 6.5  DiD on alternative outcomes",
    "alt_out = [",
    "    ('firm_age_ord',            'Firm age ordinal [falsification]'),",
    "    ('green_staff_any',         'Green staff (any)'),",
    "    ('investment_dummy',        'Investment dummy'),",
    "    ('barrier_institutional_d', 'Institutional barriers'),",
    "    ('support_knowledge_d',     'Knowledge support'),",
    "    ('green_offer_ord_d',       'Green offer (binary)'),",
    "]",
    "",
    "print('%-40s %10s %8s %8s  %7s' % ('Outcome', 'treat:post', 'SE', 'p', 'N'))",
    "print('-' * 80)",
    "for col, label in alt_out:",
    "    sub = df_s.dropna(subset=[col]).copy()",
    "    # firm_age_ord cannot appear as both LHS and RHS control",
    "    rhs = ('treat*post + C(nace_b)' if col == 'firm_age_ord'",
    "           else 'treat*post + C(nace_b) + C(firm_age_ord)')",
    "    m = smf.ols('%s ~ %s' % (col, rhs), data=sub).fit(",
    "            cov_type='cluster', cov_kwds={'groups': sub['ipscntry']})",
    "    c = m.params.get('treat:post', float('nan'))",
    "    s = m.bse.get('treat:post',   float('nan'))",
    "    p = m.pvalues.get('treat:post', float('nan'))",
    "    sg = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''",
    "    print('%-40s %+10.4f %8.4f %6.3f%-3s %7d' % (label, c, s, p, sg, int(m.nobs)))",
    "print()",
    "print('DiD: treat*post + sector FEs + firm_age FE (omitted when outcome = firm_age_ord).')",
    "print('SE clustered by ipscntry. Significant firm_age_ord coef = composition shift.')",
]))

# ── 6.6 East-West control ──────────────────────────────────────────────────────
new.append(md_cell([
    "### 6.6  OLS / DiD / DDD with East\\u2013West Control",
    "",
    "`east_west = 1` for Western EU member states, `0` for Central/Eastern EU.",
    "Adding this country-level dummy controls for unobserved geographic differences",
    "in baseline climate engagement.  In DDD models `east_west` is collinear with",
    "country-level variation in `impl`, so coefficient SEs may be inflated; the",
    "direction and significance of `treat:post:impl` provides the relevant comparison.",
    "",
    "| New model | Base | Added |",
    "|---|---|---|",
    "| s1a\\_ew / s1b\\_ew | s1a / s1b | + `east_west` (OLS) |",
    "| s2a\\_ew / s2b\\_ew | s2a / s2b | + `east_west` (DiD) |",
    "| s3b\\_ew / s3d\\_ew | s3b / s3d | + `east_west` (DDD) |",
]))

new.append(code_cell([
    "# 6.6  East-West robustness",
    "s1a_ew = smf.ols('target_engaged    ~ treat + east_west + C(nace_b) + C(firm_age_ord)',",
    "                  data=df_s).fit(**cl(df_s))",
    "s1b_ew = smf.ols('firm_growth_ord_d ~ treat + east_west + C(nace_b) + C(firm_age_ord)',",
    "                  data=df_s).fit(**cl(df_s))",
    "s2a_ew = smf.ols('target_engaged    ~ treat*post + east_west + C(nace_b) + C(firm_age_ord)',",
    "                  data=df_s).fit(**cl(df_s))",
    "s2b_ew = smf.ols('firm_growth_ord_d ~ treat*post + east_west + C(nace_b) + C(firm_age_ord)',",
    "                  data=df_s).fit(**cl(df_s))",
    "s3b_ew = smf.ols('target_engaged    ~ treat*post*impl + east_west + C(nace_b) + C(firm_age_ord)',",
    "                  data=df_d).fit(**cl(df_d))",
    "s3d_ew = smf.ols('firm_growth_ord_d ~ treat*post*impl + east_west + C(nace_b) + C(firm_age_ord)',",
    "                  data=df_d).fit(**cl(df_d))",
    "",
    "ew_specs = [",
    "    ('OLS 1a', s1a_ew, s1a, 'treat'),",
    "    ('OLS 1b', s1b_ew, s1b, 'treat'),",
    "    ('DiD 2a', s2a_ew, s2a, 'treat:post'),",
    "    ('DiD 2b', s2b_ew, s2b, 'treat:post'),",
    "    ('DDD 3b', s3b_ew, s3b, 'treat:post:impl'),",
    "    ('DDD 3d', s3d_ew, s3d, 'treat:post:impl'),",
    "]",
    "",
    "print('%-10s %11s %8s %6s   %11s %8s %6s   %9s %8s' % (",
    "      'Model', '+EW coef', 'SE', 'p', 'Base coef', 'SE', 'p', 'EW coef', 'EW SE'))",
    "print('-' * 88)",
    "for lbl, mew, mbase, key in ew_specs:",
    "    c_ew = mew.params[key];   s_ew = mew.bse[key];   p_ew = mew.pvalues[key]",
    "    c_b  = mbase.params[key]; s_b  = mbase.bse[key]; p_b  = mbase.pvalues[key]",
    "    c_e  = mew.params.get('east_west', float('nan'))",
    "    s_e  = mew.bse.get('east_west',    float('nan'))",
    "    sg   = '***' if p_ew<0.01 else '**' if p_ew<0.05 else '*' if p_ew<0.10 else ''",
    "    print('%-10s %+11.4f %8.4f %4.3f%-3s  %+11.4f %8.4f %4.3f   %+9.4f %8.4f' % (",
    "          lbl, c_ew, s_ew, p_ew, sg, c_b, s_b, p_b, c_e, s_e))",
]))

# ── Write ──────────────────────────────────────────────────────────────────────
cells.extend(new)
print("Total cells:", len(cells))

full = "\n".join("".join(c["source"]) for c in cells)
assert "## 6. Robustness" in full
assert "lg_ols"           in full
assert "chi2_contingency" in full
assert "treat_emp"        in full
assert "ddd_overlap.png"  in full
assert "alt_out"          in full
assert "s1a_ew"           in full
print("All assertions passed.")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Written.")
