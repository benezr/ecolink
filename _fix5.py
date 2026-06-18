"""
_fix5.py — insert bandwidth robustness check (6.3c) after cell 44 (6.3b code)
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
assert len(cells) == 51, f"Expected 51 cells, got {len(cells)}"
assert "treat_pb"  in "".join(cells[44]["source"]), "Cell 44 should be 6.3b code"
assert "6.4"       in "".join(cells[45]["source"]), "Cell 45 should be 6.4 header"

md = md_cell([
    "#### Bandwidth sample: firms just around the CSRD threshold",
    "",
    "A composition-effect check: restrict the sample to firms that are",
    "**near the treatment threshold on both size dimensions**.",
    "",
    "| Dimension | Bandwidth | Spans threshold at |",
    "|---|---|---|",
    "| Employees | 50\\u2013499 | 250 employees |",
    "| Revenue   | \\u20ac2 M\\u2013\\u20ac50 M | \\u20ac10 M |",
    "",
    "Both flags (`employee_50to499`, `turnover_2m_50m`) must equal 1.",
    "Within this window the original `treat` indicator still applies",
    "(\\u2265 250 employees **or** \\u2265 \\u20ac10 M revenue).",
    "",
    "If the treatment effect holds in this narrow bandwidth it is unlikely",
    "to be driven by firm characteristics that differ systematically",
    "between large and small firms far from the threshold.",
    "",
    "*Requires `employee_50to499` and `turnover_2m_50m` in the parquet.",
    "Re-run `preprocess.py` if missing.*",
])

code = code_cell([
    "# 6.3c  Bandwidth sample around the CSRD threshold",
    "if 'employee_50to499' not in df.columns or 'turnover_2m_50m' not in df.columns:",
    "    raise RuntimeError('Parquet missing bandwidth columns. Run: python preprocess.py')",
    "",
    "# ── Build bandwidth sub-samples ─────────────────────────────────────────",
    "bw_mask_s = (df_s['employee_50to499'] == 1) & (df_s['turnover_2m_50m'] == 1)",
    "bw_mask_d = (df_d['employee_50to499'] == 1) & (df_d['turnover_2m_50m'] == 1)",
    "df_s_bw = df_s[bw_mask_s].copy()",
    "df_d_bw = df_d[bw_mask_d].copy()",
    "",
    "print('Bandwidth sample (employees 50-499, revenue 2M-50M)')",
    "print('  impl=1 (df_s_bw): n=%-5d  treat=1: %d (%.1f%%)  treat=0: %d (%.1f%%)' % (",
    "      len(df_s_bw),",
    "      (df_s_bw['treat']==1).sum(), 100*(df_s_bw['treat']==1).mean(),",
    "      (df_s_bw['treat']==0).sum(), 100*(df_s_bw['treat']==0).mean()))",
    "print('  all ctry (df_d_bw): n=%-5d  treat=1: %d (%.1f%%)  treat=0: %d (%.1f%%)' % (",
    "      len(df_d_bw),",
    "      (df_d_bw['treat']==1).sum(), 100*(df_d_bw['treat']==1).mean(),",
    "      (df_d_bw['treat']==0).sum(), 100*(df_d_bw['treat']==0).mean()))",
    "print()",
    "",
    "# ── OLS / DiD / DDD on bandwidth sample, both outcomes ──────────────────",
    "outcomes_bw = [",
    "    ('target_engaged',    'Climate target', s1a, s2a, s3b, 'treat', 'treat:post', 'treat:post:impl'),",
    "    ('firm_growth_ord_d', 'Firm growth',    s1b, s2b, s3d, 'treat', 'treat:post', 'treat:post:impl'),",
    "]",
    "",
    "print('%-16s %-14s %11s %8s %8s   %11s %8s %8s' % (",
    "      'Outcome', 'Spec', 'BW coef', 'SE', 'p', 'Full coef', 'SE', 'p'))",
    "print('-' * 88)",
    "for (col, lbl, m_ols_f, m_did_f, m_ddd_f, k_ols, k_did, k_ddd) in outcomes_bw:",
    "    m_bw_ols = smf.ols('%s ~ treat + C(nace_b) + C(firm_age_ord)' % col,",
    "                        data=df_s_bw).fit(",
    "                            cov_type='cluster', cov_kwds={'groups': df_s_bw['ipscntry']})",
    "    m_bw_did = smf.ols('%s ~ treat*post + C(nace_b) + C(firm_age_ord)' % col,",
    "                        data=df_s_bw).fit(",
    "                            cov_type='cluster', cov_kwds={'groups': df_s_bw['ipscntry']})",
    "    m_bw_ddd = smf.ols('%s ~ treat*post*impl + C(nace_b) + C(firm_age_ord)' % col,",
    "                        data=df_d_bw).fit(",
    "                            cov_type='cluster', cov_kwds={'groups': df_d_bw['ipscntry']})",
    "    for spec, mbw, mf, key in [('OLS', m_bw_ols, m_ols_f, k_ols),",
    "                                ('DiD', m_bw_did, m_did_f, k_did),",
    "                                ('DDD', m_bw_ddd, m_ddd_f, k_ddd)]:",
    "        c_bw = mbw.params[key];  s_bw = mbw.bse[key];  p_bw = mbw.pvalues[key]",
    "        c_f  = mf.params[key];   s_f  = mf.bse[key];   p_f  = mf.pvalues[key]",
    "        sg   = '***' if p_bw<0.01 else '**' if p_bw<0.05 else '*' if p_bw<0.10 else ''",
    "        out_lbl = lbl if spec == 'OLS' else ''",
    "        print('%-16s %-14s %+11.4f %8.4f %6.3f%-3s  %+11.4f %8.4f %6.3f' % (",
    "              out_lbl, spec, c_bw, s_bw, p_bw, sg, c_f, s_f, p_f))",
    "    print()",
    "print('BW = bandwidth sample (n_s=%d, n_d=%d). Full = full sample.' % (",
    "      len(df_s_bw), len(df_d_bw)))",
    "print('SE clustered by ipscntry in both.')",
])

# Insert after cell 44
cells.insert(45, md)
cells.insert(46, code)

print(f"Total cells: {len(cells)}")

full = "\n".join("".join(c["source"]) for c in cells)
assert "employee_50to499"  in full
assert "turnover_2m_50m"   in full
assert "df_s_bw"           in full
assert "df_d_bw"           in full
assert "6.4"               in "".join(cells[47]["source"]), "6.4 header should now be at index 47"
print("All assertions passed.")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Written.")
