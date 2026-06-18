"""
_fix4.py — insert two below-threshold placebo cells after existing 6.3 code (cell 42)
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
assert len(cells) == 49, f"Expected 49 cells, got {len(cells)}"
# Confirm cell 42 is the existing 6.3 code cell
assert "treat_emp" in "".join(cells[42]["source"]), "Cell 42 should be the 6.3 placebo code"
# Confirm cell 43 is the 6.4 header
assert "6.4" in "".join(cells[43]["source"]), "Cell 43 should be the 6.4 header"

insert_after = 42   # insert at index 43 and 44

md = md_cell([
    "#### Below-threshold placebo treatments",
    "",
    "These placebos test whether the DiD/DDD effect is **specific to firms at the CSRD",
    "threshold** rather than a general medium-firm trend.",
    "",
    "**Placebo A** assigns treatment to firms with \\u2265 50 employees **AND** turnover",
    "> \\u20ac250k (below the CSRD thresholds of \\u2265 250 employees / \\u2265 \\u20ac10M)",
    "in the full analysis sample.",
    "",
    "**Placebo B** applies the same placebo treatment but restricts the sample to firms",
    "that do **not** qualify under the original CSRD criterion (`treat = 0`),",
    "isolating medium firms entirely below the reporting threshold.",
    "",
    "A null result in both placebos supports the interpretation that the main estimates",
    "reflect genuine CSRD-threshold effects rather than a broader size trend.",
    "",
    "*Requires `employee_50plus` and `turnover_250k` in the parquet.",
    "Re-run `preprocess.py` if these columns are missing.*",
])

code = code_cell([
    "# 6.3b  Below-threshold placebo treatments",
    "if 'employee_50plus' not in df.columns or 'turnover_250k' not in df.columns:",
    "    raise RuntimeError('Parquet missing new columns. Run: python preprocess.py')",
    "",
    "# ── Placebo treatment: employees >= 50 AND turnover > 250k ──────────────",
    "df_s_pb = df_s.copy()",
    "df_d_pb = df_d.copy()",
    "for df_ in [df_s_pb, df_d_pb]:",
    "    valid = df_['employee_50plus'].notna() & df_['turnover_250k'].notna()",
    "    df_['treat_pb'] = np.where(",
    "        valid,",
    "        ((df_['employee_50plus'] == 1) & (df_['turnover_250k'] == 1)).astype(float),",
    "        np.nan,",
    "    )",
    "",
    "df_s_pb = df_s_pb.dropna(subset=['treat_pb'])",
    "df_d_pb = df_d_pb.dropna(subset=['treat_pb'])",
    "",
    "# ── Placebo B: restrict to firms below CSRD threshold (treat_original = 0) ──",
    "df_s_pb0 = df_s_pb[df_s_pb['treat'] == 0].copy()",
    "df_d_pb0 = df_d_pb[df_d_pb['treat'] == 0].copy()",
    "",
    "# Group sizes",
    "for lbl, ds in [('Placebo A (full sample)', df_s_pb),",
    "                ('Placebo B (treat=0 only)', df_s_pb0)]:",
    "    n   = len(ds)",
    "    n1  = int(ds['treat_pb'].sum())",
    "    pct = 100 * n1 / n",
    "    print('%s:  n=%d,  treat_pb=1: %d (%.1f%%)' % (lbl, n, n1, pct))",
    "print()",
    "",
    "# ── DiD and DDD for both placebos ─────────────────────────────────────────",
    "print('%-33s %10s %8s %8s   %10s %8s %8s' % (",
    "      'Placebo', 'DiD coef', 'SE', 'p', 'DDD coef', 'SE', 'p'))",
    "print('-' * 85)",
    "for label, ds, dd in [",
    "    ('A: full (>=50 emp, >250k rev)',   df_s_pb,  df_d_pb),",
    "    ('B: excl. CSRD treated (treat=0)', df_s_pb0, df_d_pb0),",
    "]:",
    "    md_ = smf.ols(",
    "        'target_engaged ~ treat_pb*post + C(nace_b) + C(firm_age_ord)',",
    "        data=ds).fit(cov_type='cluster', cov_kwds={'groups': ds['ipscntry']})",
    "    m3_ = smf.ols(",
    "        'target_engaged ~ treat_pb*post*impl + C(nace_b) + C(firm_age_ord)',",
    "        data=dd).fit(cov_type='cluster', cov_kwds={'groups': dd['ipscntry']})",
    "    cd  = md_.params.get('treat_pb:post',       float('nan'))",
    "    sd_ = md_.bse.get('treat_pb:post',          float('nan'))",
    "    pd_ = md_.pvalues.get('treat_pb:post',      float('nan'))",
    "    c3  = m3_.params.get('treat_pb:post:impl',  float('nan'))",
    "    s3_ = m3_.bse.get('treat_pb:post:impl',     float('nan'))",
    "    p3_ = m3_.pvalues.get('treat_pb:post:impl', float('nan'))",
    "    sgd = '***' if pd_<0.01 else '**' if pd_<0.05 else '*' if pd_<0.10 else ''",
    "    sg3 = '***' if p3_<0.01 else '**' if p3_<0.05 else '*' if p3_<0.10 else ''",
    "    print('%-33s %+10.4f %8.4f %6.3f%-3s  %+10.4f %8.4f %6.3f%s' % (",
    "          label, cd, sd_, pd_, sgd, c3, s3_, p3_, sg3))",
    "",
    "print()",
    "print('Main spec reference  — DiD (s2a): treat:post = %.4f  p=%.3f' % (",
    "      s2a.params['treat:post'], s2a.pvalues['treat:post']))",
    "print('Main spec reference  — DDD (s3b): treat:post:impl = %.4f  p=%.3f' % (",
    "      s3b.params['treat:post:impl'], s3b.pvalues['treat:post:impl']))",
])

# Insert after cell 42
cells.insert(insert_after + 1, md)
cells.insert(insert_after + 2, code)

print(f"Total cells: {len(cells)}")

full = "\n".join("".join(c["source"]) for c in cells)
assert "treat_pb"         in full
assert "Placebo B"        in full
assert "employee_50plus"  in full
assert "turnover_250k"    in full
assert "6.4"              in "".join(cells[45]["source"]), "6.4 header should be at index 45"
print("All assertions passed.")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Written.")
