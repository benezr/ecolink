"""
_fix8.py — append Section 6 (Extension: resource-efficiency domain outcomes).
  Runs OLS / DiD / DDD for each of the 10 *_maturity_d domain outcomes.
  Also updates the title-cell outline.
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
assert "5.5" in "".join(cells[37]["source"]), "Cell 37 should be §5.5 east-west code"

# ─── Update title-cell outline (cell 0) ──────────────────────────────────────
cells[0]["source"] = src([
    "# Comparative Specifications — DiD Analysis",
    "",
    "**Standalone notebook** — loads pre-processed data from `analysis_data.parquet`.",
    "Run `preprocess.py` once to generate the parquet from the two Excel source files.",
    "",
    "## Outline",
    "| Section | Specification | Key coefficient |",
    "|---|---|---|",
    "| 0 | Data loading & sample construction | — |",
    "| 1 | Simple OLS | `treat` |",
    "| 2 | Difference-in-Differences | `treat:post` |",
    "| 3 | Triple Difference (DDD) | `treat:post:impl` |",
    "| 4 | Overview table | all specs side-by-side |",
    "| 5 | Robustness checks | various |",
    "| 6 | Extension: resource-efficiency domains | `treat` / `treat:post` / `treat:post:impl` |",
])

# ─── Section 6 header ────────────────────────────────────────────────────────
header = md_cell([
    "---",
    "## 6. Extension — Resource-Efficiency Domain Outcomes",
    "",
    "The main analysis uses `target_engaged` and `firm_growth_ord_d` as outcomes.",
    "Here we re-run the three core specifications — **OLS**, **DiD**, and **DDD** —",
    "using each of the ten resource-efficiency *action-maturity* indicators as the",
    "outcome variable.  Each `{domain}_maturity_d` equals 1 if the firm currently",
    "undertakes the action **or** plans to (built from the `q1`/`q2` pairs in",
    "`preprocess.py`), and 0 otherwise.",
    "",
    "| Domain | Variable |",
    "|---|---|",
    "| Water efficiency        | `water_maturity_d` |",
    "| Energy saving           | `energy_saving_maturity_d` |",
    "| Green energy            | `green_energy_maturity_d` |",
    "| Materials efficiency    | `materials_maturity_d` |",
    "| Green suppliers         | `green_suppliers_maturity_d` |",
    "| Waste reduction         | `waste_reduction_maturity_d` |",
    "| Selling residues        | `sell_residues_maturity_d` |",
    "| Recycling               | `recycling_maturity_d` |",
    "| Eco-design              | `eco_design_maturity_d` |",
    "| Other                   | `other_maturity_d` |",
    "",
    "All outcomes are binary, so OLS is a linear probability model (consistent with",
    "Sections 1–3).  Controls: sector FEs + firm-age FEs.  SE clustered by country.",
])

# ─── Section 6 code ──────────────────────────────────────────────────────────
code = code_cell([
    "# 6.  OLS / DiD / DDD for the resource-efficiency domain outcomes",
    "domain_outcomes = [",
    "    ('water_maturity_d',           'Water efficiency'),",
    "    ('energy_saving_maturity_d',   'Energy saving'),",
    "    ('green_energy_maturity_d',    'Green energy'),",
    "    ('materials_maturity_d',       'Materials efficiency'),",
    "    ('green_suppliers_maturity_d', 'Green suppliers'),",
    "    ('waste_reduction_maturity_d', 'Waste reduction'),",
    "    ('sell_residues_maturity_d',   'Selling residues'),",
    "    ('recycling_maturity_d',       'Recycling'),",
    "    ('eco_design_maturity_d',      'Eco-design'),",
    "    ('other_maturity_d',           'Other'),",
    "]",
    "",
    "def _st(p):",
    "    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''",
    "",
    "print('%-22s %9s %7s %7s   %9s %7s %7s   %9s %7s %7s' % (",
    "      'Domain', 'OLS', 'SE', 'p', 'DiD', 'SE', 'p', 'DDD', 'SE', 'p'))",
    "print('%-22s %9s %7s %7s   %9s %7s %7s   %9s %7s %7s' % (",
    "      '', 'treat', '', '', 'tr:post', '', '', 'tr:po:impl', '', ''))",
    "print('-' * 96)",
    "",
    "domain_rows = []",
    "for col, label in domain_outcomes:",
    "    sub_s = df_s.dropna(subset=[col]).copy()",
    "    sub_d = df_d.dropna(subset=[col]).copy()",
    "    m_ols = smf.ols('%s ~ treat + C(nace_b) + C(firm_age_ord)' % col,",
    "                    data=sub_s).fit(cov_type='cluster', cov_kwds={'groups': sub_s['ipscntry']})",
    "    m_did = smf.ols('%s ~ treat*post + C(nace_b) + C(firm_age_ord)' % col,",
    "                    data=sub_s).fit(cov_type='cluster', cov_kwds={'groups': sub_s['ipscntry']})",
    "    m_ddd = smf.ols('%s ~ treat*post*impl + C(nace_b) + C(firm_age_ord)' % col,",
    "                    data=sub_d).fit(cov_type='cluster', cov_kwds={'groups': sub_d['ipscntry']})",
    "    co, so, po = m_ols.params['treat'],            m_ols.bse['treat'],            m_ols.pvalues['treat']",
    "    cd, sd, pd_ = m_did.params['treat:post'],       m_did.bse['treat:post'],       m_did.pvalues['treat:post']",
    "    c3, s3, p3 = m_ddd.params['treat:post:impl'],   m_ddd.bse['treat:post:impl'],  m_ddd.pvalues['treat:post:impl']",
    "    print('%-22s %+9.4f %7.4f %5.3f%-2s %+9.4f %7.4f %5.3f%-2s %+9.4f %7.4f %5.3f%-2s' % (",
    "          label, co, so, po, _st(po), cd, sd, pd_, _st(pd_), c3, s3, p3, _st(p3)))",
    "    domain_rows.append({",
    "        'domain': col, 'label': label,",
    "        'ols_treat': co, 'ols_se': so, 'ols_p': po,",
    "        'did_treatpost': cd, 'did_se': sd, 'did_p': pd_,",
    "        'ddd_treatpostimpl': c3, 'ddd_se': s3, 'ddd_p': p3,",
    "    })",
    "",
    "print('-' * 96)",
    "print('OLS/DiD on impl=1 sample (df_s); DDD on full sample (df_d).')",
    "print('Controls: sector FEs + firm-age FEs. SE clustered by ipscntry.')",
    "print('Stars: *** p<0.01, ** p<0.05, * p<0.10.')",
    "",
    "domain_table = pd.DataFrame(domain_rows)",
    "domain_table.to_csv('domain_results.csv', index=False)",
    "print('\\nSaved: domain_results.csv')",
])

cells.extend([header, code])
print(f"Total cells: {len(cells)}")

full = "\n".join("".join(c["source"]) for c in cells)
assert "## 6. Extension"   in full
assert "domain_outcomes"   in full
assert "domain_results.csv" in full
assert "treat:post:impl"   in full
print("All assertions passed.")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Written.")
