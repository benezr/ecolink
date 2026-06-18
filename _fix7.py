"""
_fix7.py:
  1. Delete cells 34-35 (§5.4 DDD first stage / overlap)
  2. Update §5.5 header + code → §5.4, extend to also run DDD per outcome
  3. Renumber §5.6 → §5.5
  4. Update §5 robustness overview table
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\zolb\Files C\ClaudeFolder\EcoLink\comparative_specifications.ipynb")

def src(lines):
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
assert len(cells) == 40, f"Expected 40 cells, got {len(cells)}"
assert "5.4" in "".join(cells[34]["source"]), "Cell 34 should be §5.4 header"
assert "5.4" in "".join(cells[35]["source"]), "Cell 35 should be §5.4 code"
assert "5.5" in "".join(cells[36]["source"]), "Cell 36 should be §5.5 header"
assert "5.5" in "".join(cells[37]["source"]), "Cell 37 should be §5.5 code"
assert "5.6" in "".join(cells[38]["source"]), "Cell 38 should be §5.6 header"

# ─── 1. Delete §5.4 cells ────────────────────────────────────────────────────
del cells[34:36]   # cells 34 and 35 gone; cell 36 → 34, 37 → 35, 38 → 36, 39 → 37

# ─── 2. §5.4 header (was §5.5, now cell 34) ──────────────────────────────────
cells[34]["source"] = src([
    "### 5.4  DiD and DDD with Alternative Outcome Variables",
    "",
    "Running DiD **and** DDD on pre-determined firm characteristics (`firm_age_ord`)",
    "serves as a **falsification test**: a significant coefficient would indicate",
    "compositional shifts rather than genuine treatment effects.",
    "For forward-looking variables (`green_offer_ord_d`, `green_staff_any`, etc.) the",
    "estimates capture potential policy spillovers or mechanisms.",
])

# ─── 3. §5.4 code (was §5.5, now cell 35) — add DDD column ───────────────────
cells[35]["source"] = src([
    "# 5.4  DiD and DDD on alternative outcomes",
    "alt_out = [",
    "    ('firm_age_ord',            'Firm age ordinal [falsification]'),",
    "    ('green_staff_any',         'Green staff (any)'),",
    "    ('investment_dummy',        'Investment dummy'),",
    "    ('barrier_institutional_d', 'Institutional barriers'),",
    "    ('support_knowledge_d',     'Knowledge support'),",
    "    ('green_offer_ord_d',       'Green offer (binary)'),",
    "]",
    "",
    "print('%-40s %10s %8s %8s   %10s %8s %8s' % (",
    "      'Outcome', 'DiD coef', 'SE', 'p', 'DDD coef', 'SE', 'p'))",
    "print('-' * 90)",
    "for col, label in alt_out:",
    "    # DiD (impl=1 sample)",
    "    sub_s = df_s.dropna(subset=[col]).copy()",
    "    rhs_s = ('treat*post + C(nace_b)' if col == 'firm_age_ord'",
    "             else 'treat*post + C(nace_b) + C(firm_age_ord)')",
    "    m_did = smf.ols('%s ~ %s' % (col, rhs_s), data=sub_s).fit(",
    "                cov_type='cluster', cov_kwds={'groups': sub_s['ipscntry']})",
    "    # DDD (all countries)",
    "    sub_d = df_d.dropna(subset=[col]).copy()",
    "    rhs_d = ('treat*post*impl + C(nace_b)' if col == 'firm_age_ord'",
    "             else 'treat*post*impl + C(nace_b) + C(firm_age_ord)')",
    "    m_ddd = smf.ols('%s ~ %s' % (col, rhs_d), data=sub_d).fit(",
    "                cov_type='cluster', cov_kwds={'groups': sub_d['ipscntry']})",
    "    cd  = m_did.params.get('treat:post',      float('nan'))",
    "    sd  = m_did.bse.get('treat:post',         float('nan'))",
    "    pd_ = m_did.pvalues.get('treat:post',     float('nan'))",
    "    c3  = m_ddd.params.get('treat:post:impl', float('nan'))",
    "    s3  = m_ddd.bse.get('treat:post:impl',    float('nan'))",
    "    p3  = m_ddd.pvalues.get('treat:post:impl',float('nan'))",
    "    sgd = '***' if pd_<0.01 else '**' if pd_<0.05 else '*' if pd_<0.10 else ''",
    "    sg3 = '***' if p3<0.01  else '**' if p3<0.05  else '*' if p3<0.10  else ''",
    "    print('%-40s %+10.4f %8.4f %6.3f%-3s  %+10.4f %8.4f %6.3f%s' % (",
    "          label, cd, sd, pd_, sgd, c3, s3, p3, sg3))",
    "print()",
    "print('DiD: treat*post + sector FEs + firm_age FE (omitted when outcome=firm_age_ord).')",
    "print('DDD: treat*post*impl + same FEs. SE clustered by ipscntry.')",
])

# ─── 4. Renumber §5.6 → §5.5 in header (cell 36) and code comment (cell 37) ──
for idx in [36, 37]:
    s = "".join(cells[idx]["source"])
    cells[idx]["source"] = s.replace("5.6", "5.5").replace("# 5.6", "# 5.5").splitlines(keepends=True)

# ─── 5. Update §5 robustness overview table (cell 23) ────────────────────────
cells[23]["source"] = src([
    "---",
    "## 5. Robustness Checks",
    "",
    "| Subsection | Check |",
    "|---|---|",
    "| 5.1 | Logistic regression for binary growth outcome |",
    "| 5.2 | Repeated cross-section composition checks |",
    "| 5.3 | Placebo treatment definitions |",
    "| 5.4 | DiD and DDD with alternative outcome variables |",
    "| 5.5 | OLS / DiD / DDD with East\\u2013West control |",
])

# ─── Assertions ───────────────────────────────────────────────────────────────
full = "\n".join("".join(c["source"]) for c in cells)
assert "ddd_overlap"        not in full, "ddd_overlap still present!"
assert "mticker"            not in full, "overlap plot code still present!"
assert "treat*post*impl"    in full,     "DDD in alt outcomes missing!"
assert "### 5.4"            in full,     "§5.4 header missing!"
assert "### 5.5"            in full,     "§5.5 header missing!"
assert "### 5.6"            not in full, "Old §5.6 still present!"
print(f"Total cells: {len(cells)}")
print("All assertions passed.")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Written.")
