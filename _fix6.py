"""
_fix6.py — remove Section 4 (DR DDD) entirely from comparative_specifications.ipynb
  1. Delete cells 20-32 (13 cells: DR DDD header through save)
  2. Patch Overview Table (was §5 → §4): strip DR DDD rows from markdown + code
  3. Renumber §5 Overview → §4, §6 Robustness → §5, all 6.x subsections → 5.x
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
assert len(cells) == 53, f"Expected 53 cells, got {len(cells)}"
assert "Doubly-Robust" in "".join(cells[20]["source"]), "Cell 20 should be DR DDD header"
assert "dr_results"    in "".join(cells[32]["source"]), "Cell 32 should be DR DDD save"
assert "## 5. Overview" in "".join(cells[33]["source"])
assert "## 6. Robustness" in "".join(cells[36]["source"])

# ─── 0. Update title cell outline (cell 0) ───────────────────────────────────
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
])

# ─── 1. Delete cells 20-32 ────────────────────────────────────────────────────
del cells[20:33]   # removes 13 cells; cell 33 becomes cell 20, etc.

# After deletion: orig33→20, orig34→21, orig35→22, orig36→23 … orig52→39

# ─── 2. Overview Table markdown (new cell 20) ─────────────────────────────────
cells[20]["source"] = src([
    "---",
    "## 4. Overview Table",
    "",
    "All parametric specifications (Sections 1–3) side-by-side.",
    "Standard errors clustered by country (`ipscntry`).",
    "Significance: \\* p<0.1, \\*\\* p<0.05, \\*\\*\\* p<0.01.",
    "",
    "| Section | Models | Key coefficient |",
    "|---|---|---|",
    "| 1 OLS  | 1a–1d | `treat` |",
    "| 2 DiD  | 2a–2d | `treat:post` |",
    "| 3 DDD  | 3a–3d | `treat:post:impl` |",
])

# ─── 3. Overview Table code (new cell 21): drop DR DDD block ─────────────────
cells[21]["source"] = src([
    "from statsmodels.iolib.summary2 import summary_col",
    "",
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
])

# ─── 4. Overview Table save (new cell 22): drop DR DDD block ─────────────────
cells[22]["source"] = src([
    "with open('overview_table.txt', 'w', encoding='utf-8') as f:",
    "    f.write(str(comp_table))",
    "print('Saved: overview_table.txt')",
])

# ─── 5. Robustness header (new cell 23): §6 → §5 ─────────────────────────────
cells[23]["source"] = src([
    "---",
    "## 5. Robustness Checks",
    "",
    "| Subsection | Check |",
    "|---|---|",
    "| 5.1 | Logistic regression for binary growth outcome |",
    "| 5.2 | Repeated cross-section composition checks |",
    "| 5.3 | Placebo treatment definitions |",
    "| 5.4 | DDD first stage and covariate overlap |",
    "| 5.5 | DiD with alternative outcome variables |",
    "| 5.6 | OLS / DiD / DDD with East–West control |",
])

# ─── 6. Renumber all 6.x → 5.x in remaining markdown cells ──────────────────
for i in range(24, len(cells)):
    cell = cells[i]
    s = "".join(cell["source"])
    updated = s
    for sub in ["6.1","6.2","6.3","6.4","6.5","6.6"]:
        updated = updated.replace("### " + sub, "### 5." + sub[2:])
        updated = updated.replace("# 6." + sub[2:], "# 5." + sub[2:])
    if updated != s:
        cell["source"] = updated.splitlines(keepends=True)

# ─── Assertions ───────────────────────────────────────────────────────────────
full = "\n".join("".join(c["source"]) for c in cells)
assert "dr_results"      not in full, "dr_results still present!"
assert "OUTCOMES_DR"     not in full, "OUTCOMES_DR still present!"
assert "dr_did_rc"       not in full, "dr_did_rc still present!"
assert "Doubly-Robust"   not in full, "Doubly-Robust still present!"
assert "## 4. Overview"  in full,     "Section 4 Overview missing!"
assert "## 5. Robustness" in full,    "Section 5 Robustness missing!"
assert "### 5.1"          in full,    "5.1 header missing!"
assert "### 5.6"          in full,    "5.6 header missing!"
assert "## 6."            not in full, "Old §6 header still present!"

print(f"Total cells: {len(cells)}")
print("All assertions passed.")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Written.")
