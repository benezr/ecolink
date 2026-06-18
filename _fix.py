import json
from pathlib import Path

NB = Path(r"C:\Users\zolb\Files C\ClaudeFolder\EcoLink\comparative_specifications.ipynb")

def src(lines):
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Verify we're patching the right cell
assert "scr12" in "".join(nb["cells"][13]["source"]), "Expected scr12 in cell 13"

nb["cells"][13]["source"] = src([
    "# Main sample: implementing countries (impl=1), complete cases",
    "df_main  = df[df['sample_main'] == 1]",
    "df_clean = df_main.dropna(subset=[",
    '    "target_engaged", "treat", "post", "firm_age_ord", "nace_b", "ipscntry"',
    "]).copy()",
    "",
    "# DDD sample: all countries (impl=0 + impl=1)",
    "df_ddd = df.dropna(subset=[",
    '    "target_engaged", "treat", "post", "firm_age_ord", "nace_b",',
    '    "ipscntry", "implementation_country"',
    "]).copy()",
    'df_ddd["impl"] = df_ddd["implementation_country"]',
    "",
    "# Working samples (drop residual NAs on firm_growth_ord_d)",
    "df_s = df_clean.dropna(subset=['firm_growth_ord_d']).copy()  # OLS / DiD",
    "df_d = df_ddd.dropna(subset=['firm_growth_ord_d']).copy()    # DDD",
    "",
    "# Extended FE dummies for Section 4",
    'df_d["cntry_treat"] = df_d["ipscntry"].astype(str) + "_" + df_d["treat"].astype(str)',
    'df_d["cntry_post"]  = df_d["ipscntry"].astype(str) + "_" + df_d["post"].astype(str)',
    "",
    'print(f"OLS/DiD sample (impl=1 only): {df_s.shape[0]:,} obs")',
    'print(f"DDD sample (all countries):   {df_d.shape[0]:,} obs")',
    "print(f\"  impl=1 (policy countries):  {(df_d['impl']==1).sum():,}\")",
    "print(f\"  impl=0 (never-treated):     {(df_d['impl']==0).sum():,}\")",
    "",
    "def cl(df_):",
    '    """Clustered SE by country."""',
    '    return {"cov_type": "cluster", "cov_kwds": {"groups": df_["ipscntry"]}}',
])

assert "scr12" not in "".join(nb["cells"][13]["source"]), "scr12 still present!"
assert "firm_age_ord" in "".join(nb["cells"][13]["source"]), "firm_age_ord missing!"

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Fixed. Cell 13 now uses firm_age_ord.")
