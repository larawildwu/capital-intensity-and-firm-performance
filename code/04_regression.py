"""
04_regression.py
----------------
Panel regressions for the Capital Intensity research design.

Research design
---------------
Y: RoA = ib / at
X: Capital intensity = ppent / at
Moderator: Firm size = log(at)
Interaction: capital_intensity x ln_at

Input:
    data/processed/panel_with_vars.parquet

Output:
    output/tables/regression_results.csv

Models
------
(1) Pooled OLS
(2) Two-way fixed effects: firm FE + year FE
(3) Two-way fixed effects with interaction: Capital intensity x Firm size

Notes
-----
Do not use capx_intensity as a control here.
Capital intensity is the main independent variable, so capx_intensity would overlap conceptually.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects

warnings.filterwarnings("ignore")


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/panel_with_vars.parquet")
TABLE_PATH = Path("output/tables")
TABLE_PATH.mkdir(parents=True, exist_ok=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def require_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required variables are missing."""
    missing = [c for c in columns if c not in dataframe.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Available columns include: {list(dataframe.columns[:80])}"
        )


def get_se(res, var: str) -> float:
    """Return standard errors for statsmodels and linearmodels results."""
    if hasattr(res, "std_errors"):
        return res.std_errors[var]
    return res.bse[var]


def stars(p: float) -> str:
    """Return significance stars."""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)

required_cols = [
    "gvkey",
    "fyear",
    "roa",
    "capital_intensity",
    "ln_at",
    "capint_x_size",
    "leverage",
    "cash_ratio",
    "sales_growth",
]
require_columns(df, required_cols)

print(f"Loaded: {len(df):,} obs | {df['gvkey'].nunique():,} firms")
print(f"Years: {df['fyear'].min()} - {df['fyear'].max()}")


# ── Define variables ──────────────────────────────────────────────────────────
DV = "roa"
X_MAIN = "capital_intensity"
MODERATOR = "ln_at"
INTERACT = "capint_x_size"

CONTROLS = [
    "ln_at",
    "leverage",
    "cash_ratio",
    "sales_growth",
]

# Recompute interaction to be safe
df[INTERACT] = df[X_MAIN] * df[MODERATOR]

# Drop rows missing regression variables
reg_vars = [DV, X_MAIN, INTERACT] + CONTROLS
df_reg = df.dropna(subset=reg_vars).copy()

print(f"Regression sample: {len(df_reg):,} obs | {df_reg['gvkey'].nunique():,} firms")

# Panel index required by linearmodels
df_panel = df_reg.set_index(["gvkey", "fyear"])
print("Panel index set: gvkey x fyear")


# ── Model 1: Pooled OLS ───────────────────────────────────────────────────────
print("\n=== Model 1: Pooled OLS ===")

formula_ols = f"{DV} ~ {X_MAIN} + {' + '.join(CONTROLS)}"
print(f"Formula: {formula_ols}")

res1 = smf.ols(formula_ols, data=df_reg).fit(cov_type="HC3")

print(res1.summary().tables[1])
print(f"R² = {res1.rsquared:.3f} | N = {int(res1.nobs):,}")


# ── Model 2: Two-Way Fixed Effects ────────────────────────────────────────────
print("\n=== Model 2: Two-Way Fixed Effects ===")

formula_fe = (
    f"{DV} ~ {X_MAIN} + {' + '.join(CONTROLS)} "
    f"+ EntityEffects + TimeEffects"
)
print(f"Formula: {formula_fe}")

res2 = PanelOLS.from_formula(
    formula_fe,
    data=df_panel,
    drop_absorbed=True,
).fit(
    cov_type="clustered",
    cluster_entity=True,
)

print(res2.summary.tables[1])
print(f"R² within = {res2.rsquared:.3f} | N = {int(res2.nobs):,}")


# ── Model 3: Two-Way Fixed Effects + Interaction ──────────────────────────────
print("\n=== Model 3: TWFE + Interaction ===")

formula_int = (
    f"{DV} ~ {X_MAIN} + {INTERACT} + {' + '.join(CONTROLS)} "
    f"+ EntityEffects + TimeEffects"
)
print(f"Formula: {formula_int}")

res3 = PanelOLS.from_formula(
    formula_int,
    data=df_panel,
    drop_absorbed=True,
).fit(
    cov_type="clustered",
    cluster_entity=True,
)

print(res3.summary.tables[1])
print(f"R² within = {res3.rsquared:.3f} | N = {int(res3.nobs):,}")


# ── Random Effects comparison ─────────────────────────────────────────────────
print("\n=== Random Effects comparison ===")

formula_re = f"{DV} ~ {X_MAIN} + {' + '.join(CONTROLS)}"

res_re = RandomEffects.from_formula(
    formula_re,
    data=df_panel,
).fit()

fe_coef = res2.params[X_MAIN]
re_coef = res_re.params[X_MAIN]
diff = abs(fe_coef - re_coef)

print(f"Coefficient on {X_MAIN}:")
print(f"  FE:   {fe_coef:.4f} (p = {res2.pvalues[X_MAIN]:.3f})")
print(f"  RE:   {re_coef:.4f} (p = {res_re.pvalues[X_MAIN]:.3f})")
print(f"  Diff: {diff:.4f}")

if diff > 0.005:
    print("  → Non-trivial difference → prefer FE.")
else:
    print("  → Small difference → RE may be efficient, but FE remains acceptable.")


# ── Results table ─────────────────────────────────────────────────────────────
print("\n=== Side-by-side results table ===")

all_vars = [X_MAIN, INTERACT] + CONTROLS
models = [res1, res2, res3]
labels = ["(1) OLS", "(2) TWFE", "(3) TWFE+H2"]

rows = []

for var in all_vars:
    row = {"Variable": var}

    for label, res in zip(labels, models):
        if var in res.params.index:
            b = res.params[var]
            se = get_se(res, var)
            p = res.pvalues[var]
            row[label] = f"{b:.4f}{stars(p)}\n({se:.4f})"
        else:
            row[label] = ""

    rows.append(row)

results = pd.DataFrame(rows).set_index("Variable")

results.loc["Firm FE"] = ["No", "Yes", "Yes"]
results.loc["Year FE"] = ["No", "Yes", "Yes"]
results.loc["Clustered SE"] = ["No", "Yes", "Yes"]
results.loc["N"] = [
    f"{int(res1.nobs):,}",
    f"{int(res2.nobs):,}",
    f"{int(res3.nobs):,}",
]
results.loc["R²"] = [
    f"{res1.rsquared:.3f}",
    f"{res2.rsquared:.3f}",
    f"{res3.rsquared:.3f}",
]

print(results[labels].to_string())
print("\n* p<0.10  ** p<0.05  *** p<0.01")
print("SEs in parentheses. Models (2)-(3): clustered at firm level.")

results.to_csv(TABLE_PATH / "regression_results.csv")
print("\nSaved output/tables/regression_results.csv")


# ── Interpretation diagnostics ────────────────────────────────────────────────
print("\n=== H1: Capital intensity -> RoA ===")

b_x = res2.params[X_MAIN]
p_x = res2.pvalues[X_MAIN]

print(f"β(capital_intensity) = {b_x:.4f}{stars(p_x)}  (p = {p_x:.3f})")

if p_x < 0.10:
    if b_x > 0:
        print("H1 result: positive and statistically significant.")
        print("Interpretation: More capital-intensive SMEs show higher RoA within firms over time.")
    else:
        print("H1 result: negative and statistically significant.")
        print("Interpretation: More capital-intensive SMEs show lower RoA within firms over time.")
        print("Possible explanation: higher fixed costs, depreciation burden, or lower flexibility.")
else:
    print("H1 result: not statistically significant.")
    print("Interpretation: No clear within-firm relationship between capital intensity and RoA.")


print("\n=== H2: Firm size moderates Capital intensity -> RoA ===")

b_int = res3.params.get(INTERACT, np.nan)
p_int = res3.pvalues.get(INTERACT, 1.0)

print(f"β(capint_x_size) = {b_int:.4f}{stars(p_int)}  (p = {p_int:.3f})")

if p_int < 0.10:
    if b_int > 0:
        print("H2 result: positive and statistically significant moderation.")
        print("Interpretation: Larger SMEs benefit more from capital intensity.")
    else:
        print("H2 result: negative and statistically significant moderation.")
        print("Interpretation: The capital intensity effect is weaker or more negative for larger SMEs.")
else:
    print("H2 result: interaction not statistically significant.")
    print("Interpretation: No clear evidence that firm size moderates the relationship.")


print("\n=== OLS vs TWFE comparison ===")

ols_b = res1.params[X_MAIN]
fe_b = res2.params[X_MAIN]

pct_diff = abs((ols_b - fe_b) / ols_b) * 100 if ols_b != 0 else np.nan

print(f"OLS β = {ols_b:.4f}")
print(f"FE β  = {fe_b:.4f}")
print(f"Difference = {pct_diff:.1f}%")
print(f"R² OLS = {res1.rsquared:.3f}")
print(f"R² within FE = {res2.rsquared:.3f}")

if pct_diff > 20:
    print("Large OLS-FE difference → likely omitted variable bias in OLS.")
else:
    print("OLS-FE difference is moderate or small.")


print(
    """
─────────────────────────────────────────────────────────────
Interpretation guide:
  Main result: Model (2), Two-Way Fixed Effects
  H1: coefficient on capital_intensity
  H2: coefficient on capint_x_size
  Stars: *** p<0.01, ** p<0.05, * p<0.10
  SEs in parentheses
  Models (2)-(3): firm FE + year FE, clustered at firm level
─────────────────────────────────────────────────────────────
"""
)