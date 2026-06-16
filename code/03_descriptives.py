"""
03_descriptives.py
------------------
Summary statistics and exploratory figures for the Capital Intensity research design.

Research design
---------------
Y: RoA = ib / at
X: Capital intensity = ppent / at
Moderator: Firm size = log(at)
Interaction: capital_intensity x ln_at

Input:
    data/processed/panel_clean.parquet

Outputs:
    output/tables/summary_statistics.csv
    output/figures/correlation_matrix.png
    output/figures/main_relationship.png
    output/figures/dv_distribution.png
    output/figures/sample_composition.png
    data/processed/panel_with_vars.parquet

Notes
-----
Do not use capx_intensity as a control here.
Capital intensity is the main independent variable, and capx_intensity is too close conceptually.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.family": "sans-serif"})

WU_BLUE = "#002f5f"
WU_RED = "#c8102e"


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/panel_clean.parquet")
PANEL_OUT = Path("data/processed/panel_with_vars.parquet")

TABLE_PATH = Path("output/tables")
FIGURE_PATH = Path("output/figures")

TABLE_PATH.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.mkdir(parents=True, exist_ok=True)
PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def require_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required variables are missing."""
    missing = [c for c in columns if c not in dataframe.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Available columns include: {list(dataframe.columns[:60])}"
        )


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a variable at given lower and upper quantiles."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} observations | {df['gvkey'].nunique():,} firms")

required_cols = [
    "gvkey",
    "fyear",
    "loc",
    "at",
    "sale",
    "emp",
    "ib",
    "ppent",
    "dltt",
    "che",
]
require_columns(df, required_cols)


# ── 1. Data quality filters + SME filter ──────────────────────────────────────
print(f"\nStarting sample: {len(df):,} rows")

df = df[(df["at"] > 0.1) & (df["sale"] > 0)].copy()
print(f"After quality filters (at>0.1, sale>0): {len(df):,}")

df = df[df["at"] >= 1].copy()
print(f"After micro-firm filter (at>=1): {len(df):,}")

sme_mask = (df["emp"] < 0.25) | (df["at"] <= 43)
df = df[sme_mask].copy()
print(f"After SME filter: {len(df):,}")

print("\nTop countries:")
print(df["loc"].value_counts().head(8).to_string())


# ── 2. Variable construction ──────────────────────────────────────────────────
df["roa"] = df["ib"] / df["at"]

df["capital_intensity"] = df["ppent"].fillna(0) / df["at"]

df["ln_at"] = np.log(df["at"])

df["leverage"] = df["dltt"].fillna(0) / df["at"]
df["cash_ratio"] = df["che"].fillna(0) / df["at"]

df = df.sort_values(["gvkey", "fyear"]).copy()
df["sales_growth"] = df.groupby("gvkey")["sale"].pct_change()
df["sales_growth"] = df["sales_growth"].replace([np.inf, -np.inf], np.nan)

df["capint_x_size"] = df["capital_intensity"] * df["ln_at"]

research_vars = [
    "roa",
    "capital_intensity",
    "ln_at",
    "capint_x_size",
    "leverage",
    "cash_ratio",
    "sales_growth",
]

print("\nVariable coverage:")
for v in research_vars:
    n = df[v].notna().sum()
    nz = (df[v].notna() & (df[v] != 0)).sum()
    pct = n / len(df) * 100
    print(f"  {v:<22} {n:>7,} ({pct:>5.1f}%) non-zero: {nz:>7,}")


# ── 3. Drop missing core variables ────────────────────────────────────────────
CORE_VARS = ["roa", "capital_intensity", "ln_at", "leverage", "cash_ratio"]

n_before = len(df)
df = df.dropna(subset=CORE_VARS).copy()

print(f"\nDropped {n_before - len(df):,} rows with missing core variables")
print(f"Working sample: {len(df):,} firm-years | {df['gvkey'].nunique():,} firms")


# ── 4. Winsorize at 1%-99% ────────────────────────────────────────────────────
roa_raw = df["roa"].copy()

WINSORIZE_VARS = [
    "roa",
    "capital_intensity",
    "leverage",
    "cash_ratio",
    "sales_growth",
]

print("\nWinsorize ranges (1%-99%):")
for col in WINSORIZE_VARS:
    df[col] = winsorize(df[col])
    print(f"  {col:<22} [{df[col].min():>8.4f}, {df[col].max():>8.4f}]")

df["capint_x_size"] = df["capital_intensity"] * df["ln_at"]


# ── 5. Minimum 3 observations per firm ────────────────────────────────────────
obs = df.groupby("gvkey")["fyear"].count()
valid_firms = obs[obs >= 3].index

n_before = len(df)
df = df[df["gvkey"].isin(valid_firms)].copy()

print(f"\nMin 3 obs per firm: {n_before:,} -> {len(df):,}")
print(f"Final sample: {len(df):,} firm-years | {df['gvkey'].nunique():,} firms")
print(f"Years: {df['fyear'].min()} - {df['fyear'].max()}")
print(f"Mean capital intensity: {df['capital_intensity'].mean():.4f}")


# ── 6. Summary Statistics ─────────────────────────────────────────────────────
VAR_LABELS = {
    "roa": "RoA (ib/at)",
    "capital_intensity": "Capital Intensity (ppent/at)",
    "ln_at": "Firm Size (log assets)",
    "leverage": "Leverage (dltt/at)",
    "cash_ratio": "Cash Ratio (che/at)",
    "sales_growth": "Sales Growth",
}

summary = (
    df[list(VAR_LABELS.keys())]
    .rename(columns=VAR_LABELS)
    .describe(percentiles=[0.25, 0.5, 0.75])
    .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    .round(3)
)

print("\n=== Summary Statistics ===")
print(summary.to_string())

summary.to_csv(TABLE_PATH / "summary_statistics.csv")
print("Saved output/tables/summary_statistics.csv")


# ── 7. Correlation Matrix ─────────────────────────────────────────────────────
corr_vars = list(VAR_LABELS.keys())
corr = df[corr_vars].rename(columns=VAR_LABELS).corr().round(2)

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.8},
)

ax.set_title("Correlation Matrix — Research Variables", fontsize=13, pad=12, color=WU_BLUE)
fig.tight_layout()
fig.savefig(FIGURE_PATH / "correlation_matrix.png", dpi=150)
plt.close()

print("Saved output/figures/correlation_matrix.png")

print("\nPairs with |r| > 0.3:")
for i, v1 in enumerate(corr_vars):
    for v2 in corr_vars[i + 1:]:
        r = corr.loc[VAR_LABELS[v1], VAR_LABELS[v2]]
        if abs(r) > 0.3:
            print(f"  {v1} x {v2}: r = {r:.2f}")


# ── 8. Main Relationship: Capital Intensity vs RoA ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

df_plot = df.copy()
df_plot.reset_index(drop=True, inplace=True)

axes[0].scatter(
    df_plot["capital_intensity"],
    df_plot["roa"],
    alpha=0.06,
    s=6,
    color=WU_BLUE,
)

bins = pd.cut(df_plot["capital_intensity"], bins=20)
bin_means = df_plot.groupby(bins, observed=True)[["capital_intensity", "roa"]].mean()

axes[0].plot(
    bin_means["capital_intensity"],
    bin_means["roa"],
    color=WU_RED,
    lw=2.5,
    label="Bin mean",
)

axes[0].axhline(0, color="gray", lw=0.8, ls="--")
axes[0].set_xlabel("Capital Intensity (ppent/at)")
axes[0].set_ylabel("RoA")
axes[0].set_title("Capital Intensity vs. RoA", color=WU_BLUE)
axes[0].legend()

try:
    df_plot["capint_group"] = pd.qcut(
        df_plot["capital_intensity"],
        q=3,
        labels=[
            "Low capital intensity",
            "Medium capital intensity",
            "High capital intensity",
        ],
        duplicates="drop",
    )
except ValueError:
    df_plot["capint_group"] = pd.qcut(
        df_plot["capital_intensity"].rank(method="first"),
        q=3,
        labels=[
            "Low capital intensity",
            "Medium capital intensity",
            "High capital intensity",
        ],
    )

df_plot["size_bin"] = pd.cut(df_plot["ln_at"], bins=10)

for label, group in df_plot.groupby("capint_group", observed=True):
    group_reset = group.reset_index(drop=True)
    bm = group_reset.groupby("size_bin", observed=True)[["ln_at", "roa"]].mean()

    axes[1].plot(
        bm["ln_at"],
        bm["roa"],
        lw=2,
        marker="o",
        markersize=5,
        label=label,
    )

axes[1].axhline(0, color="gray", lw=0.8, ls="--")
axes[1].set_xlabel("Firm Size (log assets)")
axes[1].set_ylabel("Mean RoA")
axes[1].set_title("RoA by Firm Size and Capital Intensity\nH2 preview", color=WU_BLUE)
axes[1].legend()

fig.suptitle(
    "Capital Intensity & Firm Performance — European SMEs",
    fontsize=13,
    y=1.02,
    color=WU_BLUE,
)

fig.tight_layout()
fig.savefig(FIGURE_PATH / "main_relationship.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved output/figures/main_relationship.png")


# ── 9. DV Distribution + Median RoA by Year ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(roa_raw.clip(-1, 1), bins=60, alpha=0.5, label="Before winsorizing")
axes[0].hist(df["roa"], bins=60, alpha=0.7, label="After winsorizing")
axes[0].axvline(df["roa"].mean(), color=WU_RED, lw=2, label=f"Mean = {df['roa'].mean():.3f}")
axes[0].axvline(
    df["roa"].median(),
    color="orange",
    lw=2,
    ls="--",
    label=f"Median = {df['roa'].median():.3f}",
)
axes[0].set_xlabel("RoA")
axes[0].set_title("Distribution of RoA", color=WU_BLUE)
axes[0].legend()

yearly_median = df.groupby("fyear")["roa"].median()
yearly_n = df.groupby("fyear")["roa"].count()

axes[1].bar(yearly_median.index, yearly_median.values, color=WU_BLUE, alpha=0.8)
axes[1].axhline(0, color="black", lw=0.8, ls="--")
axes[1].set_xlabel("Fiscal Year")
axes[1].set_ylabel("Median RoA")
axes[1].set_title("Median RoA by Year", color=WU_BLUE)

for year, val, n in zip(yearly_median.index, yearly_median.values, yearly_n.values):
    axes[1].text(
        year,
        val + 0.002,
        f"n={n:,}",
        ha="center",
        va="bottom",
        fontsize=6,
        color="gray",
        rotation=90,
    )

fig.tight_layout()
fig.savefig(FIGURE_PATH / "dv_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved output/figures/dv_distribution.png")

print(f"\nRoA mean:   {df['roa'].mean():.4f}")
print(f"RoA median: {df['roa'].median():.4f}")
print(f"Negative RoA: {(df['roa'] < 0).sum():,} ({(df['roa'] < 0).mean() * 100:.1f}%)")
print(f"Positive RoA: {(df['roa'] > 0).sum():,} ({(df['roa'] > 0).mean() * 100:.1f}%)")


# ── 10. Sample Composition ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

country_counts = df["loc"].value_counts().head(10)
axes[0].barh(country_counts.index[::-1], country_counts.values[::-1], color=WU_BLUE)
axes[0].set_xlabel("Firm-year observations")
axes[0].set_title("Top 10 Countries in Sample", color=WU_BLUE)

year_counts = df["fyear"].value_counts().sort_index()
axes[1].bar(year_counts.index, year_counts.values, color=WU_BLUE)
axes[1].set_xlabel("Fiscal Year")
axes[1].set_ylabel("Observations")
axes[1].set_title("Sample Coverage by Year", color=WU_BLUE)

fig.tight_layout()
fig.savefig(FIGURE_PATH / "sample_composition.png", dpi=150)
plt.close()

print("Saved output/figures/sample_composition.png")


# ── 11. Save panel with variables ──────────────────────────────────────────────
df.to_parquet(PANEL_OUT, index=False)

print("\nSaved data/processed/panel_with_vars.parquet")
print(f"Final dataset: {df.shape[0]:,} rows | {df['gvkey'].nunique():,} firms")
print("\nDescriptives complete. Check output/tables/ and output/figures/")