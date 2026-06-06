"""
carbon_eda.py
EDA for EcoPredict Carbon.

Chạy:
    python carbon_eda.py

Đầu ra:
    outputs/eda_figures/*.png
    outputs/eda_tables/*.csv
"""

from __future__ import annotations

from pathlib import Path

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp, skew
except Exception:  # noqa: BLE001
    ks_2samp = None
    skew = None

from sklearn.model_selection import train_test_split

from carbon_utils import RANDOM_STATE, TARGET_COL, get_feature_columns, load_carbon_catalogue, translate_feature_name

warnings.filterwarnings("ignore")

DATA_PATH = Path("carbon_catalogue.csv")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "eda_figures"
TABLE_DIR = OUT_DIR / "eda_tables"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_missing_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean().values * 100).round(2),
    }).sort_values("missing_pct", ascending=False)
    miss.to_csv(TABLE_DIR / "missing_values.csv", index=False, encoding="utf-8-sig")
    return miss


def plot_missing_values(miss: pd.DataFrame) -> None:
    top = miss[miss["missing_pct"] > 0].head(18).sort_values("missing_pct", ascending=True)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.8))
    ax.barh(top["column"], top["missing_pct"], alpha=0.85)
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missing Value Rate by Column")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "missing_values_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pcf_distribution(df: pd.DataFrame) -> None:
    pcf = df[TARGET_COL].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.hist(pcf, bins=40, alpha=0.8, edgecolor="white")
    ax.axvline(pcf.median(), linestyle="--", color="black", lw=1.5, label=f"Median = {pcf.median():.2f}")
    ax.axvline(pcf.mean(), linestyle=":", color="black", lw=1.5, label=f"Mean = {pcf.mean():.2f}")
    ax.set_xlabel("PCF (kg CO₂e)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Product Carbon Footprint")
    ax.legend()
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pcf_distribution_raw.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.hist(np.log1p(pcf), bins=40, alpha=0.8, edgecolor="white")
    ax.set_xlabel("log1p(PCF)")
    ax.set_ylabel("Frequency")
    ax.set_title("Log-transformed PCF Distribution")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pcf_distribution_log.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pcf_by_industry(df: pd.DataFrame) -> None:
    top_groups = df["industry_group"].value_counts().head(10).index.tolist()
    plot_df = df[df["industry_group"].isin(top_groups)].copy()
    if plot_df.empty:
        return
    order = plot_df.groupby("industry_group")[TARGET_COL].median().sort_values(ascending=False).index.tolist()
    values = [plot_df.loc[plot_df["industry_group"].eq(g), TARGET_COL].dropna().values for g in order]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.boxplot(values, labels=order, vert=False, showfliers=False)
    ax.set_xscale("log")
    ax.set_xlabel("PCF (kg CO₂e, log scale)")
    ax.set_title("PCF Distribution by Top Industry Groups")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pcf_boxplot_by_industry_group.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pcf_by_year(df: pd.DataFrame) -> None:
    yr = df.dropna(subset=["year", TARGET_COL]).copy()
    if yr.empty:
        return
    stats = yr.groupby("year")[TARGET_COL].agg(["count", "median", "mean"]).reset_index()
    stats.to_csv(TABLE_DIR / "pcf_by_year.csv", index=False, encoding="utf-8-sig")
    fig, ax1 = plt.subplots(figsize=(8.6, 5.2))
    ax1.plot(stats["year"], stats["median"], marker="o", label="Median PCF")
    ax1.plot(stats["year"], stats["mean"], marker="s", label="Mean PCF")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("PCF (kg CO₂e)")
    ax1.set_title("PCF Trend by Reporting Year")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pcf_trend_by_year.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    feature_cols, numeric_features, _ = get_feature_columns(df)
    cols = [c for c in numeric_features + [TARGET_COL] if c in df.columns]
    corr = df[cols].corr(method="spearman")
    corr.to_csv(TABLE_DIR / "spearman_correlation_numeric.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    im = ax.imshow(corr.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([translate_feature_name(c) for c in cols], rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels([translate_feature_name(c) for c in cols], fontsize=8)
    ax.set_title("Spearman Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spearman_correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_outlier_summary(df: pd.DataFrame) -> None:
    num_cols = [c for c in ["product_weight_kg", TARGET_COL, "upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"] if c in df.columns]
    rows = []
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({
            "column": col,
            "count": int(s.shape[0]),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "q1": float(q1),
            "q3": float(q3),
            "max": float(s.max()),
            "iqr_outlier_count": int(((s < lo) | (s > hi)).sum()),
            "iqr_outlier_pct": float(((s < lo) | (s > hi)).mean() * 100),
            "skewness": float(skew(s)) if skew is not None and len(s) > 2 else np.nan,
        })
    pd.DataFrame(rows).to_csv(TABLE_DIR / "outlier_and_distribution_summary.csv", index=False, encoding="utf-8-sig")


def domain_shift_train_test(df: pd.DataFrame) -> None:
    if ks_2samp is None:
        return
    train, test = train_test_split(df, test_size=0.20, random_state=RANDOM_STATE, shuffle=True)
    numeric_cols = [c for c in get_feature_columns(df)[1] + [TARGET_COL] if c in df.columns]
    rows = []
    for col in numeric_cols:
        a = pd.to_numeric(train[col], errors="coerce").dropna()
        b = pd.to_numeric(test[col], errors="coerce").dropna()
        if len(a) > 10 and len(b) > 10:
            stat, p = ks_2samp(a, b)
            rows.append({"feature": col, "feature_vi": translate_feature_name(col), "ks_statistic": stat, "p_value": p})
    pd.DataFrame(rows).sort_values("ks_statistic", ascending=False).to_csv(TABLE_DIR / "ks_domain_shift_train_test.csv", index=False, encoding="utf-8-sig")


def main(data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    ensure_dirs()
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError("Không tìm thấy carbon_catalogue.csv. Hãy đặt file CSV cùng thư mục rồi chạy lại.")
    df = load_carbon_catalogue(data_path)
    df.to_csv(OUT_DIR / "carbon_processed_for_eda.csv", index=False, encoding="utf-8-sig")
    df.describe(include="all").transpose().to_csv(TABLE_DIR / "descriptive_statistics.csv", encoding="utf-8-sig")
    miss = save_missing_table(df)
    plot_missing_values(miss)
    plot_pcf_distribution(df)
    plot_pcf_by_industry(df)
    plot_pcf_by_year(df)
    plot_correlation_heatmap(df)
    save_outlier_summary(df)
    domain_shift_train_test(df)
    print("Đã hoàn thành EDA.")
    print(f"Bảng EDA: {TABLE_DIR}")
    print(f"Hình EDA: {FIG_DIR}")
    return df


if __name__ == "__main__":
    main()
