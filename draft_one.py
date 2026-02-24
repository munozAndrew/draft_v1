import csv
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import duckdb
from scipy import stats
import pandas as pd

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "output_draft"
ASSIGNMENTS_FILE = f"{OUTPUT_DIR}/domain_assignments.csv"
MIN_VOLUME = 100
KALSHI = "data/KALSHI/markets/*.parquet"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_domain_assignments():
    if not os.path.exists(ASSIGNMENTS_FILE):
        sys.exit(f"Missing {ASSIGNMENTS_FILE}: (complete extract_cat/build_domain.. first) ")

    mapping = {}
    
    with open(ASSIGNMENTS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["prefix"]] = row["domain"]

    domains = set(mapping.values())
    print(f"Loaded {len(mapping):,} prefix assignments across {len(domains)} domains")
    return mapping

def load_KALSHI():
    con = duckdb.connect()

    df = con.execute(f"""
        SELECT
            ticker,
            event_ticker,
            title,
            result,
            CAST(volume AS BIGINT) AS volume,
            CAST(last_price AS BIGINT) AS last_price
        FROM '{KALSHI}'
        WHERE result IN ('yes', 'no')
          AND volume >= {MIN_VOLUME}
    """).df()

    con.close()
    return df

def add_prefix(df):
    df["prefix"] = df["event_ticker"].str.split("-").str[0].fillna("")
    return df

def add_word_count(df):
    clean = df["title"].fillna("").str.replace(r"\*\*", "", regex=True).str.strip()
    df["word_count"] = clean.str.split().apply(len)
    
    return df

def add_brier(df):
    outcome = (df["result"] == "yes").astype(float)
    prediction = df["last_price"] / 100.0
    df["brier"] = (prediction - outcome) ** 2
    
    return df

def filter_to_focus(df, domain_map):
    df["domain"] = df["prefix"].map(domain_map)
    df = df[df["domain"].notna()].copy()
    return df

def brier_by_bucket(df, n_bins=5):
    
    df = df.copy()
    df["bucket"] = pd.qcut(df["word_count"], q=n_bins, labels=False, duplicates="drop")
    
    stats = df.groupby("bucket").agg( n = ("brier", "size"), brier_mean = ("brier", "mean"),
        brier_std = ("brier", "std"),
        wc_min = ("word_count", "min"),
        wc_max = ("word_count", "max"),
    ).reset_index()
    
    return stats


def spearman(df, x_col):
    rho, p = stats.spearmanr(df[x_col], df["brier"])
    return rho, p

def fig1_brier_by_word_bucket(df):
    stats = brier_by_bucket(df)
    rho, p = spearman(df, "word_count")

    labels = [
        f"{int(row.wc_min)}-{int(row.wc_max)} words\n(n={int(row.n):,})"
        for _, row in stats.iterrows()
    ]
    
    se = stats["brier_std"] / np.sqrt(stats["n"])

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(range(len(stats)), stats["brier_mean"],
                  color="#2563EB", alpha=0.85, width=0.6)
    
    ax.errorbar(range(len(stats)), stats["brier_mean"], yerr=1.96 * se,
                fmt="none", color="black", capsize=4, linewidth=1.5)

    for i, (bar, row) in enumerate(zip(bars, stats.itertuples())):
        ax.text(i, row.brier_mean + 0.002, f"{row.brier_mean:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Brier Score  (lower = more accurate)", fontsize=11)
    ax.set_title(
        "Prediction Accuracy by Question Length\n"
        "KALSHI: econ / politics / weather / awards markets",
        fontsize=12, fontweight="bold"
    )
    ax.text(0.97, 0.95, f"Spearman rho = {rho:.3f}  (p={p:.1e})",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    path = f"{OUTPUT_DIR}/fig1_brier_by_word_bucket.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_domain_comparison(df):
    domain_stats = df.groupby("domain").agg(
        n = ("brier", "size"),
        brier_mean = ("brier", "mean"),
        brier_std = ("brier", "std"),
        wc_mean = ("word_count", "mean"),
    ).sort_values("brier_mean").reset_index()

    colors = {
        "Economics / Finance":"#2563EB",
        "Politics / Government":"#DC2626",
        "Weather":"#16A34A",
        "Entertainment / Awards":"#D97706",
    }
    
    bar_colors = [colors.get(d, "#6B7280") for d in domain_stats["domain"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.barh(domain_stats["domain"], domain_stats["brier_mean"],
                     color=bar_colors, alpha=0.85)
    
    se = domain_stats["brier_std"] / np.sqrt(domain_stats["n"])
    
    ax1.errorbar(domain_stats["brier_mean"], range(len(domain_stats)), xerr=1.96 * se,
                 fmt="none", color="black", capsize=4, linewidth=1.5)
    
    for bar, (_, row) in zip(bars1, domain_stats.iterrows()):
        ax1.text(row.brier_mean + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"n={int(row.n):,}", va="center", fontsize=8.5)
        
    ax1.set_xlabel("Mean Brier Score", fontsize=11)
    ax1.set_title("Accuracy by Domain", fontsize=11, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", alpha=0.3)

    bars2 = ax2.barh(domain_stats["domain"], domain_stats["wc_mean"],
                     color=bar_colors, alpha=0.85)
    
    for bar, (_, row) in zip(bars2, domain_stats.iterrows()):
        ax2.text(row.wc_mean + 0.2, bar.get_y() + bar.get_height() / 2,
                 f"{row.wc_mean:.1f}w", va="center", fontsize=8.5)
        
    ax2.set_xlabel("Avg Word Count", fontsize=11)
    ax2.set_title("Question Complexity by Domain", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Domain-Level Accuracy vs. Question Complexity (KALSHI)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    path = f"{OUTPUT_DIR}/fig2_domain_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_category_scatter(df, min_n=50):
    cat_stats = df.groupby("prefix").agg(
        n = ("brier", "size"),
        brier_mean = ("brier", "mean"),
        wc_mean = ("word_count", "mean"),
        domain = ("domain", "first"),
    ).query(f"n >= {min_n}").reset_index()

    colors = {
        "Economics / Finance":"#2563EB",
        "Politics / Government":"#DC2626",
        "Weather":"#16A34A",
        "Entertainment / Awards":"#D97706",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for domain, color in colors.items():
        sub = cat_stats[cat_stats["domain"] == domain]
        if sub.empty:
            continue
        ax.scatter(
            sub["wc_mean"], sub["brier_mean"],
            s=np.log1p(sub["n"]) * 16,
            color=color, alpha=0.75,
            edgecolors="white", linewidth=0.5,
            label=domain,
        )
        
        for _, row in sub.iterrows():
            ax.annotate(
                row["prefix"],
                (row["wc_mean"], row["brier_mean"]),
                fontsize=6, ha="left", va="bottom",
                xytext=(3, 2), textcoords="offset points",
                color=color,
            )

    m, b, r, p, _ = stats.linregress(cat_stats["wc_mean"], cat_stats["brier_mean"])
    x_range = np.linspace(cat_stats["wc_mean"].min(), cat_stats["wc_mean"].max(), 100)
    ax.plot(x_range, m * x_range + b, color="black", linewidth=1.5,
            linestyle="--", label=f"OLS trend  r={r:.2f}, p={p:.3f}")

    ax.set_xlabel("Mean Word Count per Category", fontsize=11)
    ax.set_ylabel("Mean Brier Score  (lower = more accurate)", fontsize=11)
    ax.set_title(
        "Category-Level Complexity vs. Accuracy (KALSHI)\n"
        "Each dot = one market category  |  size proportional to log(market count)",
        fontsize=11, fontweight="bold"
    )
    
    ax.legend(fontsize=9, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    path = f"{OUTPUT_DIR}/fig3_category_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    
    domain_map = load_domain_assignments()

    df = load_KALSHI()

    df = add_prefix(df)
    df = add_word_count(df)
    df = add_brier(df)

    df = filter_to_focus(df, domain_map)

    rho, p = spearman(df, "word_count")

    fig1_brier_by_word_bucket(df)
    fig2_domain_comparison(df)
    fig3_category_scatter(df)
