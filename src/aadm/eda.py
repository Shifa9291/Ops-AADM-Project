"""Section 2-4: descriptives, data preparation checks and EDA (Figures 1-4)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import DESCRIPTIVE_COLS, CORR_COLS
from .utils import log, save_fig, record


def descriptives(df):
    desc = df[DESCRIPTIVE_COLS].agg(["mean", "std", "min", "median", "max"]).T
    desc.columns = ["Mean", "Std. Dev.", "Min", "Median", "Max"]
    log("\n=== Selected Descriptive Statistics ===")
    log(desc.round(2).to_string(float_format=lambda x: f"{x:,.2f}"))
    log("\nNumeric columns:", df.select_dtypes(include="number").shape[1])


def outlier_screen(df):
    log("\n=== IQR outlier counts (top) ===")
    counts = {}
    for c in df.select_dtypes(include="number").columns:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        counts[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    log(pd.Series(counts).loc[lambda s: s > 0].sort_values(ascending=False).to_string())


def business_checks(df):
    log("\n=== Business checks ===")
    cost_ok = np.isclose(df.Total_Operational_Cost_INR,
                         df.Holding_Cost_INR + df.Ordering_Cost_INR + df.Transportation_Cost_INR,
                         atol=0.05).mean()
    log("Total cost = Holding+Ordering+Transport :", cost_ok)
    log("Production actual > capacity           : {:.1%}".format(
        (df.Production_Actual > df.Production_Capacity).mean()))
    below_cost = df.Selling_Price_INR < df.Unit_Cost_INR
    log("Selling price < unit cost              : {:.1%}  ({} rows)".format(
        below_cost.mean(), int(below_cost.sum())))
    log("Inventory < reorder point              : {:.1%}".format((df.Inventory_Gap < 0).mean()))
    log("Inventory < safety stock               : {:.1%}".format((df.Safety_Gap < 0).mean()))
    log("Forecast accuracy exactly 0            : {:.1%}".format(
        (df.Forecast_Accuracy_Percent == 0).mean()))

    lt_band = pd.cut(df.Lead_Time_Days, [0, 6, 12, 18, 24, 30, 36, 44])
    log("\nOn-time rate by lead-time band:\n",
        df.groupby(lt_band, observed=True).On_Time_Delivery
          .apply(lambda s: (s == "Yes").mean()).round(3))
    log("\nOn-time rate by transport mode:\n",
        df.groupby("Transport_Mode").On_Time_Delivery
          .apply(lambda s: (s == "Yes").mean()).round(3))

    rule = (df.Lead_Time_Days <= 24) & (df.Supplier_Reliability_Score > 80)
    rule_acc = (rule == (df.On_Time_Delivery == "Yes")).mean()
    log("\nRule (LT<=24 & Reliability>80) accuracy on full data:", rule_acc,
        "| on-time predicted:", int(rule.sum()))
    record("eda", rule_accuracy=rule_acc)
    log("\nStockout risk vs inventory position:\n",
        pd.crosstab(df.Inventory_Gap < 0, df.Stockout_Risk, normalize="index").round(2))


def make_figures(df):
    # Fig 1 - Stockout risk distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    df["Stockout_Risk"].value_counts().reindex(["Low", "Medium", "High"]).plot(
        kind="bar", color="#1f77b4", ax=ax)
    ax.set_title("Stockout Risk Distribution"); ax.set_xlabel("Stockout Risk")
    ax.set_ylabel("Number of Transactions"); plt.xticks(rotation=0); ax.grid(False)
    plt.tight_layout(); save_fig(1, "stockout_risk_distribution", fig)

    # Fig 2 - On-time delivery distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    df["On_Time_Delivery"].value_counts().reindex(["No", "Yes"]).plot(
        kind="bar", color="#1f77b4", ax=ax)
    ax.set_title("On-Time Delivery Distribution"); ax.set_xlabel("On-Time Delivery")
    ax.set_ylabel("Number of Transactions"); plt.xticks(rotation=0); ax.grid(False)
    plt.tight_layout(); save_fig(2, "on_time_delivery_distribution", fig)

    # Fig 3 - Profit distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df.Estimated_Profit_INR, bins=30, color="#1f77b4")
    ax.set_title("Distribution of Estimated Profit"); ax.set_xlabel("Estimated Profit (INR)")
    ax.set_ylabel("Frequency"); ax.grid(False)
    plt.tight_layout(); save_fig(3, "profit_distribution", fig)
    log("Mean profit: {:,.0f} | Median profit: {:,.0f}".format(
        df.Estimated_Profit_INR.mean(), df.Estimated_Profit_INR.median()))

    # Fig 4 - Correlation matrix
    corr = df[CORR_COLS].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap="viridis", vmin=-0.5, vmax=1)
    ax.set_xticks(range(len(CORR_COLS))); ax.set_yticks(range(len(CORR_COLS)))
    ax.set_xticklabels(CORR_COLS, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(CORR_COLS, fontsize=7)
    ax.set_title("Correlation Matrix of Selected Operational and Financial Variables", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046); ax.grid(False)
    plt.tight_layout(); save_fig(4, "correlation_matrix", fig)

    extra = list(set(CORR_COLS + ["Total_Operational_Cost_INR", "Forecast_Accuracy_Percent"]))
    full_corr = df[extra].corr()
    for a, b in [("Transportation_Cost_INR", "Total_Operational_Cost_INR"),
                 ("Forecast_Accuracy_Percent", "Operational_Efficiency_Score"),
                 ("Revenue_INR", "Estimated_Profit_INR"),
                 ("Selling_Price_INR", "Estimated_Profit_INR"),
                 ("Unit_Cost_INR", "Estimated_Profit_INR")]:
        log(f"r({a}, {b}) = {full_corr.loc[a, b]:.3f}")


def run(df):
    descriptives(df)
    outlier_screen(df)
    business_checks(df)
    make_figures(df)
