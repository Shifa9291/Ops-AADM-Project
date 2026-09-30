"""Data loading and feature engineering."""
from pathlib import Path

import pandas as pd

from .utils import log

REQUIRED_COLUMNS = [
    "Stockout_Risk", "On_Time_Delivery", "Current_Inventory", "Reorder_Point",
    "Safety_Stock", "Actual_Demand", "Demand_Forecast", "Lead_Time_Days",
    "Supplier_Reliability_Score", "Unit_Cost_INR", "Selling_Price_INR",
    "Delivery_Distance_KM", "Estimated_Profit_INR", "Operational_Efficiency_Score",
    "Inventory_Turnover_Ratio", "Downtime_Hours", "Total_Operational_Cost_INR",
    "Holding_Cost_INR", "Ordering_Cost_INR", "Transportation_Cost_INR",
    "Production_Actual", "Production_Capacity", "Forecast_Accuracy_Percent",
    "Transport_Mode", "Revenue_INR", "Order_Priority", "Defect_Rate_Percent",
    "Order_Quantity", "Product_Category", "Warehouse", "Machine_Utilization_Percent",
]


def load_data(path):
    """Read OPS_DS1 (.csv or .xlsx) and validate that the needed columns exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place OPS_DS1.csv inside the data/ folder (see data/README.md) "
            "or pass --data <path>."
        )
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def basic_checks(df):
    log("Shape:", df.shape)
    log("Missing values:", int(df.isna().sum().sum()))
    log("Duplicate rows:", int(df.duplicated().sum()))
    log("\nStockout_Risk (%):\n", (df["Stockout_Risk"].value_counts(normalize=True) * 100).round(1))
    log("\nOn_Time_Delivery (%):\n", (df["On_Time_Delivery"].value_counts(normalize=True) * 100).round(1))


def engineer_features(df):
    """Capped copies (outliers kept in raw columns) + engineered gap features."""
    df = df.copy()
    df["Inventory_Turnover_Capped"] = df["Inventory_Turnover_Ratio"].clip(
        upper=df["Inventory_Turnover_Ratio"].quantile(0.95))
    df["Downtime_Hours_Capped"] = df["Downtime_Hours"].clip(
        upper=df["Downtime_Hours"].quantile(0.99))
    df["Inventory_Gap"] = df.Current_Inventory - df.Reorder_Point
    df["Safety_Gap"] = df.Current_Inventory - df.Safety_Stock
    df["Forecast_Error"] = df.Actual_Demand - df.Demand_Forecast
    return df


def make_X(df, num_feats, cat_feats):
    """Numeric features + one-hot encoded categoricals (drop_first) as float matrix."""
    return pd.concat([df[num_feats], pd.get_dummies(df[cat_feats], drop_first=True)],
                     axis=1).astype(float)
