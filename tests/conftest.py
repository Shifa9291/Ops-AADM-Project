"""Synthetic stand-in for OPS_DS1.csv so tests/CI run without the real dataset."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 400
    inv = rng.integers(20, 2000, n)
    rop = rng.integers(100, 900, n)
    ss = rng.integers(50, 500, n)
    demand = rng.integers(20, 1400, n)
    lead = rng.integers(1, 45, n)
    rel = rng.uniform(50, 100, n)
    cost = rng.uniform(100, 900, n)
    price = cost * rng.uniform(0.9, 1.8, n)
    qty = rng.integers(10, 1000, n)
    hold, order, trans = rng.uniform(1e3, 5e4, (3, n))
    gap = inv - rop
    risk = np.where(gap > 100, "Low", np.where(gap > -100, "Medium", "High"))
    risk[rng.random(n) < 0.1] = "Low"
    on_time = np.where((lead <= 24) & (rel > 80), "Yes", "No")
    df = pd.DataFrame({
        "Transaction_ID": range(n), "Product_ID": rng.integers(1, 50, n),
        "Supplier_ID": rng.integers(1, 20, n),
        "Stockout_Risk": risk, "On_Time_Delivery": on_time,
        "Current_Inventory": inv, "Reorder_Point": rop, "Safety_Stock": ss,
        "Actual_Demand": demand,
        "Demand_Forecast": demand + rng.integers(-100, 100, n),
        "Lead_Time_Days": lead, "Supplier_Reliability_Score": rel,
        "Unit_Cost_INR": cost, "Selling_Price_INR": price,
        "Delivery_Distance_KM": rng.uniform(10, 2000, n),
        "Order_Quantity": qty, "Revenue_INR": price * qty,
        "Estimated_Profit_INR": (price - cost) * qty,
        "Operational_Efficiency_Score": rng.uniform(40, 100, n),
        "Inventory_Turnover_Ratio": rng.uniform(0.5, 12, n),
        "Downtime_Hours": rng.exponential(5, n),
        "Holding_Cost_INR": hold, "Ordering_Cost_INR": order,
        "Transportation_Cost_INR": trans,
        "Total_Operational_Cost_INR": hold + order + trans,
        "Production_Capacity": rng.integers(500, 1500, n),
        "Production_Actual": rng.integers(400, 1600, n),
        "Forecast_Accuracy_Percent": rng.uniform(0, 100, n),
        "Defect_Rate_Percent": rng.uniform(0, 10, n),
        "Machine_Utilization_Percent": rng.uniform(40, 100, n),
        "Transport_Mode": rng.choice(["Road", "Rail", "Air", "Sea"], n),
        "Order_Priority": rng.choice(["Low", "Medium", "High"], n),
        "Product_Category": rng.choice(["A", "B", "C"], n),
        "Warehouse": rng.choice(["Mumbai", "Delhi", "Chennai"], n),
    })
    path = tmp_path_factory.mktemp("data") / "OPS_DS1.csv"
    df.to_csv(path, index=False)
    return path
