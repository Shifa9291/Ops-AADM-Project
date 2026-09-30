"""Central configuration: colours, seeds, feature lists."""

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Colour palette used in the report
BLUE, RED, GOLD = "#3D5570", "#9B5344", "#D9B04C"

RISK_CLASSES = ["Low", "Medium", "High"]

DESCRIPTIVE_COLS = [
    "Current_Inventory", "Reorder_Point", "Actual_Demand", "Lead_Time_Days",
    "Supplier_Reliability_Score", "Unit_Cost_INR", "Selling_Price_INR",
    "Delivery_Distance_KM", "Estimated_Profit_INR", "Operational_Efficiency_Score",
]

CORR_COLS = [
    "Current_Inventory", "Safety_Stock", "Reorder_Point", "Actual_Demand",
    "Lead_Time_Days", "Supplier_Reliability_Score", "Unit_Cost_INR",
    "Selling_Price_INR", "Transportation_Cost_INR", "Revenue_INR",
    "Estimated_Profit_INR", "Operational_Efficiency_Score",
]

# 5.1 Logistic regression - On_Time_Delivery
LOGIT_NUM = ["Lead_Time_Days", "Supplier_Reliability_Score", "Delivery_Distance_KM",
             "Downtime_Hours", "Defect_Rate_Percent", "Order_Quantity"]
LOGIT_CAT = ["Transport_Mode", "Order_Priority"]

# 5.2 / 5.3 Tree & forest - Stockout_Risk
TREE_NUM = ["Current_Inventory", "Safety_Stock", "Reorder_Point", "Inventory_Gap",
            "Safety_Gap", "Actual_Demand", "Demand_Forecast", "Forecast_Error",
            "Lead_Time_Days", "Supplier_Reliability_Score", "Order_Quantity"]
TREE_CAT = ["Product_Category", "Warehouse"]

# 5.4 Linear regression - Estimated_Profit_INR
LIN_NUM = ["Order_Quantity", "Actual_Demand", "Selling_Price_INR", "Unit_Cost_INR",
           "Lead_Time_Days", "Delivery_Distance_KM", "Transportation_Cost_INR",
           "Supplier_Reliability_Score", "Defect_Rate_Percent", "Machine_Utilization_Percent"]
LIN_CAT = ["Product_Category", "Warehouse", "Order_Priority"]
