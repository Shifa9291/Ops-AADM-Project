"""5.4 Linear Regression - Estimated_Profit_INR (Figures 20-21)."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import BLUE, RED, RANDOM_STATE, TEST_SIZE, LIN_NUM, LIN_CAT
from .data import make_X
from .utils import log, save_fig, record


def run(df):
    log("\n" + "=" * 70 + "\n5.4 LINEAR REGRESSION - ESTIMATED PROFIT\n" + "=" * 70)
    X = make_X(df, LIN_NUM, LIN_CAT)
    y = df.Estimated_Profit_INR
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression().fit(X_train_s, y_train)
    pred = model.predict(X_test_s)

    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = mean_absolute_error(y_test, pred)
    log("R-squared: {:.3f}".format(r2))
    log("RMSE     : INR {:,.0f}".format(rmse))
    log("MAE      : INR {:,.0f}".format(mae))
    record("linear_regression", r2=r2, rmse_inr=rmse, mae_inr=mae)

    coefs = pd.Series(model.coef_, index=X.columns)
    ordered = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    log("\nTop standardised coefficients (INR):\n", ordered.head(8).round(0).to_string())

    # Fig 20 - top factors
    top8 = ordered.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top8.index, top8.values, color=[BLUE if v > 0 else RED for v in top8.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Which Factors Raise or Lower Profit (Top 8)", fontsize=10)
    ax.set_xlabel("Effect on Profit (Rs., standardised)", fontsize=8); ax.tick_params(labelsize=8)
    plt.tight_layout(); save_fig(20, "profit_top_factors", fig)

    # Fig 21 - actual vs predicted
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(y_test / 1e6, pred / 1e6, s=18, alpha=0.5, color=BLUE)
    lims = [min(y_test.min(), pred.min()) / 1e6, max(y_test.max(), pred.max()) / 1e6]
    ax.plot(lims, lims, "--", color=RED, label="Perfect prediction")
    ax.set_title("Actual vs Predicted Profit - Linear Regression (Test Set)")
    ax.set_xlabel("Actual Profit (Rs. million)"); ax.set_ylabel("Predicted Profit (Rs. million)")
    ax.legend()
    plt.tight_layout(); save_fig(21, "profit_actual_vs_predicted", fig)
