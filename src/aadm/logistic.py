"""5.1 Logistic Regression - On_Time_Delivery (Figures 5-9)."""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import BLUE, RED, RANDOM_STATE, TEST_SIZE, LOGIT_NUM, LOGIT_CAT
from .data import make_X
from .utils import log, save_fig, record


def run(df):
    log("\n" + "=" * 70 + "\n5.1 LOGISTIC REGRESSION - ON-TIME DELIVERY\n" + "=" * 70)
    X = make_X(df, LOGIT_NUM, LOGIT_CAT)
    y = (df.On_Time_Delivery == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000).fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc, auc = accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_prob)
    log("Accuracy :", round(acc, 3))
    log("ROC-AUC  :", round(auc, 3))
    log(classification_report(y_test, y_pred, target_names=["Late", "On-time"], digits=2))
    record("logistic_regression", accuracy=acc, roc_auc=auc)

    # statsmodels: pseudo R2, LLR p-value, coefficient p-values
    try:
        X_sm = pd.DataFrame(X_train_s, columns=X.columns, index=X_train.index)
        sm_logit = sm.Logit(y_train, sm.add_constant(X_sm)).fit(disp=0, maxiter=200)
        log("Pseudo R-squared:", round(sm_logit.prsquared, 4))
        log("LLR p-value     :", sm_logit.llr_pvalue)
        log("\nStatsmodels p-values:\n", sm_logit.pvalues.round(4).to_string())
        record("logistic_regression", pseudo_r2=sm_logit.prsquared)
    except Exception as e:  # perfect / quasi-separation can raise errors
        log("statsmodels Logit issue (quasi-separation):", e)

    coef = pd.DataFrame({"Feature": X.columns, "Coef": clf.coef_[0],
                         "Odds_Ratio": np.exp(clf.coef_[0])}).sort_values("Coef")
    log("\nStandardised coefficients / odds ratios:\n", coef.round(3).to_string(index=False))

    # Fig 5 - coefficients
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(coef.Feature, coef.Coef, color=[BLUE if c > 0 else RED for c in coef.Coef])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Logistic Regression Coefficients (standardised) - On-Time Delivery", fontsize=10)
    ax.set_xlabel("Coefficient (positive = raises odds of on-time)", fontsize=8)
    ax.tick_params(labelsize=8)
    plt.tight_layout(); save_fig(5, "logistic_coefficients", fig)

    # Fig 6 - confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Late", "On-time"],
                yticklabels=["Late", "On-time"], annot_kws={"size": 13}, ax=ax)
    ax.set_title("Confusion Matrix - Test Set"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); save_fig(6, "logistic_confusion_matrix", fig)
    log("Correct classifications:", cm.trace(), "of", cm.sum())

    # Fig 7 - ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random guess")
    ax.set_title("ROC Curve - On-Time Delivery Model")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout(); save_fig(7, "logistic_roc_curve", fig)

    # Fig 8 - predicted probability distribution
    prob_df = pd.DataFrame({"prob": y_prob,
                            "actual": np.where(y_test.values == 1, "On-time", "Late")})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.histplot(data=prob_df, x="prob", hue="actual", bins=30, kde=True,
                 palette={"On-time": RED, "Late": BLUE}, alpha=0.6, ax=ax)
    ax.set_title("Predicted Probability of On-Time Delivery, by Actual Outcome")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Count")
    plt.tight_layout(); save_fig(8, "logistic_probability_distribution", fig)

    # Fig 9 - lead time vs reliability
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sc = ax.scatter(df.Lead_Time_Days, df.Supplier_Reliability_Score,
                    c=(df.On_Time_Delivery == "Yes").astype(int), cmap="RdBu",
                    s=14, alpha=0.6, edgecolor="k", linewidth=0.2)
    ax.axvline(24, color="black", ls="--", lw=1); ax.axhline(80, color="black", ls="--", lw=1)
    ax.set_title("On-Time Delivery Split by Lead Time and Supplier Reliability", fontsize=10)
    ax.set_xlabel("Lead Time (days)", fontsize=8)
    ax.set_ylabel("Supplier Reliability Score", fontsize=8)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("On-time (1) / Late (0)", fontsize=8)
    plt.tight_layout(); save_fig(9, "leadtime_vs_reliability", fig)
