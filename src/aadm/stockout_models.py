"""5.2 Decision Tree (Figures 10-14) and 5.3 Random Forest (Figures 15-19) - Stockout_Risk."""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.tree import DecisionTreeClassifier, plot_tree

from .config import (BLUE, RED, GOLD, RISK_CLASSES, RANDOM_STATE, TEST_SIZE,
                     TREE_NUM, TREE_CAT)
from .data import make_X
from .utils import log, save_fig, record


def run_decision_tree(df):
    log("\n" + "=" * 70 + "\n5.2 DECISION TREE - STOCKOUT RISK\n" + "=" * 70)
    X = make_X(df, TREE_NUM, TREE_CAT)
    y = df.Stockout_Risk
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Depth-3 interpretable tree
    dt3 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10,
                                 random_state=RANDOM_STATE).fit(X_train, y_train)
    p3 = dt3.predict(X_test)
    log("Depth-3 tree  -> Acc: {:.3f} | Macro-F1: {:.3f} | Leaves: {}".format(
        accuracy_score(y_test, p3), f1_score(y_test, p3, average="macro"), dt3.get_n_leaves()))

    # Tuned tree
    grid = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE),
                        {"max_depth": [2, 3, 4, 5, 6, 8, None], "min_samples_leaf": [5, 10, 20]},
                        cv=5, scoring="f1_macro").fit(X_train, y_train)
    dt = grid.best_estimator_
    p_dt = dt.predict(X_test)
    acc, f1 = accuracy_score(y_test, p_dt), f1_score(y_test, p_dt, average="macro")
    log("Best params   :", grid.best_params_)
    log("Best CV macro-F1: {:.3f}".format(grid.best_score_))
    log("Tuned tree    -> Acc: {:.3f} | Macro-F1: {:.3f} | Leaves: {}".format(
        acc, f1, dt.get_n_leaves()))
    log(classification_report(y_test, p_dt, labels=RISK_CLASSES, digits=2))
    record("decision_tree", best_params=str(grid.best_params_), cv_macro_f1=grid.best_score_,
           test_accuracy=acc, test_macro_f1=f1, leaves=int(dt.get_n_leaves()))

    # Fig 10 - train vs CV by depth
    depths = range(1, 12)
    train_acc, cv_f1 = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, min_samples_leaf=10,
                                   random_state=RANDOM_STATE).fit(X_train, y_train)
        train_acc.append(m.score(X_train, y_train))
        cv_f1.append(cross_val_score(m, X_train, y_train, cv=5, scoring="f1_macro").mean())
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(depths, train_acc, "o-", color=RED, label="Training accuracy")
    ax.plot(depths, cv_f1, "o-", color=BLUE, label="Cross-val macro-F1")
    ax.set_title("Training vs Cross-Validated Performance by Tree Depth", fontsize=10)
    ax.set_xlabel("Tree depth", fontsize=8); ax.set_ylabel("Score", fontsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout(); save_fig(10, "tree_depth_vs_performance", fig)

    # Fig 11 - depth-3 tree structure
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(dt3, feature_names=X.columns, class_names=dt3.classes_, filled=True,
              impurity=False, fontsize=8, ax=ax)
    ax.set_title("Decision Tree (depth = 3) - Stockout Risk")
    plt.tight_layout(); save_fig(11, "tree_depth3_structure", fig)

    # Fig 12 - confusion matrix (tuned tree)
    cm = confusion_matrix(y_test, p_dt, labels=RISK_CLASSES)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=RISK_CLASSES,
                yticklabels=RISK_CLASSES, annot_kws={"size": 13}, ax=ax)
    ax.set_title("Confusion Matrix - Tuned Tree (Test Set)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); save_fig(12, "tree_confusion_matrix", fig)

    # Fig 13 - feature importance
    imp = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
    log("\nTuned tree feature importance:\n", imp.head(8).round(3).to_string())
    top = imp.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(top.index, top.values, color=BLUE)
    ax.set_title("Feature Importance - Tuned Decision Tree (Stockout Risk)", fontsize=10)
    ax.set_xlabel("Importance", fontsize=8); ax.tick_params(labelsize=8)
    plt.tight_layout(); save_fig(13, "tree_feature_importance", fig)

    # Fig 14 - risk by inventory position
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for cls, col in zip(RISK_CLASSES, [BLUE, GOLD, RED]):
        s = df[df.Stockout_Risk == cls]
        ax.scatter(s.Inventory_Gap, s.Safety_Gap, s=14, alpha=0.6, color=col, label=cls)
    ax.axvline(0, color="black", ls="--", lw=1); ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_title("Stockout Risk by Inventory Position", fontsize=10)
    ax.set_xlabel("Inventory - Reorder Point", fontsize=8)
    ax.set_ylabel("Inventory - Safety Stock", fontsize=8)
    ax.legend(title="Risk", fontsize=8)
    plt.tight_layout(); save_fig(14, "stockout_by_inventory_position", fig)

    return dict(X=X, X_train=X_train, X_test=X_test, y_train=y_train,
                y_test=y_test, dt=dt, p_dt=p_dt)


def run_random_forest(ctx, fast=False):
    log("\n" + "=" * 70 + "\n5.3 RANDOM FOREST - STOCKOUT RISK\n" + "=" * 70)
    X, X_train, X_test = ctx["X"], ctx["X_train"], ctx["X_test"]
    y_train, y_test, dt, p_dt = ctx["y_train"], ctx["y_test"], ctx["dt"], ctx["p_dt"]

    param_grid = ({"n_estimators": [100], "max_depth": [None], "min_samples_leaf": [1]} if fast
                  else {"n_estimators": [100, 300], "max_depth": [6, 10, None],
                        "min_samples_leaf": [1, 5]})
    grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), param_grid,
                        cv=5, scoring="f1_macro", n_jobs=-1).fit(X_train, y_train)
    rf = grid.best_estimator_
    p_rf = rf.predict(X_test)
    acc, f1 = accuracy_score(y_test, p_rf), f1_score(y_test, p_rf, average="macro")
    log("Best RF params :", grid.best_params_)
    log("Best CV macro-F1: {:.3f}".format(grid.best_score_))
    log("RF -> Test Acc: {:.3f} | Test Macro-F1: {:.3f}".format(acc, f1))
    log(classification_report(y_test, p_rf, labels=RISK_CLASSES, digits=2))
    record("random_forest", best_params=str(grid.best_params_), cv_macro_f1=grid.best_score_,
           test_accuracy=acc, test_macro_f1=f1)

    # CV stability (same folds for both models)
    skf = StratifiedKFold(n_splits=5)
    cv_tree = cross_val_score(dt, X_train, y_train, cv=skf, scoring="f1_macro")
    cv_rf = cross_val_score(rf, X_train, y_train, cv=skf, scoring="f1_macro")
    log("\nSingle Tree  CV macro-F1: mean {:.3f}, std {:.3f}".format(cv_tree.mean(), cv_tree.std()))
    log("Random Forest CV macro-F1: mean {:.3f}, std {:.3f}".format(cv_rf.mean(), cv_rf.std()))

    # Fig 15 - stability boxplot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot([cv_tree, cv_rf], patch_artist=True, medianprops=dict(color=RED))
    ax.set_xticklabels(["Single Tree", "Random Forest"])
    for patch in bp["boxes"]:
        patch.set_facecolor("#C8D3DC")
    ax.set_title("Single Tree vs Random Forest - Stability Across Folds")
    ax.set_ylabel("Cross-validated macro-F1")
    plt.tight_layout(); save_fig(15, "cv_stability_boxplot", fig)

    # Fig 16 - RF confusion matrix
    cm = confusion_matrix(y_test, p_rf, labels=RISK_CLASSES)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=RISK_CLASSES,
                yticklabels=RISK_CLASSES, annot_kws={"size": 13}, ax=ax)
    ax.set_title("Confusion Matrix - Random Forest (Test Set)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); save_fig(16, "forest_confusion_matrix", fig)

    # Fig 17 - per-class F1
    f1_dt = f1_score(y_test, p_dt, labels=RISK_CLASSES, average=None)
    f1_rf = f1_score(y_test, p_rf, labels=RISK_CLASSES, average=None)
    xpos, w = np.arange(3), 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(xpos - w / 2, f1_dt, w, color=RED, label="Single Tree")
    ax.bar(xpos + w / 2, f1_rf, w, color=BLUE, label="Random Forest")
    ax.set_xticks(xpos); ax.set_xticklabels(RISK_CLASSES)
    ax.set_title("Per-Class F1: Single Tree vs Random Forest")
    ax.set_ylabel("F1-score"); ax.legend()
    plt.tight_layout(); save_fig(17, "per_class_f1_comparison", fig)

    # Fig 18 - effect of number of trees
    n_list = [1, 10, 50, 100] if fast else [1, 5, 10, 25, 50, 100, 200, 300]
    means, stds = [], []
    for n in n_list:
        m = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
        s = cross_val_score(m, X_train, y_train, cv=skf, scoring="f1_macro")
        means.append(s.mean()); stds.append(s.std())
    means, stds = np.array(means), np.array(stds)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(n_list, means, "o-", color=BLUE)
    ax.fill_between(n_list, means - stds, means + stds, color=BLUE, alpha=0.15)
    ax.set_title("Effect of Number of Trees on Forest Performance", fontsize=10)
    ax.set_xlabel("Number of trees (n_estimators)", fontsize=8)
    ax.set_ylabel("Cross-validated macro-F1", fontsize=8)
    plt.tight_layout(); save_fig(18, "forest_n_estimators_effect", fig)

    # Fig 19 - RF feature importance
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    log("\nRandom Forest feature importance:\n", imp.head(8).round(3).to_string())
    top = imp.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(top.index, top.values, color=BLUE)
    ax.set_title("Feature Importance - Random Forest (Stockout Risk)", fontsize=10)
    ax.set_xlabel("Importance (averaged across trees)", fontsize=8); ax.tick_params(labelsize=8)
    plt.tight_layout(); save_fig(19, "forest_feature_importance", fig)
