"""End-to-end pipeline: load -> prepare -> EDA -> 4 models."""
import warnings

from . import eda, logistic, linear, stockout_models
from .data import load_data, basic_checks, engineer_features
from .utils import setup, finish, log


def run_pipeline(data_path, out_dir="outputs", fast=False):
    warnings.filterwarnings("ignore")
    setup(out_dir)
    try:
        df = load_data(data_path)
        basic_checks(df)
        df = engineer_features(df)
        eda.run(df)
        logistic.run(df)
        ctx = stockout_models.run_decision_tree(df)
        stockout_models.run_random_forest(ctx, fast=fast)
        linear.run(df)
        log("\nDone - all 21 figures and result tables generated.")
    finally:
        finish()
