import json
import pytest

from aadm.data import load_data
from aadm.pipeline import run_pipeline


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="OPS_DS1.csv"):
        load_data(tmp_path / "nope.csv")


def test_missing_columns_detected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_data(bad)


def test_full_pipeline_creates_all_outputs(synthetic_csv, tmp_path):
    out = tmp_path / "out"
    run_pipeline(synthetic_csv, out, fast=True)
    assert len(list((out / "figures").glob("fig*.png"))) == 21
    assert (out / "results.txt").stat().st_size > 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert {"logistic_regression", "decision_tree", "random_forest",
            "linear_regression"} <= set(metrics)
