"""Output helpers: figure saving and a tee-style logger."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")  # headless backend - works on servers / GitHub Actions
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

_STATE = {"out": Path("outputs"), "log": None, "metrics": {}}


def setup(out_dir="outputs"):
    """Create the output folders and reset the log / metrics files."""
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _STATE["out"] = out
    _STATE["log"] = open(out / "results.txt", "w", encoding="utf-8")
    _STATE["metrics"] = {}


def log(*args):
    """print() to console AND results.txt."""
    text = " ".join(str(a) for a in args)
    print(text)
    if _STATE["log"]:
        _STATE["log"].write(text + "\n")


def record(section, **values):
    """Store key metrics for metrics.json."""
    _STATE["metrics"].setdefault(section, {}).update(values)


def save_fig(number, name, fig=None):
    """Save the current (or given) figure as figures/figNN_name.png and close it."""
    fig = fig or plt.gcf()
    path = _STATE["out"] / "figures" / f"fig{number:02d}_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"[saved] {path}")


def finish():
    """Write metrics.json and close the log."""
    with open(_STATE["out"] / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_STATE["metrics"], f, indent=2, default=float)
    if _STATE["log"]:
        _STATE["log"].close()
        _STATE["log"] = None
