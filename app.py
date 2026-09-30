"""Streamlit dashboard for the AADM end-term project.

Local:   streamlit run app.py
Cloud:   Main file path = app.py
"""
import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from aadm.pipeline import run_pipeline  # noqa: E402

DEFAULT_DATA = Path(__file__).parent / "data" / "OPS_DS1.csv"

SECTIONS = {
    "Exploratory analysis": range(1, 5),
    "Logistic Regression - On-Time Delivery": range(5, 10),
    "Decision Tree - Stockout Risk": range(10, 15),
    "Random Forest - Stockout Risk": range(15, 20),
    "Linear Regression - Profit": range(20, 22),
}

st.set_page_config(page_title="AADM Supply Chain Analytics", page_icon="📦", layout="wide")


@st.cache_data(show_spinner=False)
def run_cached(data_bytes: bytes, suffix: str, fast: bool):
    """Run the full pipeline once per (dataset, mode) and return figures/text/metrics."""
    work = Path(tempfile.mkdtemp())
    data_path = work / f"data{suffix}"
    data_path.write_bytes(data_bytes)
    out = work / "out"
    run_pipeline(data_path, out, fast=fast)
    figs = {int(p.name[3:5]): p.read_bytes() for p in sorted((out / "figures").glob("fig*.png"))}
    names = {int(p.name[3:5]): p.stem[6:].replace("_", " ").title()
             for p in (out / "figures").glob("fig*.png")}
    text = (out / "results.txt").read_text(encoding="utf-8")
    metrics = json.loads((out / "metrics.json").read_text())
    return figs, names, text, metrics


st.title("📦 Advanced Analytics for Decision-Making")
st.caption("Supply-chain analytics on 1,000 transactions: on-time delivery, stockout risk and profit drivers.")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload OPS_DS1 (.csv / .xlsx)", type=["csv", "xlsx"])
    fast = st.checkbox("Fast mode (smaller Random Forest grid)", value=True,
                       help="Untick to reproduce the exact report numbers. Takes longer.")
    st.markdown("---")
    st.caption("Models: Logistic Regression, Decision Tree, Random Forest, Linear Regression.")

if upload is not None:
    data_bytes, suffix = upload.getvalue(), Path(upload.name).suffix
elif DEFAULT_DATA.exists():
    data_bytes, suffix = DEFAULT_DATA.read_bytes(), ".csv"
else:
    st.info("Upload **OPS_DS1.csv** in the sidebar to run the analysis.")
    st.stop()

try:
    with st.spinner("Running the analysis pipeline..."):
        figs, names, text, metrics = run_cached(data_bytes, suffix, fast)
except (ValueError, FileNotFoundError) as e:
    st.error(str(e))
    st.stop()

lr, dt, rf, lin = (metrics.get(k, {}) for k in
                   ["logistic_regression", "decision_tree", "random_forest", "linear_regression"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Logistic Reg. accuracy", f"{lr.get('accuracy', 0):.3f}", f"AUC {lr.get('roc_auc', 0):.3f}", delta_color="off")
c2.metric("Decision Tree macro-F1", f"{dt.get('test_macro_f1', 0):.3f}", f"Acc {dt.get('test_accuracy', 0):.3f}", delta_color="off")
c3.metric("Random Forest macro-F1", f"{rf.get('test_macro_f1', 0):.3f}", f"Acc {rf.get('test_accuracy', 0):.3f}", delta_color="off")
c4.metric("Linear Reg. R²", f"{lin.get('r2', 0):.3f}", f"MAE ₹{lin.get('mae_inr', 0):,.0f}", delta_color="off")

if fast:
    st.caption("Fast mode is on, so Random Forest results may differ slightly from the final report.")

tabs = st.tabs(list(SECTIONS) + ["Full results log"])
for tab, (title, rng) in zip(tabs, SECTIONS.items()):
    with tab:
        st.subheader(title)
        cols = st.columns(2)
        for i, n in enumerate([n for n in rng if n in figs]):
            with cols[i % 2]:
                st.image(figs[n], caption=f"Fig {n}: {names[n]}")
with tabs[-1]:
    st.code(text, language="text")
    st.download_button("Download results.txt", text, file_name="results.txt")
