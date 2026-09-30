# Advanced Analytics for Decision-Making — Supply Chain Analytics

End-term project for **Advanced Analytics for Decision-Making** (PGDM E-Business, Welingkar Institute of Management Development & Research).

The project analyses **1,000 supply-chain transactions (38 fields)** and applies four predictive
techniques to answer four operational questions:

| # | Business question | Technique | Target |
|---|-------------------|-----------|--------|
| 5.1 | Which orders will be delivered on time? | Logistic Regression | `On_Time_Delivery` |
| 5.2 | Which items are at risk of a stockout? (interpretable rules) | Decision Tree | `Stockout_Risk` |
| 5.3 | Can an ensemble predict stockout risk more reliably? | Random Forest | `Stockout_Risk` |
| 5.4 | What drives profit? | Linear Regression | `Estimated_Profit_INR` |

## Key results (test set, 75/25 split, `random_state=42`)

| Model | Metric | Result |
|-------|--------|--------|
| Logistic Regression | Accuracy / ROC-AUC | 0.904 / 0.970 |
| Decision Tree (tuned, depth 6) | Accuracy / Macro-F1 | 0.904 / 0.831 |
| Random Forest (tuned) | Accuracy / Macro-F1 | 0.920 / 0.874 |
| Linear Regression | R² / RMSE / MAE | 0.672 / ₹8.25M / ₹6.09M |

Main findings: lead time and supplier reliability drive on-time delivery; the gap between current
inventory and the reorder point is the strongest predictor of stockout risk; selling price and unit
cost are the dominant profit drivers.

## Repository structure

```
├── app.py                   # Streamlit dashboard (main file for deployment)
├── run_analysis.py          # one-command command-line entry point
├── src/aadm/
│   ├── config.py            # palette, seeds, feature lists
│   ├── data.py              # loading, validation, feature engineering
│   ├── eda.py               # descriptives, outliers, business checks, Fig 1-4
│   ├── logistic.py          # 5.1  Fig 5-9
│   ├── stockout_models.py   # 5.2 Decision Tree (Fig 10-14) + 5.3 Random Forest (Fig 15-19)
│   ├── linear.py            # 5.4  Fig 20-21
│   ├── pipeline.py          # end-to-end orchestration
│   └── utils.py             # figure saving + logging
├── notebooks/AADM_code.ipynb  # original single-file notebook (Colab-friendly)
├── data/                    # put OPS_DS1.csv here
├── outputs/                 # generated figures, results.txt, metrics.json
├── tests/                   # pytest suite (uses synthetic data)
└── .github/workflows/ci.yml # runs tests on every push
```

## Quick start

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# add data/OPS_DS1.csv, then:
python run_analysis.py
```

Options:

```bash
python run_analysis.py --data data/OPS_DS1.csv --out outputs   # custom paths
python run_analysis.py --fast                                   # smaller RF grid, quick check
```

### Interactive dashboard (Streamlit)

```bash
streamlit run app.py
```

**Deploy free on Streamlit Community Cloud:** share.streamlit.io → *Create app* → pick this repo,
branch `main`, **Main file path: `app.py`**. Upload `OPS_DS1.csv` in the sidebar, or commit it to `data/`.

### Run in Google Colab instead

Open `notebooks/AADM_code.ipynb` in Colab (File → Open notebook → GitHub tab), upload
`OPS_DS1.csv` when prompted, and run.

## Live dashboard (Streamlit)

`app.py` is a Streamlit dashboard that runs the whole analysis and shows the figures and metrics in tabs.

```bash
streamlit run app.py
```

**Deploy free on Streamlit Community Cloud:**

1. Push this repo to GitHub (make sure `data/OPS_DS1.csv` is committed, or users can upload it in the sidebar).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**.
3. Repository: `<your-username>/<repo-name>` - Branch: `main` - **Main file path: `app.py`**.
4. Click **Deploy**.

## Outputs

Running the pipeline writes to `outputs/`:

- `figures/fig01_…png` to `fig21_…png` — all 21 report figures
- `results.txt` — every printed table and metric
- `metrics.json` — headline metrics per model

## Methodology notes

- **Preprocessing:** IQR outlier screening (outliers retained in raw columns, capped copies created);
  engineered features `Inventory_Gap`, `Safety_Gap`, `Forecast_Error`.
- **Encoding / scaling:** one-hot encoding with `drop_first=True`; `StandardScaler` fitted on
  training data only for logistic and linear regression.
- **Validation:** stratified train/test split for classifiers; 5-fold cross-validation with
  `GridSearchCV` (scored on macro-F1) for tree and forest.
- **Reproducibility:** fixed `random_state=42` throughout.

## Tests

```bash
pytest -q
```

The tests generate a small synthetic dataset, so they run without the real data (this is what CI uses).

## Tech stack

Python · pandas · NumPy · scikit-learn · statsmodels · matplotlib · seaborn

## Author

Shifa — PGDM E-Business, Welingkar Institute of Management Development & Research, Mumbai

## License

MIT — see [LICENSE](LICENSE).
