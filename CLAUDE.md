# Breast Cancer Detection

End-to-end supervised ML project: train a binary classifier on the Breast Cancer Wisconsin (Diagnostic) dataset and serve it through a Streamlit app that loads the trained model and makes predictions.

## Goal

1. Load and validate the data
2. Explore (EDA) and preprocess
3. Train, tune, and evaluate models
4. Persist the best model as a single artifact (preprocessing + estimator)
5. Streamlit app that loads that artifact and predicts from user input

## Dataset

- Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (UCI id 17)
- 569 samples, 30 numeric features, no missing values
- Features: mean, standard error, and "worst" of 10 cell-nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- Target: `Diagnosis` — `M` (malignant) / `B` (benign). Encode as `M=1`, `B=0` so malignant is the positive class. Class balance is ~37% M / 63% B.
- Fetch with `ucimlrepo` (`fetch_ucirepo(id=17)`) and cache the raw data in `data/raw/`. Never edit raw data; write derived data to `data/processed/`.

## Project layout (target)

```
data/
  raw/            # untouched download
  processed/      # cleaned / split data
notebooks/        # EDA and experiments only; no logic the app depends on
src/
  data.py         # load + split
  features.py     # preprocessing / feature pipeline
  train.py        # training + tuning entrypoint
  evaluate.py     # metrics and plots
  predict.py      # load artifact + predict (shared by app and tests)
models/           # serialized model artifact(s) + metadata (metrics, feature order, versions)
app/
  streamlit_app.py
tests/
pyproject.toml
```

Keep reusable logic in `src/`; notebooks and the app import from it.

## Tech stack

- Python 3.11+, managed with `uv` (preferred) — `pyproject.toml` is the source of truth for dependencies
- pandas, numpy, scikit-learn (optionally xgboost/lightgbm for comparison), matplotlib/seaborn or plotly
- joblib for model persistence
- Streamlit for the app
- pytest for tests, ruff for lint/format

## Commands

(Update as the project is built — keep these accurate.)

```bash
uv sync                              # install deps
uv run python -m src.train           # train and write models/ artifact
uv run pytest                        # run tests
uv run ruff check . && uv run ruff format .
uv run streamlit run app/streamlit_app.py
```

## ML conventions

- **Prevent leakage**: split first (stratified train/test, fixed `random_state`), then fit all scalers/imputers/selectors on train only. Use a scikit-learn `Pipeline` so preprocessing is part of the model.
- **Evaluation**: use stratified k-fold CV on train for model selection; touch the held-out test set once for the final report.
- **Metrics**: this is a medical screening task — false negatives (missed malignant) are the costly error. Report recall, precision, F1, ROC-AUC, PR-AUC, and the confusion matrix, not accuracy alone. Pick the decision threshold deliberately and document it.
- **Baselines first**: start with logistic regression, then compare tree-based models; prefer the simpler model unless the gain is meaningful. Features are highly correlated, so watch multicollinearity for linear models.
- **Reproducibility**: seed everything; record library versions, metrics, and feature order next to the model artifact.
- Save the **whole pipeline** as one artifact so the app never re-implements preprocessing.

## Streamlit app conventions

- Load the model once with `@st.cache_resource`; never retrain in the app.
- Inputs must match the training feature names and order exactly (validate against the saved metadata). 30 inputs is a lot — group by mean / SE / worst, use sensible min/max/default from the training data, and consider a CSV upload for batch prediction.
- Show predicted class, probability, and the threshold used.
- Include a clear disclaimer: this is an educational model, not a medical device, and must not be used for real diagnosis.

## Working agreements

- Don't commit large data or model binaries unless small and intentional; add `data/raw/`, `.venv/`, and notebook checkpoints to `.gitignore` as appropriate.
- Add or update tests when changing `src/` (data shapes, no-leakage split, pipeline round-trip save/load, prediction output format).
- Prefer small, focused changes; keep this file updated when the layout or commands change.
