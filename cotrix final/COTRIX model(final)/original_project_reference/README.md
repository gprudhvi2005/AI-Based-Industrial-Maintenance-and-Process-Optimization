# COTRIX Integrated Predictive Maintenance Dashboard

This package connects the Cotrix frontend to the predictive-maintenance models from the uploaded project.

## Connected models

1. Logistic Regression — existing `logistic_regression_model.pkl`
2. Random Forest — retrained from the Milestone 3 model family
3. Gradient Boosting — existing `best_predictive_maintenance_model.pkl`
4. SVM — trained from the Milestone 2 SVM workflow and saved as a bundle
5. ANN — trained from the exact architecture in `ann_predictive_maintenance.ipynb`

### Important note about Decision Tree

The uploaded `Milestone_2/decision tree.py` trains on the Iris dataset, not the AI4I predictive-maintenance dataset. It is therefore intentionally NOT connected to Cotrix as a machine-failure model.

## Run locally

### 1. Open a terminal in `backend`

Windows PowerShell:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Keep that terminal running.

### 2. Open the dashboard

Open:

```text
frontend/index.html
```

in Chrome.

The dashboard calls:

```text
http://127.0.0.1:5000
```

## What is live

- Model Center reads actual evaluation metrics generated from the uploaded project/model artifacts.
- Predict New Data calls all five connected models.
- Ensemble risk is the mean probability from the five models.
- CSV upload endpoint validates and reads a CSV file.
- The UI includes the supplied futuristic Cotrix visual direction.

## Model reproducibility note

The original Milestone 3 repository saves only the best ensemble model, not separate Random Forest and ANN model files. For Cotrix integration, those two models were retrained using the configurations documented in the repository and saved under `backend/models/`.

The SVM source did not save a model and used an interactive Colab workflow, so Cotrix saves its trained SVM together with its scaler and label encoder.
