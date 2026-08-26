# WELCOME TO COTRIX 

This COTRIX dashboard show and run **7 predictive-maintenance models** from the supplied project:

1. Logistic Regression
2. SVM
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. XGBoost
7. ANN (128 → 64 → 32)

### Important preprocessing note
The production API uses the project's Milestone 1 feature pipeline:
- removes identifiers and target-leaking failure-mode columns
- engineers `Temp difference`
- one-hot encodes Type
- standard-scales the six numeric features

The XGBoost model is adapted to this same leakage-safe 8-feature pipeline so that the dashboard's user-entered sensor values can be passed consistently to every model.

## Run on Windows

### Backend
Open CMD inside `backend`:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Keep the backend terminal open.

### Frontend
Open a second CMD inside `frontend`:

```bat
python -m http.server 5500
```

Then open:

`http://127.0.0.1:5500/index.html`

### API health check

`http://127.0.0.1:5000/health`

### What the dashboard now does
- ML Model Center displays all 7 models and live test metrics.
- Predict New Data sends one set of sensor values to all 7 models.
- The UI shows each model's failure probability.
- Ensemble risk is the mean of the 7 model probabilities.
- Health score and maintenance recommendation are calculated from the ensemble.
