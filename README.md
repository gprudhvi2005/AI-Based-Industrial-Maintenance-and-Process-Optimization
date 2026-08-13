# 🔧 AI-Based Predictive Maintenance & Process Intelligence System

> An AI-powered predictive maintenance system that leverages machine learning to analyze industrial sensor data and predict machine failures before they occur — built using the **AI4I 2020 Predictive Maintenance Dataset**.

---

## 📌 Problem Statement

Unplanned machine failures cost manufacturers billions annually in downtime. Traditional maintenance strategies are either:
- **Reactive** — fix after failure (costly & dangerous)
- **Time-based** — scheduled regardless of machine condition (wasteful)

Industrial sensors generate massive real-time data that is largely underutilised. Early failure signals are buried in high-dimensional, class-imbalanced sensor streams with no intelligent system to proactively predict failures before they occur.

---

## 🎯 Objectives

- Build an end-to-end ML pipeline for predictive maintenance
- Leverage the AI4I 2020 synthetic industrial dataset (10,000 rows)
- Handle severe class imbalance (~3.4% failure rate)
- Compare classical & advanced ML/DL models
- Provide SHAP-based model explainability

---

## 📊 Dataset — AI4I 2020

| Property | Detail |
|---|---|
| Source | UCI Machine Learning Repository |
| Size | 10,000 rows × 14 columns |
| Target | `Machine failure` (Binary: 0 = Healthy, 1 = Failed) |
| Class Balance | ~96.6% Healthy / ~3.4% Failure (Highly Imbalanced) |
| Machine Types | L (Low), M (Medium), H (High) |
| Key Features | Air Temp, Process Temp, Rotational Speed, Torque, Tool Wear |
| Failure Subtypes | TWF, HDF, PWF, OSF, RNF |

---

## 📂 Project Structure

```
AI-Based-Predictive-Maintenance-and-Process-Intelligence-System/
│
├── README.md
├── ai4i2020.csv                          # Raw dataset (10,000 rows)
├── generate_ppt.py                       # PPT generation script
│
├── Milestone_1/                          # Data Preprocessing & EDA
│   ├── ai4i2020_preprocessing_eda.ipynb  # Main EDA notebook
│   ├── ai4i2020_clean.csv                # Cleaned dataset
│   ├── X_train_scaled.csv                # Scaled training features
│   ├── X_test_scaled.csv                 # Scaled test features
│   ├── y_train.csv                       # Training labels
│   └── y_test.csv                        # Test labels
│
├── Milestone_2/                          # Classical ML Models
│   ├── logistic_regression.py            # Logistic Regression
│   ├── svm code.py                       # Support Vector Machine
│   ├── decision tree.py                  # Decision Tree
│   ├── logistic_regression_model.pkl     # Saved LR model
│   ├── svm_model.pkl                     # Saved SVM model
│   ├── decision_tree_model.pkl           # Saved DT model
│   └── feature_importance.png            # LR odds-ratio chart
│
└── Milestone_3/                          # Advanced Models
    ├── ensemble_models.py                # Random Forest + Gradient Boosting
    ├── ann.py                            # ANN (MLPClassifier)
    ├── ann_predictive_maintenance.ipynb  # ANN notebook
    ├── best_predictive_maintenance_model.pkl
    ├── ann_model.pkl
    ├── model_comparison.csv
    ├── roc_curves.png
    ├── confusion_matrices.png
    └── shap_summary.png
```

---

## 🔁 Pipeline

```
Raw Dataset (ai4i2020.csv)
        │
        ▼
Milestone 1: EDA + Preprocessing
        │   ├── Label encode Type (L/M/H)
        │   ├── Feature engineering (Temp Difference)
        │   ├── StandardScaler normalisation
        │   └── 80/20 stratified train-test split
        │
        ▼
  X_train_scaled.csv / X_test_scaled.csv / y_train.csv / y_test.csv
        │                           │
        ▼                           ▼
Milestone 2 (Classical ML)   Milestone 3 (Advanced ML)
  ├── Logistic Regression      ├── Random Forest (SMOTE + GridSearchCV)
  ├── SVM (RBF Kernel)         ├── Gradient Boosting (SMOTE + GridSearchCV)
  └── Decision Tree            └── ANN (MLPClassifier 128→64→32)
```

> **All models share a single data source** — the preprocessed CSVs from Milestone 1.

---

## 🤖 Models & Results

### Milestone 2 — Classical ML Models

| Model | Accuracy | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 83.0% | 0.2400 | 0.9069 | 0.3814 |
| SVM (RBF Kernel) | 91.3% | 0.4200 | 0.9685 | 0.6611 |
| Decision Tree | 94.8% | 0.5100 | 0.8432 | 0.7151 |

### Milestone 3 — Advanced Models

| Model | Accuracy | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Random Forest | 97.2% | 0.6543 | 0.9763 | 0.7988 |
| Gradient Boosting | 97.6% | 0.7000 | 0.9661 | 0.8158 |
| **ANN (MLP) ★** | **98.6%** | **0.7680** | **0.9810** | **0.8205** |

> ⭐ **Best Model: ANN** — highest across all metrics. Threshold optimised to 0.55 via F1 sweep.

> **Note:** PR-AUC is the most informative metric here due to heavy class imbalance (28.5:1). A high ROC-AUC alone can be misleading on imbalanced datasets.

---

## 🔬 Key Insights

- **Logistic Regression** — Fast baseline. Low PR-AUC (0.38) reveals weakness on the minority failure class despite decent ROC-AUC.
- **SVM (RBF)** — Non-linear boundary improves F1 to 0.42. PR-AUC jumps to 0.66 — better at catching actual failures.
- **Decision Tree** — Best PR-AUC (0.72) among classical models. Gini splits naturally identify sensor threshold patterns. Fully interpretable.
- **Random Forest** — Ensemble of 200 trees + SMOTE + GridSearch. Robust and stable. PR-AUC 0.80 with 97.2% accuracy.
- **Gradient Boosting** — Sequential boosting + SMOTE. F1 0.70, PR-AUC 0.82. Excellent production candidate.
- **ANN (128→64→32)** — Captures non-linear feature interactions via ReLU layers. Early stopping prevents overfitting. Best across all metrics.

---

## 🛠️ Tech Stack

| Category | Library |
|---|---|
| Data | `pandas`, `numpy` |
| ML Models | `scikit-learn` (LR, SVM, RF, GB, DT, MLP) |
| Imbalance Handling | `imbalanced-learn` (SMOTE) |
| Explainability | `shap` |
| Visualisation | `matplotlib`, `seaborn` |
| Model Persistence | `joblib` (.pkl files) |

---

## 🚀 Future Scope

- **Real-Time Streaming** — Integrate with Apache Kafka/MQTT for live sensor processing and instant alerts
- **LSTM / Transformer** — Model temporal sensor sequences for time-series failure prediction
- **Cloud Deployment** — FastAPI/Flask REST API on AWS/Azure with an operator dashboard
- **Multi-Class Failure** — Extend from binary to predicting specific failure types (TWF, HDF, PWF, OSF)
- **Federated Learning** — Train across multiple plant locations without sharing raw sensor data
- **Digital Twin** — Link ML model with digital twin simulation to validate predictions virtually

---



## ▶️ How to Run

```bash
# Install dependencies
pip install scikit-learn imbalanced-learn shap matplotlib seaborn joblib pandas numpy python-pptx

# Run Milestone 2 models
python "Milestone_2/logistic_regression.py"
python "Milestone_2/svm code.py"
python "Milestone_2/decision tree.py"

# Run Milestone 3 models
python "Milestone_3/ensemble_models.py"
python "Milestone_3/ann.py"

# Generate presentation
python generate_ppt.py
```

> All scripts must be run from the **project root directory**.

---

*Infosys Springboard Internship  — AI4I 2020 Predictive Maintenance Dataset*
