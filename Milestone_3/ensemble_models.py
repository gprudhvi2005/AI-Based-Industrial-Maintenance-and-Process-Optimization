import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Resampling & Tuning
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score
)

# Explainability
import shap

# Output directory for artifacts
OUTPUT_DIR = "Milestone_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load preprocessed data from Milestone_1 folder
print("Loading preprocessed dataset...")
X_train = pd.read_csv("Milestone_1/X_train_scaled.csv")
X_test = pd.read_csv("Milestone_1/X_test_scaled.csv")
y_train = pd.read_csv("Milestone_1/y_train.csv").values.ravel()
y_test = pd.read_csv("Milestone_1/y_test.csv").values.ravel()

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Original Training Balance (0/1): {np.bincount(y_train)}")

# 2. Handle Class Imbalance using SMOTE
print("\nApplying SMOTE resampling to training data...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Resampled Training Balance (0/1): {np.bincount(y_train_res)}")

# 3. Define Models and Hyperparameter Grids
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_and_params = {
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [8, 12, None],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced", None]
        }
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0]
        }
    )
}

best_models = {}

# 4. Perform Hyperparameter Optimization
for name, (model, params) in models_and_params.items():
    print(f"\nTuning {name} with GridSearchCV...")
    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_res, y_train_res)
    best_models[name] = grid.best_estimator_
    print(f"Best {name} Params: {grid.best_params_}")
    print(f"Best CV F1-Score: {grid.best_score_:.4f}")

# 5. Model Evaluation & Metrics Table
results = []
plt.figure(figsize=(9, 6))

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)

    results.append({
        "Model":     name,
        "Accuracy":  round(acc, 4),
        "Precision": round(prec, 4),
        "Recall":    round(rec, 4),
        "F1-Score":  round(f1, 4),
        "ROC-AUC":   round(roc_auc, 4),
        "PR-AUC":    round(pr_auc, 4)
    })
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Milestone 3 Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=300)
plt.close()

# Export Summary CSV
comparison_df = pd.DataFrame(results)
comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print("\n--- Model Performance Comparison ---")
print(comparison_df.to_string(index=False))

# 6. Confusion Matrices Plot
fig, axes = plt.subplots(1, len(best_models), figsize=(5 * len(best_models), 4))
for ax, (name, model) in zip(axes, best_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"{name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=300)
plt.close()

# 7. Model Interpretability with SHAP
best_model_name = comparison_df.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]
best_model = best_models[best_model_name]
print(f"\nCalculating SHAP Feature Importance for top model: {best_model_name}...")

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap_vals_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_to_plot, X_test, show=False)
plt.title(f"SHAP Summary Plot ({best_model_name})", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=300)
plt.close()

# 8. Save Best Model Artifact
model_export_path = os.path.join(OUTPUT_DIR, "best_predictive_maintenance_model.pkl")
joblib.dump(best_model, model_export_path)
print(f"\n[SUCCESS] Milestone 3 complete. Artifacts saved in '{OUTPUT_DIR}' directory.")