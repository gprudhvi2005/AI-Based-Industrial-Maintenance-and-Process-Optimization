import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ==========================================
# 1. Load Preprocessed Datasets
# ==========================================
X_train_scaled = pd.read_csv('Milestone_1/X_train_scaled.csv')
X_test_scaled  = pd.read_csv('Milestone_1/X_test_scaled.csv')
y_train        = pd.read_csv('Milestone_1/y_train.csv').values.ravel()
y_test         = pd.read_csv('Milestone_1/y_test.csv').values.ravel()

# ==========================================
# 2. Initialize Model & Fit on Training Data
# ==========================================
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# ==========================================
# 3. Predict on Test Data
# ==========================================
y_pred = model.predict(X_test_scaled)                 # Default 0.5 threshold
y_proba = model.predict_proba(X_test_scaled)[:, 1]     # Failure probability scores

# ==========================================
# 4. Print Confusion Matrix & Metrics
# ==========================================
print("=== Confusion Matrix (Threshold 0.50) ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))

# ROC-AUC score requires both classes (0 and 1) in y_test
if len(np.unique(y_test)) > 1:
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# ==========================================
# 5. Evaluate with Custom Threshold (0.30)
# ==========================================
custom_threshold = 0.30
y_pred_custom = (y_proba >= custom_threshold).astype(int)

print(f"\n=== Results with Custom Threshold ({custom_threshold}) ===")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom, zero_division=0))

# ==========================================
# 6. Rank High-Risk Maintenance Predictions
# ==========================================
results_df = pd.DataFrame({
    'Actual Label': y_test,
    'Predicted Label': y_pred,  # <---  standard 0.5 threshold here
    'Failure Risk (%)': np.round(y_proba * 100, 2)
})

print("\n=== Top 10 Highest Risk Predictions ===")
print(results_df.sort_values(by='Failure Risk (%)', ascending=False).head(10).to_string(index=False))

# ==========================================
# 7. ADDITION: Save Trained Model Artifacts
# ==========================================
joblib.dump(model, 'Milestone_2/logistic_regression_model.pkl')
print("\n[SUCCESS] Saved model to 'Milestone_2/logistic_regression_model.pkl'")

# ==========================================
# 8. ADDITION: Feature Importance (Odds Ratios)
# ==========================================
feature_names = X_train_scaled.columns
odds_ratios = np.exp(model.coef_[0])

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Odds Ratio': odds_ratios
}).sort_values(by='Odds Ratio', ascending=True)

print("\n=== Feature Importance (Odds Ratios) ===")
print(importance_df.sort_values(by='Odds Ratio', ascending=False).to_string(index=False))

# Plot Feature Importance
plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Odds Ratio'], color='skyblue')
plt.axvline(x=1.0, color='red', linestyle='--', label='Baseline Risk (Odds Ratio = 1.0)')
plt.xlabel('Odds Ratio (Impact on Failure Probability)')
plt.title('Logistic Regression - Feature Importance')
plt.legend()
plt.tight_layout()

# Save plot asset for project documentation
plt.savefig('Milestone_2/feature_importance.png')
print("[SUCCESS] Saved plot to 'Milestone_2/feature_importance.png'")
plt.show()