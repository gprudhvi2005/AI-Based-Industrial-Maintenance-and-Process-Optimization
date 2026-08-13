# ==========================================
# AI-Based Predictive Maintenance — Decision Tree
# Data Source: Milestone_1 preprocessed CSVs
# ==========================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay, average_precision_score
)

# ==========================================
# 1. Load Preprocessed Datasets (Milestone_1)
# ==========================================
print("Loading preprocessed data from Milestone_1...")
X_train = pd.read_csv('Milestone_1/X_train_scaled.csv')
X_test  = pd.read_csv('Milestone_1/X_test_scaled.csv')
y_train = pd.read_csv('Milestone_1/y_train.csv').values.ravel()
y_test  = pd.read_csv('Milestone_1/y_test.csv').values.ravel()

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Class distribution in y_train: {dict(zip(*np.unique(y_train, return_counts=True)))}")

# ==========================================
# 2. Train Decision Tree Model
# ==========================================
print("\nTraining Decision Tree Classifier...")
model = DecisionTreeClassifier(
    max_depth=6,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
print("Decision Tree Trained Successfully!")

# ==========================================
# 3. Predictions
# ==========================================
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ==========================================
# 4. Evaluation Metrics
# ==========================================
accuracy = (y_pred == y_test).mean()
print(f"\nTraining Accuracy: {accuracy:.4f}")

if len(np.unique(y_test)) > 1:
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC Score : {average_precision_score(y_test, y_proba):.4f}")

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# 5. Confusion Matrix Plot
# ==========================================
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Greens')
plt.title("Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.savefig('Milestone_2/dt_confusion_matrix.png', dpi=150)
print("[SUCCESS] Saved confusion matrix to 'Milestone_2/dt_confusion_matrix.png'")
plt.show()

# ==========================================
# 6. Feature Importance Plot
# ==========================================
feature_names = X_train.columns.tolist()
importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='mediumseagreen')
plt.xlabel('Gini Importance')
plt.title('Decision Tree — Feature Importances')
plt.tight_layout()
plt.savefig('Milestone_2/dt_feature_importance.png', dpi=150)
print("[SUCCESS] Saved feature importance to 'Milestone_2/dt_feature_importance.png'")
plt.show()

# ==========================================
# 7. Top 10 Highest Risk Predictions
# ==========================================
results_df = pd.DataFrame({
    'Actual Label':    y_test,
    'Predicted Label': y_pred,
    'Failure Risk (%)': np.round(y_proba * 100, 2)
})
print("\n=== Top 10 Highest Risk Predictions ===")
print(results_df.sort_values(by='Failure Risk (%)', ascending=False).head(10).to_string(index=False))

# ==========================================
# 8. Save Trained Model
# ==========================================
joblib.dump(model, 'Milestone_2/decision_tree_model.pkl')
print("\n[SUCCESS] Saved model to 'Milestone_2/decision_tree_model.pkl'")
