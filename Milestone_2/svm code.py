# ==========================================
# AI-Based Predictive Maintenance using SVM
# Data Source: Milestone_1 preprocessed CSVs
# ==========================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_auc_score, average_precision_score
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
# 2. Train SVM Model (RBF Kernel)
# ==========================================
print("\nTraining SVM model (RBF kernel)...")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
          probability=True, random_state=42)
svm.fit(X_train, y_train)
print("SVM Model Trained Successfully!")

# ==========================================
# 3. Predictions
# ==========================================
y_pred  = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)[:, 1]

# ==========================================
# 4. Evaluation Metrics
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

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
disp.plot(cmap='Blues')
plt.title("SVM - Confusion Matrix")
plt.tight_layout()
plt.savefig('Milestone_2/svm_confusion_matrix.png', dpi=150)
print("[SUCCESS] Saved confusion matrix to 'Milestone_2/svm_confusion_matrix.png'")
plt.show()

# ==========================================
# 6. Scatter Visualization (Air Temp vs Process Temp)
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(X_test['Air temperature'], X_test['Process temperature'],
            c=y_test, cmap='coolwarm', alpha=0.6)
plt.xlabel("Air Temperature (Scaled)")
plt.ylabel("Process Temperature (Scaled)")
plt.title("SVM — Machine Failure Distribution on Test Set")
plt.colorbar(label="Machine Failure")
plt.tight_layout()
plt.savefig('Milestone_2/svm_scatter.png', dpi=150)
print("[SUCCESS] Saved scatter plot to 'Milestone_2/svm_scatter.png'")
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
joblib.dump(svm, 'Milestone_2/svm_model.pkl')
print("\n[SUCCESS] Saved model to 'Milestone_2/svm_model.pkl'")
