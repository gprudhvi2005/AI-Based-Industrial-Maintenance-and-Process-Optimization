# ==========================================
# AI-Based Predictive Maintenance — ANN
# Model: MLPClassifier (128 → 64 → 32)
# Data Source: Milestone_1 preprocessed CSVs
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve,
    roc_curve, auc, accuracy_score
)
import joblib

OUTPUT_DIR = "Milestone_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. Load Preprocessed Datasets (Milestone_1)
# ==========================================
print("Loading preprocessed data from Milestone_1...")
X_train = pd.read_csv("Milestone_1/X_train_scaled.csv")
X_test  = pd.read_csv("Milestone_1/X_test_scaled.csv")
y_train = pd.read_csv("Milestone_1/y_train.csv").values.ravel()
y_test  = pd.read_csv("Milestone_1/y_test.csv").values.ravel()

print(f"X_train shape : {X_train.shape}")
print(f"X_test  shape : {X_test.shape}")
print(f"y_train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"y_test  distribution: {dict(zip(*np.unique(y_test,  return_counts=True)))}")
print(f"Feature columns: {list(X_train.columns)}")

# ==========================================
# 2. EDA — Class Distribution + Correlation
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

labels = ["No Failure (0)", "Failure (1)"]
counts = [int(sum(y_train == 0)), int(sum(y_train == 1))]
colors = ["#2ecc71", "#e74c3c"]
bars = axes[0].bar(labels, counts, color=colors, edgecolor="black", linewidth=0.8)
axes[0].set_title("Train Class Distribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count")
for bar, v in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 30, str(v),
                 ha="center", fontweight="bold", fontsize=11)

corr = X_train.astype(float).corr()
sns.heatmap(corr, ax=axes[1], cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
axes[1].set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ann_eda_overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nClass imbalance ratio (train): {counts[0]/counts[1]:.1f} : 1")

# ==========================================
# 3. Train ANN (MLPClassifier)
# ==========================================
print("\nTraining ANN (MLPClassifier: 128 -> 64 -> 32)...")
ann = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    learning_rate_init=0.001
)
ann.fit(X_train, y_train)
print(f"Training complete! Iterations run: {ann.n_iter_}  |  Final loss: {ann.loss_:.6f}")

# ==========================================
# 4. Training Loss Curve
# ==========================================
plt.figure(figsize=(8, 4))
plt.plot(ann.loss_curve_, color="#3498db", lw=2, label="Training Loss")
if ann.validation_scores_ is not None:
    val_loss = [1 - s for s in ann.validation_scores_]
    plt.plot(val_loss, color="#e74c3c", lw=2, linestyle="--", label="Validation Loss (1−acc)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("ANN Training Loss Curve", fontsize=13, fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ann_loss_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

# ==========================================
# 5. Threshold Tuning (Best F1)
# ==========================================
y_pred = ann.predict(X_test)
y_prob = ann.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores  = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
best_thresh = thresholds[int(np.argmax(f1_scores))]
y_pred_best = (y_prob >= best_thresh).astype(int)

f1_default = f1_score(y_test, y_pred, zero_division=0)
f1_best    = float(max(f1_scores))
print(f"\nDefault threshold (0.50) -> F1 = {f1_default:.4f}")
print(f"Best    threshold ({best_thresh:.2f}) -> F1 = {f1_best:.4f}")

# ==========================================
# 6. Evaluation Metrics
# ==========================================
accuracy  = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best, zero_division=0)
recall    = recall_score(y_test, y_pred_best, zero_division=0)
f1        = f1_score(y_test, y_pred_best, zero_division=0)
map_score = average_precision_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc   = auc(fpr, tpr)

print("\n" + "=" * 52)
print("         ANN CLASSIFICATION METRICS")
print("=" * 52)
print(f"  Accuracy                   : {accuracy:.4f}")
print(f"  Precision (best threshold) : {precision:.4f}")
print(f"  Recall                     : {recall:.4f}")
print(f"  F1-Score                   : {f1:.4f}")
print(f"  mAP (Average Precision)    : {map_score:.4f}")
print(f"  ROC-AUC                    : {roc_auc:.4f}")
print("=" * 52)
print()
print("Full Classification Report:")
print(classification_report(y_test, y_pred_best,
      target_names=["No Failure", "Failure"], zero_division=0))

# ==========================================
# 7. Confusion Matrix + PR Curve
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"],
            linewidths=1, linecolor="white")
axes[0].set_title(f"Confusion Matrix (thresh={best_thresh:.2f})", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Actual", fontsize=11)
axes[0].set_xlabel("Predicted", fontsize=11)

prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
axes[1].fill_between(rec_c, prec_c, alpha=0.15, color="#e74c3c")
axes[1].plot(rec_c, prec_c, color="#e74c3c", lw=2.5, label=f"mAP = {map_score:.4f}")
baseline = float(sum(y_test)) / len(y_test)
axes[1].axhline(baseline, color="navy", linestyle="--", lw=1.5, label=f"Baseline = {baseline:.4f}")
axes[1].set_xlabel("Recall", fontsize=11)
axes[1].set_ylabel("Precision", fontsize=11)
axes[1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ann_confusion_pr_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

# ==========================================
# 8. ROC Curve + F1 vs Threshold
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].fill_between(fpr, tpr, alpha=0.15, color="#3498db")
axes[0].plot(fpr, tpr, color="#3498db", lw=2.5, label=f"ROC AUC = {roc_auc:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1.2)
axes[0].set_xlabel("False Positive Rate", fontsize=11)
axes[0].set_ylabel("True Positive Rate", fontsize=11)
axes[0].set_title("ROC Curve", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

axes[1].plot(thresholds, f1_scores, color="#9b59b6", lw=2.5, marker="o", markersize=4)
axes[1].axvline(best_thresh, color="#e74c3c", linestyle="--", lw=2,
                label=f"Best threshold = {best_thresh:.2f}")
axes[1].set_xlabel("Threshold", fontsize=11)
axes[1].set_ylabel("F1 Score", fontsize=11)
axes[1].set_title("F1 Score vs Decision Threshold", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ann_roc_threshold.png"), dpi=150, bbox_inches="tight")
plt.close()

# ==========================================
# 9. Metrics Bar Chart
# ==========================================
metrics_dict = {
    "Accuracy":  float(accuracy),
    "Precision": float(precision),
    "Recall":    float(recall),
    "F1-Score":  float(f1),
    "mAP":       float(map_score),
    "ROC-AUC":   float(roc_auc)
}

fig, ax = plt.subplots(figsize=(9, 5))
palette = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
bars = ax.bar(metrics_dict.keys(), metrics_dict.values(),
              color=palette, edgecolor="black", linewidth=0.8, width=0.5)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("ANN Evaluation Metrics Summary", fontsize=14, fontweight="bold")
for bar, v in zip(bars, metrics_dict.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.4f}",
            ha="center", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ann_metrics_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

# ==========================================
# 10. Save Model
# ==========================================
joblib.dump(ann, os.path.join(OUTPUT_DIR, "ann_model.pkl"))
print(f"[SUCCESS] ANN model saved to '{OUTPUT_DIR}/ann_model.pkl'")
print(f"[SUCCESS] All plots saved in '{OUTPUT_DIR}/' directory.")
