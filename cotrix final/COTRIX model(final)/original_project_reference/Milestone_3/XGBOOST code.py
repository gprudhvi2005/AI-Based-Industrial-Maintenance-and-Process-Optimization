import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Load AI4I 2020 Predictive Maintenance Dataset
# --------------------------------------------------

df = pd.read_csv("ai4i2020.csv")

# Clean column names - remove special characters that XGBoost doesn't allow
df.columns = df.columns.str.replace(r'[\[\]<>]', '', regex=True).str.replace(' ', '_')

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())


# --------------------------------------------------
# 2. Remove unnecessary columns
# --------------------------------------------------

df = df.drop(columns=["UDI", "Product_ID"])


# --------------------------------------------------
# 3. Convert categorical Type column
# --------------------------------------------------

df = pd.get_dummies(df, columns=["Type"], drop_first=True)


# --------------------------------------------------
# 4. Separate input features and target
# --------------------------------------------------

X = df.drop(columns=["Machine_failure"])
y = df["Machine_failure"]


# --------------------------------------------------
# 5. Split data into training and testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# --------------------------------------------------
# 6. Create XGBoost model
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)


# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nXGBoost model training completed.")


# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 9. Calculate performance metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)


print("\n======================================")
print("       XGBOOST MODEL RESULTS")
print("======================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# --------------------------------------------------
# 10. Classification Report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Failure", "Failure"],
        zero_division=0
    )
)


# --------------------------------------------------
# 11. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Failure", "Failure"],
    yticklabels=["No Failure", "Failure"]
)

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 12. Feature Importance
# --------------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10, 6))

importance.plot(kind="bar")

plt.title("XGBoost Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 13. Final Summary
# --------------------------------------------------

print("\n======================================")
print("          FINAL SUMMARY")
print("======================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")