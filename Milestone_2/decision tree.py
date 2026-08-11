"""Simple Decision Tree example for the PredictiveMaintenance workspace."""

from __future__ import annotations

import sys


def load_data():
    try:
        from sklearn.datasets import load_iris

        data = load_iris()
        return data.data, data.target, data.feature_names, data.target_names
    except ImportError:
        print("scikit-learn is not installed. Install it with: pip install scikit-learn")
        sys.exit(1)


def train_decision_tree(X, y):
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y, target_names):
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    print(f"Training accuracy: {accuracy:.2%}")
    print("Example prediction: ")
    for i in range(min(5, len(X))):
        print(f"  sample {i}: predicted={target_names[predictions[i]]}, actual={target_names[y[i]]}")


def main():
    X, y, feature_names, target_names = load_data()
    print("Loaded dataset with features:", feature_names)
    model = train_decision_tree(X, y)
    evaluate_model(model, X, y, target_names)


if __name__ == "__main__":
    main()
