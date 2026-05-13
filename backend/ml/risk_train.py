from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from backend.ml.risk_features import RISK_FEATURE_NAMES, make_risk_training_dataset

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "risk_classifier.joblib"


def train_risk_model(random_state: int = 42) -> dict:
    X, y = make_risk_training_dataset(n_steps=2000, random_state=random_state)

    cutoff = int(len(X) * 0.80)
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=0.1,
        random_state=random_state,
    )
    model.fit(X_train_sc, y_train)

    accuracy = float(accuracy_score(y_test, model.predict(X_test_sc)))

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": RISK_FEATURE_NAMES,
        "model_version": "risk-v1",
        "horizon_days": 5,
        "threshold": -0.02,
        "test_accuracy": accuracy,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    return artifact


if __name__ == "__main__":
    result = train_risk_model()
    print(f"Model saved : {MODEL_PATH}")
    print(f"Version     : {result['model_version']}")
    print(f"Test accuracy: {result['test_accuracy']:.4f}")
    print(f"Features    : {result['feature_names']}")
