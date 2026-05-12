from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from backend.ml.features import FEATURE_NAMES, make_training_dataset

MODEL_VERSION = "v1"
MODEL_FILENAME = "signal_classifier.joblib"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME


def train_model(random_state: int = 42) -> dict:
    X, y = make_training_dataset(n_samples=1000, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    accuracy = float(accuracy_score(y_test, clf.predict(X_test)))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "model_version": MODEL_VERSION,
            "feature_names": FEATURE_NAMES,
            "accuracy": accuracy,
        },
        MODEL_PATH,
    )

    return {
        "model_path": str(MODEL_PATH),
        "model_version": MODEL_VERSION,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    result = train_model()
    print(result)
