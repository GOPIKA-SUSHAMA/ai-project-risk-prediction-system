"""Evaluate multi-output classification + regression."""

from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.model_selection import train_test_split

from generate_dataset import make_dataset


def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def main() -> None:
    # Load models
    enc = joblib.load("../models/ordinal_encoder.joblib")
    rf_clf = joblib.load("../models/random_forest_multioutput_classifier.joblib")
    rf_reg = joblib.load("../models/random_forest_multioutput_regressor.joblib")

    df = make_dataset(n=5000, seed=42)
    df[["industry"]] = enc.transform(df[["industry"]])

    X = df.drop(columns=["project_id", "risk_type", "impact", "probability", "response"])
    y_type = df["risk_type"]
    y_imp = df["impact"]
    y_prob = df["probability"]
    y_resp = df["response"]

    X_train, X_test, yt_tr, yt_te, yi_tr, yi_te, yp_tr, yp_te, yr_tr, yr_te = train_test_split(
        X, y_type, y_imp, y_prob, y_resp, test_size=0.2, random_state=42
    )

    pred_type_rf, pred_resp_rf = rf_clf.predict(X_test).T
    pred_imp_rf, pred_prob_rf = rf_reg.predict(X_test).T

    # Hamming loss + exact match (for the 2 classification outputs)
    type_classes = sorted(yt_te.unique())
    resp_classes = sorted(yr_te.unique())

    y_true_type = np.array([type_classes.index(l) for l in yt_te])
    y_true_resp = np.array([resp_classes.index(l) for l in yr_te])
    y_pred_type = np.array([type_classes.index(l) for l in pred_type_rf])
    y_pred_resp = np.array([resp_classes.index(l) for l in pred_resp_rf])

    y_true_clf = np.column_stack([y_true_type, y_true_resp])
    y_pred_clf = np.column_stack([y_pred_type, y_pred_resp])

    incorrect = (y_true_clf != y_pred_clf).sum()
    total = y_true_clf.size
    hamming_loss = incorrect / total
    exact_match = (y_true_clf == y_pred_clf).all(axis=1).mean()

    out = {
        "RF risk_type": evaluate_classification(yt_te, pred_type_rf),
        "RF response": evaluate_classification(yr_te, pred_resp_rf),
        "RF impact": evaluate_regression(yi_te, pred_imp_rf),
        "RF probability": evaluate_regression(yp_te, pred_prob_rf),
        "multioutput": {
            "hamming_loss": float(hamming_loss),
            "exact_match": float(exact_match),
            "mae_impact": float(mean_absolute_error(yi_te, pred_imp_rf)),
            "mae_probability": float(mean_absolute_error(yp_te, pred_prob_rf)),
        },
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import json
    main()
