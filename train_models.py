"""Train multi-output models for project risk prediction."""

from __future__ import annotations
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from generate_dataset import make_dataset


def main() -> None:
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    df = make_dataset(n=5000, seed=42)
    df.to_csv("../data/synthetic_project_risk_dataset.csv", index=False)

    # Encode categorical
    enc = OrdinalEncoder()
    df[["industry"]] = enc.fit_transform(df[["industry"]])

    X = df.drop(columns=["project_id", "risk_type", "impact", "probability", "response"])
    y_type = df["risk_type"]
    y_imp = df["impact"]
    y_prob = df["probability"]
    y_resp = df["response"]

    X_train, X_test, yt_tr, yt_te, yi_tr, yi_te, yp_tr, yp_te, yr_tr, yr_te = train_test_split(
        X, y_type, y_imp, y_prob, y_resp, test_size=0.2, random_state=42
    )

    rf_clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
    rf_reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))

    dt_clf = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
    dt_reg = MultiOutputRegressor(DecisionTreeRegressor(random_state=42))

    rf_clf.fit(X_train, pd.concat([yt_tr, yr_tr], axis=1))
    rf_reg.fit(X_train, pd.concat([yi_tr, yp_tr], axis=1))

    dt_clf.fit(X_train, pd.concat([yt_tr, yr_tr], axis=1))
    dt_reg.fit(X_train, pd.concat([yi_tr, yp_tr], axis=1))

    joblib.dump(enc, "../models/ordinal_encoder.joblib")
    joblib.dump(rf_clf, "../models/random_forest_multioutput_classifier.joblib")
    joblib.dump(rf_reg, "../models/random_forest_multioutput_regressor.joblib")
    joblib.dump(dt_clf, "../models/decision_tree_multioutput_classifier.joblib")
    joblib.dump(dt_reg, "../models/decision_tree_multioutput_regressor.joblib")

    print("Saved models to ../models/")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")


if __name__ == "__main__":
    main()
