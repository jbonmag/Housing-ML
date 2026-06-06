"""Train a housing price model and generate a Kaggle submission file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model for the Ames housing price dataset.")
    parser.add_argument("--train", default="train.csv", help="Training CSV path.")
    parser.add_argument("--test", default="test.csv", help="Test CSV path.")
    parser.add_argument("--output", default="submission.csv", help="Submission CSV path.")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    return parser


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def rmsle_cv(model: Pipeline, X: pd.DataFrame, y_log: pd.Series, folds: int) -> np.ndarray:
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = -cross_val_score(model, X, y_log, scoring="neg_mean_squared_error", cv=kf)
    return np.sqrt(scores)


def main() -> None:
    args = build_parser().parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    X = train.drop(columns=["SalePrice"])
    y_log = np.log1p(train["SalePrice"])

    test_ids = test["Id"]
    model = build_pipeline(X)

    cv_scores = rmsle_cv(model, X, y_log, args.folds)
    print("CV RMSLE scores:", np.round(cv_scores, 5))
    print(f"Mean CV RMSLE: {cv_scores.mean():.5f}")

    model.fit(X, y_log)
    predictions = np.expm1(model.predict(test))
    predictions = np.maximum(predictions, 0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
    submission.to_csv(output, index=False)
    print(f"Submission written to: {output}")


if __name__ == "__main__":
    main()
