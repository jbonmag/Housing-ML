"""Predict Ames house prices: cleaning, feature engineering, model comparison and a Kaggle submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42

# In data_description.txt, a missing value in these columns means "the house has none".
NONE_CATEGORICAL = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
ZERO_NUMERIC = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]
# Quality and condition scales, worst to best. They are ordered, so they are encoded as numbers.
QUALITY = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
ORDINAL = {
    "ExterQual": QUALITY, "ExterCond": QUALITY, "BsmtQual": QUALITY, "BsmtCond": QUALITY,
    "HeatingQC": QUALITY, "KitchenQual": QUALITY, "FireplaceQu": QUALITY,
    "GarageQual": QUALITY, "GarageCond": QUALITY, "PoolQC": QUALITY,
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
}
# Stored as numbers but they are codes, not quantities.
CODES_AS_CATEGORIES = ["MSSubClass", "MoSold", "YrSold"]
# Almost constant: Utilities has a single "NoSeWa" row in train and none in test.
DROP = ["Id", "Utilities"]


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Drop the two very large, very cheap houses that the dataset author recommends removing."""
    return train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300_000))].reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and derive features using only each row's own values, so train and test are treated alike."""
    out = df.drop(columns=[c for c in DROP + ["SalePrice"] if c in df.columns]).copy()
    out[NONE_CATEGORICAL] = out[NONE_CATEGORICAL].fillna("None")
    out[ZERO_NUMERIC] = out[ZERO_NUMERIC].fillna(0)
    out["GarageYrBlt"] = out["GarageYrBlt"].fillna(out["YearBuilt"])
    for col in CODES_AS_CATEGORIES:
        out[col] = out[col].astype(str)

    out["TotalSF"] = out["TotalBsmtSF"] + out["1stFlrSF"] + out["2ndFlrSF"]
    out["TotalBath"] = (
        out["FullBath"] + 0.5 * out["HalfBath"] + out["BsmtFullBath"] + 0.5 * out["BsmtHalfBath"]
    )
    out["TotalPorchSF"] = (
        out["OpenPorchSF"] + out["EnclosedPorch"] + out["3SsnPorch"] + out["ScreenPorch"] + out["WoodDeckSF"]
    )
    out["HouseAge"] = out["YrSold"].astype(int) - out["YearBuilt"]
    out["YearsSinceRemodel"] = out["YrSold"].astype(int) - out["YearRemodAdd"]
    out["HasGarage"] = (out["GarageArea"] > 0).astype(int)
    out["HasBasement"] = (out["TotalBsmtSF"] > 0).astype(int)
    out["Has2ndFloor"] = (out["2ndFlrSF"] > 0).astype(int)
    return out


def signed_log1p(x):
    """log1p that also accepts the few negative values (e.g. a house sold before its build year)."""
    return np.sign(x) * np.log1p(np.abs(x))


def nominal_columns(X: pd.DataFrame) -> list[str]:
    """Text columns that are not ordered scales."""
    return [c for c in X.select_dtypes(include=["object", "string"]).columns if c not in ORDINAL]


def build_preprocessor(linear: bool) -> ColumnTransformer:
    """Linear models get log-transformed, scaled numbers; tree models get them as they are."""
    ordinal_cols = list(ORDINAL)
    ordinal = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(categories=[ORDINAL[c] for c in ordinal_cols],
                                      handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    nominal = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if linear:
        numeric_steps += [("log", FunctionTransformer(signed_log1p)), ("scale", StandardScaler())]
    numeric = Pipeline(numeric_steps)

    return ColumnTransformer(
        [
            ("ord", ordinal, ordinal_cols),
            ("nom", nominal, nominal_columns),
            ("num", numeric, make_column_selector(dtype_include="number")),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    ridge = Pipeline([("prep", build_preprocessor(linear=True)), ("model", RidgeCV(alphas=np.logspace(-1, 3, 30)))])
    boosting = Pipeline(
        [
            ("prep", build_preprocessor(linear=False)),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.03, max_iter=1500, max_leaf_nodes=15, min_samples_leaf=10,
                l2_regularization=1.0, random_state=RANDOM_STATE)),
        ]
    )
    forest = Pipeline(
        [
            ("prep", build_preprocessor(linear=False)),
            ("model", RandomForestRegressor(n_estimators=400, max_features=0.33, random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    return {
        "Ridge + boosting (average)": VotingRegressor([("ridge", ridge), ("boosting", boosting)]),
        "Ridge (linear)": ridge,
        "Gradient boosting": boosting,
        "Random forest": forest,
    }


def previous_version() -> Pipeline:
    """The model this repository used before: random forest on the raw columns, Id included."""
    return Pipeline(
        [
            ("prep", ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), make_column_selector(dtype_include="number")),
                    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                      ("encode", OneHotEncoder(handle_unknown="ignore"))]),
                     make_column_selector(dtype_include=["object", "string"])),
                ]
            )),
            ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )


def rmse_log(model, X, y_log, n_repeats: int = 3) -> np.ndarray:
    """Cross-validated RMSE of log(price), the Kaggle metric."""
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    return -cross_val_score(model, X, y_log, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)


def compare_models(train: pd.DataFrame, n_repeats: int = 3) -> pd.DataFrame:
    """All models are scored on the same rows (outliers removed) and the same folds."""
    clean = remove_outliers(train)
    X, y_log = add_features(clean), np.log1p(clean["SalePrice"])
    rows = []
    scores = rmse_log(previous_version(), clean.drop(columns="SalePrice"), y_log, n_repeats)
    rows.append({"model": "Previous version (random forest, raw columns)", "rmse_log": scores.mean(), "std": scores.std()})
    for name, model in candidate_models().items():
        scores = rmse_log(model, X, y_log, n_repeats)
        rows.append({"model": name, "rmse_log": scores.mean(), "std": scores.std()})
    return pd.DataFrame(rows).sort_values("rmse_log").reset_index(drop=True)


def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame, name: str) -> tuple[Pipeline, pd.DataFrame]:
    clean = remove_outliers(train)
    model = candidate_models()[name].fit(add_features(clean), np.log1p(clean["SalePrice"]))
    prices = np.expm1(model.predict(add_features(test)))
    return model, pd.DataFrame({"Id": test["Id"], "SalePrice": prices})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Folder with train.csv and test.csv.")
    parser.add_argument("--output", default="submission.csv", help="Where to write the Kaggle submission.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats of 5-fold cross-validation.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    results = compare_models(train, args.repeats)
    print(results.to_string(index=False, float_format="%.4f"))

    best = next(m for m in results["model"] if m in candidate_models())
    _, submission = fit_and_predict(train, test, best)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Best model: {best}. Submission written to {output}")


if __name__ == "__main__":
    main()
