# Housing ML

Machine learning project for predicting housing prices in Ames, Iowa, based on Kaggle's Home Data for ML Course challenge. The repository includes the train and test CSV files, an exploratory notebook and a reproducible script to train the model and generate `submission.csv`.

## Objective

Predict `SalePrice` from numerical and categorical property features. The competition metric is RMSE over the logarithm of the sale price, which is equivalent to RMSLE on the original price scale.

## Structure

```text
.
|-- train.csv
|-- test.csv
|-- sample_submission.csv
|-- data_description.txt
|-- modelo_predictivo.ipynb
|-- src/
|   `-- train_model.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Approach

The main script uses a scikit-learn pipeline:

1. Split predictors and target.
2. Apply `log1p` to `SalePrice`.
3. Impute missing values.
4. One-hot encode categorical variables.
5. Train a `RandomForestRegressor`.
6. Run cross-validation with RMSLE.
7. Generate `submission.csv`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train the model and generate the submission file:

```bash
python src/train_model.py
```

Use custom paths:

```bash
python src/train_model.py --train train.csv --test test.csv --output outputs/submission.csv
```

## Result

The script prints cross-validation scores and creates a CSV in Kaggle submission format:

```csv
Id,SalePrice
1461,169000.1
1462,187724.1
```

## Possible Improvements

- Hyperparameter tuning with `RandomizedSearchCV`.
- Comparison against boosting models.
- Feature importance analysis.
- Outlier validation and skewed feature transformation.

## Security

The project does not require credentials or tokens. Generated artifacts are excluded from Git through `.gitignore`.
