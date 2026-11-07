# Housing-ML

Predicts house prices in Ames, Iowa, for the Kaggle [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/competitions/home-data-for-ml-course): 1,460 sold houses to learn from, 1,459 to price, 79 columns describing each one.

| model | RMSE of log price |
|---|---|
| **Ridge + gradient boosting (average)** | **0.1128** ± 0.007 |
| Ridge (linear) | 0.1160 ± 0.007 |
| Gradient boosting | 0.1235 ± 0.007 |
| Random forest | 0.1296 ± 0.005 |
| Previous version of this repository (random forest, raw columns) | 0.1384 ± 0.007 |

The error is the competition metric, measured with 5-fold cross-validation repeated 3 times, on the same 1,458 houses and the same folds for every model. An RMSE of 0.11 on log price corresponds to a typical miss of about 11%. Out of fold, the best model has a median error of 5.6%, and 76% of houses are priced within 10%. The submission is not scored here; Kaggle scores it when it is uploaded.

## What changed, and why it helps

The previous version fed the raw columns to a random forest. The error is now 18% lower, for two reasons. Preparing the data takes the same kind of random forest from 0.1384 to 0.1296. On the prepared data a linear model does better still (0.1160), and averaging it with gradient boosting gives 0.1128.

- **Missing values that mean "none".** According to `data_description.txt`, a missing pool, alley, fence, fireplace, garage or basement value means the house has none. Those become `"None"` (or 0 for areas and counts) instead of being filled with the most common value.
- **Ordered scales as numbers.** Quality and condition (`Po` < `Fa` < `TA` < `Gd` < `Ex`) and finish levels keep their order.
- **Codes as categories.** `MSSubClass` (dwelling type), `MoSold` and `YrSold` are numbers that work as labels. `Id` is dropped: it is a row number, not a property of the house.
- **New features.** Total square footage, total bathrooms, total porch area, age at sale, years since remodelling, and flags for garage, basement and second floor.
- **Two outliers removed.** Two partial sales of houses with more than 4,000 sq ft that sold for under $200,000. The dataset's author recommends removing them.
- **A linear model.** Once the numeric columns are on a log scale, ridge regression beats both tree models. Prices grow smoothly with size and quality. Averaging ridge with gradient boosting is better than either alone.

## The notebook

[`housing.ipynb`](housing.ipynb) is saved with its outputs, so it can be read on GitHub without running anything:

1. the price distribution and why the model learns its logarithm;
2. correlations, living area against price (with the outliers marked), and the median price by neighborhood;
3. what the missing values mean;
4. the features;
5. the model comparison;
6. predicted against actual prices and the error distribution, out of fold;
7. the submission.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/housing.py
```

The script prints the comparison table and writes `submission.csv`. It takes about 4 minutes, mostly for the 15 cross-validation fits of each model. Options: `--data-dir`, `--output`, `--repeats`.

## Files

```text
data/train.csv               sold houses with SalePrice (Kaggle)
data/test.csv                houses to price (Kaggle)
data/sample_submission.csv   Kaggle's submission format
data/data_description.txt    meaning of every column and value
src/housing.py               cleaning, features, pipelines, comparison, submission
housing.ipynb                exploration and results, executed
```

The data comes from Dean De Cock's [Ames Housing dataset](https://jse.amstat.org/v19n3/decock.pdf) as distributed by Kaggle, and is subject to the competition rules.
