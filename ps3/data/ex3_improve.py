import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import load_transform, create_sample_split


# ----------------------------------------------------------------------
# Helper(s)
# ----------------------------------------------------------------------
TWEEDIE_POWER = 1.5
tweedie_family = TweedieDistribution(TWEEDIE_POWER)


def print_deviance(name, y_true, y_pred, exposure):
    """Convenience function to print average Tweedie deviance."""
    dev = tweedie_family.deviance(y_true, y_pred, sample_weight=exposure)
    avg_dev = dev / exposure.sum()
    print(f"{name:>15}: {avg_dev: .4f}")


# ----------------------------------------------------------------------
# 1. Load and transform the French Motor data
# ----------------------------------------------------------------------
df = load_transform()

print("Raw data after load_transform:")
print(df.shape)
print(df.head())

# ----------------------------------------------------------------------
# 2. Create train / test split using `create_sample_split`
#    (adding a 'sample' column to df)
# ----------------------------------------------------------------------
df = create_sample_split(df)

assert "sample" in df.columns, "create_sample_split() must add a 'sample' column"

train_mask = df["sample"] == "train"
test_mask = df["sample"] == "test"

df_train = df.loc[train_mask].copy()
df_test = df.loc[test_mask].copy()

print("\nTrain / test sizes:")
print("train:", df_train.shape, "test:", df_test.shape)

# outcome & weights
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
weight = df["Exposure"].values

y_train = y[train_mask]
y_test = y[test_mask]
w_train = weight[train_mask]
w_test = weight[test_mask]

# ----------------------------------------------------------------------
# 3. Benchmark Tweedie GLM (as in glum tutorial)
# ----------------------------------------------------------------------
categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]
numeric_cols = ["BonusMalus", "Density"]

predictors = categoricals + numeric_cols

glm_categorizer = Categorizer(columns=categoricals)

X_train_glm = glm_categorizer.fit_transform(df_train[predictors])
X_test_glm = glm_categorizer.transform(df_test[predictors])

glm_benchmark = GeneralizedLinearRegressor(
    family=tweedie_family,
    l1_ratio=1.0,
    fit_intercept=True,
)

glm_benchmark.fit(X_train_glm, y_train, sample_weight=w_train)

df_train["pp_glm_benchmark"] = glm_benchmark.predict(X_train_glm)
df_test["pp_glm_benchmark"] = glm_benchmark.predict(X_test_glm)

print("\n== Benchmark Tweedie GLM ==")
print_deviance("train deviance", y_train, df_train["pp_glm_benchmark"], w_train)
print_deviance("test deviance", y_test, df_test["pp_glm_benchmark"], w_test)

print(
    "Total claim amount (test) – observed vs predicted:",
    df_test["ClaimAmountCut"].sum(),
    (df_test["Exposure"] * df_test["pp_glm_benchmark"]).sum(),
)

# ----------------------------------------------------------------------
# 4. Improved parametric model:
#    adding splines for BonusMalus and Density via a pipeline
# ----------------------------------------------------------------------


spline_cols = ["BonusMalus", "Density"]

# preprocessor: splines for numeric + one-hot for categoricals
numeric_spline_pipeline = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        (
            "splines",
            SplineTransformer(
                degree=3,
                n_knots=5,
                knots="quantile",
                include_bias=False,  # only one intercept overall (from GLM)
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num_splines", numeric_spline_pipeline, spline_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
).set_output(transform="pandas")

glm_splines = GeneralizedLinearRegressor(
    family=tweedie_family,
    l1_ratio=1.0,
    fit_intercept=True,
)

glm_spline_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("estimate", glm_splines),
    ]
)

glm_spline_pipeline.fit(df_train[predictors], y_train, estimate__sample_weight=w_train)

df_train["pp_glm_splines"] = glm_spline_pipeline.predict(df_train[predictors])
df_test["pp_glm_splines"] = glm_spline_pipeline.predict(df_test[predictors])

print("\n== GLM with splines on BonusMalus & Density ==")
print_deviance("train deviance", y_train, df_train["pp_glm_splines"], w_train)
print_deviance("test deviance", y_test, df_test["pp_glm_splines"], w_test)

print(
    "Total claim amount (test) – observed vs predicted:",
    df_test["ClaimAmountCut"].sum(),
    (df_test["Exposure"] * df_test["pp_glm_splines"]).sum(),
)

# ----------------------------------------------------------------------
# 5. LGBM Regressor model, tuned to reduce overfitting
# ----------------------------------------------------------------------

lgbm = LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=TWEEDIE_POWER,
    random_state=42,
)

lgbm_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("estimate", lgbm),
    ]
)

# simple hyper-parameter grid
param_grid = {
    "estimate__learning_rate": [0.05, 0.1],
    "estimate__n_estimators": [200, 400],
    "estimate__max_depth": [3, 5],
}

lgbm_cv = GridSearchCV(
    estimator=lgbm_pipeline,
    param_grid=param_grid,
    scoring=None, 
    cv=3,
    n_jobs=-1,
)

lgbm_cv.fit(df_train[predictors], y_train, estimate__sample_weight=w_train)

print("\nBest LGBM params:", lgbm_cv.best_params_)

best_lgbm = lgbm_cv.best_estimator_

df_train["pp_lgbm"] = best_lgbm.predict(df_train[predictors])
df_test["pp_lgbm"] = best_lgbm.predict(df_test[predictors])

print("\n== Tuned LGBM Regressor ==")
print_deviance("train deviance", y_train, df_train["pp_lgbm"], w_train)
print_deviance("test deviance", y_test, df_test["pp_lgbm"], w_test)

print(
    "Total claim amount (test) – observed vs predicted:",
    df_test["ClaimAmountCut"].sum(),
    (df_test["Exposure"] * df_test["pp_lgbm"]).sum(),
)

# ----------------------------------------------------------------------
# (Quick comparison plot of prediction distributions
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for col, label in [
    ("pp_glm_benchmark", "GLM benchmark"),
    ("pp_glm_splines", "GLM + splines"),
    ("pp_lgbm", "LGBM"),
]:
    df_test[col].plot(kind="kde", ax=ax, label=label)

ax.set_xlabel("Pure premium prediction")
ax.set_title("Distribution of predicted pure premiums (test set)")
ax.legend()
plt.tight_layout()
plt.show()
