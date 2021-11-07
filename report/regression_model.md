# Regression Model Evaluation

## Dataset & Evaluation Setting

1. [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
    1. total 20k
        1. 80% -> train & eval
        2. 20% -> test
2. 

## Metrics

1. Mean squared error (MSE)

## Model Evaluation

Refer to [model performance excel](model_eval.xlsx) and [regression notebook](../notebooks/assignment_regression.ipynb):

1. When applying linear regression, MinMaxScaler has no effects
2. When applying Lasso or Ridge, MinMaxScaler seems has negative effects
3. Ridge performed better than Lasso in this case; but ridge's and lasso's performance is the same as linear regression
4. PCA has negative effect because the feature dimension are so rare


