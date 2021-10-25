import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost.sklearn import XGBModel
from typing import List


def plot_feature_importances(model: XGBModel, feature_cols: List[str], show_feature_num=10, figsize=(20, 10)):
    """
    plot feature importance of xgboost model
    Args:
        model:
        feature_cols:
        show_feature_num:
        figsize:

    Returns:

    """
    feature_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)[:show_feature_num]
    plt.figure(figsize=figsize)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.title("Feature Importance")
    plt.show()

"""
Grid Search

xgb = XGBRegressor()

parameters = {
              'objective':['reg:squarederror'],
              'learning_rate': [
                  .03, 
                  #0.05, .07, 0.1
              ],
              'max_depth': [ 9 ],
              'min_child_weight': [1],
              'silent': [1],
              'subsample': [ #0.8, 0.9, 
                  0.95],
              'colsample_bytree': [ 0.8, #0.9, 0.95
                                  ],
              'n_estimators': [
                              #100, 
                              # 200, 300, 
                  500
                  #600, 700, 800, 900
                              ]
}
xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)
xgb_grid.fit(train_X,
         train_y)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

params = xgb_grid.best_params_
"""