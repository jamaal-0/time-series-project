from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np


class GBMForecast:

    def __init__(self):
        pass

    def evaluate_test_set_recursive(self, train_set, test_set, model, horizon=None):

        train_df = train_set.copy()
        train_df = train_df.rename(
            columns={"SalesAmount": "y"}
        )
        train_df["unique_id"] = 0
        train_df = train_df.reset_index().rename(
            columns={"Date": "ds"}
        )
        
        if horizon is None:
            horizon = len(test_set)

        forecast_df = model.predict(horizon)
        preds = forecast_df["LGBMRegressor"].values

        y_test = test_set["SalesAmount"].values
        error = mean_absolute_error(y_test, preds)
        mean_sales = np.mean(y_test)

        return error, preds.tolist(), mean_sales