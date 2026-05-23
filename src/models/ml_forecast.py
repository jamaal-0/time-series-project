import numpy as np
import pandas as pd
import itertools
from lightgbm import LGBMRegressor
from mlforecast import MLForecast


class GMBModel:
    def __init__(self):
        self.best_model = None
        self.best_params = None

    def create_param_grid(self):
        max_depth = [3, 5, 7]
        learning_rate = [0.01, 0.05, 0.1]
        n_estimators = [100, 300]

        return list(itertools.product(max_depth, learning_rate, n_estimators))

    
    def cv(self, train_set, param_grid, horizon=3):
        df = train_set.copy()
        # Rename target for MLForecast convention
        df = df.rename(columns={"SalesAmount": "y"})
        # Ensure MS (Month Start) frequency handling
        df = df.reset_index().rename(columns={"Date": "ds"})
        df["unique_id"] = 0


        best_score = np.inf
        best_params = None

        for max_depth, lr, n_estimators in param_grid:

            model = LGBMRegressor(
                max_depth=max_depth,
                learning_rate=lr,
                n_estimators=n_estimators,
                random_state=42
            )

            fcst = MLForecast(
                models=[model],
                freq="MS",
                lags=[1, 2, 3, 5, 6, 12],
                date_features=["month", "quarter", "year"]
            )

            cv_results = fcst.cross_validation(
                df,
                h=horizon,
                n_windows=5,
                step_size=horizon,
                refit=True
            )

            # Evaluate MAE
            score = np.mean(
                np.abs(cv_results["y"] - cv_results["LGBMRegressor"])
            )

            print(f"Params {(max_depth, lr, n_estimators)} -> MAE: {score:.4f}")

            if score < best_score:
                best_score = score
                best_params = (max_depth, lr, n_estimators)

        self.best_params = best_params

        print("\n========================")
        print("BEST PARAMS:")
        print(best_params)
        print("BEST SCORE:", best_score)
        print("========================")

        return best_params


    def train_final_model(self, train_set, max_depth, learning_rate, n_estimators):

        df = train_set.copy()
        df = df.rename(columns={"SalesAmount": "y"})
        df = df.reset_index().rename(columns={"Date": "ds"})
        df["unique_id"] = 0

        model = LGBMRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42
        )

        fcst = MLForecast(
            models=[model],
            freq="MS",
            lags=[1, 2, 3, 6, 12],
            date_features=["month", "quarter", "year"]
                )

        fcst.fit(df)
        self.best_model = fcst

        return fcst