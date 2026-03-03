import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

class Sarima():
    def __init__(self, sales):
        self.sales = sales
        pass

    

#carry out the cross validation

    def carry_out_cv(self, order,seasonal_order, train_size=0.8, horizon=3):
        
        """ obtain the folds for cv and carry out cv to obtain the MAE of the set of hyperparameters and the true forecast """
        self.sales = self.sales.dropna()
        n = len(self.sales)
        errors = []

        split_point = int(n * train_size)

        if split_point + horizon >= n:
            return float("inf")

        n_splits = (n - split_point) // horizon
        #print(len(self.sales))
        #print(split_point)
        #print(len(self.sales.iloc[split_point:]))

        for i in range(n_splits):
            window_end = split_point+(i*horizon)
            val_end = window_end + horizon

            if val_end > n:
                break
            train_subset = self.sales.iloc[:window_end]
            val_subset = self.sales.iloc[window_end:val_end]

            model = SARIMAX(
                train_subset,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(
                disp=False,
                method='lbfgs',
                maxiter=50
            )

            forecast = model.forecast(len(val_subset))

            errors.append(
                mean_absolute_error(val_subset, forecast)
            )
        if len(errors) == 0:
            return float("inf")
        
        return np.mean(errors)


            
        
    
    def create_param_grid(self):
        """ create the param grid for cv """
        

        # ---------- Parameter Grid ----------

        p = range(0, 3)
        d = range(0,2)           # You already differenced manually
        q = range(0, 3)

        P = range(0, 2)
        D = range(0,2)
        Q = range(0, 2)

        s = 12

        arima_params = list(itertools.product(p, d, q))
        seasonal_params = list(itertools.product(P, D, Q))

        sarima_grid = []

        for order in arima_params:
            for seasonal in seasonal_params:
                sarima_grid.append((order, (*seasonal, s)))

        return sarima_grid
    
    def hyperparameter_tuning(self, sarima_grid):
        """ do cross validation for the hyperparameters tuning to select best: p, q, P and Q """
        best_score = np.inf
        best_params = None

        for order, seasonal_order in sarima_grid:

            score = self.carry_out_cv(
                order=order,
                seasonal_order=seasonal_order
            )

            print(f"Tested {order} {seasonal_order} -> CV Score = {score:.4f}")

            if not np.isnan(score) and score < best_score:
                best_score = score
                best_params = (order, seasonal_order)

        print("\n============================")
        print("BEST MODEL FOUND")
        print("Order:", best_params[0])
        print("Seasonal:", best_params[1])
        print("CV Score:", best_score)
        print("============================")

        return best_params
    
    def forecast_best_params(self,train_series,best_order, best_seasonal):
        """ obtain best model and forecast """
        #forecast_horizon = len(test_self.sales)
        train_series = train_series.dropna()
        final_model = SARIMAX(
            train_series,
            order=best_order,
            seasonal_order=best_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        forecast = final_model.forecast(12)
        return (final_model,forecast)


