import lightgbm as lgb
import numpy as np
import pandas as pd 
import itertools
from sklearn.metrics import mean_absolute_error
from src.features.feature_engineering import featureEngineering

class GMBModel():
    def __init__(self):
        self.featureengineering = featureEngineering()

  


    def create_param_grid(self):
            """ create the param grid for cv """
            

            # ---------- Parameter Grid ----------

            max_depth = [3, 5, 7]
            learning_rate = [0.01, 0.05, 0.1]
            n_estimators = [100, 300]

            LightGBM_params = list(itertools.product(max_depth, learning_rate, n_estimators))
            #print(LightGBM_params)

            return LightGBM_params

    def cv(self, train_set,  LightGBM_params):
        

        best_score = np.inf
        best_params = None

        for max_depth, learning_rate, n_estimators in LightGBM_params:
            print('max_depth: ', max_depth)
            print('learning_rate: ', learning_rate)
            print('n_estimators: ', n_estimators)

            score = self.carry_out_cv(train_set,max_depth, learning_rate, n_estimators)

            print(f"Tested {max_depth}, {learning_rate}, {n_estimators} -> CV Score = {score:.4f}")

            if not np.isnan(score) and score < best_score:
                    best_score = score
                    best_params = (max_depth, learning_rate, n_estimators)
            
            
            print("\n============================")
            print("BEST MODEL FOUND")
            print("max_depth:", best_params[0])
            print("learning_rate:", best_params[1])
            print("n_estimators:", best_params[2])
            print("CV Score:", best_score)
            print("============================")

        return best_params

            
            
        

    def carry_out_cv(self, train_set, max_depth, learning_rate, n_estimators,threshold=0.8,horizon=3):
        features = train_set.columns.drop(['SalesAmount'])
        target = 'SalesAmount'
        n = len(train_set)
        split = int(threshold * n)
        errors = []
        n_splits = (n - split) // horizon


        #for i in cv folds 
        for i in range(n_splits):
            #window is incrementally increased by the length of horizon until val_end reaches the end of the training set
            window_end = split+(i*horizon)
            val_end = window_end + horizon

            if val_end > n:
                break

            #created the current folds for the cv
            train_subset = train_set.iloc[:window_end]
            val_subset = train_set.iloc[window_end:val_end]

            #created the X and y for the train set
            X_train = train_subset[features]
            y_train = train_subset[target]

            #created the X and y to validate on
            X_val = val_subset[features]
            y_val = val_subset[target]

        #create and fit the model
            model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            model.fit(X_train, y_train)

            # ===== Recursive Forecast =====
            #create a copy of the training fold
            history = train_subset.copy()
            recursive_preds = []

            #for each prediction to compute in this cv interation
            for _ in range(horizon):

                # Rebuild features from history
                #history_featues now contains the lag and rolling features
                history_features = self.featureengineering.create_features(history.copy())

                #pbtain the most recent observation/entry
                    # - only the most recent observation/entry is needed for tree models
                    # - given lag and rolling features as the encode history
                X_latest = history_features[features].iloc[[-1]]

                #obtain the prediction as a scaler because it is returned as an np array
                y_pred = model.predict(X_latest)[0]

                #add it to the prediction list
                recursive_preds.append(y_pred)

                #append prediction into history buffer
                new_row = history.iloc[-1:].copy()
                new_row[target] = y_pred

                history = pd.concat([history, new_row])

            # Evaluate recursive forecast vs validation window
            errors.append(
                mean_absolute_error(
                    y_val.values,
                    recursive_preds
                )
            )
        if len(errors) == 0:
            return float("inf")
            
        return np.mean(errors)

    def carry_out_DMS_cv(self, train_set, max_depth, learning_rate, n_estimators,threshold=0.8,horizon=3):
        features = train_set.columns.drop(['SalesAmount'])
        target = 'SalesAmount'
        n = len(train_set)
        split = int(threshold * n)
        errors = []
        n_splits = (n - split) // horizon

        for i in range(n_splits):
            window_end = split+(i*horizon)
            val_end = window_end + horizon

            if val_end > n:
                break
            train_subset = train_set.iloc[:window_end]
            val_subset = train_set.iloc[window_end:val_end]

            X_train = train_subset[features]
            y_train = train_subset[target]

            X_val = val_subset[features]
            y_val = val_subset[target]

            model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            model.fit(X_train, y_train)

            pred = model.predict(X_val)

            errors.append(
                    mean_absolute_error(y_val, pred)
                )
        if len(errors) == 0:
            return float("inf")
            
        return np.mean(errors)


    def train_final_model(self, train_set, max_depth, learning_rate, n_estimators):

        target = 'SalesAmount'
        features = train_set.columns.drop([target])

        X_train = train_set[features]
        y_train = train_set[target]

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )

        model.fit(X_train, y_train)

        return model