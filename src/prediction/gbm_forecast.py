
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from src.features.feature_engineering import featureEngineering 


class GBMForecast():

    def __init__(self):
        self.feature_engineering = featureEngineering()
        pass

    def evaluate_test_set_recursive(self, train_set, test_set, model):

        target = 'SalesAmount'

        # History buffer starts with full training data
        history = train_set.copy()

        recursive_preds = []

        for i in range(len(test_set)):

            # Rebuild features from current history
            history_features = self.feature_engineering.create_features(history.copy())

            # Select latest row (state representation)
            X_latest = history_features.iloc[[-1]].drop(columns=[target], errors='ignore')

            # Predict next step
            y_pred = model.predict(X_latest)[0]
            recursive_preds.append(y_pred)

            # Create new row based on last known row
            new_row = history.iloc[-1:].copy()

            # Replace target with prediction
            new_row[target] = y_pred

            # IMPORTANT: advance time index
            new_row.index = [test_set.index[i]]

            # Append prediction into history
            history = pd.concat([history, new_row])

        # Evaluate against actual test targets
        y_test = test_set[target].values

        assert len(recursive_preds) == len(y_test)

        error = mean_absolute_error(y_test, recursive_preds)
        mean_sales = np.mean(y_test)

        return error, recursive_preds, mean_sales