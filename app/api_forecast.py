import pandas as pd
from src.features.feature_engineering import featureEngineering




def recursive_forecast(history_df, model, horizon):
    feature_engineer = featureEngineering()

    target = "SalesAmount"

    history = history_df.copy()

    preds = []

    for _ in range(horizon):

        history_features = feature_engineer.create_features(history.copy())

        X_latest = history_features.drop(columns=[target], errors="ignore").iloc[[-1]]

        y_pred = model.predict(X_latest)[0]

        preds.append(y_pred)

        # Append prediction into buffer
        new_row = history.iloc[-1:].copy()
        new_row[target] = y_pred

        # Advance time index
        new_row.index = [history.index[-1] + pd.DateOffset(months=1)]

        history = pd.concat([history, new_row])

    return preds