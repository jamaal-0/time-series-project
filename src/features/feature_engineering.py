import pandas as pd
import numpy as np


class featureEngineering():

    def __init__(self):
        pass

    def create_features(self, dataframe):

        df = dataframe.copy()

        # Remove previously engineered features first (VERY IMPORTANT)
        feature_prefixes = ('lag_', 'rolling_', 'month', 'year', 'quarter')

        df = df.loc[:, ~df.columns.str.startswith(feature_prefixes)]

        # ----- Lag Features -----
        shifts = [1, 5, 6]

        for i in shifts:
            df[f'lag_{i}'] = df['SalesAmount'].shift(i)

        # ----- Rolling Features -----
        rolling_windows = [3, 6]

        for w in rolling_windows:
            df[f'rolling_mean_{w}'] = df['SalesAmount'].shift(1).rolling(w).mean()
            df[f'rolling_std_{w}'] = df['SalesAmount'].shift(1).rolling(w).std()

        # ----- Calendar Features -----
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter

        df = df.dropna()

        return df
    



    
    

