import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error


class ForecastPipeline:

    def __init__(self, data):
        dataa = pd.read_csv(data)
        self.data = dataa.copy()

    # ----------------------------------
    # Cleaning
    # ----------------------------------
    def clean(self):

        self.data["Order Date"] = pd.to_datetime(self.data["Order Date"], dayfirst=True)
        self.data = self.data.set_index("Order Date")

        self.data = self.data.dropna(subset=["Sales"])

        return self.data

    # ----------------------------------
    # Weekly Aggregation
    # ----------------------------------
    def aggregate_weekly(self):

        weekly = self.data.groupby([
            pd.Grouper(freq="W"),
            "Region",
            "Category"
        ])["Sales"].sum().reset_index()

        return weekly

    # ----------------------------------
    # Feature Engineering
    # ----------------------------------
    def create_features(self, df):

        df = df.copy()

        # Lag features
        df["lag1"] = df["Sales"].shift(1)
        df["lag2"] = df["Sales"].shift(2)
        df["lag12"] = df["Sales"].shift(12)

        # Month encoding
        df["month"] = df["Order Date"].dt.month

        df = df.dropna()

        return df

    # ----------------------------------
    # One Hot Encoding
    # ----------------------------------
    def encode_categorical(self, df):

        categorical_cols = ["Region", "Category"]

        encoder = OneHotEncoder(sparse_output=False)

        encoded = encoder.fit_transform(df[categorical_cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )

        df_numeric = df.drop(columns=categorical_cols)

        df_final = pd.concat([df_numeric, encoded_df], axis=1)

        return df_final

    # ----------------------------------
    # Train Model
    # ----------------------------------
    def train_model(self, df):

        train_size = int(len(df) * 0.8)

        train = df.iloc[:train_size]
        test = df.iloc[train_size:]

        y_train = train["Sales"]
        y_test = test["Sales"]

        exog_cols = [c for c in df.columns if c != "Sales"]

        exog_train = train[exog_cols]
        exog_test = test[exog_cols]

        # Force numeric dtype (VERY IMPORTANT)
        exog_train = exog_train.apply(pd.to_numeric, errors="coerce").fillna(0)
        exog_test = exog_test.apply(pd.to_numeric, errors="coerce").fillna(0)

        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=(1,0,2),
            seasonal_order=(1,0,0,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit(disp=False)

        forecast = fitted_model.forecast(
            steps=len(test),
            exog=exog_test
        )

        mae = mean_absolute_error(y_test, forecast)
        relative_mae = mae / np.mean(y_test)
        print('relative mae: ',relative_mae)

        return fitted_model, mae