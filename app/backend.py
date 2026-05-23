from fastapi import FastAPI
from pydantic_models import ForecastRequest, ForecastResponse
import pandas as pd
import numpy as np
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.utils import load_object, convert_history_to_dataframe


sarima_model = None
gbm_ml_forecast_mmodel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sarima_model
    global gbm_ml_forecast_mmodel

    # Load models here
    sarima_model = load_object('/Users/v/Data Science Projects/time-series-project/models/sarima_model')
    gbm_ml_forecast_mmodel = load_object('/Users/v/Data Science Projects/time-series-project/models/gbm_model')

    yield

    print("Shutting down app")



app = FastAPI(lifespan=lifespan)


@app.post('/sarima-forecast')
def sarima_forecast(horizon: int):
    forecast = sarima_model.forecast(steps=horizon)
    preds = forecast.tolist()

    return ForecastResponse(
        forecast=preds,
        horizon=horizon,
        mean_forecast=float(np.mean(preds))
    )


@app.post('/gbm-forecast', response_model=ForecastResponse)
def gbm_forecast(request: ForecastRequest):
    df = convert_history_to_dataframe(request.history)
    df = df.rename(columns={"SalesAmount": "y"})
    df = df.reset_index(drop=True)
    df["unique_id"] = 0
    df["ds"] = pd.to_datetime([obs.Date for obs in request.history])

    forecast_df = gbm_ml_forecast_mmodel.predict(
        request.horizon,
        new_df=df
    )

    preds = forecast_df["LGBMRegressor"].tolist()

    return ForecastResponse(
        forecast=preds,
        horizon=request.horizon,
        mean_forecast=float(np.mean(preds))
    )

