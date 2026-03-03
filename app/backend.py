from fastapi import FastAPI
from pydantic_models import SalesObservation, ForecastRequest, ForecastResponse
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.utils import load_object, convert_history_to_dataframe
from src.features.feature_engineering import featureEngineering
from .api_forecast import recursive_forecast


# ---------------------------
# Load models globally (cache)
# ---------------------------

sarima_model = None
gbm_model = None
gbm_features = None


# ---------------------------
# Lifespan Manager
# ---------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    global sarima_model
    global gbm_model
    global gbm_features

    # Load models here
    sarima_model = load_object('/Users/v/Data Science Projects/time-series-project/models/sarima_model')
    gbm_model = load_object('/Users/v/Data Science Projects/time-series-project/models/gbm_model')

    yield

    # Optional cleanup logic
    print("Shutting down app")


# ---------------------------
# FastAPI App Initialization
# ---------------------------

app = FastAPI(lifespan=lifespan)


@app.post('/sarima-forecast')
def sarima_forecast(horizon: int):

    forecast = sarima_model.forecast(steps=horizon)

    return {
        "forecast": forecast.tolist(),
        "horizon": horizon
    }

@app.post('/gbm-forecast', response_model=ForecastResponse)
def gbm_forecast(request:ForecastRequest):

    feature_engineer = featureEngineering()

     #convert request to dataframe
    df = convert_history_to_dataframe(request.history)

    #do the feature engineering on the request.history dataframe
    df = feature_engineer.create_features(df)
    
    #recursively predict for a given horizon
    preds = recursive_forecast(df, gbm_model, request.horizon)

    #return according to response model
    return ForecastResponse(forecast=preds)

