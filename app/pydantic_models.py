from typing import List
from pydantic import BaseModel
from datetime import datetime


class SalesObservation(BaseModel):
    Date : datetime
    SalesAmount: int
    Promotion: int
    HolidayMonth: int

class ForecastRequest(BaseModel):
    history: List[SalesObservation]
    horizon: int = 3


class ForecastResponse(BaseModel):
    forecast: list[float]
    horizon: int
    mean_forecast: float