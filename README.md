# Time Series Forecasting System

## Overview

This project is an end-to-end time-series forecasting system built using both statistical forecasting and machine learning approaches.

The project compares:

- SARIMA (Seasonal AutoRegressive Integrated Moving Average)
- LightGBM forecasting using MLForecast

The system includes:

- Data ingestion and preprocessing pipelines
- Exploratory Data Analysis (EDA)
- Walk-forward cross validation
- Recursive forecasting evaluation
- Hyperparameter tuning
- Forecasting APIs built with FastAPI

The goal of the project was to explore how classical statistical forecasting compares against machine learning forecasting on structured retail sales time-series data.

---

# Project Architecture

```text
src/
│
├── data/
│   ├── ingestion.py
│   ├── cleaning.py
│
├── features/
│   ├── feature_engineering.py
│
├── models/
│   ├── sarima_model.py
│   ├── ml_forecast.py
│
├── prediction/
│   ├── sarima_forecast.py
│   ├── ml_forecast.py
│
├── pipeline/
│   ├── pipeline.py
│
├── api/
│   ├── app.py
│
├── utils/
│   ├── utils.py
│
models/
│   ├── sarima_model.pkl
│   ├── gbm_model.pkl
│
requirements.txt
README.md
```

---

# Technologies Used

## Core Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- LightGBM
- MLForecast
- FastAPI
- PostgreSQL

---

# Problem Statement

The objective of this project is to forecast future monthly retail sales using historical sales data.

The project compares:

1. Statistical forecasting models
2. Machine learning forecasting models

The system evaluates:

- Forecast accuracy
- Recursive forecasting performance
- Time-series cross validation performance
- Error ratios relative to average sales

---

# Exploratory Data Analysis (EDA)

The project includes a full time-series EDA workflow.

## EDA Steps

### Data Integrity Checks

- Missing value analysis
- Datetime conversion
- Frequency alignment using:

```python
.df.asfreq('MS')
```

- Duplicate inspection

---

### Trend and Seasonality Analysis

Seasonal decomposition was used to inspect:

- Trend
- Seasonality
- Residual components

Both additive and multiplicative decomposition assumptions were explored.

---

### Stationarity Testing

The Augmented Dickey-Fuller (ADF) test was used to evaluate stationarity.

The project explored:

- Differencing
- Seasonal differencing
- Residual stationarity

---

### Autocorrelation Analysis

ACF and PACF plots were used to:

- Identify autoregressive structure
- Detect seasonality
- Select lag candidates
- Support SARIMA parameter selection

---

# SARIMA Forecasting Pipeline

## Hyperparameter Tuning

SARIMA hyperparameters were tuned using walk-forward cross validation.

Parameters explored:

```python
(p, d, q)
(P, D, Q, s)
```

with:

```python
s = 12
```

for monthly seasonality.

---

## Recursive Forecast Evaluation

Recursive forecasting evaluation was performed using rolling forecasting windows.

Evaluation metric:

- Mean Absolute Error (MAE)

---

## SARIMA Results

Final evaluation:

```text
MAE: 861.69
Mean Sales: 12265.8
Error Ratio: 0.0702
```

The SARIMA model outperformed the machine learning forecasting model on this dataset.

---

# MLForecast + LightGBM Pipeline

## Forecasting Framework

The machine learning forecasting system was implemented using:

- LightGBM
- MLForecast

MLForecast was used to handle:

- lag feature generation (automated)
- walk-forward cross validation
- recursive multi-step forecasting

No manual feature engineering (lags/rolling features/date features) was ultimately used due to implementation complexity and integration issues encountered with `date_features` and custom transforms.

---

# Key Findings

## SARIMA Outperformed Gradient Boosting

The statistical forecasting model outperformed the machine learning model.

Key observations:

- The dataset exhibited strong seasonality
- The data was highly autoregressive
- Exogenous variables had limited predictive signal
- SARIMA captured the time-series structure more effectively

---

## LightGBM Diagnostic Findings

During experimentation:

- Feature importance values collapsed to zero
- LightGBM frequently produced:

```text
Stopped training because there are no more leaves that meet the split requirements
```

This suggested:

- Limited nonlinear signal
- Dominant autoregressive structure
- Strong suitability for statistical forecasting methods

---

# API Development

The project includes a forecasting API built with FastAPI.

---

# API Features

## SARIMA Forecast Endpoint

```http
POST /sarima-forecast
```

Returns future forecasts using the trained SARIMA model.

---

## MLForecast Endpoint

```http
POST /gbm-forecast
```

Returns recursive forecasts using the trained MLForecast + LightGBM model.

---

# Pydantic Validation

Request and response validation was implemented using Pydantic models.

Example:

```python
class SalesObservation(BaseModel):
    Date: datetime
    SalesAmount: int
    Promotion: int
    HolidayMonth: int
```

---

# Model Persistence

Trained forecasting models were serialized and loaded using helper utility functions.

The API loads models during application startup using:

- FastAPI lifespan context manager
- Shared model state

This avoids reloading models per request.

---

# Evaluation Strategy

## Why Walk-Forward Validation?

Traditional random cross validation is inappropriate for time-series data because it introduces future leakage.

The project instead uses:

- Expanding window validation
- Sequential forecasting evaluation
- Recursive forecasting

This better simulates real-world forecasting scenarios.

---

# Future Improvements

Potential future improvements include:

- XGBoost forecasting
- CatBoost forecasting
- Prophet comparison
- Deep learning forecasting models
- Multivariate forecasting
- Automated retraining pipelines
- Forecast monitoring dashboards
- Docker deployment
- CI/CD integration

---

# How to Run

## Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Pipeline

```bash
python3 -m src.pipeline.pipeline
```

---

## Run API

```bash
uvicorn src.api.app:app --reload
```

---

# Skills Demonstrated

This project demonstrates:

- Time-series forecasting
- Statistical modeling
- Machine learning forecasting
- Walk-forward validation
- Hyperparameter tuning
- Recursive forecasting
- FastAPI backend development
- Model persistence
- Modular ML system design
- Production-style forecasting pipelines

---

# Conclusion

This project explores both classical statistical forecasting and machine learning forecasting techniques within a modular end-to-end system.

A key outcome of the project was understanding that:

- machine learning models do not always outperform statistical models
- model performance depends heavily on data structure and signal complexity
- strong seasonality and autoregressive structure can favor classical forecasting methods such as SARIMA

The project provided practical experience building forecasting systems beyond notebook experimentation, including APIs, persistence, validation workflows, and production-style architecture.

