# ============================================
# FILE: pipeline.py
# ============================================

import pandas as pd
from src.data.ingestion import Ingestion
from src.data.cleaning import Cleaning
from src.models.sarima_model import Sarima
from src.models.ml_forecast import GMBModel
from src.prediction.sarima_forecast import SarimaForecast
from src.prediction.ml_forecast import GBMForecast
from src.utils import save_object, load_object



def ingestion(ingestion_obj):
    data = ingestion_obj.read_csv()
    train_set, test_set = ingestion_obj.test_train_split(data)
    train_data_path, test_data_path = ingestion_obj.upload_csv(
        test_set,
        train_set
    )

    return train_data_path, test_data_path


def clean(cleaning_obj):
    cleaning_obj.clean_data()
    monthly_sales = cleaning_obj.obtain_monthly_sales()
    test_results = cleaning_obj.ADF_test(monthly_sales)

    return monthly_sales, test_results


def sarima_forecast(sarima_obj, monthly_sales):
    sarima_grid = sarima_obj.create_param_grid()
    best_params = sarima_obj.hyperparameter_tuning(
        sarima_grid
    )

    best_order, best_seasonal = best_params
    final_model, forecast = sarima_obj.forecast_best_params(
        monthly_sales,
        best_order,
        best_seasonal
    )

    return final_model, forecast



def sarima_forecast_pipeline(csv_name, model_name):
    ingestion_obj = Ingestion(csv_name, model_name)
    train_data_path, test_data_path = ingestion(
        ingestion_obj
    )

    cleaning_obj = Cleaning(train_data_path)
    train_sales, _ = clean(cleaning_obj)
    sarima_obj = Sarima(train_sales)
    final_model, forecast = sarima_forecast(
        sarima_obj,
        train_sales
    )

    save_object(
        "/Users/v/Data Science Projects/time-series-project/models/sarima_model",
        final_model
    )

    
    test_cleaning_obj = Cleaning(test_data_path)
    test_sales, _ = clean(test_cleaning_obj)
    loaded_model = load_object(
        "/Users/v/Data Science Projects/time-series-project/models/sarima_model"
    )

    sarima_predictor = SarimaForecast(
        loaded_model,
        test_sales
    )

    error, result = sarima_predictor.predict()
    print(error)
    mse, mean_sales = sarima_predictor.is_model_good(
        error
    )

    ratio = mse / mean_sales
    print("\n============================")
    print("SARIMA TEST RESULTS")
    print("============================")
    print('ratio',ratio)
    print('mse', mse)
    print('mean sales', mean_sales)
    print("============================")



def gbm_forecast_pipeline(csv_name, model_name):
    ingestion_obj = Ingestion(csv_name, model_name)
    data = ingestion_obj.read_csv()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")
    data = data.asfreq("MS")

    train_set, test_set = ingestion_obj.test_train_split(
        data
    )

    gbm_model = GMBModel()
    param_grid = gbm_model.create_param_grid()
    max_depth, learning_rate, n_estimators = gbm_model.cv(
        train_set,
        param_grid,
        horizon=3
    )

    model = gbm_model.train_final_model(
        train_set,
        max_depth,
        learning_rate,
        n_estimators
    )

    save_object(
        "/Users/v/Data Science Projects/time-series-project/models/gbm_model",
        model
    )

    loaded_model = load_object(
        "/Users/v/Data Science Projects/time-series-project/models/gbm_model"
    )

    gbm_forecast = GBMForecast()

    error, preds, mean_sales = (
        gbm_forecast.evaluate_test_set_recursive(
            train_set,
            test_set,
            loaded_model
        )
    )

    print("\n============================")
    print("GBM TEST RESULTS")
    print("============================")
    print(f"MAE: {error}")
    print(f"Mean Sales: {mean_sales}")
    print(f"Error Ratio: {error / mean_sales}")
    print("============================")

    return loaded_model




sarima_forecast_pipeline(
    "retail_sales_mock_data",
    "sarima"
)

model = gbm_forecast_pipeline(
    "retail_sales_mock_data",
    "gbm"
)
lgb_model = model.models_["LGBMRegressor"]
model_importance_series = pd.Series(lgb_model.feature_importances_, index=lgb_model.feature_name_).sort_values(ascending=False)
print(model_importance_series)
print(lgb_model.booster_.num_trees())