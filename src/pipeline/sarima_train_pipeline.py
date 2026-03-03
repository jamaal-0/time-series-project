import pandas as pd
from src.data.ingestion import Ingestion
from src.data.cleaning import Cleaning
from src.models.sarima_model import Sarima
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from src.utils import save_object, load_object
from src.prediction.sarima_forecast import SarimaForecast
from src.models.gbm_model import GMBModel
from src.prediction.gbm_forecast import GBMForecast
from src.features.feature_engineering import featureEngineering



def ingestion(ingestion_obj):
    data = ingestion_obj.read_csv()
    train_set,test_set = ingestion_obj.test_train_split(data)
    train_data_path, test_data_path = ingestion_obj.upload_csv(test_set,train_set)
    return train_data_path, test_data_path


def clean(Cleaning_obj):
    Cleaning_obj.clean_data()
    weekly_sales = Cleaning_obj.obtain_monthly_sales()
    print(type(weekly_sales.index))
    #box_cox_diff_weekly_sales,lam,_ = Cleaning_obj.make_stationary(weekly_sales)
    test_results = Cleaning_obj.ADF_test(weekly_sales)

    return weekly_sales, test_results


def sarima_forecast(sarima_obj,weekly_sales):
    sarima_grid = sarima_obj.create_param_grid()
    best_params = sarima_obj.hyperparameter_tuning(sarima_grid)
    best_order, best_seasonal = best_params[0], best_params[1]
    (final_model,forecast) = sarima_obj.forecast_best_params(weekly_sales, best_order, best_seasonal)

    return (final_model,forecast)



def sarima_forecast_pipeline(csv_name, model_name):
    ingestion_obj = Ingestion(csv_name, model_name)
    train_data_path, test_data_path = ingestion(ingestion_obj)

    Cleaning_obj = Cleaning(train_data_path)
    train_weekly_sales, train_test_results = clean(Cleaning_obj)

    sarima_obj = Sarima(train_weekly_sales)
    (final_model,train_forecast) = sarima_forecast(sarima_obj, train_weekly_sales)

    save_object("/Users/v/Data Science Projects/time-series-project/models/sarima_model",final_model)

    test_Cleaning_obj = Cleaning(test_data_path)
    test_weekly_sales, test_test_results = clean(test_Cleaning_obj)

    loaded_sarima_obj = load_object('/Users/v/Data Science Projects/time-series-project/models/sarima_model')
    sarima_predictor = SarimaForecast(loaded_sarima_obj, test_weekly_sales)

    error, result = sarima_predictor.predict()
    print(error)
    mse,mean_saless = sarima_predictor.is_model_good(error)
    ratio = mse/mean_saless
    print(ratio,mse,mean_saless)
    print(loaded_sarima_obj.resid.mean())
    '''
    plt.figure(figsize=(10,5))
    plt.plot(loaded_sarima_obj.resid.dropna())
    plt.title("Residuals Plot")
    plt.show()
    '''




def gbm_forecast_pipeline(csv_name,model_name):

    ingestion_obj = Ingestion(csv_name,model_name)
    data = ingestion_obj.read_csv()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")
    data = data.asfreq("MS")

    feature_engineer = featureEngineering()
    data = feature_engineer.create_features(data)

    train_set,test_set = ingestion_obj.test_train_split(data)
    train_data_path, test_data_path = ingestion_obj.upload_csv(test_set,train_set)

    #Cleaning_obj = Cleaning(train_data_path)
    #train_weekly_sales, train_test_results = clean(Cleaning_obj)
    train_weekly_sales = train_set

    gbm_model = GMBModel()
    LightGBM_params = gbm_model.create_param_grid()


    max_depth, learning_rate, n_estimators = gbm_model.cv(train_weekly_sales,  LightGBM_params)
    model = gbm_model.train_final_model(train_weekly_sales, max_depth, learning_rate, n_estimators)

    gbm_forecast = GBMForecast()

    save_object("/Users/v/Data Science Projects/time-series-project/models/gbm_model",model)
    loaded_gbm_model = load_object('/Users/v/Data Science Projects/time-series-project/models/gbm_model')


    error, pred, mean_sales = gbm_forecast.evaluate_test_set_recursive(train_set, test_set, loaded_gbm_model)
    print(f'error: {error}, pred: {pred}, mean_sales: {mean_sales}, ratio: {error/mean_sales}')


sarima_forecast_pipeline('retail_sales_mock_data','sarima')
gbm_forecast_pipeline('retail_sales_mock_data','gbm')








