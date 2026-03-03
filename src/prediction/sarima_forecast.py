import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

class SarimaForecast:

    def __init__(self, sarima_model, test_data):

        self.test_set = test_data.copy()
        self.model = sarima_model

    def predict(self):

        forecast = self.model.forecast(len(self.test_set))

        error = mean_absolute_error(self.test_set, forecast)
        mean_sales = np.mean(self.test_set)

        return error, forecast
    
    def is_model_good(self,mae):
        """
        Returns True if MAE / mean_sales < threshold
        """

        mean_sales = np.mean(self.test_set)
        #print(mean_sales)

        if mean_sales == 0:
            return False

        return (mae , mean_sales)
