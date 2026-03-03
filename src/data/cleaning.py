import numpy as np
import pandas as pd
from .ingestion import Ingestion
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox

class Cleaning():
    def __init__(self,train_data_path):
        if train_data_path:
            data = pd.read_csv(train_data_path)
            self.data = data.copy()
        else:
            data = pd.read_csv('/Users/v/Data Science Projects/time-series-project/data/train.csv')
            self.data = data.copy()


    def clean_data(self):
        #self.data = self.data.dropna(subset=["Postal Code"])

        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data = self.data.set_index("Date")

        pass

    def obtain_monthly_sales(self):
        #monthly_sales = self.data['Sales'].resample(rule='W').sum()
        monthly_sales = self.data['SalesAmount']
        monthly_sales = monthly_sales.asfreq("MS")
        return monthly_sales
    
    
    def make_stationary(self, monthly_sales):
        box_cox_monthly_sales,lam = boxcox(monthly_sales+1)
        box_cox_monthly_sales = pd.Series(box_cox_monthly_sales,index=monthly_sales.index)
        #box_cox_diff_sales = box_cox_monthly_sales.diff().dropna()

        return box_cox_monthly_sales,lam, box_cox_monthly_sales.iloc[-1]
    
    def ADF_test(self,box_cox_diff_sales):
        test_results = adfuller(box_cox_diff_sales.dropna())
        print('ADF Statistic:', test_results[0])
        print('P-Value:', test_results[1])
        print('Critical Values:')

        for thres, adf_stat in test_results[4].items():
            print(f'\t{thres}: {adf_stat:.2f}')

        return test_results
    

        

    
     
'''
if __name__ == '__main__':
    ingestion_obj = Ingestion('/Users/v/Data Science Projects/time-series-project/data/train.csv')
    data = ingestion_obj.read_csv()
    test_set,train_set = ingestion_obj.test_train_split(data)
    train_data_path, test_data_path = ingestion_obj.upload_csv(test_set,train_set)
    Cleaning_obj = Cleaning(test_data_path)
    Cleaning_obj.clean_data()
    monthly_sales = Cleaning_obj.obtain_monthly_sales()
    print(type(monthly_sales.index))
    box_cox_diff_monthly_sales,lam = Cleaning_obj.make_stationary(monthly_sales)
    test_results = Cleaning_obj.ADF_test(box_cox_diff_monthly_sales)


'''
    
