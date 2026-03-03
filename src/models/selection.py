import numpy as np
import pandas as pd


class ModelSelector():
    def __init__(self,train_set,sarima_obj,ML_obj):
        self.train_set = train_set
        self.sarima_obj = sarima_obj
        self.ML_obj =  ML_obj
        pass

    def compare(self):
        """ comapre the ML model and the sarima model that have been tested 
            on the test set in order to decide which model is best to use """
        
        pass

