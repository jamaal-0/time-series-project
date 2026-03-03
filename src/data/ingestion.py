from dataclasses import dataclass
import os
import pandas as pd 
import numpy as np

@dataclass
class DataConfig:
    csv_name: str
    model: str


    train_data_path: str = None
    test_data_path: str = None
    raw_data_path: str = None

    def __post_init__(self):
        self.train_data_path = os.path.join('data',f"train_{self.csv_name}_{self.model}.csv")
        self.test_data_path = os.path.join('data',f"test_{self.csv_name}_{self.model}.csv")
        self.raw_data_path = os.path.join('data/original',f"{self.csv_name}.csv")

class Ingestion():
    def __init__(self,csv_name, model):
        self.ingestion_config=DataConfig(csv_name, model)

        pass

    def read_csv(self):
        data = pd.read_csv(self.ingestion_config.raw_data_path)

        return data
    
    def test_train_split(self,data):
        split_index = int(len(data) * 0.8)

        train_set = data.iloc[:split_index]
        test_set = data.iloc[split_index:]

        return train_set,test_set
    
#need to add the train and test csvs to root/data folder
    def upload_csv(self,test_set:pd.DataFrame,train_set:pd.DataFrame):
        train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
        test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
        
        return(self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

    
