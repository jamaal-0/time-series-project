import pickle
import os
import pandas as pd

def save_object(file_path, obj):
    dir_path = os.path.dirname(file_path)

    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)

def load_object(file_path):
     with open(file_path, "rb") as file_obj:
         return pickle.load(file_obj)


def convert_history_to_dataframe(history_list):

    df = pd.DataFrame([obs.dict() for obs in history_list])

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    df = df.sort_index()

    return df