import pandas as pd


DATASET_PATH = "data.csv"


def load_data_csv():
    try:
        return pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=["km", "price"])
