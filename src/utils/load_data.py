import pandas as pd


def load_data_csv(path: str = "data.csv"):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
