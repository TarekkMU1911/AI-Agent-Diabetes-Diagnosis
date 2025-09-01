import os
import pandas as pd


def load_csv(path, sep=";", encoding="utf-8"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path, sep=sep, encoding=encoding)
        return df
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None


def load_all_csvs(folder, sep=";", encoding="utf-8"):

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    dataframes = {}
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            df = load_csv(path, sep=sep, encoding=encoding)
            if df is not None:
                dataframes[file] = df
    return dataframes
