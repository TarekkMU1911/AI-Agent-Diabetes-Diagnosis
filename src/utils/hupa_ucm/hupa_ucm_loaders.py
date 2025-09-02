import os
import pandas as pd



# Get numeric columns
def get_numeric_columns(df):
    return df.select_dtypes(include=["float64", "int64"]).columns

# Prepare output directory
def prepare_output(file_name, output_folder, subfolder=""):
    file_base = os.path.splitext(file_name)[0]
    save_dir = os.path.join(output_folder, subfolder, file_base)
    os.makedirs(save_dir, exist_ok=True)
    return file_base, save_dir



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
