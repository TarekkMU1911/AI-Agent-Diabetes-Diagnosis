import os
from src.utils.hupa_ucm.hupa_ucm_loaders import load_csv

# to apply a list of EDA to a single  df
def run_eda_on_file(df, file_name, funcs):
    print(f"File: {file_name}")

    for function in funcs:
        function(df, file_name)

#apply EDA functions to all CSV files in a folder
def run_eda_on_folder(data_folder, funcs, sep=";", encoding="utf-8"):
    if not os.path.exists(data_folder):
        print("Folder not found:", data_folder)
        return

    csv_files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    print("Found CSV files:", csv_files)

    for file in csv_files:
        path = os.path.join(data_folder, file)
        df = load_csv(path, sep=sep, encoding=encoding)

        if df is not None:
            run_eda_on_file(df, file, funcs)
