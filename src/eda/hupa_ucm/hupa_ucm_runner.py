import os
import sys
sys.path.append('../../utils/hupa_ucm')
from utils.hupa_ucm.hupa_ucm_loaders import load_csv


def run_eda_on_file(df, file_name, funcs):
    print("*" * 60)
    print(f"File: {file_name}")

    for func in funcs:
        func(df, file_name)


def run_eda_on_folder(data_folder, funcs, sep=";", encoding="utf-8"):
    if not os.path.exists(data_folder):
        print("Folder not found:", data_folder)
        return

    csv_files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    print("Found CSV files:", csv_files)
    print('*' * 20)

    for file in csv_files:
        path = os.path.join(data_folder, file)
        df = load_csv(path, sep=sep, encoding=encoding)

        if df is not None:
            run_eda_on_file(df, file, funcs)
        print('*' * 20)