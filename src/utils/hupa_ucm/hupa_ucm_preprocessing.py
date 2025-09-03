import os
import json

from src.utils.hupa_ucm.hupa_ucm_loaders import get_numeric_columns, prepare_output
from src.utils.hupa_ucm.hupa_ucm_daily_reports import to_daily_reports
from src.utils.hupa_ucm.hupa_ucm_features import *

#convert numpy to (python values) to use with json

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def preprocess_patient_csv(file_path, output_folder, return_json=False):
    df = pd.read_csv(file_path, sep=';')

    #clean & feature engineering
    df = basic_cleaning(df)
    df = add_time_deltas(df)
    numeric_cols = get_numeric_columns(df)
    df = add_deltas(df, numeric_cols)
    df = add_cumulative_features(df)
    df = add_event_flags(df)

    #save preprocessed csv
    base_name = os.path.basename(file_path)
    file_base, save_dir = prepare_output(base_name, output_folder, "preprocessed")
    save_csv_path = os.path.join(save_dir, f"{file_base}_preprocessed.csv")
    df.to_csv(save_csv_path, index=False)
    print(f"Saved preprocessed CSV: {save_csv_path}")

    #convert to daily reports


    patient_name = file_base.split('.')[0]
    daily_texts, daily_jsons = to_daily_reports(df, patient_name, return_json=True)

    #save daily reports in
    text_dir = os.path.join(save_dir, "text")
    json_dir = os.path.join(save_dir, "json")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # json file

    save_json_path = os.path.join(json_dir, f"{patient_name}_daily.json")
    with open(save_json_path, 'w') as file:
        json.dump(daily_jsons, file, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x))
    print(f"Saved daily JSON: {save_json_path}")

    # text file
    save_txt_path = os.path.join(text_dir, f"{patient_name}_daily.txt")
    with open(save_txt_path, 'w') as file:
        for line in daily_texts:
            file.write(line + '\n')
    print(f"Saved daily text: {save_txt_path}")

    return df, daily_texts, daily_jsons

def preprocess_all_patients(data_folder, output_folder):
    all_results = []
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder, file)
            df, daily_texts, daily_jsons = preprocess_patient_csv(file_path, output_folder)
            all_results.append((df, daily_texts, daily_jsons))
    return all_results