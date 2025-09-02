import os
import pandas as pd
import numpy as np

def basic_cleaning(df):
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    df = df.drop_duplicates()

    # Convert 'time' to datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    else:
        raise ValueError("'time' column not found in dataframe")

    df = df.sort_values('time').reset_index(drop=True)
    return df


def add_time_deltas(df):
    df['time_delta_min'] = df['time'].diff().dt.total_seconds().fillna(0) / 60
    return df


def add_deltas(df, cols):
    for col in cols:
        df[f'{col}_delta'] = df[col].diff().fillna(0)
    return df


def add_cumulative_features(df):
    # Make sure 'time' is datetime
    if not np.issubdtype(df['time'].dtype, np.datetime64):
        df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Use rolling on datetime index
    df = df.set_index('time')
    df['steps_cumsum_1h'] = df['steps'].rolling('60min').sum().fillna(0)
    df['carb_input_cumsum_1h'] = df['carb_input'].rolling('60min').sum().fillna(0)
    df['glucose_roll_mean_1h'] = df['glucose'].rolling('60min').mean().fillna(0)
    df['glucose_roll_std_1h'] = df['glucose'].rolling('60min').std().fillna(0)
    df = df.reset_index()
    return df


def normalize_features(df, cols):
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_norm'] = (df[col] - mean) / std
    return df

def add_event_flags(df):
    df['meal_flag'] = (df['carb_input'] > 0).astype(int)
    df['insulin_flag'] = (df['bolus_volume_delivered'] > 0).astype(int)
    df['activity_flag'] = (df['steps'] > 0).astype(int)
    return df


def to_text_daily(df, patient_name):
    # Add a date column
    df['date'] = df['time'].dt.date

    daily_texts = []

    for date, group in df.groupby('date'):
        # Average glucose and heart rate
        avg_glucose = group['glucose'].mean()
        min_glucose = group['glucose'].min()
        max_glucose = group['glucose'].max()

        avg_hr = group['heart_rate'].mean()
        total_steps = group['steps'].sum()
        total_carbs = group['carb_input'].sum()
        total_insulin = group['bolus_volume_delivered'].sum()

        # Count events
        meals = group['meal_flag'].sum()
        activity_periods = group['activity_flag'].sum()
        insulin_periods = group['insulin_flag'].sum()
        hypos = (group['glucose'] < 70).sum()
        hypers = (group['glucose'] > 180).sum()

        text = (
            f"Patient {patient_name} on {date}: "
            f"average glucose {avg_glucose:.1f} mg/dL (min {min_glucose}, max {max_glucose}), "
            f"average heart rate {avg_hr:.1f} bpm, total steps {total_steps}, "
            f"total carbs {total_carbs} g, total insulin {total_insulin} units. "
            f"Meals logged: {meals}, physical activity periods: {activity_periods}, "
            f"insulin doses given: {insulin_periods}.",
            f" Hypoglycemia events: {hypos}, Hyperglycemia events: {hypers}."

        )
        daily_texts.append(text)

    return daily_texts

def preprocess_patient_csv(file_path, output_folder):
    df = pd.read_csv(file_path, sep=';')
    df = basic_cleaning(df)
    df = add_time_deltas(df)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    df = add_deltas(df, numeric_cols)
    df = add_cumulative_features(df)
    df = add_event_flags(df)

    # Save preprocessed CSV
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.basename(file_path)
    save_csv_path = os.path.join(output_folder, base_name)
    df.to_csv(save_csv_path, index=False)
    print(f"Saved preprocessed CSV: {save_csv_path}")

    # Generate daily text summaries
    patient_name = base_name.split('.')[0]
    daily_texts = to_text_daily(df, patient_name)

    # Save text file per patient
    save_txt_path = os.path.join(output_folder, f"{patient_name}.txt")
    with open(save_txt_path, 'w') as f:
        for line in daily_texts:
            # Each element in line is a tuple (main text, hypo/hyper)
            f.write(''.join(line) + '\n')
    print(f"Saved daily summary text: {save_txt_path}")

    return daily_texts

def preprocess_all_patients(data_folder, output_folder):
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder, file)
            preprocess_patient_csv(file_path, output_folder)
