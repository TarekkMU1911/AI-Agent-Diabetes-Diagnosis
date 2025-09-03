import pandas as pd
import numpy as np

def basic_cleaning(df):
    df.columns = df.columns.str.strip()

    df = df.drop_duplicates()

    # Convert 'time' to datetime to use in the rolling functions
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    else:
        raise ValueError("(time) column not found in dataframe")

    df = df.sort_values('time').reset_index(drop=True)
    return df

#difference in minutes
def add_time_deltas(df):
    #in minutes

    df['time_delta_min'] = df['time'].diff().dt.total_seconds().fillna(0) / 60
    return df


def add_deltas(df, cols):
    for col in cols:
        df[f'{col}_delta'] = df[col].diff().fillna(0)
    return df


def add_cumulative_features(df):
    if not np.issubdtype(df['time'].dtype, np.datetime64):
        df['time'] = pd.to_datetime(df['time'], errors='coerce')

    #  datetime index
    # we can use rolling with a time
    # offset like '60min'
    df = df.set_index('time')
    df['steps_cum_sum_1h'] = df['steps'].rolling('60min').sum().fillna(0)
    df['carb_input_cum_sum_1h'] = df['carb_input'].rolling('60min').sum().fillna(0)
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
