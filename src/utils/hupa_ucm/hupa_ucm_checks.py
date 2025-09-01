import pandas as pd

def show_shape(df, file_name=None):
    print("\n Shape (rows, cols):", df.shape)

def show_columns(df, file_name=None):
    print("\n Columns:", df.columns.tolist())

def show_head_tail(df, file_name=None):
    print("\n Head:")
    print(df.head())
    print("\n Tail:")
    print(df.tail())

def show_info(df, file_name=None):
    print("\n Info:")
    print(df.info())

def check_missing(df, file_name=None):
    print("\n Missing values:")
    print(df.isnull().sum())

def show_describe(df, file_name=None):
    print("\n Describe (summary stats):")
    print(df.describe(include="all"))

def check_duplicates(df, file_name=None):
    print("\n Duplicates:", df.duplicated().sum())

def check_outliers(df, file_name=None):
    print("\n Possible outliers (values above 99th percentile):")
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        high = df[col].quantile(0.99)
        print(f"{col}: >{high}")