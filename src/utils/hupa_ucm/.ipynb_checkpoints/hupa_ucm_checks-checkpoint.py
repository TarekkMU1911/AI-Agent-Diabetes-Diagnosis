import pandas as pd

def show_shape(df):
    print("\n Shape (rows, cols):", df.shape)

def show_columns(df):
    print("\n Columns:", df.columns.tolist())

def show_head_tail(df):
    print("\n Head:")
    print(df.head())
    print("\n Tail:")
    print(df.tail())

def show_info(df):
    print("\n Info:")
    print(df.info())

def check_missing(df):
    print("\n Missing values:")
    print(df.isnull().sum())

def show_describe(df):
    print("\n Describe (summary stats):")
    print(df.describe(include="all"))

def check_duplicates(df):
    print("\n Duplicates:", df.duplicated().sum())
