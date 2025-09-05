import os
import matplotlib.pyplot as plt

from src.utils.hupa_ucm.hupa_ucm_loaders import get_numeric_columns, prepare_output


# Create and save boxplots for numeric df columns to visualize outliers
def plot_outliers(df, file_name, output_folder="../../../Datasets/HUPA-UCM Diabetes Dataset/EDA_Outputs/outliers"):
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) == 0:
        print(f"No numeric columns in {file_name}")
        return

    file_base, save_dir = prepare_output(file_name, output_folder, "outliers")

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.boxplot(df[col], vert=False)
        plt.title(f"Outliers in {col}")
        plt.xlabel(col)
        plt.savefig(os.path.join(save_dir, f"{col}_outliers.png"), bbox_inches="tight")
        plt.close()

    print(f"Saved outlier plots for {file_name} in {save_dir}")

# Create and save histogram for numeric df columns to visualize outliers

def plot_distributions(df, file_name, output_folder="../../../Datasets/HUPA-UCM Diabetes Dataset/EDA_Outputs/distributions"):
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) == 0:
        print(f"No numeric columns in {file_name}")
        return

    file_base, save_dir = prepare_output(file_name, output_folder, "outliers")

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.savefig(os.path.join(save_dir, f"{col}_distribution.png"), bbox_inches="tight")
        plt.close()

    print(f"Saved distribution plots for {file_name} in {save_dir}")