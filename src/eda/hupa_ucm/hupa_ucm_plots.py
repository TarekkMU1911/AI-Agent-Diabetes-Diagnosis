import os
import matplotlib.pyplot as plt


def plot_outliers(df, file_name, output_folder="../../../Datasets/HUPA-UCM Diabetes Dataset/EDA_Outputs/outliers"):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns to plot.")
        return

    file_base = os.path.splitext(file_name)[0]
    save_dir = os.path.join(output_folder, file_base)
    os.makedirs(save_dir, exist_ok=True)

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.boxplot(df[col], vert=False)
        plt.title(f"Outliers in {col}")
        plt.xlabel(col)
        plt.savefig(os.path.join(save_dir, f"{col}_outliers.png"), bbox_inches="tight")
        plt.close()

    print(f"Saved outlier plots for {file_name} in {save_dir}")


def plot_distributions(df, file_name, output_folder="../../../Datasets/HUPA-UCM Diabetes Dataset/EDA_Outputs/distributions"):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns to plot.")
        return

    file_base = os.path.splitext(file_name)[0]
    save_dir = os.path.join(output_folder, file_base)
    os.makedirs(save_dir, exist_ok=True)

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.savefig(os.path.join(save_dir, f"{col}_distribution.png"), bbox_inches="tight")
        plt.close()

    print(f"Saved distribution plots for {file_name} in {save_dir}")