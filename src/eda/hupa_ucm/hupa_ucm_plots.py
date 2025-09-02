import os
import matplotlib.pyplot as plt

#  get numeric columns & create output directory
def prepare_plotting(df, file_name, output_folder):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns to plot.")
        return None, None
    file_base = os.path.splitext(file_name)[0] # get file name without extension
    save_dir = os.path.join(output_folder, file_base)
    os.makedirs(save_dir, exist_ok=True)
    return numeric_cols, save_dir

# Create and save boxplots for numeric df columns to visualize outliers
def plot_outliers(df, file_name, output_folder="../../../Datasets/HUPA-UCM Diabetes Dataset/EDA_Outputs/outliers"):
    numeric_cols,save_dir = prepare_plotting(df, file_name, output_folder)
    if numeric_cols is None:
        return

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
    numeric_cols, save_dir = prepare_plotting(df, file_name, output_folder)
    if numeric_cols is None:
        return

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.savefig(os.path.join(save_dir, f"{col}_distribution.png"), bbox_inches="tight")
        plt.close()

    print(f"Saved distribution plots for {file_name} in {save_dir}")