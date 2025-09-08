import os
import pandas as pd
import numpy as np
from src.utils.hupa_ucm.hupa_ucm_loaders import get_numeric_columns, prepare_output

def generate_column_summary(df, file_name, output_folder):
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) == 0:
        print(f"No numeric columns in {file_name}")
        return

    file_base, save_dir = prepare_output(file_name, output_folder, "summaries")

    summary_data = []

    for col in numeric_cols:
        col_data = df[col]
        col_min = col_data.min()
        col_max = col_data.max()
        p1 = np.percentile(col_data, 1)
        p99 = np.percentile(col_data, 99)
        zeros = (col_data == 0).sum()
        below_1 = (col_data < p1).sum()
        above_99 = (col_data > p99).sum()

        summary_data.append({
            "Patient": os.path.splitext(file_name)[0],
            "Column": col,
            "Min": col_min,
            "1%": p1,
            "99%": p99,
            "Max": col_max,
            "Zeros": zeros,
            "Below_1%": below_1,
            "Above_99%": above_99
        })

    summary_df = pd.DataFrame(summary_data)

    save_path = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}_summary.csv")
    summary_df.to_csv(save_path, index=False)
    print(f"Saved summary for {file_name} in {save_path}")
