"""
Main EDA Runner for HUPA-UCM Diabetes Dataset
Integrates all EDA modules for comprehensive analysis
"""

import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.hupa_ucm.hupa_ucm_loaders import load_all_csvs
from src.utils.hupa_ucm.hupa_ucm_checks import (
    show_shape, show_columns, show_head_tail, show_info,
    check_missing, show_describe, check_duplicates
)
from src.eda.hupa_ucm.hupa_ucm_plots import plot_outliers, plot_distributions
from src.eda.hupa_ucm.hupa_ucm_summary import (
    generate_summary_stats, generate_column_summary,
    generate_correlation_summary, generate_value_counts_summary,
    generate_complete_summary
)


class HUPAEDARunner:
    """
    Complete EDA runner for HUPA-UCM Diabetes Dataset
    """

    def __init__(self, data_folder, output_base="../Datasets/HUPA-UCM Diabetes Dataset/EDA_Outputs"):
        self.data_folder = data_folder
        self.output_base = output_base
        self.dataframes = {}

        # Create output directories
        self.output_dirs = {
            'summaries': os.path.join(output_base, 'summaries'),
            'outliers': os.path.join(output_base, 'outliers'),
            'distributions': os.path.join(output_base, 'distributions'),
            'reports': os.path.join(output_base, 'reports')
        }

        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def load_data(self):
        """Load all CSV files from the data folder"""
        try:
            self.dataframes = load_all_csvs(self.data_folder)
            print(f"✅ Loaded {len(self.dataframes)} CSV files:")
            for filename in self.dataframes.keys():
                print(f"   - {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def run_basic_checks(self):
        """Run basic data quality checks on all datasets"""
        print("\n" + "=" * 80)
        print("BASIC DATA QUALITY CHECKS")
        print("=" * 80)

        for filename, df in self.dataframes.items():
            print(f"\n{'=' * 60}")
            print(f"DATASET: {filename}")
            print(f"{'=' * 60}")

            show_shape(df)
            show_columns(df)
            show_info(df)
            check_missing(df)
            check_duplicates(df)

    def run_statistical_analysis(self):
        """Run statistical analysis on all datasets"""
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS")
        print("=" * 80)

        for filename, df in self.dataframes.items():
            print(f"\n{'=' * 60}")
            print(f"DATASET: {filename}")
            print(f"{'=' * 60}")

            show_describe(df)

    def generate_summaries(self):
        """Generate comprehensive summaries for all datasets"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE SUMMARIES")
        print("=" * 80)

        for filename, df in self.dataframes.items():
            generate_complete_summary(df, filename, self.output_dirs['summaries'])

    def create_visualizations(self):
        """Create all visualizations for the datasets"""
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)

        for filename, df in self.dataframes.items():
            print(f"\n📊 Creating plots for {filename}...")

            # Create outlier plots
            plot_outliers(df, filename, self.output_dirs['outliers'])

            # Create distribution plots
            plot_distributions(df, filename, self.output_dirs['distributions'])

    def generate_report(self):
        """Generate a comprehensive EDA report"""
        report_path = os.path.join(self.output_dirs['reports'], 'eda_report.txt')

        with open(report_path, 'w') as f:
            f.write("HUPA-UCM Diabetes Dataset - EDA Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total datasets analyzed: {len(self.dataframes)}\n")
            f.write(f"Output directory: {self.output_base}\n\n")

            f.write("Datasets:\n")
            for filename, df in self.dataframes.items():
                f.write(f"- {filename}: {df.shape[0]} rows, {df.shape[1]} columns\n")

            f.write(f"\nGenerated outputs:\n")
            f.write(f"- Summary statistics: {self.output_dirs['summaries']}\n")
            f.write(f"- Outlier plots: {self.output_dirs['outliers']}\n")
            f.write(f"- Distribution plots: {self.output_dirs['distributions']}\n")

        print(f"\n📄 EDA report saved to: {report_path}")

    def run_complete_eda(self):
        """Run the complete EDA pipeline"""
        print("🚀 Starting Complete EDA Pipeline for HUPA-UCM Diabetes Dataset")
        print("=" * 80)

        # Step 1: Load data
        if not self.load_data():
            return False

        # Step 2: Basic checks
        self.run_basic_checks()

        # Step 3: Statistical analysis
        self.run_statistical_analysis()

        # Step 4: Generate summaries
        self.generate_summaries()

        # Step 5: Create visualizations
        self.create_visualizations()

        # Step 6: Generate report
        self.generate_report()

        print("\n" + "=" * 80)
        print("✅ EDA PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📁 All outputs saved to: {self.output_base}")
        print("=" * 80)

        return True

    def run_quick_overview(self):
        """Run a quick overview of all datasets"""
        if not self.load_data():
            return False

        print("\n" + "=" * 80)
        print("QUICK DATASET OVERVIEW")
        print("=" * 80)

        for filename, df in self.dataframes.items():
            print(f"\n📋 {filename}:")
            print(f"   Shape: {df.shape}")
            print(f"   Missing values: {df.isnull().sum().sum()}")
            print(f"   Duplicates: {df.duplicated().sum()}")
            print(f"   Numeric columns: {len(df.select_dtypes(include=['float64', 'int64']).columns)}")
            print(f"   Categorical columns: {len(df.select_dtypes(include=['object', 'category']).columns)}")


def main():
    """Main execution function"""

    # Configuration
    data_folder = "../Datasets/HUPA-UCM Diabetes Dataset/Preprocessed"

    # Initialize EDA runner
    eda_runner = HUPAEDARunner(data_folder)

    # Run complete EDA
    eda_runner.run_complete_eda()


if __name__ == "__main__":
    main()