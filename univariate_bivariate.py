import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load both datasets
competitor_df = pd.read_csv("competitor_dataset.csv")  # Update file path
historical_df = pd.read_csv("historical_dataset.csv")  # Update file path

# Convert 'Date' to datetime format
competitor_df["Date"] = pd.to_datetime(competitor_df["Date"])
historical_df["Date"] = pd.to_datetime(historical_df["Date"])

# ========================== #
#   1️⃣ UNIVARIATE ANALYSIS   #
# ========================== #

def univariate_analysis(df, dataset_name):
    """Perform Univariate Analysis on a given dataset."""
    print(f"\n🔹 Univariate Analysis for {dataset_name}")

    # Summary statistics
    print("Summary Statistics:")
    print(df.describe())

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Histograms
    numeric_cols = df.select_dtypes(include=['number']).columns  # Select numerical columns
    df[numeric_cols].hist(figsize=(12, 6), bins=30, edgecolor='black')
    plt.suptitle(f"Histograms for {dataset_name}", fontsize=14)
    plt.show()

    # Boxplots for outlier detection
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df[numeric_cols])
    plt.title(f"Boxplot of Numeric Variables in {dataset_name}")
    plt.show()

# Perform Univariate Analysis on both datasets
univariate_analysis(competitor_df, "Competitor Dataset")
univariate_analysis(historical_df, "Historical Dataset")

# ========================= #
#   2️⃣ BIVARIATE ANALYSIS   #
# ========================= #

def bivariate_analysis(df, dataset_name, x_col, y_col, hue_col=None):
    """Perform Bivariate Analysis on a given dataset."""
    print(f"\n🔹 Bivariate Analysis for {dataset_name}")

    # Correlation Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Matrix for {dataset_name}")
    plt.show()

    # Scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.7)
    plt.title(f"{y_col} vs. {x_col} in {dataset_name}")
    plt.show()

    # Boxplot (if category column is available)
    if hue_col:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=hue_col, y=y_col)
        plt.xticks(rotation=45)
        plt.title(f"Comparison of {y_col} Across {hue_col} in {dataset_name}")
        plt.show()

# Perform Bivariate Analysis on Competitor Dataset
bivariate_analysis(competitor_df, "Competitor Dataset", "Selling Price", "Volume Sold", "Company")

# Perform Bivariate Analysis on Historical Dataset
bivariate_analysis(historical_df, "Historical Dataset", "Open", "Close", "Symbol")
bivariate_analysis(historical_df, "Historical Dataset", "Close", "Volume", "Symbol")
