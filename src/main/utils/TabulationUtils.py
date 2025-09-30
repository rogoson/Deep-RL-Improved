import os
import pandas as pd


def format_latex_table(df):
    """
    Formats a Pandas DataFrame into a LaTeX table, boldfacing the maximum values in each numeric column.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing numeric and non-numeric columns.

    Returns:
    str: A LaTeX-formatted table as a string.
    """
    # Identify numeric columns
    numericCols = df.select_dtypes(include="number").columns

    # Boldface maximum values in numeric columns
    for col in numericCols:
        if col in ["Mean MDD (%)", "Std Dev (%)", "Maximum Drawdown"]:
            value = df[col].min()
        else:
            value = df[col].max()
        df[col] = df[col].apply(
            lambda x: f"\\textbf{{{x}}}" if x == value else str(x)
        )  # Apply LaTeX bold formatting

    # Convert to LaTeX format with escape=False to preserve LaTeX commands
    return df.to_latex(index=False, escape=False)


# # Example usage
# df = pd.DataFrame({
#     "Ticker": ["AAPL", "TSLA", "GOOGL"],
#     "Price": [150.25, 248.66, 122.87],
#     "Volume": [2000000, 1500000, 1800000]
# })

# latexTable = format_latex_table(df)
# print(latexTable)
