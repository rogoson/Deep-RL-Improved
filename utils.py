import os
from IPython.core.display import display_html
from tabulate import tabulate
import pandas as pd


def pathJoin(firstStr: str, secondStr: str):
    return os.path.join(str(firstStr), secondStr)


"""
The below is robbed from CM50268 Coursework 1 & 2 Setup Code
"""


def tabulate_neatly(table, headers=None, title=None, **kwargs):
    # Example Usage:
    # table = [["Column 1","Column 2"]]
    # table.append([Column_1_Value, Column_2_Value])
    # setup.tabulate_neatly(table, headers="firstrow", title="Table Title")
    headers = headers or "keys"
    if title is not None:
        display_html(f"<h3>{title}</h3>\n", raw=True)
    display_html(tabulate(table, headers=headers, tablefmt="html", **kwargs))


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
        maxValue = df[col].max()
        df[col] = df[col].apply(
            lambda x: f"\\textbf{{{x}}}" if x == maxValue else str(x)
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
