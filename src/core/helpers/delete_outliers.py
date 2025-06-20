import pandas as pd


def delete_outliers(df: pd.DataFrame, feature: str, lower_bound: float, upper_bound: float) -> pd.DataFrame:
    """Remove outliers based on the given bounds."""

    clean_iris_df = df[(df[feature] > lower_bound) & (df[feature] < upper_bound)]
    print(f"Number of outliers in {feature}: {len(df) - len(clean_iris_df)}")
    return clean_iris_df