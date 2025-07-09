import pandas as pd
import numpy as np


def add_petal_ratio_feature(iris_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new feature 'petal_ratio' to the iris DataFrame
    """
    df = iris_df.copy()
    df['petal_ration'] = df['PetalLengthCm'] / (df['PetalWidthCm'] + 1e-6)
    columns = df.columns.to_list()
    columns[-2], columns[-1] = columns[-1], columns[-2]
    df = df[columns]
    return df


def add_sepal_area_feature(iris_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new feature 'sepal_area' to the iris DataFrame
    """
    df = iris_df.copy()
    df['sepal_area'] = df['SepalLengthCm'] * df['SepalWidthCm']
    columns = df.columns.to_list()
    columns[-2], columns[-1] = columns[-1], columns[-2]
    df = df[columns]
    return df


def add_petal_area_feature(iris_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new feature 'petal_area' to the iris DataFrame
    """
    df = iris_df.copy()
    df['petal_area'] = df['PetalLengthCm'] * df['PetalWidthCm']
    columns = df.columns.to_list()
    columns[-2], columns[-1] = columns[-1], columns[-2]
    df = df[columns]
    return df