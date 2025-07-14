import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, SelectKBest


from typing import Tuple


def generate_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k: int = 5
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Generate top k most impactful features for classification across train + test datasets,
    and return them preserving original train/test splits.
    """

    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)

    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
    numeric_features = X_full.select_dtypes(include=np.number).columns.tolist()

    X_processed = X_full[numeric_features].copy()
    for i in range(len(numeric_features)):
        for j in range(i + 1, len(numeric_features)):
            col1 = numeric_features[i]
            col2 = numeric_features[j]
            X_processed[f'{col1}/{col2}'] = X_processed[col1] / (X_processed[col2] + 1e-8)
            X_processed[f'{col1}*{col2}'] = X_processed[col1] * X_processed[col2]

    X_processed = X_processed.replace([np.inf, -np.inf], np.nan).fillna(0)

    selector = SelectKBest(score_func=f_classif, k=min(k, X_processed.shape[1]))
    selector.fit(X_processed, y_full)
    selected_columns = X_processed.columns[selector.get_support(indices=True)].tolist()

    X_selected = X_processed[selected_columns].copy()
    X_selected.columns = [f"generated_{col}" for col in X_selected.columns]

    n_train = len(X_train)
    X_train_transformed = X_selected.iloc[:n_train].reset_index(drop=True).round(3)
    X_test_transformed = X_selected.iloc[n_train:].reset_index(drop=True).round(3)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train_transformed, y_train, X_test_transformed, y_test

