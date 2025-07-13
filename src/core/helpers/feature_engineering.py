import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, SelectKBest


def generate_top_features(
    df: pd.DataFrame,
    target_column: str = 'Species',
    k: int = 5
) -> pd.DataFrame:
    """Generate top k the most impactive features for classification task."""
    y = df[target_column]
    X = df.drop(columns=[target_column])

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()

    X_processed = X[numeric_features].copy()

    for i in range(len(numeric_features)):
        for j in range(i + 1, len(numeric_features)): 
            col1 = numeric_features[i]
            col2 = numeric_features[j]

            X_processed[f'{col1}/{col2}'] = X_processed[col1] / (X_processed[col2] + 1e-8)
            X_processed[f'{col1}*{col2}'] = X_processed[col1] * X_processed[col2]

    X_for_selection = X_processed.replace([np.inf, -np.inf], np.nan).fillna(0)

    selector = SelectKBest(score_func=f_classif, k=min(k, X_for_selection.shape[1]))

    selector.fit(X_for_selection, y)

    selected_feature_indices = selector.get_support(indices=True)
    top_feature_names = X_for_selection.columns[selected_feature_indices].tolist()

    result_df = X_for_selection[top_feature_names].copy()
    result_df[target_column] = y

    return result_df
