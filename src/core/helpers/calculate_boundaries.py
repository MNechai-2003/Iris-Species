import numpy as np
import pandas as pd
from typing import Union, List, Tuple


def boundaries(feature: pd.Series) -> Union[tuple, None]:
    feature = pd.to_numeric(feature, errors='coerce')
    feature = feature.dropna()

    if len(feature) == 0:
        return None

    percentile_25 = np.percentile(feature, 25)
    percentile_75 = np.percentile(feature, 75)

    inter_quartile_range = percentile_75 - percentile_25
    if inter_quartile_range == 0:
        return None

    lower_bound = percentile_25 - 1.5 * inter_quartile_range
    upper_bound = percentile_75 + 1.5 * inter_quartile_range
    return (lower_bound, upper_bound)


def compress_boundaries_by_class(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
    true_label_col: str,
    quantile_lower: float = 0.1,
    quantile_upper: float = 0.9,
    compression_factor: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compresses the boundaries of feature values based on their class-wise distributions
    to improve linear separability.
    """
    train_df_transformed = train_df.copy()
    test_df_transformed = test_df.copy()

    for feature in feature_names:
        compressed_feature_col = f"compressed_{feature}"
        train_df_transformed[compressed_feature_col] = np.nan
        test_df_transformed[compressed_feature_col] = np.nan

        for class_name in train_df_transformed[true_label_col].unique():
            train_mask = train_df_transformed[true_label_col] == class_name
            feature_train_series = train_df_transformed.loc[train_mask, feature]

            lower_bound = feature_train_series.quantile(quantile_lower)
            upper_bound = feature_train_series.quantile(quantile_upper)
            
            compressed_train = np.where(
                (feature_train_series >= lower_bound) & (feature_train_series <= upper_bound),
                lower_bound + (feature_train_series - lower_bound) * compression_factor,
                feature_train_series
            )
            train_df_transformed.loc[train_mask, compressed_feature_col] = compressed_train

            test_mask = test_df_transformed[true_label_col] == class_name
            feature_test_series = test_df_transformed.loc[test_mask, feature]

            compressed_test = np.where(
                (feature_test_series >= lower_bound) & (feature_test_series <= upper_bound),
                lower_bound + (feature_test_series - lower_bound) * compression_factor,
                feature_test_series
            )
            test_df_transformed.loc[test_mask, compressed_feature_col] = compressed_test

        train_df_transformed.drop(columns=feature, inplace=True)
        test_df_transformed.drop(columns=feature, inplace=True)

    return train_df_transformed, test_df_transformed
