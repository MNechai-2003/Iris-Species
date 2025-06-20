import numpy as np
import pandas as pd
from typing import Union


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
