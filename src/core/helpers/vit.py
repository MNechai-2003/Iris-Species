import logging
import pandas as pd
from statsmodels.api import OLS
from statsmodels.tools.tools import add_constant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_vif(df: pd.DataFrame, feature_1: str, feature_2: str, upper_threshold: float, lower_threshold: float) -> None:
    X = df[[feature_1]]
    X = add_constant(X)

    Y = df[feature_2]

    model = OLS(Y, X).fit()
    R_2 = model.rsquared

    vit = 1 / (1 - R_2)
    if vit > upper_threshold:
        logger.info(f"High multicollinearity between {feature_1} and {feature_2} value: {vit}")
    elif vit < lower_threshold:
        logger.info(f"Low multicollinearity between {feature_1} and {feature_2} value: {vit}")
    else:
        logger.info(f"Normal multicollinearity between {feature_1} and {feature_2} value: {vit}")
