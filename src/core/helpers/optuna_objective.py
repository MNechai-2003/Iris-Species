import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def objective_for_logistic_regression(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> float:
    """
    Objective function for Optuna optimization for LogisticRegression.

    Parameters:
        trial (optuna.Trial): The Optuna trial object.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        float: The objective value to minimize.
    """
    C = trial.suggest_loguniform('C', 1e-4, 1e2)
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1'])
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        class_weight=class_weight,
        max_iter=max_iter,
        solver=solver
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='weighted')


def objective_for_xgboost_classifier(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> float:
    """
    Objective function for Optuna optimization for XGBoostClassifier.

    Parameters:
        trial (optuna.Trial): The Optuna trial object.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        float: The objective value to minimize.
    """
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.5)
    subsample = trial.suggest_uniform('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.5, 1.0)
    gamma = trial.suggest_loguniform('gamma', 1e-8, 1.0)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-8, 10.0)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_child_weight=min_child_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return f1_score(y_test, y_pred, average='weighted')
