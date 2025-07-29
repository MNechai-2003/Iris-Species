import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold
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


def objective_for_bagging_classifier(trial, X_train, y_train, best_base_params_lr):
    """
    Objective function for Optuna to optimize hyperparameters of BaggingClassifier.
    Uses the already optimized Logistic Regression as the base estimator.
    """
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_samples = trial.suggest_float('max_samples', 0.5, 1.0)

    base_estimator = LogisticRegression(**best_base_params_lr, random_state=41)

    bagging_model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=1.0,
        bootstrap=True,
        random_state=41,
        n_jobs=-1
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
    f1_scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        bagging_model.fit(X_train_fold, y_train_fold)
        val_predictions = bagging_model.predict(X_val_fold)
        f1_scores.append(f1_score(y_val_fold, val_predictions, average='weighted'))

    return np.mean(f1_scores)