import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(X: pd.DataFrame, y: pd.Series, model) -> tuple:
    """
    Evaluate the given classification model on the provided dataset.

    Returns:
        - DataFrame with:
            original features,
            'true_label',
            'predicted_label',
            'probability_of_<class>' for each class,
            'is_correct' (bool: was prediction correct)
        - accuracy
        - precision (weighted)
        - recall (weighted)
        - f1-score (weighted)
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    class_names = model.classes_

    results_df = X.copy()
    results_df['true_label'] = y
    results_df['predicted_label'] = y_pred

    for i, class_name in enumerate(class_names):
        results_df[f'probability_of_{class_name}'] = list(map(lambda x: round(float(x), 3), y_proba[:, i]))

    results_df['is_correct'] = (y == y_pred)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

    return results_df, accuracy, precision, recall, f1
