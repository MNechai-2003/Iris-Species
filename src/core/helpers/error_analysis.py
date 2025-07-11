import sys
import pandas as pd
import numpy as np
from typing import List
from scipy.stats import shapiro, norm, mannwhitneyu

sys.path.append("/Users/nikolaynechay/Iris-Species")

from src.core.helpers.visualizations import plot_distribution_of_logits


def create_detailed_analysis(
    df: pd.DataFrame,
    weights: np.ndarray,
    bias: np.ndarray, 
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Method for creating a detailed DataFrame with all required information for error analyzing.
    Calculates logits, probabilities, feature contributions, and their differences for misclassified samples.

    Args:
        df (pd.DataFrame): Input DataFrame containing feature values, 'true_label', and 'predicted_label'.
        weights (np.ndarray): Model weights (coefficients) of shape (num_classes, num_features).
        bias (np.ndarray): Model bias (intercepts) of shape (num_classes,).
        feature_names (List[str]): List of names of the features used in the model.

    Returns:
        pd.DataFrame: A detailed DataFrame for error analysis.
    """
    detailed_analysis = []

    if not isinstance(weights, np.ndarray):
        weights = np.asarray(weights, dtype=np.float64)
    if not isinstance(bias, np.ndarray):
        bias = np.asarray(bias, dtype=np.float64)

    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    elif weights.ndim != 2:
        raise ValueError(f"Weights must be 1D or 2D. Got shape {weights.shape}.")

    if bias.ndim == 0:
        bias = np.array([bias])
    elif bias.ndim != 1:
        raise ValueError(f"Bias must be 0D (scalar) or 1D. Got shape {bias.shape}.")

    num_classes = weights.shape[0]
    for idx, row in df.iterrows():
        features = row[feature_names].values.astype(np.float64)

        true_label = int(row['true_label'])
        pred_label = int(row['predicted_label'])

        # Calculate logits: (num_classes, num_features) @ (num_features,) -> (num_classes,)
        logits = np.dot(weights, features) + bias

        # Softmax for numerical stability
        logits_stable = logits - np.max(logits)
        exp_logits = np.exp(logits_stable)
        probabilities = exp_logits / np.sum(exp_logits)

        feature_contributions = weights * features

        error_dict = {
            'original_idx': row.name,
            'true_label': true_label,
            'predicted_label': pred_label
        }

        for c in range(num_classes):
            error_dict[f'logit_class_{c}'] = round(float(logits[c]), 3)
            error_dict[f'probability_class_{c}'] = round(float(probabilities[c]), 3)
            error_dict[f'bias_class_{c}'] = round(float(bias[c]), 3)

        for i, feature_name in enumerate(feature_names):
            error_dict[f'{feature_name}_value'] = round(float(features[i]), 3) # Round value as well
            for c in range(num_classes):
                error_dict[f'{feature_name}_contribution_class_{c}'] = round(float(feature_contributions[c, i]), 3)
                error_dict[f'{feature_name}_weight_class_{c}'] = round(float(weights[c, i]), 3)

        # Calculate contribution differences ONLY for misclassified samples
        if pred_label != true_label:
            for i, feature_name in enumerate(feature_names):
                diff = feature_contributions[pred_label, i] - feature_contributions[true_label, i]
                error_dict[f'contribution_diff_{feature_name}'] = round(float(diff), 3)

            bias_diff = bias[pred_label] - bias[true_label]
            error_dict['bias_diff'] = round(float(bias_diff), 3)

        detailed_analysis.append(error_dict)

    return pd.DataFrame(detailed_analysis)


def analyze_classification_errors_simple(df: pd.DataFrame, feature_names: List[str]) -> None:
    misclassified = df[df['true_label'] != df['predicted_label']].copy()

    print("Analyse classification errors")
    print("=" * 50)
    print(f"Number of errors: {len(misclassified)}")
    print()

    results = {}

    for idx, (_, row) in enumerate(misclassified.iterrows()):
        error_num = idx + 1

        print(f"ERROR #{error_num} (ID: {row['original_idx']})")
        print(f"True class: {int(row['true_label'])} -> Predicted class: {int(row['predicted_label'])}")

        feature_impacts = []
        for feature in feature_names:
            value = row[f'{feature}_value']
            contrib_diff = row[f'contribution_diff_{feature}']

            feature_impacts.append({
                'feature': feature,
                'value': value,
                'contribution_diff': contrib_diff,
                'abs_contribution': abs(contrib_diff)
            })

        # Sort by absolute contribution
        feature_impacts.sort(key=lambda x: x['abs_contribution'], reverse=True)

        print("Feature contributions (sorted by impact):")
        for impact in feature_impacts:
            print(f"  {impact['feature']:<15}: value={impact['value']:.2f}, "
                  f"contribution={impact['contribution_diff']:.3f} ")

        main_culprit = feature_impacts[0]
        print(f"Main impact: {main_culprit['feature']} "
              f"(contribution: {main_culprit['contribution_diff']:.3f})")
        print()

        results[f'error_{error_num}'] = {
            'original_idx': row['original_idx'],
            'true_class': int(row['true_label']),
            'pred_class': int(row['predicted_label']),
            'main_culprit': main_culprit['feature'],
            'main_contribution': main_culprit['contribution_diff'],
            'all_contributions': feature_impacts
        }

    print("\nSUMMARY TABLE OF ERRORS:")
    print("-" * 70)
    print(f"{'Error':<8} {'ID':<5} {'True':<10} {'Predicted':<13} {'Main impact':<15} {'Contribution':<8}")
    print("-" * 70)

    for error_key, error_data in results.items():
        print(f"{error_key:<8} {error_data['original_idx']:<5} "
              f"{error_data['true_class']:<10} {error_data['pred_class']:<13} "
              f"{error_data['main_culprit']:<15} {error_data['main_contribution']:<8.3f}")

    print("-" * 70)

    return results

def calculate_main_statistics(correct_df: pd.DataFrame, columns: list) -> dict:
    """Calculate main statistics for particular columns in correct_df"""
    statistics = {}

    for column in columns:
        statistics[column] = {
            'mean': correct_df[column].mean(),
            'std': correct_df[column].std(),
            'IQR': correct_df[columns].quantile(0.75) - correct_df[columns].quantile(0.25),
            'min': correct_df[column].min(),
            'max': correct_df[column].max()
        }

    return statistics

                        
def analyse_logits_in_incorrect_predictions(df: pd.DataFrame, classes: list, alpha: float = 0.05) -> None:
    """Analyse logit values in incorrect predictions compared to correct predictions."""
    df['is_correct'] = df['true_label'] == df['predicted_label']

    for class_name in classes:
        correct_classified_df = df[(df['is_correct'] == True) & (df['true_label'] == class_name)]
        incorrect_classified_df = df[(df['is_correct'] == False)]

        logit_statistics = calculate_main_statistics(correct_df=correct_classified_df, columns=['logit'])

        plot_distribution_of_logits(correct_classified_df['logit'], class_name)

        for i, row in incorrect_classified_df.iterrows():
            if row['true_label'] == class_name and len(correct_classified_df) > 30:
                print(f"\nAnalyzing sample {i} with respect to TRUE class: '{class_name}'")

                stat, p_value = shapiro(correct_classified_df['logit'])
                if p_value > alpha:
                    print(f"Shapiro–Wilk test PASSED (p = {p_value:.4f}) → Logit dist is normal.")
                    z_score = (row['logit'] - logit_statistics['logit']['mean']) / logit_statistics['logit']['std']
                    p = 2 * (1 - norm.cdf(abs(z_score)))
                    if p >= alpha:
                        print(f"[Z-test] Logit = {row['logit']:.3f}, z = {z_score:.2f}, p = {p:.4f} → Within expected range.")
                    else:
                        print(f"[Z-test] Logit = {row['logit']:.3f}, z = {z_score:.2f}, p = {p:.4f} → Statistically different (reject H₀).")
                else:
                    print(f"Shapiro–Wilk test FAILED (p = {p_value:.4f}) → Non-normal distribution.")
                    _, p = mannwhitneyu(correct_classified_df['logit'], [row['logit']], alternative='two-sided')
                    if p >= alpha:
                        print(f"[Mann–Whitney] p = {p:.4f} → Consistent with correct distribution.")
                    else:
                        print(f"[Mann–Whitney] p = {p:.4f} → Logit is anomalous in TRUE class.")

            if row['predicted_label'] == class_name and len(correct_classified_df) > 30:
                print(f"\nAnalyzing sample {i} with respect to PREDICTED class: '{class_name}'")

                stat, p_value = shapiro(correct_classified_df['logit'])
                if p_value > alpha:
                    print(f"Shapiro–Wilk test PASSED (p = {p_value:.4f}) → Logit dist is normal.")
                    z_score = (row['logit'] - logit_statistics['logit']['mean']) / logit_statistics['logit']['std']
                    p = 2 * (1 - norm.cdf(abs(z_score)))
                    if p >= alpha:
                        print(f"[Z-test] Logit = {row['logit']:.3f}, z = {z_score:.2f}, p = {p:.4f} → Within expected range.")
                    else:
                        print(f"[Z-test] Logit = {row['logit']:.3f}, z = {z_score:.2f}, p = {p:.4f} → Statistically different (reject H₀).")
                else:
                    print(f"Shapiro–Wilk test FAILED (p = {p_value:.4f}) → Non-normal distribution.")
                    _, p = mannwhitneyu(correct_classified_df['logit'], [row['logit']], alternative='two-sided')
                    if p >= alpha:
                        print(f"[Mann–Whitney] p = {p:.4f} → Consistent with predicted class distribution.")
                    else:
                        print(f"[Mann–Whitney] p = {p:.4f} → Logit is anomalous in PREDICTED class.")
                        

def analyse_contributions_in_incorrect_predictions(
    df: pd.DataFrame,
    classes: list,
    feature_names: list,
    alpha: float = 0.05
) -> None:
    """Define the most impactful features in incorrect predictions."""
    contribution_cols = [col for col in df.columns if col.endswith("_contribution")]
    df['is_correct'] = df['true_label'] == df['predicted_label']
    
    for class_name in classes:
        correct_classified_df = df[(df['is_correct'] == True) & (df['true_label'] == class_name)]
        incorrect_classified_df = df[(df['is_correct'] == False)]
    
        contribution_statistics = calculate_main_statistics(correct_df=correct_classified_df, columns=contribution_cols)
        features_statistics = calculate_main_statistics(correct_classified_df, columns=[str(x)+"_value" for x in feature_names])
        for i, row in incorrect_classified_df.iterrows():
            if row['true_label'] == class_name and len(correct_classified_df) > 30:
                print(f"\nAnalyzing sample {i} with respect to TRUE class: '{class_name}'")
                for feature in feature_names:
                    stat, p_value = shapiro(correct_classified_df[str(feature)+'_value'])
                    print(f"Analysing feature: {feature}")
                    if p_value >= alpha:
                        print(f"Shapiro–Wilk test PASSED (p = {p_value:.4f}) → Feature distribution is normal.")
                        z_score = (row[str(feature)+'_value'] - features_statistics[str(feature)+'_value']['mean']) / features_statistics[str(feature)+'_value']['std']
                        p = 2 * (1 - norm.cdf(abs(z_score)))
                        if p >= alpha:
                            print(f"[Z-test] Feature = {row[str(feature)+'_value']:.3f}, z = {z_score:.2f}, p = {p:.4f} → Within expected range.")
                        else:
                            print(f"[Z-test] Feature = {row[str(feature)+'_value']:.3f}, z = {z_score:.2f}, p = {p:.4f} → Statistically different (reject H₀).")
                    else:
                        print(f"Shapiro–Wilk test FAILED (p = {p_value:.4f}) → Non-normal distribution.")
                        _, p = mannwhitneyu(correct_classified_df[str(feature)+"_value"], [row[str(feature)+'_value']], alternative='two-sided')
                        if p >= alpha:
                            print(f"[Mann–Whitney] p = {p:.4f} → Consistent with correct distribution.")
                        else:
                            print(f"[Mann–Whitney] p = {p:.4f} → Logit is anomalous in TRUE class.")
                for contribution in contribution_cols:
                    stat, p_value = shapiro(correct_classified_df[contribution])
                    print(f"Analysing contribution: {contribution}")
                    if p_value >= alpha:
                        print(f"Shapiro–Wilk test PASSED (p = {p_value:.4f}) → Contribution member distribution is normal.")
                        z_score = (row[contribution] - contribution_statistics[contribution]['mean']) / contribution_statistics[contribution]['std']
                        p = 2 * (1 - norm.cdf(abs(z_score)))
                        if p >= alpha:
                            print(f"[Z-test] Contribution member = {row[contribution]:.3f}, z = {z_score:.2f}, p = {p:.4f} → Within expected range.")
                        else:
                            print(f"[Z-test] Contribution membe = {row[contribution]:.3f}, z = {z_score:.2f}, p = {p:.4f} → Statistically different (reject H₀).")
                    else:
                        print(f"Shapiro–Wilk test FAILED (p = {p_value:.4f}) → Non-normal distribution.")
                        _, p = mannwhitneyu(correct_classified_df[contribution], [row[contribution]], alternative='two-sided')
                        if p >= alpha:
                            print(f"[Mann–Whitney] p = {p:.4f} → Consistent with correct distribution.")
                        else:
                            print(f"[Mann–Whitney] p = {p:.4f} → Logit is anomalous in TRUE class.")
            elif row['predicted_label'] == class_name and len(correct_classified_df) > 30:
                print(f"\nAnalyzing sample {i} with respect to PREDICTED class: '{class_name}'")
                for feature in feature_names:
                    stat, p_value = shapiro(correct_classified_df[str(feature)+'_value'])
                    print(f"Analysing feature: {feature}")
                    if p_value >= alpha:
                        print(f"Shapiro–Wilk test PASSED (p = {p_value:.4f}) → Feature distribution is normal.")
                        z_score = (row[str(feature)+'_value'] - features_statistics[str(feature)+'_value']['mean']) / features_statistics[str(feature)+'_value']['std']
                        p = 2 * (1 - norm.cdf(abs(z_score)))
                        if p >= alpha:
                            print(f"[Z-test] Feature = {row[str(feature)+'_value']:.3f}, z = {z_score:.2f}, p = {p:.4f} → Within expected range.")
                        else:
                            print(f"[Z-test] Feature = {row[str(feature)+'_value']:.3f}, z = {z_score:.2f}, p = {p:.4f} → Statistically different (reject H₀).")
                    else:
                        print(f"Shapiro–Wilk test FAILED (p = {p_value:.4f}) → Non-normal distribution.")
                        _, p = mannwhitneyu(correct_classified_df[str(feature)+"_value"], [row[str(feature)+'_value']], alternative='two-sided')
                        if p >= alpha:
                            print(f"[Mann–Whitney] p = {p:.4f} → Consistent with predicted class distribution.")
                        else:
                            print(f"[Mann–Whitney] p = {p:.4f} → Logit is anomalous in PREDICTED class.")
                for contribution in contribution_cols:
                    stat, p_value = shapiro(correct_classified_df[contribution])
                    print(f"Analysing contribution: {contribution}")
                    if p_value >= alpha:
                        print(f"Shapiro–Wilk test PASSED (p = {p_value:.4f}) → Contribution member distribution is normal.")
                        z_score = (row[contribution] - contribution_statistics[contribution]['mean']) / contribution_statistics[contribution]['std']
                        p = 2 * (1 - norm.cdf(abs(z_score)))
                        if p >= alpha:
                            print(f"[Z-test] Contribution member = {row[contribution]:.3f}, z = {z_score:.2f}, p = {p:.4f} → Within expected range.")
                        else:
                            print(f"[Z-test] Contribution membe = {row[contribution]:.3f}, z = {z_score:.2f}, p = {p:.4f} → Statistically different (reject H₀).")
                    else:
                        print(f"Shapiro–Wilk test FAILED (p = {p_value:.4f}) → Non-normal distribution.")
                        _, p = mannwhitneyu(correct_classified_df[contribution], [row[contribution]], alternative='two-sided')
                        if p >= alpha:
                            print(f"[Mann–Whitney] p = {p:.4f} → Consistent with predicted class distribution.")
                        else:
                            print(f"[Mann–Whitney] p = {p:.4f} → Logit is anomalous in PREDICTED class.")
                            
# analyse_logits_in_incorrect_predictions(
#     df=detailed_df,
#     classes=detailed_df['true_label'].unique()
# )

# analyse_contributions_in_incorrect_predictions(
#     df=detailed_df,
#     classes=detailed_df['true_label'].unique(),
#     feature_names=feature_names
# )