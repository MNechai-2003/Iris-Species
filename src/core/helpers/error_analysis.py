import sys
import pandas as pd
import numpy as np
from scipy.stats import shapiro, norm, mannwhitneyu

sys.path.append("/Users/nikolaynechay/Iris-Species")

from src.core.helpers.visualizations import plot_distribution_of_logits


def error_analysis(df: pd.DataFrame, weights: list, bias: float, feature_names: list) -> pd.DataFrame:
    """Method for analyze predictions errors."""

    detailed_analysis = []
    for idx, row in df.iterrows():
        features = row[feature_names].values
        true_label = row['true_label']
        pred_label = row['predicted_label']

        z = np.dot(weights, features) + bias  # logits
        feature_contributions = weights * features

        error_analysis = {
            'original_idx': row.name,
            'true_label': true_label,
            'predicted_label': pred_label,
            'logit': z,
            'probability': 1 / (1 + np.exp(-z))
        }

        for i, feature_name in enumerate(feature_names):
            error_analysis[f'{feature_name}_value'] = features[i]
            error_analysis[f'{feature_name}_contribution'] = feature_contributions[i]
            error_analysis[f'{feature_name}_weight'] = weights[i]

        error_analysis['bias'] = bias
        detailed_analysis.append(error_analysis)

    return pd.DataFrame(detailed_analysis)


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
                        


# def find_problematic_features(df: pd.DataFrame, classes: list, alpha: float = 0.05) -> None:
#     """Define the most impacted features in incorrect predictions."""
#     contribution_cols = [col for col in df.columns if col.endswith("_contribution")]
#     df['is_correct'] = df['true_label'] == df['predicted_label']
    
#     for class_name in classes:
#         correct_classified_df = df[(df['is_correct'] == True) & (df['true_label'] == class_name)]
#         incorrect_classified_df = df[(df['is_correct'] == False) & (df['true_label'] == class_name)]
        
#         logit_statistics = calculate_main_statistics(
#             correct_df=correct_classified_df,
#             columns=['logit']
#         )
        
#         contribution_statistics = calculate_main_statistics(
#             correct_df=correct_classified_df,
#             columns=contribution_cols
#         )
        
#         plot_distribution_of_logits(correct_classified_df['logit'], class_name)
#         for i, row in incorrect_classified_df.iterrows():
#             if row['true_label'] == class_name or row['predicted_label'] == class_name:
#                 if len(correct_classified_df) > 30:
#                     stat, p_value = shapiro(correct_classified_df['logit'])
#                     if p_value > alpha:
#                         print(
#                             f"Shapiro–Wilk test for class '{class_name}' passed "
#                             f"(p-value = {p_value:.4f}): no evidence to reject H₀ — "
#                             f"logit distribution can be considered normal."
#                         )
                        
#                         z_score = (row['logit'] - logit_statistics['logit']['mean']) / logit_statistics['logit']['std']
#                         p = 2 * (1 - norm.cdf(abs(z_score)))
#                         if p >= alpha:
#                             print(f"[Z-test] Sample {i}: logit = {row['logit']:.3f}, z = {z_score:.2f}, p = {p:.4f} → "
#                                   f"Logit is within expected range (not statistically different).")
#                         else:
#                             print(f"[Z-test] Sample {i}: logit = {row['logit']:.3f}, z = {z_score:.2f}, p = {p:.4f} → "
#                                   f"Logit significantly deviates from correct distribution (reject H₀).")
#                     else:
#                         print(
#                             f"Shapiro–Wilk test for class '{class_name}' failed "
#                             f"(p-value = {p_value:.4f}): reject H₀ — "
#                             f"logit distribution deviates from normality."
#                         )
                        
#                         _, p = mannwhitneyu(correct_classified_df['logit'], [row['logit']], alternative='two-sided')
#                         if p >= alpha:
#                             print(
#                                 f"[Mann–Whitney] Sample {i}: logit = {row['logit']:.3f}, p = {p:.4f} → "
#                                 f"Logit is consistent with correct distribution."
#                             )
#                         else:
#                             print(
#                                 f"[Mann–Whitney] Sample {i}: logit = {row['logit']:.3f}, p = {p:.4f} → "
#                                 f"Logit distribution mismatch — possible anomaly or high bias."
#                             )