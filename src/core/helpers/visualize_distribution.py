import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_distribution(feature: pd.Series, lower_bound: float, upper_bound: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.histplot(feature, kde=True, ax=axes[0], binwidth=0.2)
    axes[0].axvline(feature.mean(), color='gray', linestyle='--', label='Mathematical Expectation')
    axes[0].axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
    axes[0].axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
    axes[0].set_title(f"Histogram for {feature.name}")
    axes[0].legend()

    sns.boxplot(x=feature, ax=axes[1])
    axes[1].set_title(f"Boxplot for {feature.name}")

    plt.tight_layout()
    plt.show()
