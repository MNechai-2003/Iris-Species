import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.core.helpers.calculate_boundaries import boundaries


def plot_histograms(df: pd.DataFrame) -> pd.DataFrame:
    statistics_list = []

    fig, axes = plt.subplots(4, 3, figsize=(15, 15))
    axes = axes.flatten()
    index = 0

    for feature in df.columns[:-1]:
        for target in df['Species'].unique():
            filtered_data = df[df['Species'] == target]

            lower_bound, upper_bound = boundaries(filtered_data[feature])

            sns.histplot(filtered_data[feature], kde=True, color='blue', stat='density', ax=axes[index])
            axes[index].axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
            axes[index].axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
            axes[index].set_title(f'Histogram for {feature} by Species {target}')
            axes[index].set_xlabel(feature)
            axes[index].set_ylabel('Count')

            feature_statistics = filtered_data[feature].describe().to_frame().T
            feature_statistics['pair'] = f"{target}_{feature}"

            statistics_list.append(feature_statistics)

            index += 1
    statistics = pd.concat(statistics_list, ignore_index=True)
    plt.tight_layout()
    plt.show()

    return statistics
