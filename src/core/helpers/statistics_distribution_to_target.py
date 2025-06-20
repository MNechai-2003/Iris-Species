import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def statistics_distribution_to_target(df: pd.DataFrame, target: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Species", y=target, data=df)
    sns.swarmplot(x="Species", y=target, data=df, color='black', size=3, alpha=0.7)
    plt.title(f'Distribution of {target} by Species', fontsize=14)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.show()