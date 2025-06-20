import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_target_distribution(iris_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Species', data=iris_df)
    plt.title('Distribution of Iris Species')
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.show()
