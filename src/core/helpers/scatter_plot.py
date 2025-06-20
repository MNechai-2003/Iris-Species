import pandas as pd
import matplotlib.pyplot as plt


def scatter_plot(df: pd.DataFrame, x: str, y: str) -> None:
    plt.figure(figsize=(10, 6))
    for species, group in df.groupby('Species'):
        plt.scatter(group[x], group[y], label=species, alpha=0.7)

    plt.title(f'Scatter Plot of {y} & {x} by Species', fontsize=14)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.legend(title="Species")
    plt.show()
