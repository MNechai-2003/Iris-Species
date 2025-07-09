import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_distribution_of_logits(df: pd.Series, class_name: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pd.DataFrame(df), x='logit', kde=True, color='skyblue', bins=30)
    plt.title(f"Logit Distribution for Class '{class_name}'")
    plt.xlabel("Logit Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()