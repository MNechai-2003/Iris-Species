import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix_for_miss_classifications(error_df: pd.DataFrame) -> None:
    confusion = pd.crosstab(error_df['true_label'], error_df['predicted_label'])
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Greys', cbar=False)
    plt.title('Confusion Matrix for Miss Classifications')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()