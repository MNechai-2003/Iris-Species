import numpy as np
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


def plot_distributions_of_feature_by_classes(
    feature_name: str,
    general_df: pd.DataFrame,
    target_name: str,
    flag='train'
) -> None:
    """This method plots the distributions of feature by classes, displaying the mean and 3 standard deviations for each class."""
    plt.figure(figsize=(12, 6))
    unique_classes = general_df[target_name].unique()

    colors = sns.color_palette('viridis', n_colors=len(unique_classes))
    class_color_map = {target_name: colors[i] for i, target_name in enumerate(unique_classes)}

    for class_name in unique_classes:
        class_distribution_set = general_df[general_df[target_name] == class_name][feature_name].dropna()

        sns.kdeplot(class_distribution_set, label=f'{class_name} Distribution', color=class_color_map[class_name], fill=True, alpha=0.4)

        mean_value = class_distribution_set.mean()
        std_value = class_distribution_set.std()

        plt.axvline(mean_value, color=class_color_map[class_name], linestyle='--', lw=2, label=f'{class_name} Mean')

        text_position = plt.ylim()[1] * 0.5
        plt.text(mean_value, text_position, f'Mean: {mean_value:.2f}', color=class_color_map[class_name], rotation=0,
                 verticalalignment='bottom', horizontalalignment='center', fontsize=10)

        for i in range(1, 4):
            plt.axvline(mean_value + i * std_value, color=class_color_map[class_name], linestyle=':', lw=1)
            plt.axvline(mean_value - i * std_value, color=class_color_map[class_name], linestyle=':', lw=1)
            # if class_name == unique_classes[2]:
            if class_name == 'virginica':
                test_position = plt.ylim()[1] * 0.4
            else:
                test_position = plt.ylim()[1] * 0.85
            plt.text(mean_value + i * std_value, test_position, f"{i} SD = {mean_value + i * std_value:.2f}", color=class_color_map[class_name], rotation=45,
                     verticalalignment='bottom', horizontalalignment='left')
            plt.text(mean_value - i * std_value, test_position, f"-{i} SD = {mean_value - i * std_value:.2f}", color=class_color_map[class_name], rotation=45,
                     verticalalignment='bottom', horizontalalignment='right')

            plt.text(mean_value, plt.ylim()[1] * 0.2, f"{class_name}",
                        color=class_color_map[class_name], rotation=0, verticalalignment='bottom',
                        horizontalalignment='center', fontsize=10)
    if flag == 'test':
        plt.title(f"Distribution of {feature_name} by Specie on Test Set", fontsize=16)
    else:
        plt.title(f"Distribution of {feature_name} by Specie on Train Set", fontsize=16)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title=target_name)
    plt.tight_layout()
    plt.show()


def plot_boxplot_with_missclassifications(
    general_df: pd.DataFrame,
    feature_name: str,
    target_name: str,
    flag: str
) -> None:
    """This method plots a boxplot of the feature with missclassifications highlighted."""
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette('viridis', n_colors=general_df[target_name].nunique())
    ax = sns.boxplot(x=target_name, y=feature_name, data=general_df, palette=colors)
    unique_classes = general_df[target_name].unique()
    class_to_position = {class_name: i for i, class_name in enumerate(unique_classes)}
    errors_df = general_df[general_df['predicted_label'] != general_df[target_name]]

    if not errors_df.empty:
        for class_name in unique_classes:
            class_errors = errors_df[errors_df[target_name] == class_name]
            if not class_errors.empty:
                x_position = class_to_position[class_name]

                n_points = len(class_errors)
                jitter = np.random.uniform(-0.2, 0.2, n_points)
                x_positions = np.full(n_points, x_position) + jitter

                plt.scatter(x_positions, class_errors[feature_name], 
                           color='red', alpha=0.8, s=50, marker='X', 
                           edgecolors='darkred', linewidth=1,
                           label='Missclassifications' if class_name == unique_classes[0] else "")

    if flag == 'test':
        plt.title(f"Boxplot of {feature_name} by Species on Test Set")
    else:
        plt.title(f"Boxplot of {feature_name} by Species on Train Set")

    plt.xlabel('Species')
    plt.ylabel(feature_name)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()