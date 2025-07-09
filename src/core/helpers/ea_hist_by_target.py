import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_error_distribution(y_pred: pd.Series, y_true: pd.Series) -> None:
    """
    Visualize the distribution of correct and incorrect predictions by classes. 
    """

    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    df['is_correct'] = df['y_true'] == df['y_pred']

    error_counts = df.groupby(['y_true', 'is_correct']).size().reset_index(name='count')
    error_counts['status'] = error_counts['is_correct'].map({True: 'Correct', False: 'Incorrect'})

    class_metrics = df.groupby('y_true').agg({
        'is_correct': ['count', 'sum']
    }).round(3)
    class_metrics.columns = ['total', 'correct']
    class_metrics['accuracy'] = (class_metrics['correct'] / class_metrics['total'] * 100).round(1)

    plt.style.use('default')
    sns.set_palette("husl")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = {
        'Correct': '#A8E6CF',
        'Incorrect': '#FFB3BA'
    }

    sns.barplot(
        data=error_counts,
        x='y_true',
        y='count',
        hue='status',
        width=0.8,
        palette=colors,
        ax=ax1,
        edgecolor='white',
        linewidth=0.5
    )

    ax1.set_title('Distribution of correct and incorrect predictions by classes', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Target classes', fontsize=12, fontweight='semibold')
    ax1.set_ylabel('Number of predictions', fontsize=12, fontweight='semibold')
    ax1.legend(title='Status of prediction', title_fontsize=8, fontsize=9, frameon=True, fancybox=True, shadow=True, loc='upper right')

    for container in ax1.containers:
        ax1.bar_label(container, fmt='%d', fontsize=10, fontweight='semibold')

    ax1.set_ylim(0, error_counts['count'].max() * 1.15)

    accuracy_data = class_metrics.reset_index()
    bars = ax2.bar(accuracy_data['y_true'], accuracy_data['accuracy'], 
                   color='#B8B8FF', edgecolor='white', linewidth=0.5, alpha=0.8, width=0.5)

    ax2.set_title('Accuracy by classes', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Target classes', fontsize=12, fontweight='semibold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='semibold')
    ax2.set_ylim(0, 105)

    for bar, acc in zip(bars, accuracy_data['accuracy']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='semibold')

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(colors='#666666')
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.show()