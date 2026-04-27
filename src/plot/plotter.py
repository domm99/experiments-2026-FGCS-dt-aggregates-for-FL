import glob
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt


def beautify_policy_name(name):
    if 'Drift' in name:
        return 'Performance-Based Retraining'
    elif 'Time' in name:
        return 'Time-Based Retraining'
    elif 'Each' in name:
        return 'Population-Based Retraining'
    else:
        return 'Unknown name'

def beautify_training_metrics(name):
    if 'train_accuracy' in name:
        return 'Training Accuracy'
    else:
        return name

def plot_training(experiment, data_folder, current_output_dir):
    files = sorted(glob.glob(f'{data_folder}{experiment}/training*.csv'))
    metrics = [
        'train_accuracy',
        'train_precision',
        'train_recall',
        'train_f1_score',
        'val_accuracy',
        'val_precision',
        'val_recall',
        'val_f1_score',
    ]

    sns.set_style('whitegrid')

    for metric in metrics:
        hist = []
        for file in files:
            df = pd.read_csv(file)
            if metric not in df.columns:
                continue
            hist.append(df[metric].iloc[-1])
        if not hist:
            continue
        plt.scatter(range(len(hist)), hist, label=metric, s=50, marker='D')
        plt.plot(range(len(hist)), hist, label=metric, linewidth=2)
        plt.title(beautify_policy_name(experiment), fontsize=22)
        plt.ylabel(beautify_training_metrics(metric), fontsize=22)
        plt.xlabel('Train Step', fontsize=22)
        plt.tick_params(axis='both', labelsize=20)
        plt.ylim([0.4, 0.9])
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        plt.tight_layout()
        plt.savefig(f'{current_output_dir}/{metric}-{experiment}.png')
        plt.close()

def test_metrics(experiment, data_folder, current_output_dir):
    files = sorted(glob.glob(f'{data_folder}{experiment}/test*.csv'))

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    sns.set_style('whitegrid')
    for metric in metrics:
        means = []
        mins = []
        maxs = []
        for file in files:
            df = pd.read_csv(file)
            if metric not in df.columns:
                continue
            min_v = df[metric].min()
            max_v = df[metric].max()
            mean = df[metric].mean()
            means.append(mean)
            mins.append(min_v)
            maxs.append(max_v)
        if not means:
            continue

        plt.ylabel(metric.capitalize(), fontsize=22)
        plt.xlabel('Time Step', fontsize=22)
        means = means[:400]
        plt.plot(range(len(means)), means, linewidth=2)
        plt.ylim([0.4, 0.9])
        plt.xlim([-2, len(means)])
        #plt.fill_between(range(len(means)), mins, maxs, alpha=0.2)
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        plt.title(beautify_policy_name(experiment), fontsize=23)
        plt.tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        plt.savefig(f'{current_output_dir}/test_{metric}-{experiment}.png')
        plt.close()

def plot_dict_histogram(data: dict[str, int], current_output_dir: str, title: str | None = None):
    """
    Plot a bar chart from a dictionary where keys are labels
    and values are bar heights.
    """
    labels = list(data.keys())
    values = list(data.values())
    sns.set_style('whitegrid')
    # Colorblind palette with a softer/pastel-like saturation
    palette = sns.color_palette("muted", n_colors=len(labels), desat=0.55)
    plt.figure(figsize=(7, 3.5))

    for i, (label, value) in enumerate(zip(labels, values)):
        plt.bar(i, value, color=palette[i], label=label)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    plt.xlabel("")
    plt.xticks([])
    plt.ylabel("Number of trainings", fontsize=20)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(labels), 3),
        frameon=False,
        fontsize=14,
    )

    if title is not None:
        plt.title(title)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.tight_layout()
    plt.tick_params(axis='both', labelsize=17)
    plt.savefig(f'{current_output_dir}/number_of_trainings.png')
    plt.close()

if __name__ == '__main__':

    data_folder = 'data/'
    charts_output_folder = 'charts/'
    Path(charts_output_folder).mkdir(parents=True, exist_ok=True)
    experiments = ['RetrainAfterPerformanceDrift', 'RetrainAfterTime', 'RetrainEachNDTsActivated']# , 'RetrainEachNDTsActivated'], 'RetrainAfterPerformanceDrift'

    for experiment in experiments:
        current_output_dir = f'{charts_output_folder}{experiment}/'
        Path(current_output_dir).mkdir(parents=True, exist_ok=True)

        plot_training(experiment, data_folder, current_output_dir)
        test_metrics(experiment, data_folder, current_output_dir)

    plot_dict_histogram(
        {
            'Time-Based': 16,
            'Population-Based': 25,
            'Performance-Based': 7
        },
        charts_output_folder
    )
