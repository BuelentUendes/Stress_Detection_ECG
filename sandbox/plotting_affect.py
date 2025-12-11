import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_1samp
import os
import matplotlib.lines as mlines
from utils.helper_path import FIGURES_PATH, FEATURE_DATA_PATH

# Calculate label-sitting differences (reusing the logic from your previous code)
def calculate_label_sitting_difference(data, reference, affect_type="Positive_affect"):
    sitting_baseline = data[data['label'] == reference].groupby('Subject_ID')[
        affect_type].mean().reset_index()
    sitting_baseline.columns = ['Subject_ID', f"Sitting_{affect_type}"]

    condition_means = data.groupby(['Subject_ID', 'label'])[affect_type].mean().reset_index()
    merged_data = condition_means.merge(sitting_baseline, on='Subject_ID', how='left')
    merged_data[f"{affect_type}_difference"] = merged_data[affect_type] - merged_data[
        f"Sitting_{affect_type}"]

    return merged_data


def plot_positive_affect_reactivity_box(data, save_path, affect_type="Positive_affect",
                                        show_plot=True, reference="Sitting"):
    """
    Plot positive affect reactivity (difference from sitting baseline) for each experimental condition

    Parameters:
    - data: DataFrame with columns ['Subject_ID', 'label', 'Positive_affect']
    - affect_type: Which affect type to show
    - save_path: Path to save the plot
    - show_plot: Whether to display the plot
    - reference: Reference condition (default: "Sitting")
    """

    # Calculate the differences
    plot_data = calculate_label_sitting_difference(data, reference=reference, affect_type=affect_type)

    # Remove the reference condition from plotting (since it would always be 0)
    plot_data = plot_data[plot_data['label'] != reference]

    # Check for missing participants per label
    all_participants = set(plot_data["Subject_ID"].unique())
    participants_per_label = plot_data.groupby("label")["Subject_ID"].unique()

    for label in participants_per_label.index:
        missing = all_participants - set(participants_per_label[label])
        if missing:
            print(f"Missing in {label}: {missing}")

    # Perform t-tests for each label
    t_test_results = {}
    for label in plot_data["label"].unique():
        affect_reactivity_data = plot_data.loc[
            plot_data["label"] == label, f"{affect_type}_difference"
        ]

        if not affect_reactivity_data.empty:
            t_stat, p_value = ttest_1samp(affect_reactivity_data, popmean=0, nan_policy='omit')
            t_test_results[label] = {"t-statistic": t_stat, "p-value": p_value}

    print(f"T-test results for {affect_type}\n")
    print(t_test_results)
    # Calculate statistics
    affect_reactivity_statistics = plot_data[["label", f"{affect_type}_difference"]].groupby(
        ["label"]).describe()
    affect_reactivity_statistics.columns = ['_'.join(col).strip() for col in affect_reactivity_statistics.columns]
    mean_affect_reactivity = np.round(affect_reactivity_statistics[f"{affect_type}_difference_mean"], 4)

    # Set publication-quality plot aesthetics
    sns.set_style("ticks")
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out'
    })

    # Color scheme for different conditions (same as your HR function)
    colors_index = {
        'Pasat': '#E69F00',
        'Pasat_repeat': '#56B4E9',
        'Raven': '#009E73',
        'SSST_Sing_countdown': '#0072B2',
        'TA': '#D55E00',
        'TA_repeat': '#CC79A7'
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create a custom order based on mean positive affect reactivity
    ordered_labels = mean_affect_reactivity.sort_values().index.tolist()

    # Create the boxplot with customized appearance
    boxplot = sns.boxplot(
        x='label',
        y=f"{affect_type}_difference",
        data=plot_data,
        order=ordered_labels,
        palette=[colors_index[label] for label in ordered_labels],
        width=0.6,
        fliersize=4,
        linewidth=1.0,
        ax=ax,
        whiskerprops={'color': 'black', 'linestyle': '-'},
        capprops={'color': 'black'},
        medianprops={'color': 'black', 'linewidth': 1.5},
        showfliers=False,
    )

    # Apply alpha to boxes manually
    for patch in boxplot.patches:
        facecolor = patch.get_facecolor()
        patch.set_facecolor((*facecolor[:3], 0.6))

    # Add individual data points as gray points
    sns.stripplot(
        x='label',
        y=f"{affect_type}_difference",
        data=plot_data,
        order=ordered_labels,
        color='#4D4D4D',
        size=2,
        alpha=0.5,
        jitter=True,
        dodge=False,
        ax=ax
    )

    # Add a horizontal line at y=0 to indicate baseline
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

    # Add statistical significance markers
    max_vals = plot_data.groupby('label')[f"{affect_type}_difference"].max()
    y_max = max_vals.max()
    y_min = plot_data[f"{affect_type}_difference"].min()
    y_range = y_max - y_min
    y_buffer = y_range * 0.03

    significance_present = 0
    for i, label in enumerate(ordered_labels):
        if t_test_results[label]['p-value'] < 0.001:
            significance = r"$\mathit{P}$<.001"
        elif t_test_results[label]['p-value'] < 0.01:
            p = t_test_results[label]['p-value']
            significance = f"$\mathit{{P}}$={f'{p:.3f}'.lstrip('0')}"
        elif t_test_results[label]['p-value'] < 0.05:
            p = t_test_results[label]['p-value']
            significance = f"$\mathit{{P}}$={f'{p:.3f}'.lstrip('0')}"
        else:
            significance = 'ns'
            significance_present = 1

        current_max = max_vals.get(label, y_max)
        ax.text(i, current_max + y_buffer, significance,
                horizontalalignment='center', fontsize=9)

    # Customize the plot appearance
    ax.set_xlabel('Experimental Condition', fontsize=14)
    ax.set_ylabel(f"{affect_type.replace('_', ' ').title()} Difference\n(Δ from sitting baseline)", fontsize=14)

    # Improve x-tick labels for readability
    plt.xticks(rotation=45, ha='right')
    xlabels = [
        "SSST" if label.startswith("SSST") else label.replace('_', ' ').upper().replace(" REPEAT", " (repeat)")
        for label in ordered_labels
    ]
    ax.set_xticklabels(xlabels, fontsize=12)

    # Add means to the plot (diamond shape)
    for i, label in enumerate(ordered_labels):
        mean_val = mean_affect_reactivity.loc[label]
        ax.scatter(i, mean_val, marker='D', s=60, color='black', zorder=10)

    # Create legend
    legend_items = []
    legend_items.append(mlines.Line2D([], [], color='red', linestyle="--",
                                      lw=1.5, alpha=0.7, label=f"{affect_type.replace('_', ' ').title()} Difference Threshold"))
    legend_items.append(
        mlines.Line2D([], [], marker='D', color='white', markerfacecolor='black', markersize=8, label='Mean'))
    legend_items.append(mlines.Line2D([], [], marker='_', color='black', lw=1.5, markersize=10, label='Median'))
    legend_items.append(
        mlines.Line2D([], [], marker='o', color='white', markerfacecolor='#4D4D4D', alpha=0.5, markersize=6,
                      label='Individual Data Points'))

    if significance_present:
        legend_items.append(
            mlines.Line2D([], [], color='black', marker='$ns$', markersize=10, linestyle='', label='no significance'))

    # # Create legend with better formatting
    legend = ax.legend(handles=legend_items,
                       loc='upper left',
                       bbox_to_anchor=(1., 1),
                       frameon=False,
                       framealpha=0.9,
                       edgecolor='lightgray',
                       fontsize=12)


    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0.03, 0.85, 1])

    # Save the figure with high resolution
    save_path_full = os.path.join(save_path,
                                  f"boxplot_{affect_type}_difference_reference_{reference}.png")
    plt.savefig(save_path_full, dpi=500, bbox_inches='tight', format='png')

    if show_plot:
        plt.show()

    plt.close()

    # Print summary statistics
    print(f"{affect_type} Difference Statistics:")
    print("=" * 50)
    for label in ordered_labels:
        mean_val = mean_affect_reactivity.loc[label]
        p_val = t_test_results[label]['p-value']
        print(f"{label}: Mean = {mean_val:.4f}, p-value = {p_val:.4f}")

    return plot_data, t_test_results


# Assuming you have your data prepared as 'total_data_participant' from your previous code
subjective_ratings = pd.read_csv(os.path.join(FEATURE_DATA_PATH, "subjective_ratings.csv"))
# Ensure we have the required columns - rename to match expected format
column_mapping = {
    'Experimental_Condition': 'label',
}

labels_to_include = ["Sitting", "TA", "TA_repeat", "Pasat", "Pasat_repeat", "Raven", "SSST_Sing_countdown"]

# Apply column renaming
total_data_participant = subjective_ratings.rename(columns=column_mapping)
total_data_participant = total_data_participant.loc[total_data_participant["label"].isin(labels_to_include)]

plot_data, t_test_results = plot_positive_affect_reactivity_box(
    total_data_participant,
    save_path=FIGURES_PATH,
    show_plot=True
)

plot_data, t_test_results = plot_positive_affect_reactivity_box(
    total_data_participant,
    affect_type="Negative_affect",
    save_path=FIGURES_PATH,
    show_plot=True
)


