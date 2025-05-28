# Collections of helper functions that are reused across several scripts

import os
import random
import json
import yaml
from typing import Optional, Tuple, Union, Any

from mne.simulation.metrics import recall_score
from scipy import stats
import torch
from torchmetrics.functional.classification import binary_calibration_error
from numpy import ndarray
from optuna import Trial
from pandas import Series, DataFrame
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import ttest_1samp

from numpy.testing import assert_almost_equal
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier
from sklearn.calibration import calibration_curve
from sklearn.mixture import GaussianMixture

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from imblearn.over_sampling import ADASYN, SMOTE

from sklearn.feature_selection import RFE
import optuna
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from typing import Optional, Tuple, Dict, Union, List
import warnings
import umap
import time
from matplotlib.patches import Patch
from utils.helper_path import FIGURES_PATH
from sklearn.manifold import TSNE


def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        path: Path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed_number: int) -> None:
    """
    Set the seed
    :param seed_number: seed number to set
    :return: None
    """
    # Set seed for reproducibility
    torch.manual_seed(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)

def get_data_folders(input_path: str) -> list[str]:
    """
    Loads the data and from the input_path
    :param input_path: input_path to load the data from
    :return: list of participants
    """
    data_folders = [filename for filename in os.listdir(input_path) if filename.lower().endswith((".csv"))]
    return data_folders


def report_nans(df, name="data"):
    # count NaNs in each column
    nan_counts = df.isna().sum()
    # keep only columns where count > 0
    cols_with_nans = nan_counts[nan_counts > 0]
    if not cols_with_nans.empty:
        print(f"Columns in {name} with NaNs:")
        print(cols_with_nans)
    else:
        print(f"No NaNs in {name}.")


class ECGDataset:
    """
    Feature engineered dataset for the ECG dataset
    """

    def __init__(self,
                 root_dir: str,
                 test_size: Optional[float] = 0.2,
                 val_size: Optional[float] = 0.2,
                 add_participant_id: Optional[bool] = False):
        """
        :param root_dir: root directory for the data import
        :param test_size: test size split, default 0.2
        :param val_size: val size split, default 0.2
        :param add_participant_id, if we should do a within setup
        """

        assert isinstance(test_size, float), "test size needs to be a float"
        assert 0.0 <= test_size <= 1.0, "test size should be in between 0 and 1"

        assert isinstance(val_size, float), "test size needs to be a float"
        assert 0.0 <= val_size <= 1.0, "val size should be in between 0 and 1"

        self.root_dir = root_dir
        self.test_size = test_size
        self.val_size = val_size
        self.add_participant_id = add_participant_id

        self._get_data_folders()
        self._split_data()

    def _get_data_folders(self) -> None:
        """
        Gets the data folders of the root directory which we will then load as a dataset
        """
        self.data_folders = [filename for filename in os.listdir(self.root_dir) if filename.lower().endswith((".csv"))]

    def _load_data(self, data_files: list[str], add_participant_id: Optional[bool]=False) -> pd.DataFrame:
        """
        Loads the data into a pandas dataframe from the CSV files.
        :param data_files: list of data files to load from
        :return: Combined DataFrame containing data from all CSV files.
        """
        dataframes = []  # List to hold individual DataFrames

        for csv_file in data_files:
            if self.add_participant_id or add_participant_id:
                participant_idx = int(csv_file.split(".")[0])
            file_path = os.path.join(self.root_dir, csv_file)  # Construct full file path
            try:
                df = pd.read_csv(file_path)  # Read CSV file into DataFrame
                if self.add_participant_id or add_participant_id:
                    df["participant_id"] = participant_idx
                dataframes.append(df)  # Append DataFrame to the list
            except Exception as e:
                print(f"Error reading {file_path}: {e}")  # Handle exceptions

        combined_df = pd.concat(dataframes, ignore_index=True)  # Concatenate all DataFrames
        return combined_df  # Return the combined DataFrame

    def _split_data_by_condition(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data by condition within each participant, using 'Recov_standing' as the splitting point.
        All data before 'Recov_standing' goes to training, all data after goes to testing.
        """
        participants = list(set(data["participant_id"].values))
        train_frames = []
        test_frames = []

        for participant in participants:
            # Get data for this participant
            participant_df = data[data["participant_id"] == participant].copy()
            participant_df = participant_df.reset_index(drop=True)  # Reset index for proper splitting

            # Find the index where 'Recov_standing' occurs
            recov_indices = participant_df.index[participant_df["label"] == "Recov_standing"]

            if len(recov_indices) == 0:
                print(f"Warning: No 'Recov_standing' found for participant {participant}. Skipping.")
                continue

            split_idx = recov_indices[0]  # Take the first occurrence

            # Split the data at the recovery standing point
            train_data = participant_df.loc[:split_idx-1]
            test_data = participant_df.loc[split_idx:]

            # Check that filtering worked
            assert len(train_data[train_data["label"]=="Recov_standing"]) == 0, "train data contains the filter condition!"

            # Only append if we have data in both splits
            if not train_data.empty and not test_data.empty:
                train_frames.append(train_data)
                test_frames.append(test_data)
            else:
                print(f"Warning: Empty split for participant {participant}. Skipping.")

        # Combine all participants' data
        if not train_frames or not test_frames:
            raise ValueError("No valid splits found in the data. Check if 'Recov_standing' exists in labels.")

        training_data = pd.concat(train_frames, axis=0, ignore_index=True)
        testing_data = pd.concat(test_frames, axis=0, ignore_index=True)

        return training_data, testing_data


    def get_average_hr_reactivity(self, positive_class, negative_class, save_path,
                                  show_plot=True,
                                  reference="Sitting",
                                  heart_measure="hrv_mean",
                                  verbose=False):
        # We calculate the average HR reactivity based on participant id
        total_data_participant = self._load_data(self.data_folders, add_participant_id=True)
        negative_class_baseline = total_data_participant[total_data_participant["label"] == reference.capitalize()][[heart_measure, "participant_id"]]
        negative_class_baseline_hr = negative_class_baseline.groupby(['participant_id']).mean().reset_index()

        positive_class_baseline = total_data_participant[total_data_participant["category"] == positive_class][[heart_measure, "participant_id", "label"]]
        positive_class_baseline_hr_label = positive_class_baseline.groupby(["participant_id", "label"]).mean().reset_index()

        # merge now
        positive_class_baseline_hr_label = pd.merge(
            positive_class_baseline_hr_label,
            negative_class_baseline_hr,
            on="participant_id",
            how="left"
        )

        # Rename the column names (hr_mean_x is now HR_mean_experimental)
        positive_class_baseline_hr_label = positive_class_baseline_hr_label.rename(columns={f"{heart_measure}_x": "hrv_mean_experiment_condition",
                                                 f"{heart_measure}_y": "hrv_mean_baseline_condition"})

        positive_class_baseline_hr_label["hrv_reactivity"] = positive_class_baseline_hr_label["hrv_mean_experiment_condition"] - positive_class_baseline_hr_label["hrv_mean_baseline_condition"]
        all_participants = set(positive_class_baseline_hr_label["participant_id"].unique())
        participants_per_label = positive_class_baseline_hr_label.groupby("label")["participant_id"].unique()

        # logging what participants are missing
        if verbose:
            for label in participants_per_label.index:
                missing = all_participants - set(participants_per_label[label])
                if missing:
                    print(f"Missing in {label}: {missing}")

        # Perform t-tests for each label
        t_test_results = {}

        for label in positive_class_baseline_hr_label["label"].unique():
            hr_reactivity_data = positive_class_baseline_hr_label.loc[
                positive_class_baseline_hr_label["label"] == label, "hrv_reactivity"
            ]

            if not hr_reactivity_data.empty:
                t_stat, p_value = ttest_1samp(hr_reactivity_data, popmean=0, nan_policy='omit')
                t_test_results[label] = {"t-statistic": t_stat, "p-value": p_value}

        hr_reactivity_statistics = positive_class_baseline_hr_label[["label", "hrv_reactivity"]].groupby(["label"]).describe()
        hr_reactivity_statistics.columns = ['_'.join(col).strip() for col in hr_reactivity_statistics.columns]

        mean_hr_reactivity = np.round(hr_reactivity_statistics["hrv_reactivity_mean"], 4)
        unique_experiment_conditions = set(positive_class_baseline_hr_label["label"].unique())

        colors_index = {
            'Pasat': '#E69F00',
            'Pasat_repeat': '#56B4E9',
            'Raven': '#009E73',
            'SSST_Sing_countdown': '#0072B2',
            'TA': '#D55E00',
            'TA_repeat': '#CC79A7'
        }

        plt.figure(figsize=(10, 6))

        for experiment_condition in unique_experiment_conditions:
            hr_reactivity_data = \
            positive_class_baseline_hr_label[positive_class_baseline_hr_label['label'] == experiment_condition][
                "hrv_reactivity"]
            if not hr_reactivity_data.empty:
                p_value = t_test_results[experiment_condition]['p-value']
                # Get the mean corresponding to this experiment condition
                mean_condition = mean_hr_reactivity.loc[experiment_condition]

                label = (f"{experiment_condition.replace('_', ' ')}: Mean: {mean_condition:.3f} "
                         f"(p < 0.05)"
                         if p_value < 0.05 else f"{experiment_condition.replace('_', ' ')}")

                sns.kdeplot(hr_reactivity_data, color=colors_index[experiment_condition],
                            label=label, fill=True, alpha=0.3)

        # Customize the plot
        plt.xlabel('Heart Rate Reactivity (Δ HRV relative to baseline)')
        plt.ylabel('Density')
        plt.legend(loc='upper left',  title="Task & Statistics", bbox_to_anchor=(1.05, 1))
        save_path = os.path.join(save_path, f"histogram_heart_rate_variability_reactivity_label_measure_{heart_measure}_reference_{reference}.png")
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        if show_plot:
            plt.show()
            plt.close()
        plt.close()

    def get_average_hr_reactivity_box(self, positive_class, negative_class, save_path,
                                      show_plot=True,
                                      reference="Sitting",
                                      exclude_recovery= False,
                                      heart_measure="hrv_mean"):
        # We calculate the average HR reactivity based on participant id
        total_data_participant = self._load_data(self.data_folders, add_participant_id=True)

        if (reference.lower() == "sitting") or (reference.lower() == "standing"):
            col_name = "label"
        elif reference.lower() == "baseline":
            col_name = "category"
        else:
            raise ValueError("Please input either reference 'sitting' or 'baseline'")

        # Build the filtering condition based on whether recovery should be excluded
        if exclude_recovery:
            filter_condition = total_data_participant[col_name] == reference
        else:
            filter_condition = (
                    (total_data_participant[col_name] == reference) |
                    (total_data_participant[col_name] == f"Recov_{reference.lower()}")
            )

        # Filter the DataFrame and select the desired columns
        negative_class_baseline = total_data_participant.loc[filter_condition, [heart_measure, "participant_id"]]
        negative_class_baseline_hr = negative_class_baseline.groupby(['participant_id']).mean().reset_index()

        positive_class_baseline = total_data_participant[total_data_participant["category"] == positive_class][
            [heart_measure, "participant_id", "label"]]
        positive_class_baseline_hr_label = positive_class_baseline.groupby(
            ["participant_id", "label"]).mean().reset_index()

        # merge now
        positive_class_baseline_hr_label = pd.merge(
            positive_class_baseline_hr_label,
            negative_class_baseline_hr,
            on="participant_id",
            how="left"
        )

        # Rename the column names (hr_mean_x is now HR_mean_experimental)
        positive_class_baseline_hr_label = positive_class_baseline_hr_label.rename(
            columns={f"{heart_measure}_x": "hrv_mean_experiment_condition",
                     f"{heart_measure}_y": "hrv_mean_baseline_condition"})

        positive_class_baseline_hr_label["hrv_reactivity"] = positive_class_baseline_hr_label[
                                                                 "hrv_mean_experiment_condition"] - \
                                                             positive_class_baseline_hr_label[
                                                                 "hrv_mean_baseline_condition"]
        all_participants = set(positive_class_baseline_hr_label["participant_id"].unique())
        participants_per_label = positive_class_baseline_hr_label.groupby("label")["participant_id"].unique()

        # logging what participants are missing
        for label in participants_per_label.index:
            missing = all_participants - set(participants_per_label[label])
            if missing:
                print(f"Missing in {label}: {missing}")

        # Perform t-tests for each label
        t_test_results = {}

        for label in positive_class_baseline_hr_label["label"].unique():
            hr_reactivity_data = positive_class_baseline_hr_label.loc[
                positive_class_baseline_hr_label["label"] == label, "hrv_reactivity"
            ]

            if not hr_reactivity_data.empty:
                t_stat, p_value = ttest_1samp(hr_reactivity_data, popmean=0, nan_policy='omit')
                t_test_results[label] = {"t-statistic": t_stat, "p-value": p_value}

        hr_reactivity_statistics = positive_class_baseline_hr_label[["label", "hrv_reactivity"]].groupby(
            ["label"]).describe()
        hr_reactivity_statistics.columns = ['_'.join(col).strip() for col in hr_reactivity_statistics.columns]

        mean_hr_reactivity = np.round(hr_reactivity_statistics["hrv_reactivity_mean"], 4)

        # Set publication-quality plot aesthetics
        sns.set_style("ticks")  # Use ticks instead of whitegrid
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 12,
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'xtick.direction': 'out',
            'ytick.direction': 'out'
        })

        # Color scheme for different conditions
        colors_index = {
            'Pasat': '#E69F00',
            'Pasat_repeat': '#56B4E9',
            'Raven': '#009E73',
            'SSST_Sing_countdown': '#0072B2',
            'TA': '#D55E00',
            'TA_repeat': '#CC79A7'
        }

        fig, ax = plt.subplots(figsize=(10, 7))
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Prepare data for boxplot
        plot_data = positive_class_baseline_hr_label[['label', 'hrv_reactivity']].copy()

        # Create a custom order based on mean HRV reactivity
        ordered_labels = mean_hr_reactivity.sort_values().index.tolist()

        # Create the boxplot with customized appearance
        boxplot = sns.boxplot(
            x='label',
            y='hrv_reactivity',
            data=plot_data,
            order=ordered_labels,
            palette=[colors_index[label] for label in ordered_labels],
            width=0.6,
            fliersize=4,
            linewidth=1.0,
            ax=ax,
            boxprops={'facecolor': 'none', 'edgecolor': 'none'},  # Remove box fill and border
            whiskerprops={'color': 'black', 'linestyle': '-'},
            capprops={'color': 'black'},
            medianprops={'color': 'black', 'linewidth': 1.5},
            showfliers=False,
        )

        # Add individual data points as gray points
        sns.stripplot(
            x='label',
            y='hrv_reactivity',
            data=plot_data,
            order=ordered_labels,
            color='gray',
            size=3.5,
            alpha=0.5,
            jitter=True,
            dodge=True,
            ax=ax
        )

        # Add a horizontal line at y=0 to indicate baseline
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

        # Add statistical significance markers
        # Calculate y-buffer dynamically
        max_vals = plot_data.groupby('label')['hrv_reactivity'].max()
        y_max = max_vals.max()
        y_min = plot_data['hrv_reactivity'].min()
        y_range = y_max - y_min
        y_buffer = y_range * 0.03  # Smaller buffer for tighter placement

        # Add significance markers
        significance_present = 0
        for i, label in enumerate(ordered_labels):
            if t_test_results[label]['p-value'] < 0.001:
                significance = '***'
            elif t_test_results[label]['p-value'] < 0.01:
                significance = '**'
            elif t_test_results[label]['p-value'] < 0.05:
                significance = '*'
            else:
                significance = 'ns'
                significance_present = 1

            # Get the max value for this specific label
            current_max = max_vals.get(label, y_max)

            # Place the significance marker slightly above the max value
            ax.text(i, current_max + y_buffer, significance,
                    horizontalalignment='center', fontsize=12, fontweight='bold')

        label_plot =  heart_measure.split("_")[0].upper()

        # Customize the plot appearance
        ax.set_xlabel('Task', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{label_plot} Reactivity\n(Δ from baseline)', fontsize=14, fontweight='bold')

        # Improve x-tick labels for readability
        plt.xticks(rotation=45, ha='right')
        ax.set_xticklabels([label.replace('_', ' ') for label in ordered_labels], fontsize=12)

        # Add subtle grid for y-axis only
        # ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        # ax.set_axisbelow(True)

        # Add means to the plot (diamond shape)
        for i, label in enumerate(ordered_labels):
            mean_val = mean_hr_reactivity.loc[label]
            # Diamond marker for mean
            ax.scatter(i, mean_val, marker='D', s=60, color='black', zorder=10)

        # Create a clean, professional legend - import required modules
        import matplotlib.lines as mlines
        from matplotlib.patches import Patch

        # Create legend items with better formatting
        legend_items = []

        # Add a section header
        # legend_items.append(mlines.Line2D([], [], color='white', marker='', linestyle='', label='Elements'))

        # Add main elements
        legend_items.append(mlines.Line2D([], [], color='red', linestyle="--",
                                          lw=1.5, alpha=0.7, label=f'{label_plot} reactivity threshold'))
        legend_items.append(
            mlines.Line2D([], [], marker='D', color='white', markerfacecolor='black', markersize=8, label='Mean'))
        legend_items.append(mlines.Line2D([], [], marker='_', color='black', lw=1.5, markersize=10, label='Median'))
        legend_items.append(
            mlines.Line2D([], [], marker='o', color='white', markerfacecolor='gray', alpha=0.4, markersize=6,
                          label='Individual data points'))

        # Add significance section
        legend_items.append(mlines.Line2D([], [], color='white', marker='', linestyle='', label=' '))
        legend_items.append(
            mlines.Line2D([], [], color='white', marker='', linestyle='', label='Statistical Significance'))
        legend_items.append(mlines.Line2D([], [], color='black', marker='$*$', markersize=5, linestyle='', label='p < 0.05'))
        legend_items.append(mlines.Line2D([], [], color='black', marker='$**$', markersize=10, linestyle='', label='p < 0.01'))
        legend_items.append(mlines.Line2D([], [], color='black', marker='$***$', markersize=15, linestyle='', label='p < 0.001'))

        if significance_present:
            legend_items.append(
                mlines.Line2D([], [], color='black', marker='$ns$', markersize=10, linestyle='', label='no significance'))

        # Create legend with better formatting
        legend = ax.legend(handles=legend_items,
                           loc='upper left',
                           bbox_to_anchor=(1.05, 1),
                           frameon=True,
                           framealpha=0.9,
                           edgecolor='lightgray',
                           fontsize=11)

        # Add descriptive text about the calculation method (we should not use this)
        # if (reference.lower() == "sitting") or (reference.lower() == "standing"):
        #     if reference.lower() == "standing" and not exclude_recovery:
        #         reference = "standing (and recovery standing)"
        #     fig.text(0.5, 0.01, f"{label_plot} reactivity calculated as mean {label_plot} during experimental task minus mean {label_plot} during {reference.lower()} baseline",
        #              ha='center', fontsize=10, fontstyle='italic')
        # else:
        #     fig.text(0.5, 0.01, f"HRV reactivity calculated as mean {label_plot} during experimental task minus mean {label_plot} during baseline (sitting + recovery sitting)",
        #              ha='center', fontsize=10, fontstyle='italic')

        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0.03, 0.85, 1])

        # Save the figure with high resolution
        save_path = os.path.join(save_path,
                                 f"boxplot_heart_rate_variability_reactivity_label_measure_{heart_measure}_reference_{reference}.png")
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')

        # Also save as vector format for journal publication
        # vector_save_path = os.path.join(save_path,
        #                                 f"boxplot_heart_rate_variability_reactivity_label_measure_{heart_measure}_reference_{reference}.pdf")
        # plt.savefig(vector_save_path, bbox_inches='tight', format='pdf')

        if show_plot:
            plt.show()

        plt.close()

    def plot_histogram(self,
                       column: str,
                       x_label: Optional[str] = None,
                       use_density: Optional[bool] = True,
                       save_path: Optional[str] = None,
                       save_name: Optional[str] = None,
                       show_plot=True,
                       show_baseline=True,
                       plot_subcategory: Optional[bool] = False,
                       category_to_plot: Optional[str] = "mental_stress") -> None:
        """
        Plots a histogram of the specified column, separated by category.

        Args:
            column: Name of the column to plot (e.g., 'hr_mean')
            x_label: str: label for the x-axis of the histogram
            use_density: bool: if set, we normalize the data and the pdf is then shown
            save_path: Optional path to save the plot. If None, plot is displayed.
            save_name: Optional: name of the resulting plot
            plot_subcategory: Optional: If set, we plot the subcategory labels
            category_to_plot: Optional: Which category to focus on when plotting labels
        """
        plt.figure(figsize=(10, 6))

        # Define colors and categories
        # old one
        if show_baseline:
            colors = {
                'black': '#000000',
                'orange': '#E69F00',
                'Sitting and recovery': '#000000',
                'low_physical_activity': '#009E73',
                'moderate_physical_activity': '#F0E442',
                'blue': '#0072B2',
                'mental_stress': '#0072B2',
                'high_physical_activity': '#d84315',
                'Sitting': '#9d9d9d',
            }

        else:
            colors = {
                'black': '#000000',
                'orange': '#E69F00',
                'low_physical_activity': '#009E73',
                'moderate_physical_activity': '#F0E442',
                'blue': '#0072B2',
                'mental_stress': '#0072B2',
                'high_physical_activity': '#d84315',
                'Sitting': '#9d9d9d',
            }


        # Silver color palette
        # https: // www.color - hex.com / color - palette / 1057579

        if plot_subcategory:
            label_data = self.total_data[self.total_data["category"]==category_to_plot][[column, "label"]]
            unique_labels = list(label_data["label"].unique())

            for sub_label in unique_labels:
                unique_label_data = label_data[label_data["label"]==sub_label][column]
                print(f"{sub_label} mean is {unique_label_data.mean()}")
                if not unique_label_data.empty:
                    plt.hist(unique_label_data,
                             bins=100,
                             alpha=0.25,
                             label=sub_label,
                             density=use_density)

            # Customize the plot
            plt.xlabel(f"{x_label}")
            plt.ylabel('Probability Density') if use_density else plt.ylabel('Count')
            plt.legend()

            # Save or show the plot
            if save_path:
                if save_name is not None:
                    save_path = os.path.join(save_path, f"{save_name}_label.png")
                else:
                    save_path = os.path.join(save_path, f"histogram_{x_label}_label.png")

                plt.savefig(save_path, dpi=500, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                plt.close()

        # Plot histograms for each category
        for category, color in colors.items():
            if category == "Sitting and recovery":
                category_name = "baseline"
                category_data = self.total_data[(self.total_data['category'] == category_name) | (self.total_data["label"] == category_name)][column]
            else:
                category_data = \
                self.total_data[(self.total_data['category'] == category) | (self.total_data["label"] == category)][
                    column]
            if not category_data.empty:
                sns.kdeplot(category_data, color=color, label=category.replace('_', ' ').title(), fill=True, alpha=0.6)

        # Customize the plot
        plt.xlabel(x_label)
        plt.ylabel('Density' if use_density else 'Frequency')
        plt.legend()

        # Save or show the plot
        if save_path:
            if save_name is not None:
                save_path = os.path.join(save_path, f"{save_name}.png")
            else:
                save_path = os.path.join(save_path, f"histogram_{column}_baseline_included_{show_baseline}.png")

            plt.savefig(save_path, dpi=500, bbox_inches='tight')
            plt.close()
        if show_plot:
            plt.show()
            plt.close()

        plt.close()

    def _split_data(self) -> tuple:
        """
        Splits the dataset into train, validation, and test sets based on participant CSV files.
        :param test_size: Proportion of the dataset to include in the test split
        :param val_size: Proportion of the dataset to include in the validation split
        :return: Tuple of (train_data, val_data, test_data)
        """
        # Use the filenames as participant identifiers
        participant_files = self.data_folders

        # Get the total files
        self.total_data = self._load_data(participant_files)

        self.number_mental_stress = self.total_data[self.total_data["category"] == "mental_stress"].count(axis=1)
        self.number_baseline = self.total_data[self.total_data["category"] == "baseline"].count(axis=1)
        self.number_lpa = self.total_data[self.total_data["category"] == "low_physical_activity"].count(axis=1)
        self.number_mpa = self.total_data[self.total_data["category"] == "moderate_physical_activity"].count(axis=1)
        self.number_hpa = self.total_data[self.total_data["category"] == "high_physical_activity"].count(axis=1)

        # Get the percentages:
        self.mental_stress_per = len(self.number_mental_stress) / len(self.total_data)
        self.baseline_per = len(self.number_baseline) / len(self.total_data)
        self.lpa_per = len(self.number_lpa) / len(self.total_data)
        self.mpa_per = len(self.number_mpa) / len(self.total_data)
        self.hpa_per = len(self.number_hpa) / len(self.total_data)

        train_files, test_files = train_test_split(participant_files, test_size=self.test_size)
        val_size_adjusted = self.val_size / (1 - self.test_size)  # Adjust validation size based on remaining data
        train_files, val_files = train_test_split(train_files, test_size=val_size_adjusted)

        # Here we should split the dataset intro train_feature_selection, val_feature_selection
        train_files_feature_selection, val_files_feature_selection = train_test_split(
            train_files, test_size=0.2
        )

        self.train_feature_selection = self._load_data(train_files_feature_selection)
        self.val_feature_selection = self._load_data(val_files_feature_selection)

        self.train_data = self._load_data(train_files)
        self.val_data = self._load_data(val_files)
        self.test_data = self._load_data(test_files)

        # Find the idx where split should occur "Recovery standing"
        if self.add_participant_id:
            self.train_data_within, self.test_data_within = self._split_data_by_condition(self.total_data)
            # Get rid of the participant id as we do not need it anymore
            self.train_data_within = self.train_data_within.drop(["participant_id"], axis=1)
            self.test_data_within = self.test_data_within.drop(["participant_id"], axis=1)
            self.train_data = self.train_data.drop(["participant_id"], axis=1)
            self.val_data = self.val_data.drop(["participant_id"], axis=1)
            self.test_data = self.test_data.drop(["participant_id"], axis=1)

    def get_feature_selection_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        returns the datasets split for the feature selection process
        :return:
        """
        return self.train_feature_selection, self.val_feature_selection

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        returns the datasets split in train, val and test data
        :return:
        """
        return self.train_data, self.val_data, self.test_data


#ToDo: Really refactor this code!
def encode_data(
        data: pd.DataFrame,
        positive_class: str,
        negative_class: str,
        leave_one_out: bool,
        leave_out_stressor_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # First drop data that is not either in the positive class or negative class
    # How can make all labels lower case and rename ssst
    data["label"] = data["label"].str.lower()
    data["label"] = data["label"].str.replace('ssst_sing_countdown', 'ssst', regex=False)

    positive_single_classes = ["ssst", "raven", "ta", "pasat", "pasat_repeat", "ta_repeat"]
    # Special case for any_physical_activity vs non_physical_activity
    if positive_class == "any_physical_activity" and negative_class == "non_physical_activity":
        possible_physical_activities = ["low_physical_activity", "moderate_physical_activity", "high_physical_activity"]

        # Keep all data - we want to classify between physical activities and everything else
        # Encode as 1 for any physical activity and 0 for everything else
        data.loc[:, 'category'] = data['category'].apply(
            lambda x: 1 if x in possible_physical_activities else 0)  # Encode classes

    elif positive_class in positive_single_classes and not leave_one_out:
        if negative_class == "base_lpa_mpa":
            possible_physical_activities = ["baseline", "low_physical_activity", "moderate_physical_activity"]

            data = data[
                (data['label'] == positive_class) |
                (data['category'].isin(possible_physical_activities))
                ]

            data.loc[:, 'category'] = data['category'].apply(
                lambda x: 0 if x in possible_physical_activities else 1)  # Encode classes

        else:
            data = data[
                (data['label'] == positive_class) |
                (data['category'] == negative_class)
                ]

            data.loc[:, 'category'] = data['category'].apply(lambda x: 0 if x == negative_class else 1)  # Encode classes

    elif (negative_class == "any_physical_activity") or (positive_class == "any_physical_activity"):
        possible_physical_activities =  ["low_physical_activity", "moderate_physical_activity", "high_physical_activity"]

        if negative_class == "any_physical_activity":
            data = data[
                (data['category'] == positive_class) | (data['category'].isin(possible_physical_activities))]  # Filter relevant classes
        else:
            data = data[
                (data['category'] == negative_class) | (data['category'].isin(possible_physical_activities))]

        # Then label the data 1 for positive and 0 for negative
        if positive_class == "any_physical_activity":
            data.loc[:, 'category'] = data['category'].apply(lambda x: 1 if x in possible_physical_activities else 0)  # Encode classes
        else:
            data.loc[:, 'category'] = data['category'].apply(
                lambda x: 1 if x == positive_class else 0)  # Encode classes

    elif (negative_class == "base_lpa_mpa") or (positive_class == "base_lpa_mpa"):
        possible_physical_activities =  ["baseline", "low_physical_activity", "moderate_physical_activity"]

        if negative_class == "base_lpa_mpa":
            data = data[
                (data['category'] == positive_class) | (data['category'].isin(possible_physical_activities))]  # Filter relevant classes
        else:
            data = data[
                (data['category'] == negative_class) | (data['category'].isin(possible_physical_activities))]

        # Then label the data 1 for positive and 0 for negative
        if positive_class == "base_lpa_mpa":
            data.loc[:, 'category'] = data['category'].apply(lambda x: 1 if x in possible_physical_activities else 0)  # Encode classes
        else:
            data.loc[:, 'category'] = data['category'].apply(
                lambda x: 1 if x == positive_class else 0)  # Encode classes

    elif (negative_class == "low_moderate_physical_activity") or (positive_class == "low_moderate_physical_activity"):
        possible_physical_activities =  ["low_physical_activity", "moderate_physical_activity"]

        if negative_class == "low_moderate_physical_activity":
            data = data[
                (data['category'] == positive_class) | (data['category'].isin(possible_physical_activities))]  # Filter relevant classes
        else:
            data = data[
                (data['category'] == negative_class) | (data['category'].isin(possible_physical_activities))]

        # Then label the data 1 for positive and 0 for negative
        if positive_class == "low_moderate_physical_activity":
            data.loc[:, 'category'] = data['category'].apply(lambda x: 1 if x in possible_physical_activities else 0)  # Encode classes
        else:
            data.loc[:, 'category'] = data['category'].apply(
                lambda x: 1 if x == positive_class else 0)  # Encode classes

    elif negative_class == "rest":
        # By rest we mean everything without high physical_activity
        data = data[(data["category"] != "high_physical_activity")]
        data.loc[:, 'category'] = data['category'].apply(lambda x: 1 if x == positive_class else 0)  # Encode classes

    elif (negative_class == "non_physical_activity") or (positive_class == "non_physical_activity"):
        possible_physical_activities = ["low_physical_activity", "moderate_physical_activity", "high_physical_activity"]

        if negative_class == "non_physical_activity":
            data = data[
                (data['category'] == positive_class) | (
                    ~data['category'].isin(possible_physical_activities))]  # Filter relevant classes
        else:
            data = data[
                (data['category'] == negative_class) | ~(data['category'].isin(possible_physical_activities))]

        # Then label the data 1 for positive and 0 for negative
        if positive_class == "non_physical_activity":
            data.loc[:, 'category'] = data['category'].apply(
                lambda x: 1 if x not in possible_physical_activities else 0)  # Encode classes
        else:
            data.loc[:, 'category'] = data['category'].apply(
                lambda x: 1 if x == positive_class else 0)  # Encode classes

    else:
        column_name_negative_class = "label"  if negative_class in ["walking_own_pace", "standing", "sitting"] else "category"
        data = data[(data['category'] == positive_class) | (data[column_name_negative_class].str.lower() == negative_class)]  # Filter relevant classes
        # Then label the data 1 for positive and 0 for negative
        data.loc[:, 'category'] = data['category'].apply(lambda x: 1 if x == positive_class else 0)  # Encode classes

    if leave_one_out:
        if any(data.label.str.lower().str.startswith(leave_out_stressor_name)):
            data = data[~data.label.str.lower().str.startswith(leave_out_stressor_name)]
            assert not any(
                data.label.str.lower().str.startswith(leave_out_stressor_name)), "leave out operation went wrong!"
        else:
            print(f"We could not find {leave_out_stressor_name} in our dataset")

    x = data.drop(columns=["category", "label"]).reset_index(drop=True) if "label" in list(data.columns) \
        else data.drop(columns=["category"]).reset_index(drop=True)
    label = data["label"].reset_index(drop=True) if "label" in list(data.columns) \
        else None

    # In case we have the column label here we will keep it so we can track it
    # The target label needs to be an integer
    y = data["category"].astype(int).reset_index(drop=True)

    return x, label, y


def handle_missing_data(data: pd.DataFrame, drop_values = True) -> pd.DataFrame:
    original_data_len = len(data)


    # Identify rows and columns with infinity values
    inf_mask = data.isin([np.inf, -np.inf])
    rows_with_inf = inf_mask.any(axis=1)
    cols_with_inf = inf_mask.any(axis=0)

    print(f"Rows with infinity values: {rows_with_inf.sum()}")
    print(f"Columns with infinity values:")
    for col in data.columns[cols_with_inf]:
        inf_count = inf_mask[col].sum()
        print(f"  - {col}: {inf_count} infinity values ({(inf_count / len(data)) * 100:.2f}%)")

    # We drop the colum completely:
    # Identify rows and columns with NaN values
    nan_mask = data.isna()
    rows_with_nan = nan_mask.any(axis=1)
    cols_with_nan = nan_mask.any(axis=0)

    # Get rows with infinity values
    print(f"Rows with NaN values: {rows_with_nan.sum()}")
    print(f"Columns with NaN values:")
    for col in data.columns[cols_with_nan]:
        nan_count = nan_mask[col].sum()
        print(f"  - {col}: {nan_count} NaN values ({(nan_count / len(data)) * 100:.2f}%)")

    # Drop both data now (infinity data and missing values)
    if drop_values:
        clean_data = data[~data.isin([np.inf, -np.inf]).any(axis=1)]
        clean_data = clean_data.dropna()

        dropped_percent = ((original_data_len - len(clean_data)) / original_data_len) * 100
        print(f"Dropping these rows removed {np.round(dropped_percent, 4)}% of the original data")

    # Else we can impute the data with KNN imputer for instance
    else:
        raise NotImplementedError("We have not yet implemented the imputation method")

    return clean_data


def resample_data(data: pd.DataFrame,
                  positive_class: str,
                  negative_class: str,
                  downsample: bool) -> pd.DataFrame:
    """
    Resample the data to balance classes, either via upsampling minority or downsampling majority.

    Args:
        data: Input DataFrame containing the data
        positive_class: Label of the positive class
        negative_class: Label of the negative class
        downsample: If True, downsample majority class; if False, upsample minority

    Returns:
        Balanced DataFrame with equal class distributions

    Note:
        Upsampling minority class will create duplicates for highly imbalanced data
    """
    # Split data by class
    df_positive = data[data["category"] == positive_class]
    df_negative = data[data["category"] == negative_class]

    # Determine majority and minority classes
    if len(df_positive) >= len(df_negative):
        majority_df, minority_df = df_positive, df_negative
    else:
        majority_df, minority_df = df_negative, df_positive

    # Perform resampling
    if downsample:
        print(f"We downsample!")
        resampled_majority = resample(majority_df,
                                    replace=False,
                                    n_samples=len(minority_df),
                                    random_state=42)
        balanced_data = pd.concat([resampled_majority, minority_df])
    else:
        print(f"We upsample!")
        resampled_minority = resample(minority_df,
                                    replace=True,
                                    n_samples=len(majority_df),
                                    random_state=42)
        balanced_data = pd.concat([resampled_minority, majority_df])

    # Shuffle the final dataset
    balanced_data = balanced_data.sample(frac=1, replace=False, random_state=42).reset_index(drop=True)

    # Verify balancing
    class_counts = balanced_data["category"].value_counts()
    assert class_counts[positive_class] == class_counts[negative_class], \
        f"Resampling failed: classes are not balanced. Counts: {class_counts}"

    return balanced_data


def analyze_feature_distributions(df: pd.DataFrame, alpha: float = 0.05):
    """
    Analyzes each feature, tests for normality, applies multiple transformations,
    and provides visualization of before/after transformations.

    Args:
        df: DataFrame with features to analyze
        alpha: Significance level for normality test
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import boxcox

    # Store results for each feature
    feature_stats = {}
    transformed_df = df.copy()

    # Analyze each feature
    for column in df.columns:
        # Skip columns with all missing values
        if df[column].isnull().all():
            continue

        # Get non-null values for analysis
        data = df[column].dropna()

        # Calculate basic statistics
        stats_summary = {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'missing_pct': df[column].isnull().mean() * 100
        }

        # Test for normality
        # Use Shapiro-Wilk for smaller samples (works best for n<50)
        # Use D'Agostino-Pearson for larger datasets
        if len(data) < 50:
            if len(data) >= 3:  # Shapiro-Wilk requires at least 3 samples
                normality_stat, p_value = stats.shapiro(data)
                stats_summary['normality_test'] = 'Shapiro-Wilk'
            else:
                p_value = None
                stats_summary['normality_test'] = 'Too few samples'
        else:
            sample = data.sample(n=5000) if len(data) > 5000 else data  # Limit sample size for performance
            normality_stat, p_value = stats.normaltest(sample)  # D'Agostino-Pearson test
            stats_summary['normality_test'] = 'D\'Agostino-Pearson'

        stats_summary['p_value'] = p_value
        stats_summary['is_normal'] = p_value > alpha if p_value is not None else None

        # Always try log transform if data is positive
        log_transform_possible = data.min() > 0
        if log_transform_possible:
            # Apply log transform
            transformed_df[column + '_log'] = np.log1p(df[column])  # log(1+x) to handle zeros
            log_data = transformed_df[column + '_log'].dropna()

            # Check normality after log transformation
            if len(log_data) >= 3:
                if len(log_data) < 50:
                    _, p_log = stats.shapiro(log_data)
                else:
                    log_sample = log_data.sample(n=5000) if len(log_data) > 5000 else log_data
                    _, p_log = stats.normaltest(log_sample)
                stats_summary['p_value_log'] = p_log
                stats_summary['is_normal_log'] = p_log > alpha

            # Try Box-Cox transformation if appropriate
            try:
                transformed_data, lambda_param = boxcox(data)
                transformed_df[column + '_boxcox'] = transformed_data
                stats_summary['lambda'] = lambda_param

                # Check normality after Box-Cox transformation
                if len(transformed_data) >= 3:
                    if len(transformed_data) < 50:
                        _, p_boxcox = stats.shapiro(transformed_data)
                    else:
                        transformed_sample = transformed_data[:5000] if len(
                            transformed_data) > 5000 else transformed_data
                        _, p_boxcox = stats.normaltest(transformed_sample)
                    stats_summary['p_value_boxcox'] = p_boxcox
                    stats_summary['is_normal_boxcox'] = p_boxcox > alpha
            except (ValueError, np.linalg.LinAlgError):
                # If Box-Cox fails, note it in the stats
                stats_summary['boxcox_failed'] = True

        # Determine best transformation
        if log_transform_possible:
            if stats_summary.get('is_normal', False):
                stats_summary['best_transform'] = 'none'
            elif stats_summary.get('is_normal_log', False) and not stats_summary.get('is_normal_boxcox', False):
                stats_summary['best_transform'] = 'log1p'
            elif stats_summary.get('is_normal_boxcox', False) and not stats_summary.get('is_normal_log', False):
                stats_summary['best_transform'] = 'boxcox'
            elif stats_summary.get('is_normal_boxcox', False) and stats_summary.get('is_normal_log', False):
                # If both work, choose the one with higher p-value
                if stats_summary.get('p_value_boxcox', 0) > stats_summary.get('p_value_log', 0):
                    stats_summary['best_transform'] = 'boxcox'
                else:
                    stats_summary['best_transform'] = 'log1p'
            else:
                # Neither achieved normality
                stats_summary['best_transform'] = 'none_effective'
        else:
            stats_summary['best_transform'] = 'none'
            stats_summary['transform_note'] = 'Data contains zero or negative values'

        feature_stats[column] = stats_summary

    # Function to visualize each feature
    def view_feature(feature_index=0, show_plot=False, save_path=None, save_feature_plots=False):
        features = list(feature_stats.keys())
        if feature_index >= len(features) or feature_index < 0:
            print(f"Index out of range. Please select an index between 0 and {len(features) - 1}")
            return

        feature = features[feature_index]
        stats_dict = feature_stats[feature]

        # Organize the figure in a 3x3 grid: histograms in left column, Q-Q plots in middle, stats on right
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.8])

        # Original data row
        ax_hist_orig = fig.add_subplot(gs[0, 0])  # Original histogram
        ax_qq_orig = fig.add_subplot(gs[0, 1])  # Original Q-Q plot

        # Log transform row
        ax_hist_log = fig.add_subplot(gs[1, 0])  # Log histogram
        ax_qq_log = fig.add_subplot(gs[1, 1])  # Log Q-Q plot

        # Box-Cox row
        ax_hist_boxcox = fig.add_subplot(gs[2, 0])  # Box-Cox histogram
        ax_qq_boxcox = fig.add_subplot(gs[2, 1])  # Box-Cox Q-Q plot

        # Stats panel on right (spans all rows)
        ax_stats = fig.add_subplot(gs[:, 2])

        # Set figure title
        fig.suptitle(f"Feature Analysis: {feature} (Index: {feature_index}/{len(features) - 1})", fontsize=16)

        # Original distribution histogram and Q-Q plot
        sns.histplot(df[feature].dropna(), kde=True, ax=ax_hist_orig)
        ax_hist_orig.set_title(f"Original Distribution\nSkewness: {stats_dict['skewness']:.2f}")

        stats.probplot(df[feature].dropna(), plot=ax_qq_orig)
        ax_qq_orig.set_title(f"Q-Q Plot (Original)\np-value: {stats_dict.get('p_value', 'N/A'):.4e}")

        # Log transformation plots (if possible)
        log_transform_possible = stats_dict.get('p_value_log') is not None
        if log_transform_possible:
            sns.histplot(transformed_df[feature + '_log'].dropna(), kde=True, ax=ax_hist_log)
            ax_hist_log.set_title(
                f"Log Transformed (log1p)\nSkewness: {stats.skew(transformed_df[feature + '_log'].dropna()):.2f}")

            stats.probplot(transformed_df[feature + '_log'].dropna(), plot=ax_qq_log)
            ax_qq_log.set_title(f"Q-Q Plot (Log)\np-value: {stats_dict.get('p_value_log', 'N/A'):.4e}")
        else:
            ax_hist_log.set_visible(False)
            ax_qq_log.set_visible(False)

        # Box-Cox transformation plots (if possible)
        boxcox_transform_possible = 'p_value_boxcox' in stats_dict and not stats_dict.get('boxcox_failed', False)
        if boxcox_transform_possible:
            sns.histplot(transformed_df[feature + '_boxcox'].dropna(), kde=True, ax=ax_hist_boxcox)
            ax_hist_boxcox.set_title(f"Box-Cox Transformed (λ={stats_dict.get('lambda', 'N/A'):.4f})\n" +
                                     f"Skewness: {stats.skew(transformed_df[feature + '_boxcox'].dropna()):.2f}")

            stats.probplot(transformed_df[feature + '_boxcox'].dropna(), plot=ax_qq_boxcox)
            ax_qq_boxcox.set_title(f"Q-Q Plot (Box-Cox)\np-value: {stats_dict.get('p_value_boxcox', 'N/A'):.4e}")
        else:
            ax_hist_boxcox.set_visible(False)
            ax_qq_boxcox.set_visible(False)

        # Clear the stats axis and set no frame
        ax_stats.axis('off')

        # Create the stats text
        stats_text = [
            f"Feature: {feature}",
            f"Mean: {stats_dict['mean']:.4f}",
            f"Median: {stats_dict['median']:.4f}",
            f"Min: {stats_dict['min']:.4f}",
            f"Max: {stats_dict['max']:.4f}",
            f"Std Dev: {stats_dict['std']:.4f}",
            f"Skewness: {stats_dict['skewness']:.4f}",
            f"Kurtosis: {stats_dict['kurtosis']:.4f}",
            f"Missing: {stats_dict['missing_pct']:.1f}%",
            f"\nNormality Tests:",
            f"Test type: {stats_dict['normality_test']}",
            f"Original p-value: {stats_dict.get('p_value', 'N/A'):.4e}",
            f"Original is normal: {'Yes' if stats_dict.get('is_normal') else 'No'}"
        ]

        if log_transform_possible:
            stats_text.extend([
                f"\nLog Transform:",
                f"Log p-value: {stats_dict.get('p_value_log', 'N/A'):.4e}",
                f"Log is normal: {'Yes' if stats_dict.get('is_normal_log', False) else 'No'}"
            ])

        if boxcox_transform_possible:
            stats_text.extend([
                f"\nBox-Cox Transform:",
                f"Lambda: {stats_dict.get('lambda', 'N/A'):.4f}",
                f"Box-Cox p-value: {stats_dict.get('p_value_boxcox', 'N/A'):.4e}",
                f"Box-Cox is normal: {'Yes' if stats_dict.get('is_normal_boxcox', False) else 'No'}"
            ])

        # Add best transformation recommendation
        stats_text.extend([
            f"\nRecommendation:",
            f"Best transform: {stats_dict.get('best_transform', 'none').replace('_', ' ').title()}"
        ])

        # Add notes for cases where neither transform worked
        if stats_dict.get('best_transform') == 'none_effective':
            stats_text.extend([
                f"\nNote: Neither transformation achieved normality.",
                f"Consider:",
                f"- Square root transformation",
                f"- Yeo-Johnson transformation",
                f"- Quantile normalization",
                f"- Using non-parametric methods"
            ])

        # Add notes if transforms couldn't be applied
        if not log_transform_possible or not boxcox_transform_possible:
            stats_text.append(f"\nNote: Some transformations couldn't be applied.")
            if 'transform_note' in stats_dict:
                stats_text.append(stats_dict['transform_note'])

        # Display the stats text in a nicely formatted box
        ax_stats.text(0.05, 0.95, '\n'.join(stats_text),
                      transform=ax_stats.transAxes,
                      fontsize=11,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round,pad=1', facecolor='orange', alpha=0.2))

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to make room for the title

        # Save figure:
        if save_feature_plots:
            plt.savefig(os.path.join(save_path, f'feature_plot_{feature}.png'),
                        dpi=400, format="png")
        if show_plot:
            plt.show()

        return fig

    # Function to run all features and display them automatically
    def run_all_features():
        features = list(feature_stats.keys())
        print(f"Analyzing {len(features)} features...")

        for i in range(len(features)):
            fig = view_feature(i)
            plt.close(fig)  # Clean up after displaying

        print("Analysis complete!")

    return view_feature, run_all_features, transformed_df, feature_stats


def prepare_data(train_data: pd.DataFrame,
                 val_data: pd.DataFrame,
                 test_data: Optional[pd.DataFrame] = None,
                 imputation_method: Optional[str] = "knn",
                 positive_class: Optional[str] = "mental_stress",
                 negative_class: Optional[str] = "baseline",
                 resampling_method: Optional[str] = None,
                 balance_positive_sublabels: Optional[bool] = False,
                 balance_sublabels_method: Optional[str] = "upsample",
                 scaler: Optional[str] = None,
                 use_quantile_transformer: Optional[bool] = False,
                 use_subset: Optional[list[bool]] = None,
                 save_path: Optional[str] = None,
                 save_feature_plots: bool = False,
                 leave_one_out: Optional[bool] = False,
                 leave_out_stressor_name:Optional[str]= None,
                 ) -> tuple:
    """
    Prepares the data for scikit-learn models. Can handle both 2-way (train/val) and 3-way (train/val/test) splits.

    Args:
        train_data: DataFrame containing the training data
        val_data: DataFrame containing the validation data
        test_data: Optional DataFrame containing the test data. If None, assumes 2-way split
        imputation_method: Optional. Str. Either 'drop', 'knn', 'knn_subset', or 'iterative_imputer'
        positive_class: str, which category to be encoded as 1
        negative_class: str, which category to be encoded as 0
        resampling_method: str, resampling method to use. Options: None, "downsample", "upsample", "smote", "adasyn"
        balance_positive_sublabels: bool, whether to balance sublabels within the positive class
        balance_sublabels_method: str, "upsample" or "downsample" the sublabels within positive class
        scaler: StandardScaler instance for normalization
        use_quantile_transformer: If set, we transform the features to normal distribution first
        use_subset: bool, list of bool to indicate which features should be included or not
        save_path: str. Save path to plot the feature plots and save them so we can see what is going on
        save_feature_plots: bool. Save feature plots which are then saved in save path
        leave_one_out: Optional[bool]: If we use the leave one out setting,
        leave_out_stressor_name: Optional[str]: If leave one out setting is used, which stressor to leave out

    Returns:
        If test_data is provided:
            Tuple of ((X_train, y_train, label_train), (X_val, y_val, label_val), (X_test, y_test, label_test), feature_names)
        If test_data is None:
            Tuple of ((X_train, y_train, label_train), (X_val, y_val, label_val), feature_names)
    """
    # Old code

    try:
        overall_label_distribution = train_data['category'].value_counts().to_dict()

        for key, value in val_data['category'].value_counts().to_dict().items():
            overall_label_distribution[key] += value

        for key, value in test_data['category'].value_counts().to_dict().items():
            overall_label_distribution[key] += value

        print(f" The overall label distribution is {overall_label_distribution}")
    except TypeError:
        overall_label_distribution = 0

    assert imputation_method in ["drop", "knn", "knn_subset", "iterative_imputer"], \
        "Please use as imputation method either 'knn', 'drop' or 'knn_subset'."

    # lf_feature.Power is useless as it does not vary at all. Check this!
    if 'lf_Feature.POWER' in train_data.columns:
        # Check if sampen is included in the data columns
        train_data.drop('lf_Feature.POWER', axis=1, inplace=True)
    if (test_data is not None) and ('lf_Feature.POWER' in test_data.columns):
        test_data.drop('lf_Feature.POWER', axis=1, inplace=True)
    if 'lf_Feature.POWER' in val_data.columns:
        val_data.drop('lf_Feature.POWER', axis=1, inplace=True)

    # Sampen
    if imputation_method == "drop":
        if 'sampen' in train_data.columns:
        # Check if sampen is included in the data columns
            train_data.drop('sampen', axis=1, inplace=True)
        if (test_data is not None) and ('sampen' in test_data.columns):
            test_data.drop('sampen', axis=1, inplace=True)
        if 'sampen' in val_data.columns:
            val_data.drop('sampen', axis=1, inplace=True)

        train_data = handle_missing_data(train_data)

        if val_data is not None:
            val_data = handle_missing_data(val_data)
        if test_data is not None:
            test_data = handle_missing_data(test_data)

    # First, if requested, balance the sublabels within the positive class
    if balance_positive_sublabels:
        print(f"Balancing sublabels within positive class '{positive_class}' using {balance_sublabels_method} method")
        # train_data = balance_sublabels(train_data, positive_class, balance_sublabels_method)

        upsample = balance_sublabels_method.lower() == "upsample"
        train_data = balance_sublabels(train_data, positive_class, upsample)

    # If no resampling, just shuffle and encode the data
    train_data = train_data.sample(frac=1, replace=False, random_state=42).reset_index(drop=True)
    x_train, label_train, y_train = encode_data(train_data, positive_class, negative_class,
                                               leave_one_out, leave_out_stressor_name)

    # We have not yet cleaned up missing values, so we replace them with nan values first
    x_train = x_train.replace([np.inf, -np.inf], np.nan)

    # Could we handle the missing data here?
    if val_data is not None:
        x_val, label_val, y_val = encode_data(val_data, positive_class, negative_class,
                                              leave_one_out, leave_out_stressor_name)
        x_val = x_val.replace([np.inf, -np.inf], np.nan)

    if test_data is not None:
        # In  test data we never leave out any stressor, only test and val data
        leave_one_out = False

        x_test, label_test, y_test = encode_data(test_data, positive_class, negative_class,
                                                 leave_one_out, leave_out_stressor_name)
        x_test = x_test.replace([np.inf, -np.inf], np.nan)

    # Ensure the length of use_subset matches the number of features
    if use_subset is not None:
        assert len(use_subset) == x_train.shape[1], \
            f"Length of use_subset ({len(use_subset)}) must match number of features ({x_train.shape[1]})"

        # Filter features using boolean mask
        x_train = x_train.iloc[:, use_subset]

        if val_data is not None:
            x_val = x_val.iloc[:, use_subset]
        if test_data is not None:
            x_test = x_test.iloc[:, use_subset]

    feature_names = list(x_train.columns.values)

    # If positive values only -> maybe log transform
    # Else min-max scaling
    if test_data is not None:
        viewer, run_all, transformed_df, stats = analyze_feature_distributions(x_test)

    if save_feature_plots:
        for feature_idx in range(len(feature_names)):
            print(f"We are plotting and saving feature plot {feature_idx}/{len(feature_names)}")
            viewer(feature_idx, save_path=save_path, save_feature_plots=save_feature_plots)  # View the first feature

    # Apply scaling after resampling if requested
    if scaler is not None:
        assert scaler.lower() in ["min_max", "standard_scaler"], \
            "please set a valid scaler. Options: 'min_max', 'standard_scaler'"

        if use_quantile_transformer:
            print("We use the quantile transformer")
            quantile_transformer_obj = QuantileTransformer(n_quantiles=1000, output_distribution="normal")
            x_train = pd.DataFrame(quantile_transformer_obj.fit_transform(x_train),
                                   columns=x_train.columns, index=x_train.index)
            if val_data is not None:
                x_val = pd.DataFrame(quantile_transformer_obj.transform(x_val),
                                     columns=x_val.columns, index=x_val.index)
            if test_data is not None:
                x_test = pd.DataFrame(quantile_transformer_obj.transform(x_test),
                                      columns=x_test.columns, index=x_test.index)

        # List of special columns to always scale with MinMaxScaler
        special_cols = [col for col in ["nn20", "nn50", "wmax"] if col in x_train.columns]

        if scaler.lower() == "standard_scaler":
            # For standard_scaler, use StandardScaler on other columns and MinMaxScaler on special columns.
            other_cols = [col for col in x_train.columns if col not in special_cols]

            # Instantiate scalers
            scaler_obj = StandardScaler()
            min_max_scaler = MinMaxScaler()

            # Create copies to hold scaled data
            x_train = x_train.copy()
            # Fit and transform the "other" columns
            if other_cols:
                x_train[other_cols] = scaler_obj.fit_transform(x_train[other_cols])
            # Fit and transform the special columns with MinMaxScaler
            if special_cols:
                x_train[special_cols] = min_max_scaler.fit_transform(x_train[special_cols])

            # Apply the same transformation on validation and test sets if available.
            if val_data is not None:
                x_val = x_val.copy()
                if other_cols:
                    x_val[other_cols] = scaler_obj.transform(x_val[other_cols])
                if special_cols:
                    x_val[special_cols] = min_max_scaler.transform(x_val[special_cols])
            if test_data is not None:
                x_test_scaled = x_test.copy()
                if other_cols:
                    x_test[other_cols] = scaler_obj.transform(x_test[other_cols])
                if special_cols:
                    x_test[special_cols] = min_max_scaler.transform(x_test[special_cols])
        else:
            # Otherwise, use the specified scaler (likely MinMaxScaler) on the entire dataset.
            scaler_obj = MinMaxScaler()  # or your alternative scaler if needed
            x_train = pd.DataFrame(scaler_obj.fit_transform(x_train),
                                          columns=x_train.columns, index=x_train.index)
            if val_data is not None:
                x_val = pd.DataFrame(scaler_obj.transform(x_val),
                                            columns=x_val.columns, index=x_val.index)
            if test_data is not None:
                x_test = pd.DataFrame(scaler_obj.transform(x_test),
                                             columns=x_test.columns, index=x_test.index)

    # Here we should use the KNN imputer then
    if imputation_method in ["knn", "iterative_imputer"]:

        if overall_label_distribution != 0:
            report_nans(x_train, "train")
            report_nans(x_val, "validation")
            report_nans(x_test, "test")

        imputer = KNNImputer(n_neighbors=5, copy=False) if imputation_method == "knn" else IterativeImputer(max_iter=10, random_state=0)

        imputer.fit(x_train)
        x_train = imputer.transform(x_train)
        if val_data is not None:
            x_val = imputer.transform(x_val)
        if test_data is not None:
            x_test = imputer.transform(x_test)

        # Check if any missing values are still present:
        assert np.isnan(x_train).sum() == 0, "Imputation did not work!"

    # Apply resampling only to training data (this now happens after sublabel balancing if enabled)
    if resampling_method in ["downsample", "upsample"]:
        do_downsampling = resampling_method == "downsample"
        train_data = resample_data(train_data, positive_class, negative_class, downsample=do_downsampling)
        x_train, label_train, y_train = encode_data(train_data, positive_class, negative_class,
                                                  leave_one_out, leave_out_stressor_name)

    elif resampling_method == "smote":
        # SMOTE doesn't preserve the original label information, so we need to make sure
        # we're only generating synthetic samples for the positive class after sublabel balancing
        smote = SMOTE(random_state=42)
        x_train, y_train = smote.fit_resample(x_train, y_train)
        # Note: label_train will be lost during SMOTE resampling

    elif resampling_method == "adasyn":
        adasyn = ADASYN(random_state=42)
        x_train, y_train = adasyn.fit_resample(x_train, y_train)
        # Note: label_train will be lost during ADASYN resampling

    # # For better clustering with UMAP
    # fig, ax, embedding = visualize_dimensionality_reduction(
    #     x_train, y_train, label_train,
    #     method='umap',
    #     include_baseline=False,
    #     # UMAP adjustments for tighter clusters
    #     n_neighbors=5,
    #     min_dist=0.00000,
    #     metric='euclidean',
    #     subset_size=250,
    # )
    #
    # # For better clustering with t-SNE
    # fig, ax, embedding = visualize_dimensionality_reduction(
    #     x_train, y_train, label_train,
    #     method='tsne',
    #     include_baseline=False,
    #     min_dist=0.0,
    #     # t-SNE adjustments
    #     perplexity=25,
    #     n_iter=2000,
    #     early_exaggeration=18.0,
    #     learning_rate=150  #
    # )

    # Return appropriate tuple based on whether test_data was provided
    if test_data is not None and val_data is not None:
        return (x_train, y_train, label_train), (x_val, y_val, label_val), (x_test, y_test, label_test), feature_names
    else:
        if val_data is not None:
            return (x_train, y_train, label_train), (x_val, y_val, label_val), feature_names
        else:
            return (x_train, y_train, label_train), None, (x_test, y_test, label_test), feature_names


def normalize_data(train_data: pd.DataFrame) -> tuple:
    """
    Normalizes the training data and returns the scaler for future use.
    :param train_data: DataFrame containing the training data
    :return: Tuple of (normalized_train_data, scaler)
    """
    scaler = StandardScaler()
    features = train_data.drop(columns=['target'])  # Replace 'target' with your actual target column name
    normalized_train_data = scaler.fit_transform(features)  # Normalize features
    return normalized_train_data, scaler


def get_ml_model(model: str, params: dict = None):
    """
    Returns the machine learning model initialized with the specified configuration settings.

    Args:
        model (str): The name of the machine learning model to initialize.
                     Options include 'DT', 'RF', 'AdaBoost', 'LDA', 'KNN', 'LR', 'XGBoost', 'QDA'.
        params (dict, optional): A dictionary of parameters to initialize the model.
                                 If None, default parameters will be used.

    Raises:
        ValueError: If the specified model name is invalid.

    Returns:
        object: An instance of the specified machine learning model initialized with the given parameters.
    """
    # Default parameters for each model
    default_params = {
        "dt": {"random_state": 42},
        "rf": {"random_state": 42, "bootstrap": False, "n_jobs": -1},
        "adaboost": {"base_estimator": DecisionTreeClassifier(criterion='entropy', min_samples_split=20)},
        "lda": {},
        "knn": {"n_jobs": -1},
        "lr": {"n_jobs": -1},
        "xgboost": {},
        "qda": {},
        "svm": {"kernel": "rbf", "C": 1.0, "gamma": 0.7},
        "random_baseline": {"strategy": "prior"},
        "gmm": {"n_components": 2, 'covariance_type': 'diag', "n_init": 50, "random_state": 42},
    }

    # Map model names to their corresponding classes
    model_classes = {
        "dt": DecisionTreeClassifier,
        "rf": RandomForestClassifier,
        "adaboost": AdaBoostClassifier,
        "lda": LinearDiscriminantAnalysis,
        "knn": KNeighborsClassifier,
        "lr": LogisticRegression,
        "xgboost": xgb.XGBClassifier,
        "qda": QuadraticDiscriminantAnalysis,
        "svm": SVC,
        "random_baseline": DummyClassifier,
        "gmm": GaussianMixture,
    }

    if model.lower() not in model_classes:
        raise ValueError('Invalid model')

    if params is None:
        params = default_params[model.lower()]

    cls = model_classes[model.lower()](**params)  # Initialize the model with parameters

    return cls


def get_data_balance(train_data:np.array, val_data: np.array, test_data: np.array) -> np.array:
    """
    Calculates the imbalance of the dataset overall
    """

    overall_data_len = len(train_data) + len(val_data) + len(test_data)
    percentage_train = len(train_data) / overall_data_len
    percentage_val = len(val_data) / overall_data_len
    percentage_test = len(test_data) / overall_data_len

    assert_almost_equal((percentage_train + percentage_val + percentage_test), 1.0, decimal=5)

    class_1_train = np.mean(train_data)
    class_1_val = np.mean(val_data)
    class_1_test = np.mean(test_data)

    data_balance = np.round(percentage_train * class_1_train + percentage_val * class_1_val +  percentage_test * class_1_test, 4)
    return data_balance


def get_idx_per_subcategory(y_data, label, positive_class=True, include_other_class=True,
                            random_seed: Optional[int]=42):

    combined_df = pd.concat([y_data, label], axis=1)
    class_1 = len(y_data[y_data == 1.0])
    ratio_1 = np.round((class_1 / len(y_data)), 4)

    label_df = combined_df[combined_df["category"] == 1.0 if positive_class else combined_df["category"] == 0.0]
    if include_other_class:
        # Some metrics such as roc_auc need to have negative examples as well
        other_class = combined_df[combined_df["category"] == 0.0 if positive_class else combined_df["category"] == 1.0]
        other_class_idx = list(other_class.index.values)

    subcategories = list(set(label_df["label"].values))

    idx_per_subcategory = {}

    for category in subcategories:
        # Create a controlled random state using the seed
        rng = np.random.RandomState(random_seed)
        idx_values = list(label_df[label_df["label"] == category].index.values)
        replace_arg = False
        # we need to sample len(x) (1-ratio) / ratio(1) to get the same ratio 1/0
        if include_other_class:
            # Use the controlled random state for sampling
            size = int(((1 - ratio_1) * len(idx_values)) / ratio_1)
            # If we have only one subcategory we run into issues as then the sample is too small
            if (len(subcategories)) == 1 and size > len(other_class_idx):
                replace_arg = True
                size = len(other_class_idx)

            sampled_negative_class = sorted(rng.choice(
                other_class_idx, replace=replace_arg, size=size
            ))
            idx_values.extend(list(sampled_negative_class))

        idx_per_subcategory[category] = idx_values

    return idx_per_subcategory


def evaluate_ml_model_score(
        ml_model: BaseEstimator,
        data: tuple[np.ndarray, np.ndarray],
        threshold: float,
        score:str="f1",
):

    # Get the predicted probabilities
    predicted_proba = ml_model.predict_proba(data[0])[:, 1]
    predicted_classes = np.where(predicted_proba >= threshold, 1.0, 0.0)

    if score == "f1":
    # Get the f1 score now
        score_at_threshold = np.round(
            metrics.f1_score(data[1], predicted_classes), 4
        )
    elif score == "precision":
    # Get the precision score
        score_at_threshold = np.round(
            metrics.precision_score(data[1], predicted_classes), 4
        )

    elif score == "recall":
    # Get the recall score
        score_at_threshold = np.round(
            metrics.recall_score(data[1], predicted_classes), 4
        )

    return score_at_threshold

def find_best_threshold_score(ml_model: BaseEstimator,
                                 val_data: tuple[np.ndarray, np.ndarray],
                                 min_threshold: float = 0.3,
                                 max_threshold: float = 1.0,
                                 step_size: float = 0.025,
                                 score: str="f1",
                                 ):

    thresholds_to_test = np.arange(min_threshold, max_threshold, step_size)

    model_performance_various_thresholds = {
        threshold: evaluate_ml_model_score(ml_model, val_data, threshold=threshold, score=score) for
        threshold in thresholds_to_test
    }

    best_threshold_performance_pair = (min_threshold, 0.0)

    for threshold, performance in model_performance_various_thresholds.items():
        if performance > best_threshold_performance_pair[1]:
            best_threshold_performance_pair = (threshold, performance)

    return best_threshold_performance_pair[0]


def evaluate_classifier(ml_model: BaseEstimator,
                        train_data: tuple[np.ndarray, np.ndarray],
                        val_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
                        test_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
                        save_path: str = None,
                        save_name: str = None,
                        verbose: bool = True) -> dict[str, float]:
    """
    Evaluates the trained machine learning model and gets the performance metrics
    :param ml_model: scikit-learn model
    :param train_data: tuple, with 0 being the x_data and 1 the labels
    :param val_data: tuple, with 0 being the x_data and 1 the labels
    :param test_data: tuple, with 0 being the x_data and 1 the labels
    :param save_path: str, path where to save the results
    :param verbose: bool, flag for verbose output
    :param save_name: str, name of the json file
    :return: dictionary with the performance metrics
    """

    def round_result(value: float) -> float:
        return np.round(value, 4)

    idx_per_per_subcategory = get_idx_per_subcategory(test_data[1], test_data[2])

    def get_pr_curve(y_true:np.array, y_score: np.array) -> float:
        pr_auc = metrics.average_precision_score(y_true, y_score)
        return pr_auc

    results = {
        'proportion class 1': get_data_balance(train_data[1], val_data[1], test_data[1]) if val_data is not None else None,
        'train_balanced_accuracy': round_result(metrics.balanced_accuracy_score(train_data[1], ml_model.predict(train_data[0]))),
        'val_balanced_accuracy': round_result(metrics.balanced_accuracy_score(val_data[1], ml_model.predict(val_data[0]))) if val_data is not None else None,
        'test_balanced_accuracy': round_result(metrics.balanced_accuracy_score(test_data[1], ml_model.predict(test_data[0]))),
    }

    # Find best threshold for F1 score:
    best_val_threshold_f1_score = find_best_threshold_score(
        ml_model,
        val_data,
        min_threshold= 0.1,
        max_threshold = 1.0,
        step_size= 0.01,
        score="f1",
    )

    best_val_threshold_precision_score = find_best_threshold_score(
        ml_model,
        val_data,
        min_threshold= 0.1,
        max_threshold = 1.0,
        step_size= 0.01,
        score="precision",
    )

    best_val_threshold_recall_score = find_best_threshold_score(
        ml_model,
        val_data,
        min_threshold= 0.1,
        max_threshold = 1.0,
        step_size= 0.01,
        score="recall",
    )

    results["f1_score_threshold"] = best_val_threshold_f1_score
    results["val_f1_score"] = evaluate_ml_model_score(
        ml_model, val_data, threshold=best_val_threshold_f1_score, score="f1",
    )
    results["test_f1_score"] = evaluate_ml_model_score(
        ml_model, test_data, threshold=best_val_threshold_f1_score, score="f1",
    )

    results["precision_score_threshold"] = best_val_threshold_precision_score
    results["val_precision_score"] = evaluate_ml_model_score(
        ml_model, val_data, threshold=best_val_threshold_precision_score, score="precision",
    )
    results["test_precision_score"] = evaluate_ml_model_score(
        ml_model, test_data, threshold=best_val_threshold_precision_score, score="precision",
    )

    results["recall_score_threshold"] = best_val_threshold_recall_score
    results["val_recall_score"] = evaluate_ml_model_score(
        ml_model, val_data, threshold=best_val_threshold_recall_score, score="recall",
    )
    results["test_recall_score"] = evaluate_ml_model_score(
        ml_model, test_data, threshold=best_val_threshold_recall_score, score="recall",
    )

    # Binary classification
    if len(train_data[1].unique()) == 2:
        # Add here the PR-recall curve! instead of F1
        results['train_pr_auc'] = round_result(get_pr_curve(train_data[1], ml_model.predict_proba(train_data[0])[:, 1]))
        results['val_pr_auc'] = round_result(get_pr_curve(val_data[1], ml_model.predict_proba(val_data[0])[:, 1])) if val_data is not None else None
        results['test_pr_auc'] = round_result(get_pr_curve(test_data[1], ml_model.predict_proba(test_data[0])[:, 1]))

        for key, idx in idx_per_per_subcategory.items():
            results["test_pr_auc" + f"_{key}"] = round_result(
                get_pr_curve(test_data[1][idx], ml_model.predict_proba(test_data[0][idx])[:, 1])
            )

        for key, idx in idx_per_per_subcategory.items():
            results["test_balanced_auc" + f"_{key}"] = round_result(
                metrics.balanced_accuracy_score(test_data[1][idx], ml_model.predict(test_data[0][idx]))
            )

        # ROC AUC
        results['train_roc_auc'] = round_result(metrics.roc_auc_score(train_data[1], ml_model.predict_proba(train_data[0])[:, 1]))
        results['val_roc_auc'] = round_result(metrics.roc_auc_score(val_data[1], ml_model.predict_proba(val_data[0])[:, 1])) if val_data is not None else None
        results['test_roc_auc'] = round_result(metrics.roc_auc_score(test_data[1], ml_model.predict_proba(test_data[0])[:, 1]))

        for key, idx in idx_per_per_subcategory.items():
            results["test_roc_auc" + f"_{key}"] = round_result(
                metrics.roc_auc_score(test_data[1][idx], ml_model.predict_proba(test_data[0][idx])[:, 1])
            )

    else:
        raise NotImplementedError("We have not yet implemented multiclass classification")

    if verbose:
        print(results)

    # Save results to a JSON file
    if save_name is None:
        save_name = "performance_metrics.json"

    with open(os.path.join(save_path, save_name), 'w') as f:
        json.dump(results, f, indent=4)  # Save results in JSON format

    return results

def get_resampled_data(X_test: Union[np.ndarray, pd.DataFrame],
                       y_test: Union[np.ndarray, pd.Series],
                       seed: Optional[int] = 42) -> tuple[Union[np.ndarray, pd.DataFrame],
                                                                    Union[np.ndarray, pd.Series]]:
    n_samples = len(X_test)
    # Resample the dataset with replacement
    # Create a controlled random state for reproducibility
    rng = np.random.RandomState(seed)

    # Resample the dataset with replacement using the controlled random state
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    # indices = np.random.choice(n_samples, size=n_samples, replace=True)

    # Handle both pandas and numpy inputs
    if isinstance(X_test, pd.DataFrame):
        X_bootstrapped = X_test.iloc[indices]
    else:
        X_bootstrapped = X_test[indices]

    if isinstance(y_test, pd.Series):
        y_bootstrapped = y_test.iloc[indices]
    else:
        y_bootstrapped = y_test[indices]

    return X_bootstrapped, y_bootstrapped


def get_bootstrapped_cohens_kappa(
        ml_model_1: BaseEstimator,
        threshold_1: float,
        ml_model_2: BaseEstimator,
        threshold_2: float,
        test_data: tuple[np.ndarray, np.ndarray],
        bootstrap_samples: int,
        bootstrap_method: str,
):

    X_test, y_test, label_test = test_data

    # Initialize results dictionary
    results = {
        'cohen kappa': [],
    }

    for idx in range(bootstrap_samples):
        X_bootstrap, y_bootstrap = get_resampled_data(X_test, y_test, seed=idx)
        # Get predictions
        y_proba_predictions_model_1 = ml_model_1.predict_proba(X_bootstrap)[:, 1]
        y_pred_1 = np.where(y_proba_predictions_model_1 >= threshold_1, 1.0, 0.0)
        y_proba_predictions_model_2 = ml_model_2.predict_proba(X_bootstrap)[:, 1]
        y_pred_2 = np.where(y_proba_predictions_model_2 >= threshold_2, 1.0, 0.0)

        results["cohen kappa"].append(
            cohen_kappa_score(y_pred_1, y_pred_2)
        )

    # Calculate confidence intervals and means
    final_results = get_confidence_interval_mean(results, bootstrap_method)

    return final_results


def get_performance_metric_bootstrapped(model, X_bootstrap, y_bootstrap, f1_threshold):
    # Get predictions
    y_pred_proba = model.predict_proba(X_bootstrap)[:, 1]
    y_pred = model.predict(X_bootstrap)

    # Calculate metrics
    roc_auc = metrics.roc_auc_score(y_bootstrap, y_pred_proba)
    pr_auc = metrics.average_precision_score(y_bootstrap, y_pred_proba)
    balanced_accuracy = metrics.balanced_accuracy_score(y_bootstrap, y_pred)
    f1_score = evaluate_ml_model_score(
        model, (X_bootstrap, y_bootstrap), threshold=f1_threshold, score="f1",
)
    accuracy = metrics.accuracy_score(y_bootstrap, y_pred)

    precision_score = metrics.precision_score(y_bootstrap, y_pred)
    recall_score = metrics.recall_score(y_bootstrap, y_pred)

    return roc_auc, pr_auc, balanced_accuracy, f1_score, accuracy


def get_confidence_interval_mean(results: dict, bootstrap_method: str) -> dict:
    # Calculate confidence intervals and means
    final_results = {}
    for metric in results.keys():
        values = np.array(results[metric])
        mean_val = np.mean(values)
        if bootstrap_method == "quantile":
            ci_lower = np.percentile(values, 2.5)  # 2.5th percentile for lower bound
            ci_upper = np.percentile(values, 97.5)  # 97.5th percentile for upper bound
        else:
            raise NotImplementedError("We have not implemented 'se' and 'BCa'")

        final_results[metric] = {
            'mean': np.round(mean_val, 4),
            'ci_lower': np.round(ci_lower, 4),
            'ci_upper': np.round(ci_upper, 4)
        }

    return final_results


def bootstrap_test_performance(
        model: BaseEstimator,
        test_data: tuple[np.ndarray, np.ndarray],
        bootstrap_samples: int,
        bootstrap_method: str,
        f1_score_threshold: float,
        bootstrap_subcategories: bool = True,
        leave_one_out: bool = False,
        leave_out_stressor_name: str = "ssst",


) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]] | None]:
    """
    Performs bootstrap resampling to estimate model performance metrics and their confidence intervals.

    This function repeatedly samples the test data with replacement to create bootstrap samples,
    evaluates the model on each sample, and calculates performance metrics along with their
    95% confidence intervals.

    Args:
        model: Trained classifier model that implements predict_proba and predict methods
        test_data: tuple of (X_test, y_test) containing:
            - X_test: array-like of shape (n_samples, n_features)
            - y_test: array-like of shape (n_samples,) with true labels
        bootstrap_samples: int, number of bootstrap iterations (default: 1000)
        bootstrap_method: str, which method to use for bootstrap samples.
        f1_score_threshold: float: Threshold to determine the positive class
        bootstrap_subcategories: bool: If we bootstrap also each subcategory
        leave_one_out: bool: If we leave a stressor out
        leave_out_stressor_name: str, name of the left-out-stressor

    Returns:
        dict: Dictionary containing performance metrics and their confidence intervals:
            {
                'roc_auc': {'mean': float, 'ci_lower': float, 'ci_upper': float},
                'pr_auc': {'mean': float, 'ci_lower': float, 'ci_upper': float},
                'balanced_accuracy': {'mean': float, 'ci_lower': float, 'ci_upper': float}
                'f1_score': {"mean": float, "ci_lower" float, "ci_upper": float}
            }
    """
    X_test, y_test, label_test = test_data

    # Initialize results dictionary
    # These results include all stressors. So if I have the left-out mental stressor,
    # Then the evaluation also includes the one that was left-out, which potentially drives the results down
    # Actually, I should have a proper left out evaluation, where I only evaluate on the stressors that the model
    # has seen during training!
    results = {
        'roc_auc': [],
        'pr_auc': [],
        'balanced_accuracy': [],
        'f1_score': [],
        'accuracy': [],
    }

    if leave_one_out:
        # In distributin refers to the performance regarding only known stressors
        results_in_distribution = {
            'roc_auc': [],
            'pr_auc': [],
            'balanced_accuracy': [],
            'f1_score': [],
            'accuracy': [],
        }

        # Boolean mask: labels that DO NOT start with "ssst" (case-insensitive)
        mask = ~label_test.str.lower().str.startswith(leave_out_stressor_name)

        # Apply the mask to each component
        X_test_left = X_test[mask]
        y_test_left = y_test[mask]
        label_test_left = label_test[mask]

    if bootstrap_subcategories:
        idx_per_subcategory = get_idx_per_subcategory(y_test, label_test, positive_class=True, include_other_class=True)

        subcategory_results = {
            category: {
                'roc_auc': [],
                'pr_auc': [],
                'balanced_accuracy': [],
                'f1_score': [],
                'accuracy': [],
            }
            for category in idx_per_subcategory.keys()
        }

    for idx in range(bootstrap_samples):
        X_bootstrap, y_bootstrap = get_resampled_data(X_test, y_test, seed=idx)

        # Get predictions
        roc_auc, pr_auc, balanced_accuracy_score, f1_score, accuracy = get_performance_metric_bootstrapped(
            model, X_bootstrap, y_bootstrap, f1_score_threshold)

        results['roc_auc'].append(roc_auc)
        results['pr_auc'].append(pr_auc)
        results['balanced_accuracy'].append(balanced_accuracy_score)
        results['f1_score'].append(f1_score)
        results['accuracy'].append(accuracy)

        if leave_one_out:
            X_bootstrap_left, y_bootstrap_left = get_resampled_data(X_test_left, y_test_left, seed=idx)

            # Get predictions
            roc_auc, pr_auc, balanced_accuracy_score, f1_score, accuracy = get_performance_metric_bootstrapped(
                model, X_bootstrap_left, y_bootstrap_left, f1_score_threshold)

            results_in_distribution['roc_auc'].append(roc_auc)
            results_in_distribution['pr_auc'].append(pr_auc)
            results_in_distribution['balanced_accuracy'].append(balanced_accuracy_score)
            results_in_distribution['f1_score'].append(f1_score)
            results_in_distribution['accuracy'].append(accuracy)

        if bootstrap_subcategories:
            for category, idx_category in idx_per_subcategory.items():
                # Get the subcategory data
                X_subcategory = X_test.iloc[idx_category] if isinstance(X_test, pd.DataFrame) else X_test[idx_category]
                y_subcategory = y_test.iloc[idx_category] if isinstance(y_test, pd.Series) else y_test[idx_category]

                # Perform bootstrap resampling on the subcategory data
                X_bootstrap, y_bootstrap = get_resampled_data(X_subcategory, y_subcategory, seed=idx)

                # Get predictions
                roc_auc, pr_auc, balanced_accuracy_score, f1_score, accuracy = get_performance_metric_bootstrapped(
                    model, X_bootstrap, y_bootstrap, f1_score_threshold)

                subcategory_results[category]['roc_auc'].append(roc_auc)
                subcategory_results[category]['pr_auc'].append(pr_auc)
                subcategory_results[category]['balanced_accuracy'].append(balanced_accuracy_score)
                subcategory_results[category]['f1_score'].append(f1_score)
                subcategory_results[category]['accuracy'].append(accuracy)

    # Calculate confidence intervals and means
    final_results = get_confidence_interval_mean(results, bootstrap_method)
    if leave_one_out:
        final_results_in_distribution = get_confidence_interval_mean(results_in_distribution, bootstrap_method)

    if bootstrap_subcategories:
        final_results_subcategories = {}

        for category, results_category in subcategory_results.items():
            final_results_subcategories[category] = get_confidence_interval_mean(results_category, bootstrap_method)

    return (final_results, final_results_subcategories if bootstrap_subcategories else None,
            final_results_in_distribution if leave_one_out else None)


def load_yaml_config_file(path_to_yaml_file: str):
    """
    Loads a yaml file
    :param path_to_yaml_file:
    :return: the resulting dictionary
    """
    try:
        with open(path_to_yaml_file) as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print("We could not find the yaml file that you specified")


class FeatureSelectionPipeline:
    """
    Simplified pipeline for feature selection using cross-validation.
    """
    # Set the class attributes
    # Map model names to their corresponding classes
    model_classes = {
        "dt": DecisionTreeClassifier,
        "rf": RandomForestClassifier,
        "adaboost": AdaBoostClassifier,
        "lda": LinearDiscriminantAnalysis,
        "knn": KNeighborsClassifier,
        "lr": LogisticRegression,
        # "xgboost": GradientBoostingClassifier,
        "xgboost": xgb.XGBClassifier,
        "qda": QuadraticDiscriminantAnalysis,
        "svm": SVC,
        "random_baseline": DummyClassifier
    }

    def __init__(self,
                 base_estimator: BaseEstimator,
                 feature_names: str,
                 n_features_range: list[int],
                 n_splits: int = 5,
                 n_trials: int = 15,
                 scoring: str = 'roc_auc',
                 random_state: int = 42):
        """
        Initialize the pipeline.

        Args:
            base_estimator: Base model for feature selection and final model
            n_features_range: List of number of features to try
            n_splits: Number of cross-validation splits
            n_trials: Number of trials for the bayesian hyperparameter tuning
            scoring: Metric to optimize ('roc_auc' or 'balanced_accuracy')
            random_state: Random seed
        """
        self.base_estimator = base_estimator
        self.feature_names = feature_names
        self.n_features_range = n_features_range
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.scoring = scoring
        self.random_state = random_state

        self.best_features_mask = None
        self.feature_importance = None
        self.cv_results = None

    def _objective(
            self,
            trial: Trial,
            train_data: tuple,
            val_data: tuple,
            model_type: str,
            metric: str = "roc_auc",
        ) -> float:
        """
        Objective function for Optuna optimization.
        Returns validation balanced accuracy as the optimization metric.
        """

        # Define hyperparameter search space based on model type
        if isinstance(self.base_estimator, LogisticRegression):

            params = {
                'C': trial.suggest_float('C', 0.01, 1, log=True),
                'penalty': "l2",
                'max_iter': 2000,
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'n_jobs': -1,
            }

        elif isinstance(self.base_estimator, RandomForestClassifier):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 25),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 25),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'n_jobs': -1,
            }
        elif isinstance(self.base_estimator, xgb.XGBClassifier):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 25, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1.0, log=True),
                'objective': 'binary:logistic',

                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),

                'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),

                'use_label_encoder': False,
                'n_jobs': -1
            }
        elif isinstance(self.base_estimator, DecisionTreeClassifier):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
            }
        elif isinstance(self.base_estimator, AdaBoostClassifier):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
            }
        elif isinstance(self.base_estimator, KNeighborsClassifier):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2),  # 1 for manhattan_distance, 2 for euclidean_distance
                'leaf_size': trial.suggest_int('leaf_size', 20, 50),
                "n_jobs": -1
            }
        elif isinstance(self.base_estimator, LinearDiscriminantAnalysis):
            params = {
                'solver': trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen']),
                'shrinkage': trial.suggest_float('shrinkage', 0.0, 1.0) if trial.suggest_categorical('use_shrinkage',
                                                                                                     [True,
                                                                                                      False]) else None,
                'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
            }
        elif isinstance(self.base_estimator, QuadraticDiscriminantAnalysis):
            params = {
                'reg_param': trial.suggest_float('reg_param', 0.0, 1.0),
                'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
            }

        elif isinstance(self.base_estimator, SVC):
            params = {
                "C": trial.suggest_float("C", 0.0, 5.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }

        elif isinstance(self.base_estimator, DummyClassifier):
            params = {
                "strategy": "prior"
            }

        else:
            raise ValueError(f"Hyperparameter optimization not implemented for model type: {model_type}")

        # Create and train model
        model = get_ml_model(model_type, params)
        model.fit(train_data[0], train_data[1])

        # Evaluate on validation set
        if metric == "accuracy":
            val_pred = model.predict(val_data[0])
            val_score = metrics.balanced_accuracy_score(val_data[1], val_pred)
        elif metric == "roc_auc":
            val_score = metrics.roc_auc_score(val_data[1], model.predict_proba(val_data[0])[:, 1])

        return val_score

    def find_best_hyperparameter_base_estimator(self,
                                              train_data: tuple,
                                              val_data: tuple,
                                              n_trials: int = 15,
                                              save_path: Optional[str] = None) -> dict:
        """
        Find best hyperparameters for base estimator using Optuna optimization.
        Results are cached to avoid redundant optimization.

        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            n_trials: Number of optimization trials
            save_path: Path to cache hyperparameters. If None, no caching is used.

        Returns:
            dict: Best hyperparameters
        """
        # Create Optuna study
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        # Get model type string from base_estimator class
        model_type = None
        for key, cls in self.model_classes.items():
            if isinstance(self.base_estimator, cls):
                model_type = key
                break

        if model_type is None:
            raise ValueError("Unknown base estimator type")

        # Define objective function wrapper
        def objective(trial):
            return self._objective(trial, train_data, val_data, model_type, self.scoring)

        # Run optimization
        study.optimize(objective, n_trials=n_trials)

        # Get best parameters
        best_params = study.best_params

        return best_params

    def fit(self,
            train_data: tuple[np.ndarray, np.ndarray],
            val_data: tuple[np.ndarray, np.ndarray],
            feature_names: list[str] = None,
            save_path: Optional[str] = None) -> None:
        """
        Fit the feature selection pipeline using provided train/val sets.

        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            feature_names: List of feature names
            save_path: Path to cache hyperparameters. If None, no saving is done.
        """
        X_train, y_train, _ = train_data
        X_val, y_val, _ = val_data

        # First find best hyperparameters
        best_params = self.find_best_hyperparameter_base_estimator(
            train_data,
            val_data,
            n_trials=self.n_trials,
            save_path=save_path
        )

        # Create optimized base estimator
        optimized_estimator = clone(self.base_estimator).set_params(**best_params)

        # Try each number of features
        scores = []
        feature_importance_scores = np.zeros(X_train.shape[1])
        selected_features_count = np.zeros(X_train.shape[1])

        history_feature_selection = {
            str(name): 0 for name in feature_names
        }

        counter = 0
        for n_features in self.n_features_range:
            counter += 1
            print(f"Evaluating {n_features} features")

            # Feature selection using optimized estimator
            rfe = RFE(
                estimator=clone(optimized_estimator),
                n_features_to_select=n_features
            )

            # Create and fit pipeline
            pipeline = Pipeline([
                ('rfe', rfe),
                ('model', clone(optimized_estimator))
            ])
            print(f"Fitting estimator")
            pipeline.fit(X_train, y_train)

            # Score on validation set
            if self.scoring == 'roc_auc':
                score = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
            else:
                score = balanced_accuracy_score(y_val, pipeline.predict(X_val))
            print(f"The score is {score}")
            scores.append(score)

            # Track feature importance
            feature_importance_scores += rfe.ranking_
            selected_features_count += rfe.support_

            for idx, (key, value) in enumerate(history_feature_selection.items()):
                history_feature_selection[key] += bool(selected_features_count[idx])

        for key, value in history_feature_selection.items():
            history_feature_selection[key] = np.round((value / counter) * 100, 4)

        # We sort the values from most often chosen to least often
        history_feature_selection = sorted(history_feature_selection.items(), key=lambda x: x[1], reverse=True)

        # Find best number of features
        best_n_features = self.n_features_range[np.argmax(scores)]

        # Final feature selection with best number of features
        final_rfe = RFE(
            estimator=clone(optimized_estimator),
            n_features_to_select=best_n_features
        )
        final_rfe.fit(X_train, y_train)

        # Store results
        self.best_features_mask = final_rfe.support_
        self.cv_results = {
            'scores': [float(score) for score in scores],  # Convert numpy floats to Python floats
            'best_score': float(np.max(scores)),  # Convert numpy float to Python float
            'best_n_features': int(best_n_features),  # Convert numpy int to Python int
            'best_params': best_params,
            'feature_selection_mask': [(feature_name, bool(mask)) for feature_name, mask in
                                       zip(self.feature_names, self.best_features_mask)],
        }
        self.feature_importance = {
            str(name): {  # Ensure keys are strings
                'importance_score': float(score),  # Convert numpy float to Python float
                'selected': bool(count)  # Convert numpy bool to Python bool
            }
            for name, score, count in zip(
                feature_names,
                feature_importance_scores,
                selected_features_count
            )
        }

        with open(os.path.join(save_path, "feature_selection_results.json"), 'w') as f:
            json.dump(self.cv_results, f, indent=4)  # Save results in JSON format

        with open(os.path.join(save_path, "feature_importance_report.json"), "w") as f:
            json.dump(self.feature_importance, f, indent=4)

        with open(os.path.join(save_path, "feature_importance_total_selected.json"), "w") as f:
            json.dump(history_feature_selection, f, indent=4)


def plot_calibration_curve(y_test: np.array, predictions: np.array,
                           n_bins: int,
                           bin_strategy: str,
                           resampling_method: str,
                           save_path: str):
    """
    Code adapted from: https://endtoenddatascience.com/chapter11-machine-learning-calibration
    Produces a calibration plot and saves it locally. The raw data behind the plot is also written locally.
    :param y_test: y_test series
    :param predictions: predictions series
    :param n_bins: number of bins for the predictions
    :param bin_strategy: uniform - all bins have the same width; quantile - bins have the same number of observations
    :param resampling_method: str - which resampling method is used to handle imbalanced data
    :save_path: save_path for the figure
    """
    try:
        prob_true, prob_pred = calibration_curve(y_test, predictions, n_bins=n_bins, strategy=bin_strategy)

        ece = binary_calibration_error(
            torch.from_numpy(predictions), torch.from_numpy(np.asarray(y_test)), n_bins=n_bins, norm="l1"
        )

        brier_score = np.mean((y_test-predictions)**2)
        calibration_df = pd.DataFrame({'prob_true': prob_true, 'prob_pred': prob_pred})
        calibration_df["ece"] = ece.item()
        calibration_df["brier score"] = np.round(brier_score, 4)

        fig, ax = plt.subplots()
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f'model ECE: {np.round(ece, 4)}')
        line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle="--", label="perfect calibration")
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability in Each Bin')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'{resampling_method}_{bin_strategy}_{n_bins}_calibration_plot.png'), dpi=400, format="png")
        plt.clf()

        print(f"The ECE is {ece}. The brier score is {brier_score}")
        calibration_df.to_csv(os.path.join(save_path,
                                           f'{bin_strategy}_{n_bins}_calibration_summary_{resampling_method}.csv'), index=False)
    except Exception as e:
        print(e)


def min_max_scaling(values):
    max_value = max(values)
    min_value = min(values)

    return [
        (value - min_value)/(max_value - min_value) for value in values
    ]


def get_feature_importance_model(model, feature_names, normalize_values=False):

    if isinstance(model, LogisticRegression):
        feature_coefficients = model.coef_[0]

        #Normalize it
        if normalize_values:
            feature_coefficients = min_max_scaling(feature_coefficients)

        feature_importance_coeff = {
            name: coeff for name, coeff in zip(
                feature_names, feature_coefficients
            )
        }
    # Now we will sort the values according to absolute value from high to low
        sorted_feature_importance_coeff = sorted(
            feature_importance_coeff.items(), key=lambda x: abs(x[1]), reverse=True
        )

        return sorted_feature_importance_coeff

    else:
        return None


def plot_feature_importance(feature_coeffs, num_features=20, figsize=(10, 7), save_path=None):
    """
    Plots the most important features based on absolute coefficient values.

    Args:
        feature_coeffs (list of tuples): List where each tuple contains (feature_name, coefficient).
        num_features (int): Number of top features to display (default: 20).
        figsize (tuple): Figure size.
        save_path (str, optional): Path to save the figure.
    """
    feature_coeffs = np.array(feature_coeffs, dtype=object)

    feature_names = feature_coeffs[:, 0]
    coefficients = feature_coeffs[:, 1].astype(float)

    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    feature_names = feature_names[sorted_indices][:num_features]
    coefficients = coefficients[sorted_indices][:num_features]

    fig, ax = plt.subplots(figsize=figsize)

    # Old color scheme
    # colors = ['#B2182B' if coef < 0 else '#2166AC' for coef in coefficients]
    colors = ['#D73027' if coef < 0 else '#56B4E9' for coef in coefficients]

    y_positions = np.arange(len(feature_names))

    ax.barh(y_positions, coefficients, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_names, fontsize=14)
    ax.set_xlabel("Feature Coefficient", fontsize=16, fontweight='bold')
    ax.set_ylabel("Feature Name", fontsize=16, fontweight='bold')
    ax.set_title(f"Top {num_features} Most Important Features", fontsize=18, fontweight='bold', pad=15)

    ax.axvline(0, color='black', linestyle="--", linewidth=1.5)

    max_coef = max(abs(coefficients))
    plt.xlim(-1.5 * max_coef, 1.5 * max_coef)  # Expanding limits by 20% ensures no overlap

    for y, coef in zip(y_positions, coefficients):
        offset = 0.02 * max(abs(coefficients))
        ha = 'left' if coef > 0 else 'right'
        ax.text(coef + (offset * np.sign(coef)), y, f'{coef:.3f}', ha=ha, va='center', fontsize=13, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.xaxis.set_tick_params(width=1.2, length=6)
    ax.yaxis.set_tick_params(width=1.2, length=6)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Invert y-axis to put the most important feature on the top
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=False)

    plt.close()



def plot_feature_subset_comparison(results: dict, metric: str, figures_path_root: str, comparison: str) -> None:
    """
    Plot model performance across feature subsets with confidence intervals.

    Args:
        results: Nested dict {model_type: {feature_set: results_dict}}
        metric: Performance metric to plot ('roc_auc', 'pr_auc', etc.)
        figures_path_root: Path to save the figure.
        comparison: What comparison is plotted.
    """
    plt.figure(figsize=(10, 6))  # Adjust figure size

    # Set figure style for publication
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'figure.dpi': 300,
    })

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Get unique models and feature sets
    model_types = sorted(results.keys())
    feature_sets = sorted({fs for model in results.values() for fs in model.keys()})

    # Reduce spacing between models
    x = np.arange(len(model_types)) * 0.8  # Reduce model spacing
    width = 0.08  # Smaller width for tighter grouping

    handles = []
    for idx, feature_set in enumerate(feature_sets):
        means, ci_lower, ci_upper = [], [], []

        for model_type in model_types:
            result = results[model_type].get(feature_set)
            if result and metric in result:
                means.append(result[metric]['mean'])
                ci_lower.append(result[metric]['ci_lower'])
                ci_upper.append(result[metric]['ci_upper'])
            else:
                means.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)

        means = np.array(means)
        ci_lower = np.array(ci_lower)
        ci_upper = np.array(ci_upper)

        # Compute x positions
        x_pos = x + (idx - len(feature_sets)/2 + 0.5) * width

        # Color scheme: Distinguish full vs. subset
        # color = COLORS_DICT[feature_set]  # Define your colors for full vs. subset
        # # hatch = '' if 'full' in feature_set.lower() else '//'  # Hatch pattern for subsets

        valid_idx = ~np.isnan(means)
        if np.any(valid_idx):
            # handle = plt.errorbar(x_pos[valid_idx], means[valid_idx],
            #                       yerr=[means[valid_idx] - ci_lower[valid_idx],
            #                             ci_upper[valid_idx] - means[valid_idx]],
            #                       fmt='o', capsize=4, capthick=1.8, markersize=7,
            #                       color=color, label=feature_set, elinewidth=1.8)

            handle = plt.errorbar(x_pos[valid_idx], means[valid_idx],
                                  yerr=[means[valid_idx] - ci_lower[valid_idx],
                                        ci_upper[valid_idx] - means[valid_idx]],
                                  fmt='o', capsize=4, capthick=1.8, markersize=7,
                                  label=feature_set, elinewidth=1.8)

            # Prevent overlapping labels
            y_positions = []
            offset = 0.015 * (max(means[valid_idx]) - min(means[valid_idx]))

            for pos, mean in zip(x_pos[valid_idx], means[valid_idx]):
                new_y = mean
                while any(abs(new_y - y) < offset for y in y_positions):
                    new_y -= offset

                plt.text(pos, new_y, f'{mean:.3f}', ha='center', va='bottom',
                         color='black', fontsize=12, fontweight="bold")
                y_positions.append(new_y)

            handles.append(handle)

    # Customize plot
    plt.xlabel('Model Type')
    plt.ylabel(metric.replace('_', ' ').title())

    # Set x-ticks to model types
    plt.xticks(x, model_types)

    # Adjust legend positioning
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title="Number of Features")

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save figure
    plt.tight_layout()
    save_path = os.path.join(figures_path_root, f'{comparison}_feature_subset_comparison_{metric}.png')
    # plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.show()
    plt.close()


def balance_sublabels(data: pd.DataFrame, positive_class: str, upsample: bool = True) -> pd.DataFrame:
    """
    Balances different labels within the positive class to ensure equal representation of each of the labels.

    Args:
        data: DataFrame containing the data
        positive_class: The category name that represents the positive class
        upsample: If True, upsample minority labels; if False, downsample majority labels

    Returns:
        DataFrame with balanced sublabels within the positive class
    """
    # Get only the positive class data
    positive_data = data[data['category'] == positive_class].copy()
    # Get the negative class data (we'll keep this unchanged)
    negative_data = data[data['category'] != positive_class].copy()

    # Count occurrences of each label in the positive class
    label_counts = positive_data['label'].value_counts()

    # Determine target count based on upsampling or downsampling
    if upsample:
        target_count = label_counts.max()
    else:
        target_count = label_counts.min()

    # Resample each sublabel
    balanced_positive_data = pd.DataFrame()

    for label, count in label_counts.items():
        label_data = positive_data[positive_data['label'] == label]

        if count == target_count:
            # No resampling needed
            resampled_data = label_data
        elif count < target_count:
            # Upsample
            resampled_data = resample(
                label_data,
                replace=True,
                n_samples=target_count,
                random_state=42
            )
        else:
            # Downsample
            # We know that TA and Pasat each has a repeat condition, so actually we should divide their numbers by 2
            # so if they are summed up, they are in equal proportions or leave on of the two out.
            resampled_data = resample(
                label_data,
                replace=False,
                n_samples=int(target_count/2) if label.lower() != "raven" else target_count,
                random_state=42
            )

        balanced_positive_data = pd.concat([balanced_positive_data, resampled_data])

    # Combine balanced positive data with the negative data
    balanced_data = pd.concat([balanced_positive_data, negative_data])

    # Check if it worked:
    label_counts_check = balanced_positive_data['label'].value_counts()

    # Shuffle the final dataset
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_data


def visualize_umap_clusters(
        x_data: Union[np.ndarray, pd.DataFrame],
        y_data: Union[np.ndarray, pd.Series],
        labels: Union[np.ndarray, pd.Series, None] = None,
        n_neighbors: int = 30,
        min_dist: float = 0.01,
        n_components: int = 2,
        metric: str = 'euclidean',
        random_state: int = 42,
        save_path: str = None,
        title: str = "UMAP Projection of Data",
        save_name: str = "umap_visualization.png",
        figsize: tuple = (12, 10),
        show_plot: bool = True,
        use_subset: bool = True,
        subset_size: int = 500,
        label_points: bool = False,
        highlight_outliers: bool = True,
        label_color_map: Optional[dict] = None,
        colormap_categorical: str = 'tab10',
        colormap_continuous: str = 'viridis',
        include_baseline: bool = True,
        alpha_main: float = 0.7,
        alpha_baseline: float = 0.2,
        verbose: bool = True,
        point_size: int = 40,
        dpi: int = 300
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Creates and saves a UMAP visualization of data clusters, optimized for publication quality.

    Args:
        x_data: Feature data for UMAP projection
        y_data: Binary class labels (1 for mental stress, 0 for baseline/reference)
        labels: Detailed labels for subcategories (like 'Pasat', 'TA', etc.)
        n_neighbors: UMAP hyperparameter for local neighborhood size
        min_dist: UMAP hyperparameter for minimum distance between embedded points
        n_components: Number of dimensions for UMAP output (usually 2)
        metric: Distance metric for UMAP
        random_state: Random seed for reproducibility
        save_path: Directory to save the visualization
        title: Plot title
        save_name: Filename for the saved figure
        figsize: Figure dimensions (width, height) in inches
        show_plot: Whether to display the plot
        use_subset: If True, uses a subset of data for faster computation
        subset_size: Maximum number of samples to use if use_subset is True
        label_points: If True, adds text labels to representative points for each cluster
        highlight_outliers: If True, adds a light outline to points that may be outliers
        label_color_map: Dictionary mapping labels to specific colors
        colormap_categorical: Colormap name for categorical data
        colormap_continuous: Colormap name for continuous data
        include_baseline: Whether to include baseline (class 0) samples in visualization
        alpha_main: Opacity of main data points
        alpha_baseline: Opacity of baseline data points
        verbose: Whether to print progress messages
        point_size: Size of scatter plot points
        dpi: Resolution for saved figure

    Returns:
        Tuple of (figure, axes, embedding) where embedding is the UMAP projection
    """
    import umap
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import time
    import seaborn as sns
    from matplotlib.patches import Patch
    from sklearn.preprocessing import StandardScaler

    start_time = time.time()

    if verbose:
        print("Starting UMAP visualization process...")

    # Convert inputs to numpy arrays if they're pandas objects
    if isinstance(x_data, pd.DataFrame):
        x_data_np = x_data.values
    else:
        x_data_np = x_data

    if isinstance(y_data, pd.Series):
        y_data_np = y_data.values
    else:
        y_data_np = y_data

    # Handle labels
    if labels is not None:
        if isinstance(labels, pd.Series):
            labels_np = labels.values
        else:
            labels_np = labels
    else:
        # If no sublabels provided, use binary class labels
        labels_np = np.array(['Positive' if y == 1 else 'Baseline' for y in y_data_np])

    # Filter data based on include_baseline parameter
    if not include_baseline:
        mask = (y_data_np == 1)
        x_data_np = x_data_np[mask]
        labels_np = labels_np[mask]
        y_data_np = y_data_np[mask]
        if verbose:
            print(f"Filtered out baseline data, {len(x_data_np)} samples remaining")

    # Use a subset of data for faster computation if requested
    if use_subset and len(x_data_np) > subset_size:
        # Ensure we get a representative sample by stratifying on labels
        unique_labels = np.unique(labels_np)
        indices = []

        samples_per_label = subset_size // len(unique_labels)
        for label in unique_labels:
            label_indices = np.where(labels_np == label)[0]
            if len(label_indices) > samples_per_label:
                # Randomly select samples_per_label indices for this label
                selected = np.random.choice(label_indices, samples_per_label, replace=False)
            else:
                # Use all indices for this label
                selected = label_indices
            indices.extend(selected)

        # Convert to numpy array and shuffle
        indices = np.array(indices)
        np.random.shuffle(indices)

        # Subset the data
        x_data_subset = x_data_np[indices]
        y_data_subset = y_data_np[indices]
        labels_subset = labels_np[indices]

        if verbose:
            print(f"Using a subset of {len(x_data_subset)} samples for UMAP computation")
    else:
        x_data_subset = x_data_np
        y_data_subset = y_data_np
        labels_subset = labels_np

    # Standardize the data
    scaler = StandardScaler()
    x_data_scaled = scaler.fit_transform(x_data_subset)

    if verbose:
        print(f"Data prepared, starting UMAP fitting...")

    # Create and fit UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        verbose=verbose
    )

    umap_embedding = reducer.fit_transform(x_data_scaled)

    if verbose:
        print(f"UMAP fitting completed in {time.time() - start_time:.2f} seconds")

    # Setup default label colors if not provided
    if label_color_map is None:
        # Preset colors for common mental stress categories
        label_color_map = {
            'Pasat': '#E69F00',
            'Pasat_repeat': '#56B4E9',
            'Raven': '#009E73',
            'SSST_Sing_countdown': '#0072B2',
            'TA': '#D55E00',
            'TA_repeat': '#CC79A7',
            'Baseline': '#999999',  # Gray for baseline
            'Positive': '#33A02C',  # Green for generic positive class
            'Negative': '#999999'   # Gray for generic negative class
        }

    # Create a colorblind-friendly categorical colormap for labels not in the map
    unique_labels = np.unique(labels_subset)
    cmap = plt.cm.get_cmap(colormap_categorical, len(unique_labels))
    for i, label in enumerate(unique_labels):
        if label not in label_color_map:
            label_color_map[label] = cmap(i)

    # Create the plot with publication-quality settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'figure.titlesize': 18
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create a dictionary to store points for each label
    label_points_dict = {}

    # Plot baseline points first if they should be included but with lower alpha
    if include_baseline:
        baseline_mask = (y_data_subset == 0)
        if np.any(baseline_mask):
            ax.scatter(
                umap_embedding[baseline_mask, 0],
                umap_embedding[baseline_mask, 1],
                c=label_color_map.get('Baseline', '#999999'),
                s=point_size * 0.7,  # Slightly smaller points for baseline
                alpha=alpha_baseline,
                edgecolors='none',
                label='Baseline'
            )

    # Plot points for each unique label (for mental stress class)
    for label in unique_labels:
        # Skip baseline labels when plotting the mental stress categories
        if label == 'Baseline' or label == 'Negative':
            continue

        mask = (labels_subset == label)
        points = ax.scatter(
            umap_embedding[mask, 0],
            umap_embedding[mask, 1],
            c=label_color_map.get(label, cmap(list(unique_labels).index(label))),
            s=point_size,
            alpha=alpha_main,
            edgecolors='black' if highlight_outliers else 'none',
            linewidths=0.5 if highlight_outliers else 0,
            label=label.replace('_', ' ')
        )

        # Store points for this label for later labeling
        label_points_dict[label] = {
            'x': umap_embedding[mask, 0],
            'y': umap_embedding[mask, 1]
        }

    # Label representative points for each cluster if requested
    if label_points:
        for label, points in label_points_dict.items():
            # Find the point closest to the centroid of this cluster
            if len(points['x']) > 0:
                centroid_x = np.mean(points['x'])
                centroid_y = np.mean(points['y'])

                # Find index of point closest to centroid
                distances = np.sqrt((points['x'] - centroid_x)**2 + (points['y'] - centroid_y)**2)
                closest_idx = np.argmin(distances)

                # Place text label near this point
                ax.text(
                    points['x'][closest_idx],
                    points['y'][closest_idx],
                    label.replace('_', ' '),
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
                )

    # Customize the plot
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xlabel('UMAP Dimension 1', fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontweight='bold')

    # Add grid with light alpha
    ax.grid(True, linestyle='--', alpha=0.3)

    # Customize spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Create a clean, visually appealing legend
    handles, labels = ax.get_legend_handles_labels()

    # If we have many labels, place the legend outside the plot
    if len(handles) > 5:
        legend = ax.legend(
            handles, labels,
            title='Categories',
            title_fontsize=12,
            fontsize=10,
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            frameon=True,
            framealpha=0.9,
            edgecolor='lightgray'
        )
    else:
        legend = ax.legend(
            handles, labels,
            title='Categories',
            title_fontsize=12,
            fontsize=10,
            loc='best',
            frameon=True,
            framealpha=0.9,
            edgecolor='lightgray'
        )

    # Add a text box with UMAP parameters
    param_text = (
        f"UMAP Parameters:\n"
        f"n_neighbors: {n_neighbors}\n"
        f"min_dist: {min_dist}\n"
        f"metric: {metric}"
    )

    # Place the text box in the upper right corner
    text_box = ax.text(
        0.97, 0.97, param_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray')
    )

    # Add a footer with computation info
    footer_text = f"Computation time: {time.time() - start_time:.2f}s | N = {len(x_data_subset)} samples"
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=8, style='italic')

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        full_path = os.path.join(FIGURES_PATH, save_name)
        print(f"We save here {full_path}")
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"Visualization saved to {full_path}")

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # Return the figure, axes, and embedding for further use if needed
    return fig, ax, umap_embedding

def visualize_dimensionality_reduction(
        x_data: Union[np.ndarray, pd.DataFrame],
        y_data: Union[np.ndarray, pd.Series],
        labels: Union[np.ndarray, pd.Series, None] = None,
        method: str = 'umap',  # Options: 'umap', 'tsne', 'both'
        # UMAP parameters
        n_neighbors: int = 30,
        min_dist: float = 0.01,
        # t-SNE parameters
        perplexity: int = 40,
        learning_rate: int = 200,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        # Common parameters
        n_components: int = 2,
        metric: str = 'euclidean',
        random_state: int = 42,
        save_path: str = FIGURES_PATH,
        title: str = "Dimensionality Reduction",
        save_name: str = "visualization.png",
        figsize: tuple = (12, 10),
        show_plot: bool = True,
        use_subset: bool = True,
        subset_size: int = 500,
        label_points: bool = False,
        highlight_outliers: bool = True,
        label_color_map: Optional[dict] = None,
        colormap_categorical: str = 'tab10',
        colormap_continuous: str = 'viridis',
        include_baseline: bool = True,
        merge_repeats: bool = True,
        alpha_main: float = 0.7,
        alpha_baseline: float = 0.2,
        verbose: bool = True,
        point_size: int = 40,
        dpi: int = 300
) -> Union[Tuple[plt.Figure, plt.Axes, np.ndarray],
           Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, np.ndarray]]]:
    """
    Creates and saves visualizations using UMAP and/or t-SNE dimensionality reduction techniques.

    Args:
        x_data: Feature data for projection
        y_data: Binary class labels (1 for mental stress, 0 for baseline/reference)
        labels: Detailed labels for subcategories (like 'Pasat', 'TA', etc.)
        method: Dimensionality reduction method: 'umap', 'tsne', or 'both'

        # UMAP specific parameters
        n_neighbors: UMAP hyperparameter for local neighborhood size
        min_dist: UMAP hyperparameter for minimum distance between embedded points

        # t-SNE specific parameters
        perplexity: t-SNE hyperparameter controlling the balance of local/global structure
        learning_rate: t-SNE hyperparameter for step size in optimization
        n_iter: Number of iterations for t-SNE optimization
        early_exaggeration: t-SNE hyperparameter affecting cluster separation

        # Common parameters
        n_components: Number of dimensions for output (usually 2)
        metric: Distance metric for the algorithm
        random_state: Random seed for reproducibility
        save_path: Directory to save the visualization
        title: Plot title
        save_name: Filename for the saved figure
        figsize: Figure dimensions (width, height) in inches
        show_plot: Whether to display the plot
        use_subset: If True, uses a subset of data for faster computation
        subset_size: Maximum number of samples to use if use_subset is True
        label_points: If True, adds text labels to representative points for each cluster
        highlight_outliers: If True, adds a light outline to points that may be outliers
        label_color_map: Dictionary mapping labels to specific colors
        colormap_categorical: Colormap name for categorical data
        colormap_continuous: Colormap name for continuous data
        include_baseline: Whether to include baseline (class 0) samples in visualization
        alpha_main: Opacity of main data points
        alpha_baseline: Opacity of baseline data points
        verbose: Whether to print progress messages
        point_size: Size of scatter plot points
        dpi: Resolution for saved figure

    Returns:
        If method is 'umap' or 'tsne':
            Tuple of (figure, axes, embedding)
        If method is 'both':
            Tuple of (figure, {'umap': umap_axes, 'tsne': tsne_axes},
                     {'umap': umap_embedding, 'tsne': tsne_embedding})
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import time
    import seaborn as sns
    from matplotlib.patches import Patch
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    import umap
    import os

    method = method.lower()
    if method not in ['umap', 'tsne', 'both']:
        raise ValueError("Method must be one of 'umap', 'tsne', or 'both'")

    start_time = time.time()

    if verbose:
        print(f"Starting {method.upper()} visualization process...")

    # Convert inputs to numpy arrays if they're pandas objects
    if isinstance(x_data, pd.DataFrame):
        x_data_np = x_data.values
    else:
        x_data_np = x_data

    if isinstance(y_data, pd.Series):
        y_data_np = y_data.values
    else:
        y_data_np = y_data

    # Handle labels
    if labels is not None:
        if isinstance(labels, pd.Series):
            labels_np = labels.values
        else:
            labels_np = labels
    else:
        # If no sublabels provided, use binary class labels
        labels_np = np.array(['Positive' if y == 1 else 'Baseline' for y in y_data_np])

    # Filter data based on include_baseline parameter
    if merge_repeats:
        labels_np = np.array([label.split("_")[0] for label in labels_np])

    if not include_baseline:
        mask = (y_data_np == 1)
        x_data_np = x_data_np[mask]
        labels_np = labels_np[mask]
        y_data_np = y_data_np[mask]
        if verbose:
            print(f"Filtered out baseline data, {len(x_data_np)} samples remaining")

    # Use a subset of data for faster computation if requested
    if use_subset and len(x_data_np) > subset_size:
        # Ensure we get a representative sample by stratifying on labels
        unique_labels = np.unique(labels_np)
        indices = []

        samples_per_label = subset_size // len(unique_labels)
        for label in unique_labels:
            label_indices = np.where(labels_np == label)[0]
            if len(label_indices) > samples_per_label:
                # Randomly select samples_per_label indices for this label
                selected = np.random.choice(label_indices, samples_per_label, replace=False)
            else:
                # Use all indices for this label
                selected = label_indices
            indices.extend(selected)

        # Convert to numpy array and shuffle
        indices = np.array(indices)
        np.random.shuffle(indices)

        # Subset the data
        x_data_subset = x_data_np[indices]
        y_data_subset = y_data_np[indices]
        labels_subset = labels_np[indices]

        if verbose:
            print(f"Using a subset of {len(x_data_subset)} samples for computation")
    else:
        x_data_subset = x_data_np
        y_data_subset = y_data_np
        labels_subset = labels_np

    # Standardize the data
    standardize = False
    if standardize:
        scaler = StandardScaler()
        x_data_scaled = scaler.fit_transform(x_data_subset)
    else:
        x_data_scaled = x_data_subset

    # Setup default label colors if not provided
    if label_color_map is None:
        # Preset colors for common mental stress categories
        label_color_map = {
            'Pasat': '#E69F00',
            'Pasat_repeat': '#56B4E9',
            'Raven': '#009E73',
            'SSST_Sing_countdown': '#0072B2',
            'TA': '#D55E00',
            'TA_repeat': '#CC79A7',
            'Baseline': '#999999',  # Gray for baseline
            'Positive': '#33A02C',  # Green for generic positive class
            'Negative': '#999999'   # Gray for generic negative class
        }

    # Create a colorblind-friendly categorical colormap for labels not in the map
    unique_labels = np.unique(labels_subset)
    cmap = plt.cm.get_cmap(colormap_categorical, len(unique_labels))
    for i, label in enumerate(unique_labels):
        if label not in label_color_map:
            label_color_map[label] = cmap(i)

    # Create the plot with publication-quality settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'figure.titlesize': 18
    })

    # Create figure and axes based on the method
    if method == 'both':
        fig, axs = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]), dpi=dpi)
        ax_umap, ax_tsne = axs
        axes_dict = {'umap': ax_umap, 'tsne': ax_tsne}
        embeddings_dict = {}
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Compute and plot UMAP if requested
    if method in ['umap', 'both']:
        if verbose:
            print(f"Data prepared, starting UMAP fitting...")

        # Create and fit UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
            verbose=verbose
        )

        umap_embedding = reducer.fit_transform(x_data_scaled)

        if method == 'both':
            embeddings_dict['umap'] = umap_embedding
            current_ax = ax_umap
            current_ax.set_title(f"UMAP Projection", fontweight='bold', pad=20)
        else:
            current_ax = ax

        if verbose:
            print(f"UMAP fitting completed in {time.time() - start_time:.2f} seconds")

        # Plot the UMAP results
        create_scatter_plot(
            embedding=umap_embedding,
            ax=current_ax,
            y_data=y_data_subset,
            labels=labels_subset,
            label_color_map=label_color_map,
            include_baseline=include_baseline,
            alpha_main=alpha_main,
            alpha_baseline=alpha_baseline,
            point_size=point_size,
            highlight_outliers=highlight_outliers,
            label_points=label_points,
            unique_labels=unique_labels,
            cmap=cmap
        )

        # Add UMAP parameter info
        if method != 'both':
            param_text = (
                f"UMAP Parameters:\n"
                f"n_neighbors: {n_neighbors}\n"
                f"min_dist: {min_dist}\n"
                f"metric: {metric}"
            )
            add_parameter_textbox(current_ax, param_text)

    # Compute and plot t-SNE if requested
    if method in ['tsne', 'both']:
        if verbose:
            tsne_start_time = time.time()
            print(f"Data prepared, starting t-SNE fitting...")

        # Create and fit t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            early_exaggeration=early_exaggeration,
            metric=metric,
            random_state=random_state,
            verbose=1 if verbose else 0
        )

        tsne_embedding = tsne.fit_transform(x_data_scaled)

        if method == 'both':
            embeddings_dict['tsne'] = tsne_embedding
            current_ax = ax_tsne
            current_ax.set_title(f"t-SNE Projection", fontweight='bold', pad=20)
        else:
            current_ax = ax

        if verbose:
            print(f"t-SNE fitting completed in {time.time() - tsne_start_time:.2f} seconds")

        # Plot the t-SNE results
        create_scatter_plot(
            embedding=tsne_embedding,
            ax=current_ax,
            y_data=y_data_subset,
            labels=labels_subset,
            label_color_map=label_color_map,
            include_baseline=include_baseline,
            alpha_main=alpha_main,
            alpha_baseline=alpha_baseline,
            point_size=point_size,
            highlight_outliers=highlight_outliers,
            label_points=label_points,
            unique_labels=unique_labels,
            cmap=cmap
        )

        # Add t-SNE parameter info
        if method != 'both':
            param_text = (
                f"t-SNE Parameters:\n"
                f"perplexity: {perplexity}\n"
                f"learning_rate: {learning_rate}\n"
                f"n_iter: {n_iter}"
            )
            add_parameter_textbox(current_ax, param_text)

    # Set main title
    if method == 'both':
        plt.suptitle(title, fontweight='bold', y=0.98, fontsize=18)
    else:
        current_ax.set_title(title, fontweight='bold', pad=20)

    # Add a footer with computation info
    footer_text = f"Computation time: {time.time() - start_time:.2f}s | N = {len(x_data_subset)} samples"
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=8, style='italic')

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        method_prefix = method if method != 'both' else 'umap_tsne'
        full_path = os.path.join(save_path, f"{method_prefix}_{save_name}")
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"Visualization saved to {full_path}")

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # Return the appropriate results based on method
    if method == 'both':
        return fig, axes_dict, embeddings_dict
    elif method == 'umap':
        return fig, current_ax, umap_embedding
    else:  # method == 'tsne'
        return fig, current_ax, tsne_embedding

def create_scatter_plot(embedding, ax, y_data, labels, label_color_map,
                        include_baseline, alpha_main, alpha_baseline,
                        point_size, highlight_outliers, label_points,
                        unique_labels, cmap):
    """Helper function to create scatter plots for dimensionality reduction visualizations."""
    # Create a dictionary to store points for each label
    label_points_dict = {}

    # Plot baseline points first if they should be included but with lower alpha
    if include_baseline:
        baseline_mask = (y_data == 0)
        if np.any(baseline_mask):
            ax.scatter(
                embedding[baseline_mask, 0],
                embedding[baseline_mask, 1],
                c=label_color_map.get('Baseline', '#999999'),
                s=point_size * 0.7,  # Slightly smaller points for baseline
                alpha=alpha_baseline,
                edgecolors='none',
                label='Baseline'
            )

    # Plot points for each unique label (for mental stress class)
    for label in unique_labels:
        # Skip baseline labels when plotting the mental stress categories
        if label == 'Baseline' or label == 'Negative':
            continue

        mask = (labels == label)
        points = ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=label_color_map.get(label, cmap(list(unique_labels).index(label))),
            s=point_size,
            alpha=alpha_main,
            edgecolors='black' if highlight_outliers else 'none',
            linewidths=0.5 if highlight_outliers else 0,
            label=label.replace('_', ' ')
        )

        # Store points for this label for later labeling
        label_points_dict[label] = {
            'x': embedding[mask, 0],
            'y': embedding[mask, 1]
        }

    # Label representative points for each cluster if requested
    if label_points:
        for label, points in label_points_dict.items():
            # Find the point closest to the centroid of this cluster
            if len(points['x']) > 0:
                centroid_x = np.mean(points['x'])
                centroid_y = np.mean(points['y'])

                # Find index of point closest to centroid
                distances = np.sqrt((points['x'] - centroid_x)**2 + (points['y'] - centroid_y)**2)
                closest_idx = np.argmin(distances)

                # Place text label near this point
                ax.text(
                    points['x'][closest_idx],
                    points['y'][closest_idx],
                    label.replace('_', ' '),
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
                )

    # Customize the plot
    ax.set_xlabel('Dimension 1', fontweight='bold')
    ax.set_ylabel('Dimension 2', fontweight='bold')

    # Add grid with light alpha
    ax.grid(True, linestyle='--', alpha=0.3)

    # Customize spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Create a clean, visually appealing legend
    handles, labels = ax.get_legend_handles_labels()

    # If we have many labels, place the legend outside the plot
    if len(handles) > 5:
        legend = ax.legend(
            handles, labels,
            title='Categories',
            title_fontsize=12,
            fontsize=10,
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            frameon=True,
            framealpha=0.9,
            edgecolor='lightgray'
        )
    else:
        legend = ax.legend(
            handles, labels,
            title='Categories',
            title_fontsize=12,
            fontsize=10,
            loc='best',
            frameon=True,
            framealpha=0.9,
            edgecolor='lightgray'
        )

def add_parameter_textbox(ax, param_text):
    """Helper function to add parameter info textbox to the plot."""
    # Place the text box in the upper right corner
    ax.text(
        0.97, 0.97, param_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray')
    )

# For backward compatibility
def visualize_umap_clusters(
        x_data: Union[np.ndarray, pd.DataFrame],
        y_data: Union[np.ndarray, pd.Series],
        labels: Union[np.ndarray, pd.Series, None] = None,
        n_neighbors: int = 30,
        min_dist: float = 0.01,
        n_components: int = 2,
        metric: str = 'euclidean',
        random_state: int = 42,
        save_path: str = None,
        title: str = "UMAP Projection of Data",
        save_name: str = "umap_visualization.png",
        figsize: tuple = (12, 10),
        show_plot: bool = True,
        use_subset: bool = True,
        subset_size: int = 500,
        label_points: bool = False,
        highlight_outliers: bool = True,
        label_color_map: Optional[dict] = None,
        colormap_categorical: str = 'tab10',
        colormap_continuous: str = 'viridis',
        include_baseline: bool = True,
        alpha_main: float = 0.7,
        alpha_baseline: float = 0.2,
        verbose: bool = True,
        point_size: int = 40,
        dpi: int = 300
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Legacy function that calls visualize_dimensionality_reduction with UMAP method.
    Maintained for backward compatibility.
    """
    return visualize_dimensionality_reduction(
        x_data=x_data,
        y_data=y_data,
        labels=labels,
        method='umap',
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        save_path=save_path,
        title=title,
        save_name=save_name,
        figsize=figsize,
        show_plot=show_plot,
        use_subset=use_subset,
        subset_size=subset_size,
        label_points=label_points,
        highlight_outliers=highlight_outliers,
        label_color_map=label_color_map,
        colormap_categorical=colormap_categorical,
        colormap_continuous=colormap_continuous,
        include_baseline=include_baseline,
        alpha_main=alpha_main,
        alpha_baseline=alpha_baseline,
        verbose=verbose,
        point_size=point_size,
        dpi=dpi
    )

