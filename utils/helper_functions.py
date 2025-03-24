# Collections of helper functions that are reused across several scripts

import os
import random
import json
import yaml
from typing import Optional, Tuple, Union, Any

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
                                  heart_measure="hrv_mean"):
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
                                      heart_measure="hrv_mean"):
        # We calculate the average HR reactivity based on participant id
        total_data_participant = self._load_data(self.data_folders, add_participant_id=True)

        if reference.lower() == "sitting":
            column = "label"
        elif reference.lower() == "baseline":
            column = "category"
        else:
            raise ValueError("Please input either reference 'sitting' or 'baseline'")

        negative_class_baseline = total_data_participant[total_data_participant[column] == reference][
            [heart_measure, "participant_id"]]
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

        # Customize the plot appearance
        ax.set_xlabel('Task', fontsize=14, fontweight='bold')
        ax.set_ylabel('HRV Reactivity\n(Δ from baseline)', fontsize=14, fontweight='bold')

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
                                          lw=1.5, alpha=0.7, label='HR reactivity threshold'))
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

        # Add descriptive text about the calculation method
        if reference.lower() == "sitting":
            fig.text(0.5, 0.01, f"HRV reactivity calculated as HRV during experimental task minus HRV during {reference.lower()} baseline",
                     ha='center', fontsize=10, fontstyle='italic')
        else:
            fig.text(0.5, 0.01, f"HRV reactivity calculated as HRV during experimental task minus HRV during baseline (sitting + recovery sitting)",
                     ha='center', fontsize=10, fontstyle='italic')

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
        colors = {
            'black': '#000000',
            'orange': '#E69F00',
            'baseline': '#56B4E9',
            'low_physical_activity': '#009E73',
            'moderate_physical_activity': '#F0E442',
            'blue': '#0072B2',
            'mental_stress': '#D55E00',
            'high_physical_activity': '#CC79A7'
        }

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
            plt.xlabel(f"{column}")
            plt.ylabel('Probability Density') if use_density else plt.ylabel('Count')
            plt.legend()

            # Save or show the plot
            if save_path:
                if save_name is not None:
                    save_path = os.path.join(save_path, f"{save_name}_label.png")
                else:
                    save_path = os.path.join(save_path, f"histogram_{column}_label.png")

                plt.savefig(save_path, dpi=500, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                plt.close()

        # Plot histograms for each category
        for category, color in colors.items():
            category_data = self.total_data[self.total_data['category'] == category][column]
            if not category_data.empty:
                sns.kdeplot(category_data, color=color, label=category.replace('_', ' ').title(), fill=True, alpha=0.6)

        # Customize the plot
        plt.xlabel(column.replace('_', ' ').title())
        plt.ylabel('Density' if use_density else 'Frequency')
        plt.legend()

        # Save or show the plot
        if save_path:
            if save_name is not None:
                save_path = os.path.join(save_path, f"{save_name}.png")
            else:
                save_path = os.path.join(save_path, f"histogram_{column}.png")

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


#Todo: Extend to multiclass classification
def encode_data(data: pd.DataFrame, positive_class: str, negative_class: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # First drop data that is not either in the positive class or negative class
    if negative_class != "rest":
        data = data[(data['category'] == positive_class) | (data['category'] == negative_class)]  # Filter relevant classes
        # Then label the data 1 for positive and 0 for negative
        data.loc[:, 'category'] = data['category'].apply(lambda x: 1 if x == positive_class else 0)  # Encode classes

    else:
        # By rest we mean everything without high physical_activity
        data = data[(data["category"] != "high_physical_activity")]
        data.loc[:, 'category'] = data['category'].apply(lambda x: 1 if x == positive_class else 0)  # Encode classes

    # Split data into x_data and y_data
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


def prepare_data(train_data: pd.DataFrame,
                 val_data: pd.DataFrame,
                 test_data: Optional[pd.DataFrame] = None,
                 imputation_method: Optional[str] = "knn",
                 positive_class: Optional[str] = "mental_stress",
                 negative_class: Optional[str] = "baseline",
                 resampling_method: Optional[str] = None,
                 scaler: Optional[str] = None,
                 use_quantile_transformer: Optional[bool] = False,
                 use_subset: Optional[list[bool]] = None) -> tuple:

    """
    Prepares the data for scikit-learn models. Can handle both 2-way (train/val) and 3-way (train/val/test) splits.

    Args:
        train_data: DataFrame containing the training data
        val_data: DataFrame containing the validation data
        test_data: Optional DataFrame containing the test data. If None, assumes 2-way split
        imputation_method: Optional. Str. Either 'drop', 'knn' or 'knn_subset'
        positive_class: str, which category to be encoded as 1
        negative_class: str, which category to be encoded as 0
        scaler: StandardScaler instance for normalization
        use_quantile_transformer: If set, we transform the features to normal distribution first
        resampling_method: str, resampling method to use. Options: None, "downsample", "upsample", "smote", "adasyn"
        use_subset: bool, list of bool to indicate which features should be included or not

    Returns:
        If test_data is provided:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names)
        If test_data is None:
            Tuple of ((X_train, y_train), (X_val, y_val), feature_names)
    """

    # Old code
    assert imputation_method in ["drop", "knn", "knn_subset", "iterative_imputer"], \
        "Please use as imputation method either 'knn', 'drop' or 'knn_subset'."

    # Sampen
    if imputation_method == "drop":
        if 'sampen' in train_data.columns:
        # Check if sampen is included in the data columns
            train_data.drop('sampen', axis=1, inplace=True)
        if 'sampen' in test_data.columns:
            test_data.drop('sampen', axis=1, inplace=True)
        if 'sampen' in val_data.columns:
            val_data.drop('sampen', axis=1, inplace=True)

        train_data = handle_missing_data(train_data)

        if val_data is not None:
            val_data = handle_missing_data(val_data)
        if test_data is not None:
            test_data = handle_missing_data(test_data)

    # If no resampling, just shuffle and encode the data
    train_data = train_data.sample(frac=1, replace=False, random_state=42).reset_index(drop=True)
    x_train, label_train, y_train = encode_data(train_data, positive_class, negative_class)

    # We have not yet cleaned up missing values, so we replace them with nan values first
    x_train = x_train.replace([np.inf, -np.inf], np.nan)

    # Could we handle the missing data here?
    if val_data is not None:
        x_val, label_val, y_val = encode_data(val_data, positive_class, negative_class)
        x_val = x_val.replace([np.inf, -np.inf], np.nan)

    if test_data is not None:
        x_test, label_test, y_test = encode_data(test_data, positive_class, negative_class)
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

    # Apply scaling after resampling if requested
    if scaler is not None:
        assert scaler.lower() in ["min_max", "standard_scaler"], \
            "please set a valid scaler. Options: 'min_max', 'standard_scaler'"

        # Let's do quantile transformation first before applying scaler
        if use_quantile_transformer:
            print("We use the quantile transformer")
            quantile_transformer_obj = QuantileTransformer(n_quantiles=1000, output_distribution="normal")
            x_train = quantile_transformer_obj.fit_transform(x_train)
        scaler_obj = StandardScaler() if scaler.lower() == "standard_scaler" else MinMaxScaler()
        x_train = scaler_obj.fit_transform(x_train)
        if val_data is not None:
            if use_quantile_transformer:
                x_val = quantile_transformer_obj.transform(x_val)
            x_val = scaler_obj.transform(x_val)
        if test_data is not None:
            if use_quantile_transformer:
                x_test = quantile_transformer_obj.transform(x_test)
            x_test = scaler_obj.transform(x_test)

    # Here we should use the KNN imputer then
    if imputation_method in ["knn", "iterative_imputer"]:
        imputer = KNNImputer(n_neighbors=5, copy=False) if imputation_method == "knn" else IterativeImputer(max_iter=10, random_state=0)

        imputer.fit(x_train)
        x_train = imputer.transform(x_train)
        x_val = imputer.transform(x_val)
        x_test = imputer.transform(x_test)

        # Check if any missing values are still present:
        assert np.isnan(x_train).sum() == np.isnan(x_val).sum() == np.isnan(x_val).sum() == 0, "Imputation did not work!"

    # Apply resampling only to training data
    if resampling_method in ["downsample", "upsample"]:
        do_downsampling = resampling_method == "downsample"
        train_data = resample_data(train_data, positive_class, negative_class, downsample=do_downsampling)
        x_train, y_train = encode_data(train_data, positive_class, negative_class)

    elif resampling_method == "smote":
        smote = SMOTE(random_state=42)
        x_train, y_train = smote.fit_resample(x_train, y_train)

    elif resampling_method == "adasyn":
        adasyn = ADASYN(random_state=42)
        x_train, y_train = adasyn.fit_resample(x_train, y_train)

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
        # "xgboost": GradientBoostingClassifier,
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


def get_idx_per_subcategory(y_data, label, positive_class=True, include_other_class=True):

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
        idx_values = list(label_df[label_df["label"]==category].index.values)
        # we need to sample len(x) (1-ratio) / ratio(1) to get the same ratio 1/0
        if include_other_class:
            sampled_negative_class = np.random.choice(
                other_class_idx, replace=False, size=int(((1-ratio_1) * len(idx_values))/ratio_1)
            )
            idx_values.extend(list(sampled_negative_class))
        idx_per_subcategory[category] = idx_values

    return idx_per_subcategory



def evaluate_classifier(ml_model: BaseEstimator,
                        train_data: tuple[np.ndarray, np.ndarray],
                        val_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
                        test_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
                        save_path: str = None,
                        save_name: str = None,
                        verbose: bool = False) -> dict[str, float]:
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


def get_performance_metric_bootstrapped(model, X_bootstrap, y_bootstrap):
    # Get predictions
    y_pred_proba = model.predict_proba(X_bootstrap)[:, 1]
    y_pred = model.predict(X_bootstrap)

    # Calculate metrics
    roc_auc = metrics.roc_auc_score(y_bootstrap, y_pred_proba)
    pr_auc = metrics.average_precision_score(y_bootstrap, y_pred_proba)
    balanced_accuracy = metrics.balanced_accuracy_score(y_bootstrap, y_pred)

    return roc_auc, pr_auc, balanced_accuracy


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


def bootstrap_test_performance(model: BaseEstimator,
                             test_data: tuple[np.ndarray, np.ndarray],
                             bootstrap_samples: int,
                             bootstrap_method: str,
                             bootstrap_subcategories: bool = True) -> tuple[dict[str, dict[str, float]], 
                                                                         dict[str, dict[str, dict[str, float]]] | None]:
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
    
    Returns:
        dict: Dictionary containing performance metrics and their confidence intervals:
            {
                'roc_auc': {'mean': float, 'ci_lower': float, 'ci_upper': float},
                'pr_auc': {'mean': float, 'ci_lower': float, 'ci_upper': float},
                'balanced_accuracy': {'mean': float, 'ci_lower': float, 'ci_upper': float}
            }
    """
    X_test, y_test, label_test = test_data

    # Initialize results dictionary
    results = {
        'roc_auc': [],
        'pr_auc': [],
        'balanced_accuracy': []
    }

    if bootstrap_subcategories:

        idx_per_subcategory = get_idx_per_subcategory(y_test, label_test, positive_class=True, include_other_class=True)

        subcategory_results = {
            category: {
                'roc_auc': [],
                'pr_auc': [],
                'balanced_accuracy': []
            }
            for category in idx_per_subcategory.keys()
        }

    for idx in range(bootstrap_samples):
        X_bootstrap, y_bootstrap = get_resampled_data(X_test, y_test, seed=idx)
        # Get predictions
        roc_auc, pr_auc, balanced_accuracy_score = get_performance_metric_bootstrapped(model, X_bootstrap, y_bootstrap)

        results['roc_auc'].append(roc_auc)
        results['pr_auc'].append(pr_auc)
        results['balanced_accuracy'].append(balanced_accuracy_score)

        if bootstrap_subcategories:
            for category, idx_category in idx_per_subcategory.items():
                # Get the subcategory data
                X_subcategory = X_test.iloc[idx_category] if isinstance(X_test, pd.DataFrame) else X_test[idx_category]
                y_subcategory = y_test.iloc[idx_category] if isinstance(y_test, pd.Series) else y_test[idx_category]
                
                # Perform bootstrap resampling on the subcategory data
                X_bootstrap, y_bootstrap = get_resampled_data(X_subcategory, y_subcategory, seed=idx)
                
                # Get predictions
                roc_auc, pr_auc, balanced_accuracy_score = get_performance_metric_bootstrapped(model, X_bootstrap,
                                                                                               y_bootstrap)

                subcategory_results[category]['roc_auc'].append(roc_auc)
                subcategory_results[category]['pr_auc'].append(pr_auc)
                subcategory_results[category]['balanced_accuracy'].append(balanced_accuracy_score)

    # Calculate confidence intervals and means
    final_results = get_confidence_interval_mean(results, bootstrap_method)

    if bootstrap_subcategories:
        final_results_subcategories = {}

        for category, results_category in subcategory_results.items():
            final_results_subcategories[category] = get_confidence_interval_mean(results_category, bootstrap_method)

    return final_results, final_results_subcategories if bootstrap_subcategories else None


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


def plot_calibration_curve(y_test: np.array, predictions: np.array, n_bins: int,  bin_strategy: str,
                           save_path: str):
    """
    Code adapted from: https://endtoenddatascience.com/chapter11-machine-learning-calibration
    Produces a calibration plot and saves it locally. The raw data behind the plot is also written locally.
    :param y_test: y_test series
    :param predictions: predictions series
    :param n_bins: number of bins for the predictions
    :param bin_strategy: uniform - all bins have the same width; quantile - bins have the same number of observations
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
        plt.savefig(os.path.join(save_path, f'{bin_strategy}_{n_bins}_calibration_plot.png'), dpi=400, format="png")
        plt.clf()

        print(f"The ECE is {ece}. The brier score is {brier_score}")
        calibration_df.to_csv(os.path.join(save_path,
                                           f'{bin_strategy}_{n_bins}_calibration_summary.csv'), index=False)
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