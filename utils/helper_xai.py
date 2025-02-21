import os
import matplotlib.pyplot as plt
import numpy as np
import shap
import warnings


def create_shap_decision_plot(
    model, 
    explainer, 
    test_data, 
    feature_names, 
    prediction_filter='correct',
    confidence_threshold=0.9,
    figures_path=None,
    study_name=None,
    feature_selection=False
):
    """
    Creates and saves a SHAP decision plot for high-confidence predictions.
    
    Args:
        model: Trained model with predict_proba and predict methods
        explainer: SHAP explainer object
        test_data: Tuple of (X_test, y_test) or just X_test features
        feature_names: List of feature names
        prediction_filter: String indicating which predictions to analyze (default: 'correct')
            - 'correct': only high-confidence correct predictions
            - 'incorrect': only high-confidence incorrect predictions
            - 'all': all high-confidence predictions regardless of correctness
        confidence_threshold: Minimum probability threshold for "certain" predictions (default: 0.9)
        figures_path: Directory path to save the plot (default: None)
        study_name: Name of the study for the output file (default: None)
        feature_selection: Boolean indicating if feature selection was used (default: False)
    
    Returns:
        bool: True if plot was created and saved successfully, False otherwise
    """
    try:
        # Validate prediction_filter parameter
        valid_filters = ['correct', 'incorrect', 'all']
        if prediction_filter not in valid_filters:
            raise ValueError(f"prediction_filter must be one of {valid_filters}")

        # Handle test_data being either a tuple or feature matrix
        if isinstance(test_data, tuple):
            X_test, y_test, _ = test_data
        else:
            X_test = test_data
            y_test = model.predict(X_test)  # Use model predictions as ground truth if no labels provided

        # Get predictions and probabilities
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        # Filter for high-confidence predictions based on prediction_filter
        high_confidence = y_pred_proba[:, 1] >= confidence_threshold
        
        if prediction_filter == 'correct':
            filter_mask = high_confidence & (y_pred == y_test)
            filter_desc = "correct"
        elif prediction_filter == 'incorrect':
            filter_mask = high_confidence & (y_pred != y_test)
            filter_desc = "incorrect"
        else:  # 'all'
            filter_mask = high_confidence
            filter_desc = "all"

        filtered_instances = X_test[filter_mask]

        if len(filtered_instances) == 0:
            print(f"No {filter_desc} predictions meet the confidence threshold of {confidence_threshold}")
            return False

        # Calculate SHAP values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Get expected value
            if hasattr(explainer, 'expected_value'):
                expected_value = explainer.expected_value
                if isinstance(expected_value, list):
                    expected_value = expected_value[1]  # Get value for positive class
            else:
                expected_value = explainer(filtered_instances).base_values[0]

            shap_values = explainer(filtered_instances)

            # Create decision plot
            plt.figure(figsize=(12, 8))
            shap.decision_plot(
                expected_value,
                shap_values.values,
                filtered_instances,
                feature_names=feature_names,
                feature_order='hclust',
                feature_display_range=slice(-1, -X_test.shape[1], -1),
                auto_size_plot=True,
                show=False,
                link='logit'
            )

            # Save plot if path is provided
            if figures_path and study_name:
                save_name = f"{study_name}_shap_decision_{filter_desc}_confident_predictions"
                save_name += "_feature_selection" if feature_selection else ""
                save_name += ".png"
                plt.savefig(
                    os.path.join(figures_path, save_name),
                    dpi=500,
                    bbox_inches='tight'
                )
            
            plt.close()
            return True

    except Exception as e:
        print(f"Error creating SHAP decision plot: {str(e)}")
        return False

def create_shap_beeswarm_plot(
    explainer,
    test_data,
    figures_path=None,
    study_name=None,
    feature_selection=False,
    max_display=11
):
    """
    Creates and saves a SHAP beeswarm plot.
    
    Args:
        explainer: SHAP explainer object
        test_data: Features to generate SHAP values for
        figures_path: Directory path to save the plot (default: None)
        study_name: Name of the study for the output file (default: None)
        feature_selection: Boolean indicating if feature selection was used (default: False)
        max_display: Maximum number of features to display in the plot (default: 11)
    
    Returns:
        bool: True if plot was created and saved successfully, False otherwise
    """
    try:
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer(test_data)

        # Create beeswarm plot
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap_values, show=False, max_display=max_display)
        plt.tight_layout()

        # Save plot if path is provided
        if figures_path and study_name:
            save_name = f"{study_name}_shap_beeswarm"
            save_name += "_feature_selection" if feature_selection else ""
            save_name += ".png"
            plt.savefig(
                os.path.join(figures_path, save_name),
                dpi=500,
                bbox_inches='tight'
            )
        
        plt.close()
        return True, shap_values

    except Exception as e:
        print(f"Error creating SHAP beeswarm plot: {str(e)}")
        return False, None

def create_shap_dependence_plots(
    shap_values,
    feature_names,
    figures_path=None,
    study_name=None,
    feature_selection=False,
    n_top_features=5
):
    """
    Creates and saves SHAP dependence plots for top features.
    
    Args:
        shap_values: SHAP values object from explainer
        feature_names: List of feature names
        figures_path: Directory path to save the plots (default: None)
        study_name: Name of the study for the output files (default: None)
        feature_selection: Boolean indicating if feature selection was used (default: False)
        n_top_features: Number of top features to plot (default: 5)
    
    Returns:
        bool: True if plots were created and saved successfully, False otherwise
        list: List of top feature names
    """
    try:
        # Get the top features by mean absolute SHAP value
        feature_importance = np.abs(shap_values.values).mean(0)
        top_features_idx = np.argsort(feature_importance)[-n_top_features:][::-1]
        top_features = [feature_names[i] for i in top_features_idx]

        # Create dependence plots for top features
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            shap.plots.scatter(shap_values[:, feature], show=False)
            plt.title(f"SHAP Dependence Plot - {feature}")
            plt.tight_layout()
            
            if figures_path and study_name:
                save_name = f"{study_name}_shap_dependence_{feature.replace(' ', '_')}"
                save_name += "_feature_selection" if feature_selection else ""
                save_name += ".png"
                plt.savefig(
                    os.path.join(figures_path, save_name),
                    dpi=500,
                    bbox_inches='tight'
                )
            plt.close()
            
        return True, top_features

    except Exception as e:
        print(f"Error creating SHAP dependence plots: {str(e)}")
        return False, None