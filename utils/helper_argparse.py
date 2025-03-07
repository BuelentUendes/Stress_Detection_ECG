import argparse


def validate_scaler(value: str) -> str:
    if value.lower() not in ["standard_scaler", "min_max", None]:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value}. Choose from 'standard_scaler' or 'min_max'.")
    return value.lower()

def validate_feature_subset(value: str) -> str:
    feature_subset = value.replace(" ", "").split(",")
    return feature_subset


def validate_category(value: str) -> str:
    valid_categories = ['high_physical_activity', 'mental_stress', 'baseline',
                        'low_physical_activity', 'moderate_physical_activity', 'rest']
    if value.lower() not in valid_categories:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value}. "
                                         f"Choose from options in {valid_categories}.")
    return value.lower()


def validate_target_metric(value: str) -> str:
    if value.lower() not in ["roc_auc", "accuracy"]:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value}. Choose from 'standard_scaler' or 'min_max'.")
    return value.lower()


def validate_ml_model(value: str) -> str:
    valid_ml_models = ['dt', 'rf', 'adaboost', 'lda', 'knn', 'lr', 'xgboost', 'qda', 'random_baseline', 'svm', 'gmm']
    if value.lower() not in valid_ml_models:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value}. "
                                         f"Choose from options in {valid_ml_models}.")
    return value.lower()


def validate_resampling_method(value: str) -> str:
    implemented_sampling_methods = ['upsample', 'downsample', 'smote', 'none', 'adasyn']
    if value.lower() not in implemented_sampling_methods:
        raise argparse.ArgumentTypeError(f"Invalid choice: {value}. "
                                         f"Choose from options in {implemented_sampling_methods}.")
    return value.lower()
