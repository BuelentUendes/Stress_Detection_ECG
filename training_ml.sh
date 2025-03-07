#!/bin/bash

# shellcheck disable=SC2068
# We set the number of trials to 25 optuna
models=("lr" "xgboost" "rf") # Random forest is expensive
sample_frequencies=(1000 512 256 128) # we use four different window sizes # For now we do it 1000
negative_classes=("baseline" "low_physical_activity" "moderate_physical_activity")

for model in ${models[@]}
do
  for frequency in ${sample_frequencies[@]}
  do
    for class in ${negative_classes[@]}
    do
      if [ "$class" = "baseline" ]; then
        python3 main_training.py --model_type $model --sample_frequency $frequency --negative_class $class --add_calibration_plots --bootstrap_test_results --verbose --bootstrap_subcategories --do_hyperparameter_tuning --n_trials 25
      else
        python3 main_training.py --model_type $model --sample_frequency $frequency --negative_class $class --add_calibration_plots --bootstrap_test_results --verbose --resampling_method smote --do_hyperparameter_tuning --n_trials 25 --bootstrap_subcategories
      fi
    done
  done
done