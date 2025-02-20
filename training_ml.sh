#!/bin/bash

# shellcheck disable=SC2068
# We set the number of trials to 25 optuna
models=("lr" "rf" "xgboost")
sample_frequencies=(128 256 512 1000) # we use four different window sizes
negative_classes=("baseline" "low_physical_activity" "moderate_physical_activity")

for model in ${models[@]}
do
  for frequency in ${sample_frequencies[@]}
  do
    for class in ${negative_classes[@]}
    do
      if [ "@class" != "baseline" ]; then
          python3 main_training.py --model_type $model --window_size $frequency --negative_class $class --do_hyperparameter_tuning --n_trials 25 --add_calibration_plots --bootstrap_test_results --verbose
      else
        python3 main_training.py --model_type $model --window_size $frequency --negative_class $class --do_hyperparameter_tuning --n_trials 25 --add_calibration_plots --bootstrap_test_results --verbose --resampling_method smote
      fi
    done
  done
done