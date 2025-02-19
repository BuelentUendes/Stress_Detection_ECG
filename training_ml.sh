#!/bin/bash

# shellcheck disable=SC2068
# We set the number of trials to 25 optuna
models=("lr" "rf" "xgboost")
sample_frequencies=(128 ) # we use four different window sizes
negative_classes=("baseline" "low_physical_activity" "moderate_physical_activity")

for model in ${models[@]}
do
  for window_len in ${window_size[@]}
  do
    for class in ${negative_classes[@]}
    do
      python3 main_training.py --model_type $model --window_size "$window_len" --negative_class $class --n_trials 25
    done
  done
done