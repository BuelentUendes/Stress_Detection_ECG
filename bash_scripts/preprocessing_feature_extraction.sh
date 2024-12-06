#!/bin/bash

# Bash script for running sever
sample_frequencies=(1000 512 256 128)

# shellcheck disable=SC2068
# shellcheck disable=SC1073d
# shellcheck disable=SC1061
# shellcheck disable=SC1061
for sample_frequency in ${sample_frequencies[@]}
do
  python3 preprocessing.py --sample_frequency $sample_frequency
  python3 feature_extraction.py --sample_frequency $sample_frequency
done