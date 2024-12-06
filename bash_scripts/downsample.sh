#!/bin/bash

# Bash script for running sever
sample_frequencies=(512 256 128)

# shellcheck disable=SC2068
# shellcheck disable=SC1073d
# shellcheck disable=SC1061
# shellcheck disable=SC1061
for sample_frequency in ${sample_frequencies[@]}
do
  python3 downsample.py --desired_sampling_rate $sample_frequency
done