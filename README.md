# Electrocardiogram-Based Mental Stress Detection Amid Everyday Activities Using Machine Learning: Model Development and Validation Study
**Authors:** [Buelent Uendes](https://buelentuendes.github.io/), Alex Antonides, Sjors van de Ven, Denise Johanna van der Mee, Eco de Geus, and Mark Hoogendoorn

## About The Project
Using a dataset consisting of 127 participants who performed various experimental conditions, this project aims to: 
- **evaluate ML models' ability to distinguish mental-stress episodes from a composite "no-stress" background**, including rest and low- to moderate-intensity activities
- **assess** their **generalizability** to new stressors and participants
- **test robustness to lower sampling rates and fewer features**, exploring their suitability for lightweight wearables.

### Repository structure
The repository is structured as follows.

- bash_scripts/ contains bash scripts.
- data/ contains the source code as well as the datasets used in this study.
- utils/ contains helper functions for the project.
- src/ contains the main source code for the project.
- sandbox/ contains archived and experimental scrips (not needed for the results of the project).

The main files to reproduce the results obtained in our paper are the *.py files.

**Important:** The data folder is currently empty, as our dataset is not publicly available and is only available upon request.

### Stress-in-Action (SiA)-Kit
The SiA-Kit is a package that provides human-readable and extendable pipelines to repeat the experiments described in the project. The builder pattern that is used throughout the pipelines, makes it easier to understand what is happening behind the scenes. 

### Getting started
This project uses Python 3.9+ and additional dependencies managed via pip.
Please first create and activate a virtual environment via:

```bash
virtualenv <NAME_OF_YOUR_VIRTUALENV>
source <NAME_OF_YOUR_VIRTUALENV>/bin/activate
```
Install the requirements:
```bash
pip3 install -r requirements.txt 
```

### Workflow
Once having access to the dataset, obtaining the main results would require running the following scripts:

1. Optional: Downsample the ECG signal
```bash
python3 downsample.py
```

2. Preprocess the ECG signal

```bash
python3 preprocessing.py
```

3. Feature extraction

```bash
python3 feature_extraction.py
```

4. Training the ML model

```bash
python3 main_training.py
```

5. Plotting results

```bash
python3 main_figures.py
```

6. Optional: Statistical Analysis

```bash
python3 run_statistical_analysis.py
```

Each script uses command line arguments to adjust hyperparameters. 

### Citation
If you found this work useful in your research, please consider citing:
```bibtex
Uendes B, Antonides A, van de Ven S, van der Mee D, de Geus E, Hoogendoorn M
Electrocardiogram-Based Mental Stress Detection Amid Everyday Activities Using Machine Learning: Model Development and Validation Study
J Med Internet Res 2026;28:e80450
URL: https://www.jmir.org/2026/1/e80450
DOI: 10.2196/80450
```

### Acknowledgements
This work is funded by [Stress in Action](https://stress-in-action.nl/). 
The research project [Stress in Action](https://stress-in-action.nl/) is financially supported by the Dutch Research Council and the Dutch Ministry of Education, Culture and Science (NWO gravitation grant number 024.005.010).