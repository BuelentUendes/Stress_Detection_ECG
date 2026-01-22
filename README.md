# Electrocardiogram-Based Mental Stress Detection Amid Everyday Activities Using Machine Learning: Model Development and Validation Study
**Authors:** [Buelent Uendes](https://buelentuendes.github.io/), Alex Antonides, Sjors van de Ven, Denise Johanna van der Mee, Eco de Geus, and Mark Hoogendoorn

## About The Project
Using a dataset consisting of 127 participants who performed various experimental conditions, this project aims to: 
- **evaluate ML models' ability to distinguish mental-stress episodes from a composite "no-stress" background**, including rest and low- to moderate-intensity activities
- **assess** their **generalizability** to new stressors and participants
- **test robustness to lower sampling rates and fewer features**, exploring their suitability for lightweight wearables.

### Repository structure
**To be added**

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

### Acknowledgements
This work is funded by [Stress in Action](https://stress-in-action.nl/). 
The research project [Stress in Action](https://stress-in-action.nl/) is financially supported by the Dutch Research Council and the Dutch Ministry of Education, Culture and Science (NWO gravitation grant number 024.005.010).