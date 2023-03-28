# TOTAL PERSPECTIVE VORTEX

Le but de ce projet est d'entrainer un modèle de machine learning à faire des predictions à partir de données d'EEG: **Electro-encéphalogramme**.\
Il s'agit de prédire à partir de ces données, à quel mouvement (A ou B) la personne est en train de penser.

|||
|--|--|
|![](./assets/EEG.png)|![](./assets/gauche.png)![](./assets/droite.png)|


## General instructions:
You’ll have to process data coming from cerebral activity, with machine learning algo-
rithms.

The data was mesured during a motor imagery experiment, where people had to
do or imagine a hand or feet movement. Those people were told to think or do a move-
ment corresponding to a symbol displayed on screen. The results are cerebral signals
with labels indicating moments where the subject had to perform a certain task.

You’ll have to code in Python as it provides **MNE**, a library specialized in **EEG** data
processing and, scikit-learn, a library specialized in machine learning.\
The subject focuses on implementing the algorithm of dimensionality reduction, to
further transform filtered data before classification.\
This algorithm will have to be in-
tegrated within **sklearn** so you’ll be able to use sklearn tools for classification and score validation.

## Usage:

1. Parse Data::
```bash
python data_parsing.py
```
And for more info, look at the notebook "vizualize.ipynb"

2. Train
```bash
python treatment_pipeline.py
```
this will start a training on each runs and each subjects. It will save the models, the data and the results inside the src folder where you should launch the script. It will also display the score of your training.

To start training only on a specific subject and run, do the following:
```
python treatment_pipeline.py --train --run 0 --subject 1
```
For more info, do:
```
python treatment_pipeline.py -h
```

3. Predict
```bash
python treatment_pipeline.py --predict --run 0 --subject 1
```
If no training was done on this specific subject, it will give you an error, asking you to first launch the training on this same run and subject.
