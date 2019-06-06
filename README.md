# mdlearn
Machine learning of the thermodynamic properties of molecular liquids

* Run `gen-fingerprint.py` to calculate fingerprints.
* Run `split-data.py` to split original data to train/validate/test datasets based on molecule.
* Run `train-alkane-npt.py` to train the model. Use `--part xxx` to select partition cache file generated in previous step. Otherwise, the data are splitted randomly, which cause artifacts in prediction.
* Run `score-alkane-npt.py` to check the performance on test dataset.
* Run `predict-alkane-npt.py` to predict thermo properties for new molecules.
* `rfe-alkane-npt.py` is used for feature selection.

See our publication for details
`Predicting Thermodynamic Properties of Alkanes by High-throughput Force Field Simulation and Machine Learning`
https://doi.org/10.1021/acs.jcim.8b00407

Original developer is Yanze Wu
https://github.com/flipboards
