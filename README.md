# mdlearn
Machine learning of the thermodynamic properties of molecular liquids

This is the enhanced version of our previously published work  
`Predicting Thermodynamic Properties of Alkanes by High-throughput Force Field Simulation and Machine Learning`  
https://doi.org/10.1021/acs.jcim.8b00407

## Steps
*Following is an example of learning critical temperature of hydrocarbon using two simple fingerprints*

* Calculate fingerprints
```
./gen-fp.py -i ../data/nist-CH-tc.txt -e wyz,simple -o fp
```
* Split data to train/validate datasets using 5-Fold cross-validation
```
./split-data.py -i ../data/nist-CH-tc.txt -o fp
```
* Train the model  
```
./train.py -i ../data/nist-CH-tc.txt -t tc -f fp/fp_wyz,fp/fp_simple -p fp/part-1.txt --epoch 40
```
* Predict property for new molecules
```
./predict.py -o out -f wyz,simple -i CCCCCC
```
