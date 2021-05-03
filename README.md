## Crystal Systems and Space Groups Prediction of Inorganic Materials Using Machine Learning


## Premise

This is a random forest machine learning model with a new feature set combined with the standard composition features such as Magpie descriptors for effective space group prediction for inorganic materials. 

If you find our software is useful, please cite it as:<br >

#### Yuxin Li, Rongzhi Dong, Wenhui Yang, Jianjun Hu*, Crystal Systems and Space Groups Prediction of Inorganic Materials Using Machine Learning

Developed in 2021.4-30 at <br />
School of Mechanical Engineering<br />
Guizhou University, Guiyang, China <br />

Machine Learning and Evolution Laboratory<br />
Department of Computer Science and Engineering<br />
University of South Carolina, Columbia, USA<br />


## Performance on Materials Project dataset

Our model of space group prediction in cubic material is trained with the dataset of 'ML/cubic.csv' by useing the 'ML/RF_of_us.py'
, and the dataset used for other crystal system training can be downloaded from here [data.csv](https://figshare.com/s/9cfe81a3b087618353c8).
Moreover, the two previous work frameworks for space group classification are also put in the ML folder.

Prediction performance for space groups over different crystal systems （10 fold cross validation)
|Crystal system|data set size |   Accuracy  |     MCC     |   Precision |   Recall  |   F1 score  |
|-------------|---------------|-------------|-------------|-------------|-----------|-------------|
Cubic         |     17367     | 0.961±0.006 | 0.945±0.008 | 0.960±0.005 |0.961±0.006| 0.959±0.006 |
Hexagonal     |      8201     | 0.909±0.008 | 0.888±0.010 | 0.908±0.008 |0.909±0.008| 0.906±0.008 |
Trigonal      |      9429     | 0.824±0.012 | 0.797±0.014 | 0.823±0.013 |0.824±0.012| 0.818±0.012 |
Tetragonal    |     12675     | 0.849±0.013 | 0.832±0.015 | 0.846±0.013 |0.849±0.013| 0.840±0.014 |
Orthorhombic  |     22392     | 0.755±0.005 | 0.729±0.006 | 0.759±0.005 |0.755±0.005| 0.746±0.006 |
Monoclinic    |     23024     | 0.712±0.009 | 0.647±0.011 | 0.715±0.010 |0.712±0.009| 0.703±0.010 |
Triclinic     |      9440     | 0.835±0.013 | 0.665±0.026 | 0.835±0.013 |0.835±0.013| 0.834±0.013 |
<!--- img src="performance1.png" width="800"--->

## Environment Setup

To use this machine learning model, you need to create an environment with the correct dependencies. Using `Anaconda` this can be accomplished with the following commands:

```bash
conda create --name SG_predict python=3.6
conda activate SG_predict
conda install --channel conda-forge pymatgen
pip install matminer
pip install scikit-learn==0.24.1
```

## Setup

Once you have setup an environment with the correct dependencies you can install by the following commands:

```bash
conda activate SG_predict
git clone https://github.com/Yuxinya/SG_predict
cd SG_predict
pip install -e .
```

Pre-trained models are stored in google drive. Download the file `model.zip` from from the [figshare](https://figshare.com/s/62da0bce61e4ff038bf7). After downing the file, copy it to `SG_predict` and extract it. the `model` folder should be in the `SG_predict` directory after the extraction is completed. If you do not do this, the model can only make claccification by providing the crystal system information.
## Example Use

In order to test your installation you can run the following example from your `SG_predict` directory:

```sh
cd /path/to/SG_predict/
python predict.py -i full_formula -s crystal_system

for example:
python predict.py -i Zn24Si24Bi16O96 -s cubic
python predict.py -i Zn24Si24Bi16O96
```

The following cyrstal_system values are accepted
```
crystal     # crystal system unknown. 
cubic
hexagonal
trigonal
tetragonal
orthorhombic
monoclinic
triclinic
```

You can also use this algorithm to train, test and predict data
```sh
cd /path/to/SG_predict/
python predict.py -data the data you provideed -type train, test or predict

for example:
python model.py -data data/train.csv -type train
python model.py -data data/test.csv -type test
python model.py -data data/predict.csv -type predict
```
The following .csv format are accepted for train and test
|formula       |space_group    |
|--------------|---------------|
|Na8Al6Si6S1O28|     195       |
|Na4Cl4O12     |      198      |


The following .csv format are accepted for predict
|formula       |
|--------------|
|Na8Al6Si6S1O28|
|Na4Cl4O12     |
