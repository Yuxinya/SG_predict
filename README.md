## Crystal Systems and Space Groups Prediction of Inorganic Materials Using Machine Learning


## Premise

This is a random forest machine learning model with a new feature set combined with the standard composition features such as Magpie descriptors for effective space group prediction for inorganic materials. 

If you find our software is useful, please cite it as:<br >

#### Yuxin Li, Wenhui Yang, Rongzhi Dong, Jianjun Hu*, Crystal Systems and Space Groups Prediction of Inorganic Materials Using Machine Learning

Developed in 2021.4-30 at <br />
School of Mechanical Engineering<br />
Guizhou University, Guiyang, China <br />

Machine Learning and Evolution Laboratory<br />
Department of Computer Science and Engineering<br />
University of South Carolina, Columbia, USA<br />


## Performance on Materials Project dataset

Our model of space group prediction in cubic material is trained with the dataset of 'ML/data.csv' by useing the 'ML/RF_of_us.py'
, and the dataset used for other crystal system training can be downloaded from here [data.csv](https://figshare.com/s/9cfe81a3b087618353c8).

Prediction performance in terms of accuracy score for space groups over different crystal systems （10 fold cross validation)
|Crystal system|data set size |   accuracy  |
|-------------|---------------|-------------|
Cubic         |               | 0.961±0.006 |
Hexagonal     |               | 0.909±0.008 |
Trigonal      |               | 0.824±0.012 |
Tetragonal    |               | 0.849±0.013 |
Orthorhombic  |               | 0.755±0.005 |
Monoclinic    |               | 0.712±0.009 |
Triclinic     |               | 0.835±0.013 |
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

