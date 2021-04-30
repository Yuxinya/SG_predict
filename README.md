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

Our models are trained with the dataset of 'data/data.csv' by useing the RF_of_us.py
<!-- , and the dataset can be downloaded from here [data.zip](https://figshare.com/s/1411919c94be680136cd). It also includes the dataset named 'Enhanced_magpie.csv' for the baseline algorithm. -->

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
conda activate mlatticeabc
git clone https://github.com/usccolumbia/SG_Prediction
cd SG_Prediction
pip install -e .
```

<!-- Pre-trained models are stored in google drive. Download the file `model.zip` from from the [figshare](https://figshare.com/s/d478c42bbe2e21c045b3). After downing the file, copy it to `MLatticeABC` and extract it. the `Model` folder should be in the `MLatticeABC` directory after the extraction is completed.
## Example Use

In order to test your installation you can run the following example from your `MLatticeABC` directory:

```sh
cd /path/to/MLatticeABC/
python predict.py -i full_formula -s crystal_system

for example:
python predict.py -i Mn16Zn24Ge24O96 -s cubic
python predict.py -i Mn16Zn24Ge24O96
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
 -->
