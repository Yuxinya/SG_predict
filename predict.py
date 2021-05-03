from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from pymatgen.core.composition import Composition
import numpy as np 
import pandas as pd 
import bz2 
import _pickle as cPickle
import argparse

feature_calculators = MultipleFeaturizer([
    cf.ElementProperty.from_preset(preset_name="magpie"),
    cf.Stoichiometry(),
    cf.ValenceOrbital(props=['frac']),
    cf.IonProperty(fast=True),
    cf.BandCenter(),
    cf.ElementFraction(),
    ])

def generate(fake_df, ignore_errors=False):
    fake_df = np.array([fake_df])
    fake_df = pd.DataFrame(fake_df)
    fake_df.columns = ['full_formula']
    # print(fake_df)
    fake_df = StrToComposition().featurize_dataframe(fake_df, "full_formula", ignore_errors=ignore_errors)
    fake_df = fake_df.dropna()
    fake_df = feature_calculators.featurize_dataframe(fake_df, col_id='composition', ignore_errors=ignore_errors);
    fake_df["NComp"] = fake_df["composition"].apply(len)
    return fake_df

def mlmd(x):
    ls = []
    comp=Composition(x)
    redu = comp.get_reduced_formula_and_factor()[1]
    most=comp.num_atoms
    data=np.array(list(comp.as_dict().values()))
    # print(data)
    a = max(data)
    i = min(data)
    m = np.mean(data)
    # v = np.var(data)
    var = np.var(data/most)
    # var2 = np.var(data/redu)
    ls.append([most,a,i,m,redu,var,])
    df = pd.DataFrame(ls)
    return(df)

def decompress_pickle(file): 
  data = bz2.BZ2File(file, 'rb') 
  data = cPickle.load(data) 
  return data

def main():

    print('----------Predicting----------')
    print('The descriptors we use in this model are as follows:')
    print('Element Property statistics of Magpie: Atomic Number, Mendeleev Number, Atomic Weight, Melting Temperature, Periodic Table Row and Column, Covalent Radius, Electronegativity, The number of Valence e ineach Orbital(s, p. d, f, total), The number of unilled e ineach orbital (s, p. d, f, total), Ground State Volume, Ground State Band Gap Energy, Ground State Magnetic Moment, Space Group Number')
    print('Additional Predictors of Magpie: Stoichiometry p-norm (p=0,2,3,5.7), Elemental Fraction, Fraction of Electrons in each Orbital, Band Center, lon Property (possible to formionic compound, ionic charge')
    print('Added Predictors in this work: Total Atom Number, Maximum Atom Number, Minimum Atom Number, Average Atom Number, Specific Value, Atom Number Variance')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--formula',  type=str, 
                        help="The input crystal formula.")
    parser.add_argument('-s','--crystal_system',  type=str, default='crystal',
                        help="The input crystal system.")
    args = parser.parse_args()
    form = args.formula
    system = args.crystal_system
    ext_magpie = generate(form)
    m = mlmd(form)
    result = ext_magpie.join(m)
    result = result.iloc[:,2:]
    dirs = 'Model'
    # print(result)
    if system == 'crystal' :
        forest =  decompress_pickle(dirs+'/Crystal.pbz2')
        y_pred = forest.predict(result)
        print('The predicted space group is',y_pred[0])
    if system == 'cubic' or  system == 'Cubic':
        forest = decompress_pickle(dirs+'/Cubic.pbz2') #  python predict.py -i K12Mn16O32  -s cubic
        y_pred = forest.predict(result)
        print('The predicted space group is',y_pred[0])
    if system == 'hexagonal' or system == 'Hexagonal':
        forest =  decompress_pickle(dirs+'/Hexagonal.pbz2')
        y_pred = forest.predict(result)
        print('The predicted space group is',y_pred[0])
    if system == 'trigonal' or system == 'Trigonal':
        forest =  decompress_pickle(dirs+'/Trigonal.pbz2')
        y_pred = forest.predict(result)
        print('The predicted space group is',y_pred[0])
    if system == 'tetragonal' or system == 'Tetragonal':
        forest =  decompress_pickle(dirs+'/Tetragonal.pbz2')
        y_pred = forest.predict(result)
        print('The predicted space group is',y_pred[0])
    if system == 'orthorhombic' or system == 'Orthorhombic':
        forest =  decompress_pickle(dirs+'/Orthorhombic.pbz2')
        y_pred = forest.predict(result)
        print('The predicted space group is',y_pred[0])
    if system == 'monoclinic' or system == 'Monoclinic':
        forest =  decompress_pickle(dirs+'/Monoclinic.pbz2')
        y_pred = forest.predict(result)
        print('The predicted space group is',y_pred[0])
    if system == 'triclinic' or system == 'Triclinic':
        forest =  decompress_pickle(dirs+'/Triclinic.pbz2')
        y_pred = forest.predict(result)
        print('The predicted space group is',y_pred[0])
    print('-----------Complete-----------')

if __name__ == "__main__":
	main()
