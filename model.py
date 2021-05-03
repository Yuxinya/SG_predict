from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from pymatgen.core.composition import Composition

# import tensorflow as tf 
# from keras.layers import LeakyReLU
import re
import numpy as np 
import pandas as pd 
from pandas.core.frame import DataFrame

from scipy import stats

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

# import matplotlib.pyplot as plt 

# from sklearn.metrics import r2_score

# from keras import backend as K

from sklearn.ensemble import RandomForestClassifier


from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

from sklearn.metrics import f1_score, matthews_corrcoef ,make_scorer,recall_score,roc_auc_score
# from sklearn.metrics import top_k_accuracy_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import bz2 
import pickle 
import _pickle as cPickle
import os
import joblib
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
    """
        generate feature from a dataframe with a "formula" column that contains 
        chemical formulas of the compositions.
    """
    fake_df = StrToComposition().featurize_dataframe(fake_df, "formula", ignore_errors=ignore_errors)
    fake_df = fake_df.dropna()
    fake_df = feature_calculators.featurize_dataframe(fake_df, col_id='composition', ignore_errors=ignore_errors);
    fake_df["NComp"] = fake_df["composition"].apply(len)
    return fake_df


def ext_magpie(input):
	# print(input)
	# # input='{1}{0}{1}'.format(input,"'")
	# print(input)
	formula = pd.read_csv(input)
	# print(formula)
	ext_magpie = generate(formula)
	return(ext_magpie)
# ext_magpie('train.csv')

def mlmdd(input):
	# input='{1}{0}{1}'.format(input,"'")
	y = pd.read_csv(input).iloc[:,0].values
	# print(y)
	ls = []
	for i in y:
		comp=Composition(i)
		redu = comp.get_reduced_formula_and_factor()[1]
		# redu_for = comp.get_reduced_formula_and_factor()[0]
		# redu_data=np.array(list(comp.as_dict().values()))
		most=comp.num_atoms
		data=np.array(list(comp.as_dict().values()))
		# print(list(data))
		# l = len(data)
		# s = sum(data)
		# print(s)

		a = max(data)
		# print(a)
		i = min(data)
		m = np.mean(data)
		# v = np.var(data)
		var = np.var(data/most)
		# var2 = np.var(data/redu)
		ls.append([most,a,i,m,redu,var,])
	df = pd.core.frame.DataFrame(ls)
	return(df)
# mlmdd('train.csv')


def get_features(input):
	mlmd = mlmdd(input)
	ext_mag = ext_magpie(input)
	result = ext_mag.join(mlmd)
	# print(result)
	return(result)
# features = get_features('train.csv')
# print(features)
def compressed_pickle(title, data):
	with bz2.BZ2File(title + '.pbz2', 'w') as f: 
		cPickle.dump(data, f)
def decompress_pickle(file): 
	data = bz2.BZ2File(file, 'rb') 
	data = cPickle.load(data) 
	return data

def input():
	# features = get_features('train.csv')
	# print(features)
	parser = argparse.ArgumentParser()
	parser.add_argument('-data','--data',  type=str, 
	                    help="The input crystal formula.")
	parser.add_argument('-type','--type',  type=str, default='crystal',
	                    help="The input crystal system.")
	args = parser.parse_args()
	form = args.data
	system = args.type
	# print(form)
	# form='{1}{0}{1}'.format(form,"'")
	# print(form)
	dirs = 'model'
	if system == 'train':
		print('----------training----------')
		# print(pd.read_csv(form))
		df = get_features(form)
		# print(df[0])
		x = df.iloc[:,3:].fillna(0).values
		y = df.iloc[:,1].fillna(0).values 
		# print(x)
		# print(y)
		p = np.random.permutation(range(len(x)))
		x_train,y_train = x[p],y[p] #Disrupt the order of data
		forest = RandomForestClassifier(criterion='entropy',n_estimators=100,max_features=80,max_depth=None,min_samples_leaf=1,min_samples_split=2,n_jobs=-1) 
		forest.fit(x_train,y_train)
		
		if not os.path.exists(dirs):
			os.makedirs(dirs)
		compressed_pickle(dirs+'/model', forest)
		print('----------complete----------')

	if system == 'test':
		print('----------testing----------')
		df = get_features(form)
		x = df.iloc[:,3:].fillna(0).values
		y = df.iloc[:,1].fillna(0).values 
		forest = decompress_pickle(dirs+'/model.pbz2')
		acc = forest.score(x, y)
		print('The accuracy score is:', acc)
		print('----------complete----------')
	# data = pd.read_csv(form)
	# print(data)

	if system == 'predict':
		print('----------predict----------')
		df = get_features(form)
		x = df.iloc[:,2:].fillna(0).values
		forest = decompress_pickle(dirs+'/model.pbz2')
		y=forest.predict(x).reshape(-1,1)
		# print(y)
		data = pd.read_csv(form)
		result = np.hstack((data,y))
		result = pd.DataFrame(result,columns=['formula','space_group'])
		print(result)
		result.to_csv('data/predict_result.csv',index=0)
		print('----------complete----------')


if __name__ == "__main__":
	input()
