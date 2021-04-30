import numpy as np
import pandas as pd 

from sklearn.metrics import f1_score, matthews_corrcoef ,make_scorer,recall_score,roc_auc_score
from sklearn.metrics import top_k_accuracy_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# x = pd.read_csv('data/data.csv').iloc[:9440,5:].fillna(0).values 
# y = pd.read_csv('data/data.csv').iloc[:9440,1].fillna(0).values 
# x = pd.read_csv('data/data.csv').iloc[9440:32464,5:].fillna(0).values 
# y = pd.read_csv('data/data.csv').iloc[9440:32464,1].fillna(0).values
# x = pd.read_csv('data/data.csv').iloc[32464:54856,5:].fillna(0).values 
# y = pd.read_csv('data/data.csv').iloc[32464:54856,1].fillna(0).values
# x = pd.read_csv('data/data.csv').iloc[54856:67531,5:].fillna(0).values 
# y = pd.read_csv('data/data.csv').iloc[54856:67531,1].fillna(0).values
# x = pd.read_csv('data/data.csv').iloc[67531:76960,5:].fillna(0).values 
# y = pd.read_csv('data/data.csv').iloc[67531:76960,1].fillna(0).values
# x = pd.read_csv('data/data.csv').iloc[76960:85161,5:].fillna(0).values 
# y = pd.read_csv('data/data.csv').iloc[76960:85161,1].fillna(0).values

#  cubic data are as follows
x = pd.read_csv('data/data.csv').iloc[85161:,5:].fillna(0).values 
y = pd.read_csv('data/data.csv').iloc[85161:,1].fillna(0).values

p = np.random.permutation(range(len(x)))
x,y = x[p],y[p] #Disrupt the order of data

forest = RandomForestClassifier(criterion='entropy',n_estimators=100,max_features=80,max_depth=None,min_samples_leaf=1,min_samples_split=2,n_jobs=-1) 

scoring = {
'accuracy' : 'accuracy',
# 'f1_micro' : 'f1_micro',
# 'f1_macro' : 'f1_macro',
'f1_weighted' : 'f1_weighted',
'MCC':make_scorer(matthews_corrcoef) ,
}
scores = cross_validate(forest, x, y, scoring=scoring, cv=10, )

accuracy = scores['test_accuracy']
mcc = scores['test_MCC']
f1_weighted = scores['test_f1_weighted']

print('accuracy   :','%.3f'%np.mean(accuracy),'+-','%.3f'%np.std(accuracy,ddof=1))
print('test_MCC   :','%.3f'%np.mean(mcc),'+-','%.3f'%np.std(mcc,ddof=1))
print('f1_weighted:','%.3f'%np.mean(f1_weighted),'+-','%.3f'%np.std(f1_weighted,ddof=1))