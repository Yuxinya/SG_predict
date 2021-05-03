from sklearn.preprocessing import MinMaxScaler,StandardScaler
# import tensorflow as tf 
import re
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score,f1_score,matthews_corrcoef ,make_scorer,recall_score,roc_auc_score,accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import RMSprop,Adam,SGD

def main_K():
	#  cubic data are as follows
	X = pd.read_csv('cubic.csv').iloc[:,5:-6].fillna(0).values.astype(float)
	Y = pd.read_csv('cubic.csv').iloc[:,1].fillna(0).values
	# encode class values as integers
	encoder = LabelEncoder()
	encoded_Y = encoder.fit_transform(Y)
	# convert integers to dummy variables (one hot encoding)
	dummy_y = np_utils.to_categorical(encoded_Y)
	input_dim,output_dim=len(X[0]),len(dummy_y[0])

	kf = KFold(n_splits=10,shuffle=True)
	ls = []
	for train, test in kf.split(X):
	    # print(len(train),train, len(test),test)
	    x_train,x_test, y_train, y_test = X[train],X[test],dummy_y[train],dummy_y[test]
	    res = NN(x_train,x_test,y_train,y_test,input_dim,output_dim)
	    ls.append(res)
	return(ls)


def NN(x_train,x_test,y_train,y_test,input_dim,output_dim):
	scaler = StandardScaler()
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)
	model = Sequential()
	model.add(keras.Input(shape=(input_dim)))
	model.add(BatchNormalization())
	model.add(Dense(units=230,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(rate=0.2))
	model.add(Dense(units=178,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(rate=0.2))
	model.add(Dense(units=280,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(rate=0.2))
	model.add(Dense(units=244,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(rate=0.2))
	model.add(Dense(units=181,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(rate=0.2))
	model.add(Dense(units=output_dim,activation='softmax'))
	# print(model.summary())

	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	early_stopping = EarlyStopping(monitor='loss',patience=30,verbose=1)
	model.fit(x=x_train,y=y_train, epochs=2000, batch_size=256, verbose=2,callbacks=[early_stopping,])
	acc = model.evaluate(x_test, y_test)
	y_pred = model.predict(x_test)
	y_pred = np.argmax(y_pred,axis=1)
	y_test = np.argmax(y_test,axis=1)

	a = accuracy_score(y_test,y_pred)
	mcc = matthews_corrcoef(y_test,y_pred)
	f1 = f1_score(y_test, y_pred, average='weighted')
	res = [a,mcc,f1]
	print(res)
	return(res)



result = main_K()
print('accuracy   :','%.3f'%np.mean(result,axis=0)[0],'+-','%.3f'%np.std(result,axis=0)[0])
print('test_MCC   :','%.3f'%np.mean(result,axis=0)[1],'+-','%.3f'%np.std(result,axis=0)[1])
print('f1_weighted:','%.3f'%np.mean(result,axis=0)[2],'+-','%.3f'%np.std(result,axis=0)[2])