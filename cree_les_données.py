#! /usr/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, Sequential, load_model
#
from keras import backend as K
from tensorflow.keras.layers import Input, Dropout, Flatten, Permute, Reshape, Lambda
from tensorflow.keras.layers import Dense, Activation, Multiply
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import LayerNormalization
#
from keras_nlp.layers import PositionEmbedding, TransformerEncoder, TransformerDecoder
#
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
#
from tensorflow.keras.utils import Progbar
#
from random import randint, random, shuffle
from math import log, tanh
from os import system
import concurrent.futures
from tqdm import tqdm
import time
import struct as st

def transpose_2d(x): return K.permute_dimensions(x, (0, 2, 1))
def transpose_output_shape(input_shape): return (input_shape[0], input_shape[2], input_shape[1])

tf_logistique = lambda x: 1/(1+tf.exp(-         x ))
np_logistique = lambda x: 1/(1+np.exp(-np.array(x)))
logistique    = lambda x: 1/(1+np.exp(-         x ))

def ema(l,K):
	e = [l[0]]
	for a in l[1:]: e.append(e[-1]*(1-K) + K*a)
	return e

def custom_loss(y_true, y_pred):
	y0 = y_pred[:, 0:1]
	y1 = y_pred[:, 1:2]
	#
	w   = y_true[:, 0:1]
	#yh = y_true[:, 1:2]
	#
	y = tf.tanh      (y0)
	h = tf_logistique(y1)
	#
	_y = tf.stop_gradient(y)
	_h = tf.stop_gradient(h)
	#
	Y = tf.pow(tf.sign(w)     - y0, 2) * _h #(0+yh) * _h
	H = tf.pow(tf.sign(w*_y)  - y1, 2) *  1 #(1-yh)
	#
	return tf.reduce_mean(Y) + tf.reduce_mean(H)

#	======================================================================

def montrer(l, i, N):
	ligne, intervs = l[0], l[1][i]
	plt.plot(ligne)
	plt.plot([len(ligne)-N*i+j*i+1 for j in range(N)], [random()-.5 for j in range(N)]);
	plt.show()

#	======================================================================

from données import binance_btcusdt_15m, bitget_btcusdt_1H, bitget_btcusdt_15m, eurousdt, CAC_40_Données_Historiques

if __name__ == "__main__":
	df, Close, la_Date = binance_btcusdt_15m(verbose=True)
else:
	df, Close, la_Date = binance_btcusdt_15m()

I = 'OUnix','CUnix','ODate','CDate','Symbol','Open','High','Low','Close','qaV','trades','btcVol','usdtVol'

VALIDATION = 2048

Expertises = [
	[60*(df['Close']/df['Close'].ewm(com=5   ).mean()-1),	(1,        ),],	#0
	[40*(df['Close']/df['Close'].ewm(com=25  ).mean()-1),	(1,8       ),],	#1
	[20*(df['Close']/df['Close'].ewm(com=250 ).mean()-1),	(1,8,64,   ),],	#2
	#[10*(df['Close']/df['Close'].ewm(com=1000).mean()-1),	(1,8,64,256),],	#3
	#
	[ .2*(df['trades']/df['trades'].ewm(com=5   ).mean()),	(1,8       ),],	#4
	[ .1*(df['trades']/df['trades'].ewm(com=100 ).mean()),	(1,8,64    ),],	#5
	#[ .1*(df['trades']/df['trades'].ewm(com=1000).mean()),	(1,8,64,256),],	#6
]
for i in range(len(Expertises)): Expertises[i][0] = list(Expertises[i][0])

if __name__ == "__main__":
	for l,_ in Expertises: plt.plot(l)
	plt.show()
	#
	A = int(1+(len(Expertises))**.5)
	fig, ax = plt.subplots(A,A)
	for i,(l,_) in enumerate(Expertises): ax[i//A][i%A].plot(l)
	plt.show()

N = 32

#montrer(Expertises[6], 3, N)
nb_expertises = sum([1 for _,e in Expertises for i in e])
print(f"Expertises : {nb_expertises}")

MAX_I  = max([I for _,Is in Expertises for I in Is])
T      = len(df)
DEPART = N * MAX_I

assert T-DEPART > VALIDATION

print(f'DEPART={DEPART} T={T} VALIDATION={VALIDATION}')
print(f'Train T = {T-DEPART-VALIDATION}')
print(f'Test  T = {VALIDATION}')

SORTIES = 2

#	============================================================	#
if __name__ == "__main__":
	print("Création des données ...")

	X = np.zeros((T-DEPART, nb_expertises, N))

	for t in tqdm(range(DEPART,T), desc="Création des données : ", ncols=100, bar_format="{l_bar}{bar:20}{r_bar}", colour="green"):#range(DEPART, T):
		ie = 0
		for l,Is in Expertises:
			for I in Is:
				for n in range(N):
					X[t-DEPART][ie][N-n-1] = l[t - n*I]
				ie += 1

	#X = np.array([[[l[t-n*I] for n in range(N)] for l,Is in Expertises for I in Is] for t in range(DEPART, T)])
	Y = np.array([[100*(df['Close'][t+1]/df['Close'][t]-1 if t!=T-1 else 0.0)] for t in range(DEPART, T)])

	print("Séparation de train et test ...")
	X_train, Y_train = X[:-VALIDATION], Y[:-VALIDATION]
	X_test , Y_test  = X[-VALIDATION:], Y[-VALIDATION:]
	print("Données écrites !")

	print(X_train[0])
	print(Y_train[0])

	#print([eval(i).shape for i in ('X_train', 'Y_train', 'X_test', 'Y_test')])

	for la_liste in 'X_train', 'Y_train', 'X_test', 'Y_test':
		with open(la_liste, 'wb') as co:
			arr = eval(la_liste).flatten()
			co.write(st.pack('f'*len(arr), *arr))