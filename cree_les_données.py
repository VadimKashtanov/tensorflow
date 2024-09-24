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

def ema(l,K):
	e = [l[0]]
	for a in l[1:]: e.append(e[-1]*(1-1/K) + a/K)
	return np.array(e)

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

infos = 'OUnix','CUnix','ODate','CDate','Symbol','Open','High','Low','Close','qaV','trades','btcVol','usdtVol'

VALIDATION = 2048

"""Expertises = [
	[60*(df['Close']/df['Close'].ewm(com=5   ).mean()-1),	(1,        ),],
	[40*(df['Close']/df['Close'].ewm(com=25  ).mean()-1),	(1,8       ),],
	[20*(df['Close']/df['Close'].ewm(com=250 ).mean()-1),	(1,8,64,   ),],
	[10*(df['Close']/df['Close'].ewm(com=1000).mean()-1),	(1,8,64,256),],
	#
	[ .2*(df['trades']/df['trades'].ewm(com=5   ).mean()),	(1,8       ),],
	[ .1*(df['trades']/df['trades'].ewm(com=100 ).mean()),	(1,8,64    ),],
	[ .1*(df['trades']/df['trades'].ewm(com=1000).mean()),	(1,8,64,256),],
]"""

contextualiser = lambda arr, _ema: (arr - _ema)/(arr + _ema)

N = 8
MAX_I    = 8
MAX_ROLL = N*MAX_I

Expertises = [
	#[df['Close']/ema(df['Close'], K=1  ) -1,	(1,  ),],
	#[100*(df['Close']/ema(df['Close'], K=1.5  ) -1),	(1,  ),],
	[1000*(df['Close']/ema(df['Close'], K=2  ) -1),	(1,  ),],
	#[100*(df['Close']/ema(df['Close'], K=5  ) -1),	(1,  ),],
	#[100*(df['Close']/ema(df['Close'], K=10  ) -1),	(1,  ),],

	#[20*(df['Close']/df['Close'].ewm(com=250 ).mean()-1),	(64, ),],
	#[10*(df['Close']/df['Close'].ewm(com=1000).mean()-1),	(256,),],
]
for i in range(len(Expertises)): Expertises[i][0] = list(Expertises[i][0])[MAX_ROLL:]

#montrer(Expertises[6], 3, N)
nb_expertises = sum([1 for _,e in Expertises for i in e])
print(f"Expertises : {nb_expertises}")

if __name__ == "__main__":
	fig,ax = plt.subplots(2,2)
	L = 300
	f = lambda x: x*(x>0)
	for j,(l,_) in enumerate(Expertises): ax[0][0].plot(l[-L:], label=f'{j}')
	for j,(l,_) in enumerate(Expertises): ax[0][1].plot(np.convolve(l[-L:], np.array([-1, -1, 3, 0, 0]), "same"), label=f'{j}')
	for j,(l,_) in enumerate(Expertises): ax[1][1].plot(f(np.convolve(l[-L:], np.array([-1, -1, 3, 0, 0]), "same")), label=f'{j}')
	ax[0][0].legend()
	ax[1][0].plot(np.array(df['Close'][MAX_ROLL:][-L:]))
	for i in range(2):
		for j in range(2):
			ax[i][j].grid(True)
	plt.show()
	#exit(0)
	#
	A = int(1+(len(Expertises))**.5)
	fig, ax = plt.subplots(A,A)
	for i,(l,_) in enumerate(Expertises): ax[i//A][i%A].plot(l)
	plt.show()

T      = len(Expertises[0][0])
DEPART = N * MAX_I

assert T-DEPART > VALIDATION

print(f'DEPART={DEPART} T={T} VALIDATION={VALIDATION}')
print(f'Train T = {T-DEPART-VALIDATION}')
print(f'Test  T = {VALIDATION}')

SORTIES = 2

#	============================================================	#
if __name__ == "__main__":
	print("Création des données ...")

	#X = np.zeros((T-DEPART, nb_expertises, N))
	X = np.zeros((T-DEPART, N, nb_expertises))

	for t in tqdm(range(DEPART,T), desc="Création des données : ", ncols=100, bar_format="{l_bar}{bar:20}{r_bar}", colour="green"):#range(DEPART, T):
		ie = 0
		#	Transpose (infos*expertises) <-> N (car conv1d prend les choses a l'envers)
		for l,Is in Expertises:
			for I in Is:
				for n in range(N):
					#X[t-DEPART][ie][N-n-1] = l[t - n*I]
					X[t-DEPART][N-n-1][ie] = l[t - n*I]
				ie += 1

	#X = np.array([[[l[t-n*I] for n in range(N)] for l,Is in Expertises for I in Is] for t in range(DEPART, T)])
	Y = np.array([[100*(df['Close'][t+1]/df['Close'][t]-1 if t!=T-1 else 0.0)] for t in range(DEPART, T)])

	#	================== Montrer ================

	plt.imshow(X[0]); plt.show()

	A = int(1+(nb_expertises)**.5)
	fig, ax = plt.subplots(A,A)
	for i in range(nb_expertises): ax[i//A][i%A].plot([X[0][n][i] for n in range(N)])
	plt.show()

	#	===========================================

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