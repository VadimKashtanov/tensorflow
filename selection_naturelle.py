#! /usr/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, Sequential
#
from tensorflow.keras.layers import Input, Dropout, Flatten, Permute, Reshape, Lambda, Concatenate
from tensorflow.keras.layers import Dense, Activation, Multiply
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import LayerNormalization
#
from keras_nlp.layers import PositionEmbedding, TransformerEncoder, TransformerDecoder
#
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
#
from random import randint, random, shuffle
from math import log, tanh
from os import system
import struct as st

tf_logistique = lambda x: 1/(1+tf.exp(-         x ))
np_logistique = lambda x: 1/(1+np.exp(-np.array(x)))
logistique    = lambda x: 1/(1+np.exp(-         x ))

def ema(l,K):
	e = [l[0]]
	for a in l[1:]:
		e.append(e[-1]*(1-1/K) + a/K)
	return e

#	============================================================	#

def algorithme_de_selection(elements, P, enfants):
	return [
		(i,i) for i in range(P)
	] + [
		(p,P+p*enfants+e) for p in range(P) for e in range(enfants)
	]

#	============================================================	#

def custom_confident_loss(y_true, y_pred):
	y0 = y_pred[:, 0:1]
	y1 = y_pred[:, 1:2]
	#
	w  = y_true#[:, 0:1]
	#yh = y_true[:, 1:2]
	#
	y = tf.tanh      (y0)
	h = tf_logistique(y1)
	#
	_y = tf.stop_gradient(y)
	_h = tf.stop_gradient(h)
	#
	Y = tf.pow(tf.sign(w)     - y0, 2) * _h#(0+yh) * _h
	H = tf.pow(tf.sign(w*_y)  - y1, 2) *  1#(1-yh)
	#
	return tf.reduce_mean(Y) + tf.reduce_mean(H)

#	============================================================	#

#	ema(K0)[i*I0] / ema(K1)[i*(I0+I1) - n*(I0+I1+I2)] - 1

from données import binance_btcusdt_15m, bitget_btcusdt_1H, bitget_btcusdt_15m, eurousdt, CAC_40_Données_Historiques

df, Close, la_Date = binance_btcusdt_15m()

df = df[1:].reset_index()

#	Là ou c'est = 0
print(df[['Close_change', 'Volume_change', 'macd_change']][(df[['Close_change', 'Volume_change', 'macd_change']] == 0).any(axis=1)])

print(df)
print(df[['Close_change', 'Volume_change', 'macd_change']])

VALIDATION = 2048

Informations = 3; infos = ('Close_change', 'Volume_change', 'macd_change')
Expertises = 5
N = 32

MAX_I0 = 128
MAX_I1 = 4
DEPART = (1+N)*MAX_I0*MAX_I1

print(f'DEPART={DEPART}')

assert len(df)-DEPART > 1024+VALIDATION

SORTIES = 2

#	============================================================	#

class Parametres:
	def __init__(self):
		self.parametres = []

	def cloner(self):
		p = Parametres()
		for nom,val,valeurs_possibles in self.parametres: p(nom,val, valeurs_possibles)
		return p

	def muter(self):
		i = randint(0, len(self)-1)
		(nom, valeur, valeurs_possibles) = self.parametres[i]
		#
		j = valeurs_possibles.index(valeur)
		#
		choix = [-2,-1,+1,-2,  -10,+10]
		#
		nouveau_j = max(min(j+choice(choix), len(self.parametres)-1), 0)
		#
		nouvelle_valeur = valeurs_possibles[nouveau_j]
		#
		self.parametres[i] = (nom, nouvelle_valeur, valeurs_possibles)
		return self

	def __len__(self):
		return len(self.parametres)

	def __call__(self, nom, valeur, valeurs_possibles):
		self.parametres += [(nom, valeur, valeurs_possibles)]

	def __getitem__(self, elm):
		return [val for nom,val,_ in self.parametres if nom==elm][0]

def melanger_deux_params(p0, p1):
	p = Parametres()
	for i in range(len(p0)):
		if random()>.5: p(p0.parametres[i])
		else :			p(p1.parametres[i])
	return p

def cree_le_nouveau_model(mdl_initale, nouveaux_parametres):
	mdl = cree_le_modèle(nouveaux_parametres)
	mdl.set_weights(mdl_initale.get_weights())
	return mdl

def nouvelle_population(P, enfants, population, scores):
	scores = sorted(enumerate(scores), lambda x: x[1])

	elements = [s for s,_ in scores]
	regle = algorithme_de_selection(elements, P, enfants)

	nouveaux_parametres = [
		melanger_deux_params(
			population[ scores[r0] ][1],
			population[ scores[r1] ][1]
		).muter().muter().muter()
		for r0,r1 in regle
	]

	models = [
		population[scores[elm]][0]
		for elm in range(P*(1+enfants))
	]

	return [
		(cree_le_nouveau_model(models[elm], nouveaux_parametres[elm]), nouveaux_parametres[elm])
		for elm in range(P*(1+enfants))
	]

###########################################################################

'''	bloque = [
	[
		[ emaK0(t - ...)/emaK1(t - ...) ]<*N>
	]<*(Informations*Extractions)>
]<*(T-DEPART)>
'''

"""def entrées_un_model(parametres):
	_I0, _I1, _ema_K0, _ema_K1 = [], [], [], []
	#
	for info,nom_info in enumerate(infos):
		_df_info = list(df[nom_info])
		for e in range(Expertises):
			I0 = parametres[f'{info}_{e}_I0']
			I1 = parametres[f'{info}_{e}_I1']
			K0 = parametres[f'{info}_{e}_K0']
			K1 = parametres[f'{info}_{e}_K1']
			#
			#_ema_K0 += [ ema(_df_info, K0) ]
			#_ema_K1 += [ ema(_df_info, K1) ]
			_I0 += [I0]
			_I1 += [I1]
	#
	ret = []
	for t in range(DEPART, len(df)):
		ret += [\
				[i
					[
						ema_K0[t - i*I0]  /  ema_K1[t - i*I0*I1] - 1
					for i in range(N)]
				for (I0,I1, ema_K0,ema_K1) in zip(_I0,_I1, _ema_K0,_ema_K1)]
		]
	return ret"""

ecrireI = lambda co, i: co.write(st.pack('I', i))
ecriref = lambda co, f: co.write(st.pack('f'*len(f), *f))

def entrées_un_model(params:list):
	with open('instructions_rapiditée', 'wb') as co:
		ecrireI(co, Informations)
		ecrireI(co, Expertises)
		ecrireI(co, len(df))
		ecrireI(co, N)
		ecrireI(co, DEPART)
		ecrireI(co, VALIDATION)
		for i in infos:
			ecriref(co, df[i])
		#
		ecrireI(co, len(params))
		for m in range(len(params)):
			for i in range(Informations):
				for e in range(Expertises):
					I0 = params[m][f'{i}_{e}_I0']
					I1 = params[m][f'{i}_{e}_I1']
					K0 = params[m][f'{i}_{e}_K0']
					K1 = params[m][f'{i}_{e}_K1']
					ecrireI(co, K0)
					ecrireI(co, K1)
					ecrireI(co, I0)
					ecrireI(co, I1)
	#
	status = system("./creation_de_données")
	print(f'status = {status}')
	#
	ret_train = []
	ret_test  = []
	for m in range(len(params)):
		with open(f'X_bloques_par_mdl_{m}_train', 'rb') as co:
			_T = (len(df) - DEPART - VALIDATION)
			_len = _T*N*(Informations*Expertises)
			#
			bins = co.read()
			arr = st.unpack('f'*_len, bins)
			arr = np.array(arr).reshape((_T, Informations*Expertises, N))
			ret_train += [arr]
		system(f'rm X_bloques_par_mdl_{m}_train')
		with open(f'X_bloques_par_mdl_{m}_test', 'rb') as co:
			_T = VALIDATION
			_len = _T*N*(Informations*Expertises)
			#
			bins = co.read()
			arr = st.unpack('f'*_len, bins)
			arr = np.array(arr).reshape((_T, Informations*Expertises, N))
			ret_test += [arr]
		system(f'rm X_bloques_par_mdl_{m}_test')
	#
	print("Données écrites")
	return ret_train, ret_test

######################################################

def sorties_un_model():
	T = len(df)
	ret_train = np.array( list(df['Close_change'][DEPART+1:T-VALIDATION+1])        ).reshape((T-DEPART-VALIDATION,1,))
	ret_test  = np.array( list(df['Close_change'][         T-VALIDATION+1:T])+[0.0]).reshape((VALIDATION,1))
	return ret_train, ret_test

######################################################

class Union_des_Modèles():
	def __init__(self, modèles:list):
		self.modèles = modèles

		self.i = []
		self.x = []
		for i in range(len(modèles)):
			self.i += [Input(shape=(Informations*Expertises,N), name=f'entrée {i}')]
			self.x += [
				self.modèles[i][0]( self.i[-1] )
			]
		#
		x = Concatenate()(self.x)
		x = Reshape((len(modèles), SORTIES))(x)
		#
		#print(x)
		self.model = Model(self.i, x)
		self.model.summary()

#### Modèle initiale ####

p = Parametres()

K0 = list(range(200))
K1 = list(range(200))
I0 = list(range(MAX_I0))
I1 = list(range(MAX_I1))
#I2 = list(range( 16))
#n  = list(range(MAX_n))

for inf in range(Informations):
	for e in range(Expertises):
		p(f'{inf}_{e}_K0', 1, K0)	#K0
		p(f'{inf}_{e}_K1', 1, K1)	#K1
		p(f'{inf}_{e}_I0', 1, I0)	#I0
		p(f'{inf}_{e}_I1', 1, I1)	#I1
#		p(f'{inf}_{e}_I2', 0, I2)	#I2
#		p(f'{inf}_{e}_n' , 1,  n)	#n

dropout = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

dropout0 = p('dropout0', 0.30, dropout)
dropout1 = p('dropout1', 0.30, dropout)
dropout2 = p('dropout2', 0.30, dropout)
dropout3 = p('dropout3', 0.30, dropout)

activs = [0,1,2,3]

activ0 = p('activ0', 0, activs)
activ1 = p('activ1', 0, activs)
activ2 = p('activ2', 0, activs)
activ3 = p('activ3', 0, activs)

#########################

def cree_le_modèle(p:Parametres):
	#
	dropout0 = p['dropout0']
	dropout1 = p['dropout1']
	dropout2 = p['dropout2']
	dropout3 = p['dropout3']
	#
	activ0 = ['relu', 'gelu', 'linear', 'logistic'][ p['activ0']]
	activ1 = ['relu', 'gelu', 'linear', 'logistic'][ p['activ1']]
	activ2 = ['relu', 'gelu', 'linear', 'logistic'][ p['activ2']]
	activ3 = ['relu', 'gelu', 'linear', 'logistic'][ p['activ3']]
	#
	return Sequential([
		Input((Informations*Expertises, N)),
		#
		#Conv1D(16, 3, activation=activ0),	#32-2 = 30
		#MaxPooling1D(pool_size=3),			#30/3 = 10
		#Dropout(dropout0),
		#
		#Conv1D(16, 3, activation=activ1),	#10-2 = 8
		#Dropout(dropout1),
		#
		Flatten(),
		#
		#DenseDropConnect()
		#Dense(64, activation=activ2),
		#Dropout(dropout2),
		#x = x + Dense(128)(x)
		#x = x + Dense(128)(x)
		#Dense(32, activation=activ3),
		#Dropout(dropout3),
		#
		Dense(SORTIES)
	])

mdl_initale = cree_le_modèle(p)
parametres_initiaux = p

#########################

P = 3
enfants = 2

population = [
	(cree_le_nouveau_model(mdl_initale, parametres_initiaux), parametres_initiaux.cloner())
	for _ in range(P*(1+enfants))
]

history = []

for année in range(ANNEES:=10):
	pop_X_train, pop_X_test = entrées_un_model([params for _,params in population]) #j'ai pas finie le .cu
	Y_____train, Y_____test = sorties_un_model()		#il faut faire le split train, validation
	print("[0] Données écrites")					#Y n'est pas *len(pop) tf.const([y0,y1,y2..]) - tf.cont([w])
	#
	udm = Union_des_Modèles(population)
	udm.model.compile(optimizer=Adam(learning_rate=1e-5), loss=custom_confident_loss)
	#
	print(pop_X_train[0][0])
	#
	print(pop_X_train[0].shape, Y_____train.shape)
	loss = udm.model.evaluate(pop_X_test, Y_____test)
	print(loss)
	pred = udm.model.predict(pop_X_test)
	print(pred)
	#
	history.append(udm.model.fit(
		pop_X_train, Y_____train,
		epochs=10, batch_size=128,
		validation_data=(pop_X_test,Y_____test)
	))
	score = list(enumerate(udm.model.evaluate(population, pop_X_test, Y_____test).numpy()))
	print(scores)
	#
	if (année != ANNEES-1):
		population = nouvelle_population(P, enfants, population, scores)