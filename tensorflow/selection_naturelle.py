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

def ema(l,K):
	e = [l[0]]
	for a in l[1:]:
		e.append(e[-1]*(1-K) + K*a)
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
	w  = y_true[:, 0:1]
	yh = y_true[:, 1:2]
	#
	y = tf.tanh      (y0)
	h = tf_logistique(y1)
	#
	_y = tf.stop_gradient(y)
	_h = tf.stop_gradient(h)
	#
	Y = tf.pow(tf.sign(w)     - y0, 2) * (0+yh) * _h
	H = tf.pow(tf.sign(w*_y)  - y1, 2) * (1-yh)
	#
	return tf.reduce_mean(Y) + tf.reduce_mean(H)

#	============================================================	#

#	ema(K0)[i*I0] / ema(K1)[i*(I0+I1) - n*(I0+I1+I2)] - 1

from données import btcusdt_1H, btcusdt_15m, eurousdt, CAC_40_Données_Historiques

df, Close, la_Date = btcusdt_15m()

print(df)
print(df[['Close_change', 'Volume_change', 'macd_change']])

VALIDATION = 2048

Informations = 3; infos = ('Close_change', 'Volume_change', 'macd_change')
Expertises = 5
N = 32

MAX_I0 = 64
MAX_n  = 16
DEPART = N*MAX_I0*MAX_n

print(f'DEPART={DEPART}')

assert len(df)-DEPART > 1024

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
		[ courbe.iloc(t - ...) ]<*N>
	]<*(Informations*Extractions)>
]<*(T-DEPART)>
'''

def une_ligne(_df_info, nom, parametres):
	#ema(K0)[t - i*I0] / ema(K1)[t - i*(I0+I1) - n*(I0+I1+I2)] - 1
	#ema(K0)[t - i*I0] / ema(K1)[t - (i-n)*(I0+I1)] - 1
	#ema(K0)[t - i*I0] / ema(K1)[t - i*n*I0] - 1
	#
	a = ema(np.array(_df_info), K=parametres[nom+'_K0'])
	b = ema(np.array(_df_info), K=parametres[nom+'_K1'])
	#
	n = parametres[nom+'_n']
	#
	I0 = parametres[nom+'_I0']
	#I1 = parametres[nom+'_I1']
	#I2 = parametres[nom+'_I2']
	#
	return [[ a[t - i*I0]/b[t - i*n*I0] -1 for i in range(N)] for t in range(DEPART, len(_df_info))]

def entree_sortie_un_model(_df, parametres):
	lignes = [
		une_ligne(_df[infos[info]], f'{info}_{e}', parametres)
		for info in range(Informations) for e in range(Expertises)
	]
	bloques_X = []
	bloques_Y = []
	for t in range(DEPART, len(_df)):
		bloques_X.append([
			lignes[l][t]
			for l in range(len(lignes))
		])
		bloques_Y.append([
			float(_df['Close_change'])
		])
	return np.array(bloques_X), np.array(bloques_Y)

def params_to_compile(_df, parametres):
	X_chaque_mdl, Y_chaque_mdl = [], []
	#
	for p in parametres:
		bloques_X, bloques_Y = entree_sortie_un_model(_df, p)
		X_chaque_mdl += [bloques_X]
		Y_chaque_mdl += [bloques_Y]
	#
	return X_chaque_mdl, Y_chaque_mdl

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
#I1 = list(range( 32))
#I2 = list(range( 16))
n  = list(range(MAX_n))

for inf in range(Informations):
	for e in range(Expertises):
		p(f'{inf}_{e}_K0', 1, K0) #K0
		p(f'{inf}_{e}_K1', 1, K1) #K1
		p(f'{inf}_{e}_I0', 1, I0) #I0
		#p(f'{inf}_{e}_I1', 1, I1) #I1
		#p(f'{inf}_{e}_I2', 0, I2)#I2
		p(f'{inf}_{e}_n' , 1,  n) #n

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
		Conv1D(16, 3, activation=activ0),	#32-2 = 30
		MaxPooling1D(pool_size=3),			#30/3 = 10
		Dropout(dropout0),
		#
		Conv1D(16, 3, activation=activ1),	#10-2 = 8
		Dropout(dropout1),
		#
		Flatten(),
		#
		#DenseDropConnect()
		Dense(64, activation=activ2),
		Dropout(dropout2),
		#x = x + Dense(128)(x)
		#x = x + Dense(128)(x)
		Dense(32, activation=activ3),
		Dropout(dropout3),
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

	bloquesX_par_mdl = [
		
	]
	entree_sortie_un_model(_df, p)
	bloques_X_train, bloques_Y_train = params_to_compile(df, [params for _,params in population])
	#
	X_train, X_test = bloques_X_train[:-VALIDATION], bloques_X_train[-VALIDATION:]
	Y_train, Y_test = bloques_Y_train[:-VALIDATION], bloques_Y_train[-VALIDATION:]
	#
	X_train, X_test = np.array(X_train), np.array(X_test)
	Y_train, Y_test = np.array(Y_train), np.array(Y_test)
	#
	udm = Union_des_Modèles(population)
	udm.model.compile(optimizer=Adam(learning_rate=1e-5), loss=custom_confident_loss)
	#
	print(X_train.shape, Y_train.shape)
	loss, accuracy = udm.model.evaluate(X_test, Y_test)
	print(loss, accuracy)
	#
	history.append(udm.model.fit(
		X_train, Y_train,
		epochs=10, batch_size=128,
		validation_data=(X_test,Y_test)
	))
	score = list(enumerate(udm.model.evaluate(population, X_test, Y_test).numpy()))
	print(scores)
	#
	if (année != ANNEES-1):
		population = nouvelle_population(P, enfants, population, scores)