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

Informations = 3
Expertises = 5
N = 32

#	============================================================	#

class Parametres:
	def __init__(self):
		self.parametres = []

	def cloner(self):
		p = Parametres()
		for val,intervalle in self.parametres: p(val, intervalle)
		return p

	def muter(self):
		raise Exception("Err")

	def __call__(self, nom, valeur, intervalle):
		self.parametres += [(nom, valeur, intervalle)]

	def __getitem__(self, elm):
		return [val for nom,val,_ in self.parametres if nom==elm][0]

def params_to_compile(X_train, Y_train, parametres):
	return (X_test_chaque_mdl, Y_test_chaque_mdl), (X_train_chaque_mdl, Y_train_chaque_mdl)

def cree_le_nouveau_model(mdl_initale, nouveaux_parametres):
	mdl = cree_le_modèle(nouveaux_parametres)
	mdl.set_weights(mdl_initale.get_weights())
	return mdl

def muter_les_params(P, E, params, scores):
	pass

def nouvelle_population(population, scores):
	pass

class Union_des_Modèles():
	def __init__(self, modèles:list):
		self.modèles = modèles

		self.i = []
		self.x = []
		for i in range(len(modèles)):
			self.i += [Input(shape=(I*E*N,), name=f'entrée {i}')]
			x = self.i[-1]
			x = self.modèles[i](x)
			self.x += [x]
		#
		x = Concatenate()(self.x)
		x = Reshape((len(modèles), 3))(x)
		#
		model = Model(inputs=self.i, output=x)

def evaluer(i, sortie):
	return Model(inputs=i, output=sortie).evaluate()

#### Modèle initiale ####

p = Parametres()

for inf in range(Informations):
	for e in range(Expertises):
		p(f'{inf}_{e}_K0', 1.0, ']1.0;100.0[') #K0
		p(f'{inf}_{e}_K1', 1.0, ']1.0;100.0[') #K1
		p(f'{inf}_{e}_I0', 1, '[1;64]') #I0
		p(f'{inf}_{e}_I1', 1, '[1;32]') #I1
		p(f'{inf}_{e}_I2', 0, '[1;16]') #I2

dropout0 = p('dropout0', 0.30, '[0.0; 1.0[')
dropout1 = p('dropout1', 0.30, '[0.0; 1.0[')
dropout2 = p('dropout2', 0.30, '[0.0; 1.0[')
dropout3 = p('dropout3', 0.30, '[0.0; 1.0[')

activ0 = p('activ0', 0, '[0;3]')
activ1 = p('activ1', 0, '[0;3]')
activ2 = p('activ2', 0, '[0;3]')
activ3 = p('activ3', 0, '[0;3]')

#########################

# Strategies :
#	1 : chaque enfant a des params nouveaux, et s'entrainne sur 100 echopes
#	2 : chaque années les enfants heritent des params anciens et s'entrainnent 20 echopes
#	3 : chaque années les enfants héritent de 90% des poids

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
		Conv1D(16, 3, activation=activ0, input_shape=(Informations*Expertises,N)),	#32-2 = 30
		MaxPooling1D(pool_size=3),			#30/3 = 10
		Dropout(dropout0),
		#
		Conv1D(16, 3, activation=activ1),	#10-2 = 8
		Dropout(dropout1),
		#
		#DenseDropConnect()
		Dense(64, activation=activ2),
		Dropout(dropout2),
		#x = x + Dense(128)(x)
		#x = x + Dense(128)(x)
		Dense(32, activation=activ3),
		Dropout(dropout3),
		#
		Dense(2)
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
	(X_test_chaque_mdl, Y_test_chaque_mdl), (X_train_chaque_mdl, Y_train_chaque_mdl) = params_to_compile(
		X_train, Y_train,
		[params for _,params in population]
	)
	#
	udm = Union_des_Modèles(population)
	udm = Model(udm.i, udm.model)
	udm.compile(optimizer=Adam(learning_rate=1e-5), loss=custom_confident_loss)
	#
	print(udm.evaluate(X_test, Y_test, loss=custom_confident_loss))
	#
	history.append(udm.fit(
		X_train, Y_train,
		epochs=10, batch_size=128,
		validation_data=(X_test,Y_test)
	))
	score = list(enumerate(udm.evaluate(population, X_test_chaque_mdl, Y_test_chaque_mdl).numpy()))
	print(scores)
	#
	if (année != ANNEES-1):
		population = nouvelle_population(population, scores)