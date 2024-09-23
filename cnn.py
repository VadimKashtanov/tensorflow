#! /usr/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, Sequential, load_model
from keras.saving import register_keras_serializable
#
from keras import backend as K
from tensorflow.keras.layers import Input, Dropout, Flatten, Permute, Reshape, Lambda
from tensorflow.keras.layers import Dense, Activation, Multiply
from tensorflow.keras.layers import Conv1D, SeparableConv1D, DepthwiseConv1D, Conv1DTranspose, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D
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

@register_keras_serializable()
def t2d(x): return tf.transpose(x, perm=[0, 2, 1])
@register_keras_serializable()
def tos(input_shape): return (input_shape[0], input_shape[2], input_shape[1])

def mul(l):
	a = 1
	for e in l: a*=e
	return a

tf_logistique = lambda x: 1/(1+tf.exp(-         x ))
np_logistique = lambda x: 1/(1+np.exp(-np.array(x)))
logistique    = lambda x: 1/(1+np.exp(-         x ))

def ema(l,K):
	e = [l[0]]
	for a in l[1:]: e.append(e[-1]*(1-K) + K*a)
	return e

@register_keras_serializable()
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
	return (tf.reduce_mean(Y) + tf.reduce_mean(H))/2

#	============================================================	#

from cree_les_données import df, VALIDATION, N, nb_expertises, T, DEPART, SORTIES

X_train = (T-DEPART-VALIDATION, N, nb_expertises)#(T-DEPART-VALIDATION, nb_expertises, N)
Y_train = (T-DEPART-VALIDATION, 1)
X_test  = (         VALIDATION, N, nb_expertises)#(         VALIDATION, nb_expertises, N)
Y_test  = (         VALIDATION, 1)

for la_liste in 'X_train', 'Y_train', 'X_test', 'Y_test':
	with open(la_liste, 'rb') as co:
		bins = co.read()
		exec(f"{la_liste} = np.array(st.unpack('f'*mul({la_liste}), bins)).reshape({la_liste})")

#	============================================================	#

def ffn(M, N):
	return Sequential([
		Dropout(0.20),
		Dense(N, activation='relu'),
		Dropout(0.20),
		Dense(M),
	])

if __name__ == "__main__":
	entree = Input((N, nb_expertises))#Input((nb_expertises, N))
	x = entree
	x = Reshape((N,nb_expertises,1) )(x)
	x = Dense(20, activation='sigmoid')(x)
	x = Reshape((N,nb_expertises,20))(x)
	#
	#x = Dropout(0.10)(x)
	#
	#Conv1D, SeparableConv1D, DepthwiseConv1D, Conv1DTranspose,
	x = Conv2D(32, (3,3))(x)	#8*10 -> 6*8
	#x = Conv1D(32, 3)(x)	#8 -> 6
	x = MaxPooling2D((2,2))(x)	#6*8  -> 3*4
	#x = Dropout(0.10)(x)
	#
	#x = Conv1D(32, 3)(x)		#7 -> 5
	#x = AveragePooling1D(2)(x)	#10 -> 5
	#x = Dropout(0.10)(x)
	#
	x = Flatten()(x)
	#
"""	M = 16
	x = Dense(M)(x)
	x = x + ffn(M, M*2)(x)"""
	#
	R = 32
	x = Dense(R)(x)
	x = ffn(R, R*2)(x)
	#
	x = Dense(SORTIES)(x)

	model = Model(entree, x)
	model.compile(optimizer=Adam(learning_rate=1e-5), loss=custom_loss)
	model.summary()

	############################ Entrainnement #########################

	# Callbacks
	meilleur_validation = ModelCheckpoint('meilleur_model.h5.keras', monitor='val_loss', save_best_only=True)
	meilleur_train      = ModelCheckpoint('dernier__model.h5.keras', monitor='loss'    , save_best_only=True)

	history = model.fit(X_train, Y_train, epochs=100, batch_size=256, validation_data=(X_test,Y_test), shuffle=True,
		callbacks=[meilleur_validation, meilleur_train]
	)

	plt.plot(history.history['loss'    ], label='Train')
	plt.plot(history.history['val_loss'], label='Test ')
	plt.legend()
	plt.show()