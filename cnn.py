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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
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
	Y = tf.pow(tf.sign(w)     - y0, 2)# * _h #(0+yh) * _h
	H = 0*tf.pow(tf.sign(w*_y)  - y1, 2)# *  1 #(1-yh)
	#
	return (tf.reduce_mean(Y) + tf.reduce_mean(H))/2

#	============================================================	#

from cree_les_données import df, VALIDATION, N, nb_expertises, T, DEPART, SORTIES

X_train = (T-DEPART-VALIDATION, nb_expertises, N)
Y_train = (T-DEPART-VALIDATION, 1)
X_test  = (         VALIDATION, nb_expertises, N)
Y_test  = (         VALIDATION, 1)

for la_liste in 'X_train', 'Y_train', 'X_test', 'Y_test':
	with open(la_liste, 'rb') as co:
		bins = co.read()
		exec(f"{la_liste} = np.array(st.unpack('f'*mul({la_liste}), bins)).reshape({la_liste})")

#	============================================================	#

if __name__ == "__main__":
	entree = Input((nb_expertises, N))
	x = entree
	#
	#
	x = Lambda(t2d, output_shape=tos)(x)
	x = Conv1D(128, 5)(x)		#32 -> 28
	x = AveragePooling1D(2)(x)	#28 -> 14
	x = Lambda(t2d, output_shape=tos)(x)
	x = Dropout(0.20)(x)
	#
	#
	x = Lambda(t2d, output_shape=tos)(x)
	x = Conv1D(32, 3)(x)		#14 -> 12
	x = AveragePooling1D(2)(x)	#12 -> 6
	x = Lambda(t2d, output_shape=tos)(x)
	x = Dropout(0.20)(x)
	#
	#
	x = Flatten()(x)
	#
	x = Dense(256, activation='relu')(x); x = Dropout(0.50)(x)
	x = Dense(64)(x); x = Dropout(0.30)(x)
	x = Dense(SORTIES)(x)

	model = Model(entree, x)
	model.compile(optimizer=Adam(learning_rate=1e-5), loss=custom_loss)
	model.summary()

	############################ Entrainnement #########################

	# Callbacks
	meilleur_validation = ModelCheckpoint('meilleur_model.h5.keras', monitor='val_loss', save_best_only=True)
	meilleur_train      = ModelCheckpoint('dernier__model.h5.keras', monitor='loss'    , save_best_only=True)

	history = model.fit(X_train, Y_train, epochs=50, batch_size=128, validation_data=(X_test,Y_test), shuffle=True,
		callbacks=[meilleur_validation, meilleur_train]
	)

	plt.plot(history.history['loss'    ], label='Train')
	plt.plot(history.history['val_loss'], label='Test ')
	plt.legend()
	plt.show()