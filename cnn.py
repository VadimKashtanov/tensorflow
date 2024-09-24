#! /usr/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.models import Model, Sequential, load_model
from keras.saving import register_keras_serializable
#
from keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input, Dropout, Flatten, Permute, Reshape, Lambda, concatenate
from tensorflow.keras.layers import Dense, Activation, Multiply
from tensorflow.keras.layers import Conv1D, SeparableConv1D, DepthwiseConv1D, Conv1DTranspose, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import LayerNormalization, BatchNormalization
#
from keras_nlp.layers import PositionEmbedding, TransformerEncoder, TransformerDecoder
#
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
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

@register_keras_serializable()
class GaussianActivation(Layer):
	def call(self, x): return tf.exp(-tf.square(x) * 5)

def mul(l):
	a = 1
	for e in l: a*=e
	return a

tf_logistique = lambda x: 1/(1+tf.exp(-         x ))
np_logistique = lambda x: 1/(1+np.exp(-np.array(x)))
logistique    = lambda x: 1/(1+np.exp(-         x ))

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
	#Y = tf.pow(tf.sign(w)     - y0, 2) * tf.abs()# * _h #(0+yh) * _h
	#Y = tf.pow(tf.sign(w) - y0, 2) * 10*tf.abs(w)*_h# * _h #(0+yh) * _h
	#H = tf.pow(tf.sign(w*_y)  - y1, 2 + 2*tf.whitch(tf.sign(w*_y) > 0)) *  1  #(1-yh)
	#
	#Y = tf.pow(tf.sign(w) - y0, 2) * tf.abs(w) * h + tf.pow(1 - h, 2)
	#l = tf.pow(w - tf.stop_gradient(y0), 2)
	# = l / tf.reduce_mean(l)
	#Y = tf.pow(tf.sign(w)-y0, 2)/2 * tf.pow(w, 2)#tf.pow(w - y0, 2)# * tf.abs(w)# * (1 - tf.exp(-l*l))
	#
	#Y = -( tf_logistique(tf.sign(w))*tf.math.log(tf_logistique(y)) + (1-tf_logistique(tf.sign(w)))*tf.math.log(1-tf_logistique(y)))
	Y = tf.pow(tf.sign(w) - y0, 2)/2
	#
	return (tf.reduce_mean(Y))# + tf.reduce_mean(H))/2


#	Position maximale : 5 unitées
#
#		En gros c'est une perte si ça tombe jamais en dessous/dessus de la cible.
#		C'est des position court terme, mais pas de 1 unitées (comme j'ai toujours fais)
#
#	J'avais deja pensé ça y a 1 ans, mais je l'ai pas fais.
#
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

class Bruit(Callback):
	def __init__(self, data, labels):
		super(Bruit, self).__init__()
		self.data = data
		self.labels = labels

	def on_epoch_begin(self, epoch, logs=None):
		self.data = X_train + np.random.normal(0, 0.01, self.data.shape)
		print(f"Époque {epoch+1}: Données modifiées.")

	def on_batch_begin(self, batch, logs=None):
		# Optionnel : Modifie les données batch par batch
		pass

#	============================================================	#

def ffn(M, N):
	return Sequential([
		Dropout(0.20),
		Dense(N, activation='relu'),
		Dropout(0.20),
		Dense(M),
	])

if __name__ == "__main__":
	#
	#	Faire le systeme de trade ouvert 5 (ou 4) unitées de temps
	#
	entree = Input((N, nb_expertises))#Input((nb_expertises, N))
	x = entree
	#
	#x = Dropout(0.10)(x)
	#
	#Conv1D, SeparableConv1D, DepthwiseConv1D, Conv1DTranspose,
	x = Conv1D(64, 7)(x)	#8 -> 4
	#x = x*x
	#x = AveragePooling1D(2)(x)		#8 -> 4
	#
	x = Flatten()(x)
	x = Dropout(0.30)(x)
	#
	x = Dense(128, activation='relu')(x); x = Dropout(0.10)(x)
	#
	x = Dense(SORTIES)(x)

	model = Model(entree, x)
	#model.compile(optimizer=SGD(learning_rate=1e-6), loss=custom_loss)
	model.compile(optimizer=Adam(learning_rate=1e-5), loss=custom_loss)
	model.summary()

	############################ Entrainnement #########################

	# Callbacks
	meilleur_validation = ModelCheckpoint('meilleur_model.h5.keras', monitor='val_loss', save_best_only=True)
	meilleur_train      = ModelCheckpoint('dernier__model.h5.keras', monitor='loss'    , save_best_only=True)
	#
	bruit               = Bruit(X_train, Y_train)
	#
	#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0000001)

	history = model.fit(X_train, Y_train, epochs=200, batch_size=256, validation_data=(X_test,Y_test), shuffle=True,
		callbacks=[
			meilleur_validation, meilleur_train,
			#bruit,
			#reduce_lr
		]
	)

	plt.plot(history.history['loss'    ], label='Train')
	plt.plot(history.history['val_loss'], label='Test ')
	plt.legend()
	plt.show()

	print("======================================================================")

	def display_layer_outputs(model, input_data):
		# Créer un modèle qui retourne les sorties de chaque couche
		layer_outputs = [layer.output for layer in model.layers]
		activation_model = Model(inputs=model.input, outputs=layer_outputs)

		# Obtenir les valeurs de chaque couche
		activations = activation_model.predict(input_data)

		# Afficher les valeurs de chaque couche
		for layer_name, activation in zip([layer.name for layer in model.layers], activations):
			print(f"Layer: {layer_name}, Activation shape: {activation.shape}, Activation values:\n{activation}")

	display_layer_outputs(model, X_train[0:1])

	print("======================================================================")

	for layer in model.layers:
		if 'conv' in layer.name:
			# Obtenir les poids de la couche (les kernels sont dans layer.get_weights()[0])
			kernels = layer.get_weights()[0]
			
			# Normaliser les valeurs des kernels pour les afficher correctement
			min_val = np.min(kernels)
			max_val = np.max(kernels)
			kernels = (kernels - min_val) / (max_val - min_val)

			# Déterminer le nombre de filtres et la taille des kernels
			num_filters = kernels.shape[-1]
			kernel_size = kernels.shape[0]

			# Créer un plot pour afficher les filtres
			A = int(1 + num_filters**.5)
			fig, ax = plt.subplots(A, A)
			
			for i in range(num_filters):
				for ie in range(nb_expertises):
					ax[i//A][i%A].plot(kernels[:, ie, i])
			plt.show()