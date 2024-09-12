import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_nlp
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Flatten, Permute, Reshape, Lambda, Activation, Multiply
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from random import randint, random, shuffle
from math import log, tanh
from os import system

logistique =

def ema(l,K):
	e = [l[0]]
	for a in l[1:]:
		e.append(e[-1]*(1-K) + K*a)
	return e

#PRIORITEE_Y = 
#PRIORITEE_H = 0.70

def custom_confident_loss(y_true, y_pred):
	y0 = y_pred[:, 0:1]
	y1 = y_pred[:, 1:2]
	w  = y_true[:, 0:1]
	#
	y = y0
	h = 1 / (1 + tf.exp(-1*y1))
	w = w
	#
	_y = tf.stop_gradient(y)
	_h = tf.stop_gradient(h)
	#
	l = (1+tf.sign(w*_y))/2
	#
	#o = abs(w)/(1 + abs(w-_y))
	#o = len(o)*o/sum(o)
	#
	dy  = -_h*(w-y)#*abs(w)
	dh  = --(l - h)
	#dh += S - sum(h)
	#	-- Intégration des dy, dh --
	Y = _h*tf.square(w-y)
	H = tf.square(l-h)
	#
	return tf.reduce_mean(Y) + tf.reduce_mean(H)

"""	Le modèle choisis lui même si il préfère prendre plus de petites delta que des gros.
	y = pred w  [   -inf       ;       +inf   ]
	h = %invest [     0%       ;     100%     ]
	c = choix   [petits deltas ; grands deltas]
"""

def custom_confident_loss(y_true, y_pred):
	y0 = y_pred[:, 0:1]
	y1 = y_pred[:, 1:2]
	w  = y_true[:, 0:1]
	#
	y = y0
	h = 1 / (1 + tf.exp(-1*y1))
	w = w
	#
	_y = tf.stop_gradient(y)
	_h = tf.stop_gradient(h)
	#
	l = (1+tf.sign(w*_y))/2
	#
	#	-- Intégration des dy, dh --
	Y = _h*tf.square(w-y)
	H = tf.square(tf.sign(w*_y)-y1)*tf.abs(w)
	#
	return tf.reduce_mean(Y) + tf.reduce_mean(H)

def custom_confident_loss(y_true, y_pred):
	y0 = y_pred[:, 0:1]
	y1 = y_pred[:, 1:2]
	w  = y_true[:, 0:1]
	#
	y = tf.tanh(y0)
	h = 1 / (1 + tf.exp(-y1))
	w = w
	#
	_y = tf.stop_gradient(y)
	_h = tf.stop_gradient(h)
	#
	l = (1+tf.sign(w*_y))/2
	#
	#o = abs(w)/(1 + abs(w-_y))
	#print(l.shape)
	#
	#	-- Intégration des dy, dh --
	"""T = (_y.shape[1] if _y.shape[1] != None else 1)
	alterner = tf.math.mod(tf.range(T), 2)
	alterner = tf.reshape(alterner, _y.shape)
	alterner = tf.cast(alterner, tf.float32)"""
	#
	Y = _h*tf.square(tf.sign(w)-y)                      # * (  alterner)
	H = tf.square(tf.sign(w*_y)-y1) * tf.abs(w)# * (1-alterner)
	#dH = (2*x*(x<0.5)+2*(x-1)*(x>0.5))+2*(sum(x)-0.3)/len(x)
	H += tf.where(h <= 0.5, 2 * h, 2 * (1-h)) + tf.square(tf.reduce_mean(h) - 0.10)
	#
	#
	return tf.reduce_mean(Y) + tf.reduce_mean(H)# + tf.square(0.7 - tf.reduce_mean(h))

###########################################################################################################
#######################################         Couches       #############################################
###########################################################################################################

class DropConnectDense(tf.keras.layers.Layer):
	def __init__(self, units, dropconnect_rate=0.0, activation=None, **kwargs):
		super(DropConnectDense, self).__init__(**kwargs)
		self.units = units
		self.dropconnect_rate = dropconnect_rate
		self.dense_layer = tf.keras.layers.Dense(units, activation=None)  # Activation is applied later
		self.activation = tf.keras.activations.get(activation)  # Retrieve activation function

	def call(self, inputs, training=False):
		weights, biases = self.dense_layer.kernel, self.dense_layer.bias

		if training:
			# Apply DropConnect to the weights only during training
			dropconnect_mask = tf.random.uniform(tf.shape(weights)) >= self.dropconnect_rate
			weights = tf.where(dropconnect_mask, weights, tf.zeros_like(weights))

		outputs = tf.matmul(inputs, weights) + biases
		
		if self.activation:
			outputs = self.activation(outputs)  # Apply the activation function

		return outputs

	def build(self, input_shape):
		self.dense_layer.build(input_shape)
		super(DropConnectDense, self).build(input_shape)

	def get_config(self):
		config = super(DropConnectDense, self).get_config()
		config.update({
			'units': self.units,
			'dropconnect_rate': self.dropconnect_rate,
			'activation': tf.keras.activations.serialize(self.activation)
		})
		return config

def DenseT(*args, **kargs):
	return tf.keras.Sequential([
		Permute((2,1)),
		Dense(*args, **kargs),
		Permute((2,1))
	])

def FFN(d, *args, **kargs):
	return Sequential([
		Dense(d*2, activation='gelu', *args, **kargs),
		Dense(d,   *args, **kargs)
	])

def Dense_reshape(nouvelle_dimention, d, *args, **kargs):
	return Sequential([
		Dense(d, *args, **kargs),
		Reshape(nouvelle_dimention)
	])

"""def Moe(Expertises, *args, **kargs):
	expertises = Dense(128*4, use_bias=False)(x)
	expertises = Reshape((4,128))(expertises)
	Dense_reshape
	___softmax = Dense(4, use_bias=False, activation='softmax')(x)
	___softmax = Reshape((4,  1))(___softmax)
	x = Multiply()([expertises, ___softmax])
	x = Lambda(lambda z: tf.reduce_sum(z, axis=0))(x)
	return Sequential([
		Dense_reshape((Expertises,128), 128*Expertises, *args, **kargs),
		Dense_reshape((Expertises,  1),   1*Expertises, *args, **kargs),
	])"""

class Moe(Model):
	def __init__(self, d, Expertises, *args, **kargs):
		self.d = d
		self.Expertises = Expertises
		self.args = args
		self.kargs = kargs

	def __call__(self, x):
		d, Expertises = self.d, self.Expertises
		#
		expertises = Dense(Expertises*d, *self.args, **self.kargs)(x)
		expertises = Reshape((Expertises, d))(expertises)
		#
		___softmax = Dense(Expertises*1, activation='softmax')(x)
		___softmax = Reshape((Expertises, 1))(___softmax)
		#
		x = Multiply()([expertises, ___softmax])
		x = Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)
		return x

#	=================================================================================
#	===================================== Dar =======================================
#	=================================================================================

from données import btcusdt_1H, btcusdt_15m, eurousdt, CAC_40_Données_Historiques

#	------ Choix du CSV ------
df, Close, la_Date = btcusdt_15m()#btcusdt_1H()#eurousdt()

print(df)

#exit(0)

#	====================== One Hot ====================

L = 16 + 1 + 16

one_hot = lambda l,L: [l[i*int(   len(l)/(L-1)  )] for i in range(L-1)]+[l[-1]]
copier  = lambda l: [i for i in l]

#	One-hot : 	Chaque valeur, va etre transformée en un vecteur <L>.
#				Le point duquel elle est le plus proche sera = 1. Le reste du vecteur sera = 0
parties = ('Close_change',)#, 'Volume_change')
frontieres_des_one_hot = {
	partie : one_hot(sorted(copier(df[partie])), L) for partie in parties #xL (L frontières)
}

print(" ===== frontieres_des_one_hot ===== ")
for partie in parties:
	print(f'{partie} : {[round(frontiere,3) for frontiere in frontieres_des_one_hot[partie]]}')

def nombre_vers_distance(x,d):
	distances = [abs(x-_d) for _d in d]
	indexe = distances.index(min(distances))
	return [(1.0 if indexe==i else 0.0) for i in range(len(d))]

#  Un bloque de données est : Mot*Mots = (L*parties) * N
N = 32#32#64

X = []
Y = []

INTERV = [1]#,5]#,25]

sources = {
	I : {
		partie : ema([df[partie].values[i] for i in range(len(df[partie]))], K=1/I)
		for partie in parties
	}
	for I in INTERV
}

for i in range(N*max(INTERV), len(df)):
	_x = []
	for I in INTERV:
		for n in range(N):
			_l_ = []
			for partie in parties:
				_l_ += nombre_vers_distance(sources[I][partie][i-n*I], frontieres_des_one_hot[partie])
			#
			_x.append(_l_)	#append un mot
	#
	_y = [(0 if i==len(df)-1 else df['Close_change'].values[i+1])]
	#
	X.append(_x)
	Y.append(_y)

print(X[0])
print(Y[0])

LIGNES_TEST = 2048#4096

T = len(X)#LIGNES_TEST + 365*24 * (5+4+1)
X = np.array(X)[-T:]
Y = np.array(Y)[-T:]

X_test, X_train = X[-LIGNES_TEST:], X[:-LIGNES_TEST] 
Y_test, Y_train = Y[-LIGNES_TEST:], Y[:-LIGNES_TEST]

print(f" ==== X test ====                X_train.shape={X_train.shape}")
print(X_test)
print(f" ==== Y test ====                Y_train.shape={Y_train.shape}")
print(Y_test)

##################################################################################
###############################      Modèle     ##################################
##################################################################################

d_model = 32#64#128
dropout = 0.3
mots    = L*len(parties)
head_size = 16
num_heads = 8
ff_dim    = d_model*2

entree = Input(shape=(N*len(INTERV), L*len(parties)))
#x = Dense (d_model)(entree)
x = entree
#
#
"""x = Dropout(0.05)(x)
x = Dense (32*2, activation='gelu')(entree)
#x = Dropout(0.10)(x)
x = DropConnectDense(32, dropconnect_rate=0.20)(x)"""
#x = Dense (d_model)(x)
#
#x = LayerNormalization(epsilon=1e-6)(x)
"""
x = transformer_encoder(x, d_model, mots, head_size, num_heads, ff_dim, dropout, normale=True)"""
#
#x = keras_nlp.layers.TransformerEncoder(intermediate_dim=head_size, num_heads=num_heads, activation="gelu")(x)
#
x = Flatten()(x)
#x = DropConnectDense(units=256, prob=0.3, activation="tanh", use_bias=True)(x)
#x = DropConnectDense(128, dropconnect_rate=0.3, activation='tanh')(x)
#x = Dropout(0.5)(x)
#x = Dropout(0.35)(x)
#x = DropConnectDense(units=512, prob=0.05, activation='gelu')(x)
#x = Dense(512, activation='gelu')(x)#; x = Dropout(0.20)(x)
#x = DropConnectDense(64, dropconnect_rate=0.30)(x)
x = Dense(256)(x); x = Dropout(0.30)(x)
x = Dense(128)(x); x = Dropout(0.30)(x)
#
#x = Moe(d=128, Expertises=4)(x)
#
#x = LayerNormalization(epsilon=1e-6)(x)
#
#x = Dense(128, use_bias=False)(x); x = Dropout(0.2)(x)
#x = Dense(128, activation='gelu', use_bias=False)(x)#, kernel_regularizer=l2(0.001), use_bias=False)(x)
#x = Dropout(0.3)(x)
#
#x = Moe(d=2, Expertises=4, use_bias=False)(x)
x = Dense(2, use_bias=False)(x)#, kernel_regularizer=l2(0.001), use_bias=False)(x)
#x = DropConnectDense(units=2, prob=0.3, use_bias=True)(x)

#	-------------------------
#
model = Model(entree, x)
model.compile(optimizer=Adam(learning_rate=1e-5), loss=custom_confident_loss) #'mse'
model.summary()

############################ Entrainnement #########################

history = model.fit(X_train, Y_train, epochs=300, batch_size=64, validation_data=(X_test,Y_test), shuffle=True)

plt.plot(history.history['loss'],     label='Perte d\'entraînement')
plt.plot(history.history['val_loss'], label='Perte de validation')
plt.legend()
plt.show()

############################    Tester     #########################

Y_pred = model.predict(X_test)

print(Y_test)
print(Y_pred)

Y_test          = list(i for i,  in Y_test)
predictions     = list(y for y,h in Y_pred)
investissements = list(h for y,h in Y_pred)

plt.plot(Y_test,          label='Valeurs réelles',  color='blue')
plt.plot(predictions,     label='Prédictions',      color='red')
plt.plot(investissements, label='% Investissement', color='green')

#	Flèches
for t in range(LIGNES_TEST):
	pred       = predictions[t]
	point_prix = Y_test     [t]
	if pred >= 0.0: plt.plot([t, t+1], [point_prix, point_prix + 0.03], 'g')
	else:           plt.plot([t, t+1], [point_prix, point_prix - 0.03], 'r')
#	Mini Fleches --- 
for t in range(1,LIGNES_TEST):
	pred       = predictions[t]-predictions[t-1]
	point_prix = Y_test     [t]
	if pred >= 0.0: plt.plot([t, t], [point_prix, point_prix + 0.05], 'g')
	else:           plt.plot([t, t], [point_prix, point_prix - 0.05], 'r')
#
plt.xlabel('Index')
plt.ylabel('Pourcentage de changement du prix de clôture')
plt.title('Prédictions vs Valeurs Réelles')
plt.legend()
plt.show()

####################################################################
############################    Tester     #########################
####################################################################

prixs = fermetures = list(df[Close  ][-LIGNES_TEST:])
dates =              list(df[la_Date][-LIGNES_TEST:])

signe = lambda x: (1 if x>=0 else -1)

print(" Y_test     Y_pred(signe)   Y_pred(invest)   Prixs     %Prochain   Date")
invest_pourcent = 0
for t in range(LIGNES_TEST):
	w,y,h = Y_test[t], predictions[t], investissements[t]
	#
	invest_pourcent += logistic(h)
	#
	p1   = (0 if t==LIGNES_TEST-1 else prixs[t+1])
	p0   = prixs[ t ]
	#
	ch = p1/p0 - 1
	#
	print("%+7f | %+7f %+7f(%+7f) | %+7f$ %6s%% | %s %s" % (
		w,
		#
		tanh(y),
		logistic(h),h,
		#
		prixs[t],
		(str(ch*100)[:5] if t != LIGNES_TEST-1 else '?'),
		#
		dates[t],
		(f'\033[92m +$ \033[0m' if signe(w) == signe(y) else '\033[91m -$ \033[0m')
	))

print(f"Investissement moyen {round(100*invest_pourcent/LIGNES_TEST,3)} %")

##################################################################################
############################          Gains      #################################
##################################################################################

for sng in [+1]:#,-1]:

	fig, ax = plt.subplots(2,2)

	h_max = (1+max(investissements))/2

	for L in [1,10,25,50,125]:
		u0 = 100
		u1 = 100
		u2 = 100
		u3 = 100
		_u0 = []
		_u1 = []
		_u2 = []
		_u3 = []
		for t in range(LIGNES_TEST-1):
			w,y,h = Y_test[t], tanh(predictions[t]), logistique(investissements[t])
			p1   = prixs[t+1]
			p0   = prixs[ t ]
			#
			u0 += sng * u0 * L *       y  * (p1/p0-1) * h# / h_max
			u1 += sng * u1 * L *       y  * (p1/p0-1) * 1
			u2 += sng * u2 * L * signe(y) * (p1/p0-1) * h# / h_max
			u3 += sng * u3 * L * signe(y) * (p1/p0-1) * 1
			#
			if u0 < 0: u0 = 0
			if u1 < 0: u1 = 0
			if u2 < 0: u2 = 0
			if u3 < 0: u3 = 0
			#
			_u0 += [u0]
			_u1 += [u1]
			_u2 += [u2]
			_u3 += [u3]
		#
		ax[0][0].plot(_u0, label=f'x{L}')
		ax[0][1].plot(_u1, label=f'x{L}')
		ax[1][0].plot(_u2, label=f'x{L}')
		ax[1][1].plot(_u3, label=f'x{L}')
	#
	ax[0][0].set_title('      y *h'); ax[0][0].legend()
	ax[0][1].set_title('      y *1'); ax[0][1].legend()
	ax[1][0].set_title('signe(y)*h'); ax[1][0].legend()
	ax[1][1].set_title('signe(y)*1'); ax[1][1].legend()
	#
	print(f"Dernière prédiction : {predictions[-1]}")
	#
	plt.show()