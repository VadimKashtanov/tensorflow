import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_nlp
import tensorflow_probability as tfp
#from tfmo.vision.layers import PositionalEncoding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Flatten, Permute, Reshape, Lambda, Activation, Multiply
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from random import randint, random, shuffle
from math import log, tanh
from os import system

COEF_LOGISTIQUE = 1

tf_logistique = lambda x: 1/(1+tf.exp(-COEF_LOGISTIQUE*         x )) # -1 car le modèle tend a etre = 0
np_logistique = lambda x: 1/(1+np.exp(-COEF_LOGISTIQUE*np.array(x))) # ducoup ça tend a donner h=0.5 constement
logistique    = lambda x: 1/(1+np.exp(-COEF_LOGISTIQUE*         x )) # donc on punnit ça

def ema(l,K):
	e = [l[0]]
	for a in l[1:]:
		e.append(e[-1]*(1-K) + K*a)
	return e
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
"""

def custom_confident_loss(y_true, y_pred):
	y0 = y_pred[:, 0:1]
	y1 = y_pred[:, 1:2]
	y2 = y_pred[:, 2:3]
	#
	w  = y_true[:, 0:1]
	yh = y_true[:, 1:2]
	#
	#
	y = tf.tanh      (y0)
	h = tf_logistique(y1)
	c = tf_logistique(y2)
	#
	#
	_y = tf.stop_gradient(y)
	_h = tf.stop_gradient(h)
	_c = tf.stop_gradient(c)
	#
	#
	Y = tf.pow(                       w      -        tf.tanh(y0), 2) * tf_logistique(y2)
	H = tf.pow(               tf.sign(w*_y)  -                y1 , 2) * tf_logistique(y2)
	C = tf.pow(tf.reduce_mean(tf.sign(w*_y)) - tf.reduce_mean(y2), 2)
	#
	return tf.reduce_mean(Y) + tf.reduce_mean(H) + C

###########################################################################################################
#######################################         Couches       #############################################
###########################################################################################################

class DropConnectDense(tf.keras.layers.Layer):
	def __init__(self, units, dropconnect_rate=0.0, activation=None, **kwargs):
		super(DropConnectDense, self).__init__(**kwargs)
		self.units = units
		self.dropconnect_rate = dropconnect_rate
		self.dense_layer = tf.keras.layers.Dense(units, **kwargs)  # Activation is applied later
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
print(df[['Close_change', 'Volume_change']])

#	====================== One Hot ====================

L = 10 + 1 + 10

one_hot = lambda l,L: [l[i*int(   len(l)/(L-1)  )] for i in range(L-1)]+[l[-1]]
#one_hot = lambda l,L: np.arange(min(l),max(l)+1e-10, (max(l)-min(l))/(L-1))
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

INTERV = [1]#,6]

sources = {
	I : {
		partie : ema([df[partie].values[i] for i in range(len(df[partie]))], K=1/I)
		for partie in parties
	}
	for I in INTERV
}

######### Manière d'encoder les valeurs dans des vecteurs #########

def plus_proche(x,d):
	distances = [abs(x-_d) for _d in d]
	indexe = distances.index(min(distances))
	return [(1.0 if indexe==i else 0.0) for i in range(len(d))]

def pourcent_deux_plus_proches(x,d):
	if x>d[-1]: return [0 for _ in d[:-1]]+[1]
	if x<d[ 0]: return [1] + [0 for _ in d[1:]]
	#
	distances = [abs(x-_d) for _d in d]
	indexe_plus_proche = distances.index(min(distances))
	indexe_second      = indexe_plus_proche + (1 if x>d[indexe_plus_proche] else -1)
	#
	ret = [0 for _ in d]
	#
	premier = distances[indexe_plus_proche]
	second  = distances[indexe_second     ]
	#
	assert premier+second != 0
	#
	ret[indexe_plus_proche] = 1-premier/(premier+second)
	ret[indexe_second     ] = 1-second /(premier+second)
	#
	return ret

nombre_vers_distance = pourcent_deux_plus_proches#plus_proche

##################################################################

#  Un bloque de données est : Mot*Mots = (L*parties) * N
N = 16#32#32#32#64

LIGNES_TEST = 4096

X_train = []
Y_train = []
X_test = []
Y_test = []

T_total = len(df)

T = LIGNES_TEST+2048#len(df)#365*24* (5+4+1)

A,B = max([N*max(INTERV), T_total-T]), T_total #Les LIGNES_TEST derniers, sont les données de validation

for i in range(A, B):
	_x = []
	for I in INTERV:
		for n in range(N):
			_l_ = []
			for partie in parties: _l_ += nombre_vers_distance(sources[I][partie][i-n*I], frontieres_des_one_hot[partie])
			#
			_x += [_l_]	#append un mot
	#
	_y = [(0 if i==T_total-1 else df['Close_change'].values[i+1])]
	
	if i > B-LIGNES_TEST-1:
		X_test.append(_x)
		Y_test.append(_y+[0.0])
	else:	#ça peut etre plus que cas que ça aussi
		X_train.append(_x)
		Y_train.append(_y+[0.0])
		#
		X_train.append(_x)
		Y_train.append(_y+[1.0])

	#+[0 ou 1] c'est pour optimiser séparément y et h dans la loss
	#0 et 1, car les données seront shuffle=True, et donc séparé, et donc aléatoirement entrainnées

print(X_train[0])
print(Y_train[0])

X_train, Y_train, X_test, Y_test = [np.array(a) for a in (X_train, Y_train, X_test, Y_test)]

print(f" ==== X test ====                X_train.shape={X_train.shape}")
print(X_test)
print(f" ==== Y test ====                Y_train.shape={Y_train.shape}")
print(Y_test)

##################################################################################
###############################      Modèle     ##################################
##################################################################################

d_model = 16#32#64#128
dropout = 0.3
mots    = L*len(parties)
head_size = 16
num_heads = 8
ff_dim    = d_model*2

entree = Input(shape=(N*len(INTERV), L*len(parties)))
#x = Dense(d_model)(entree)
#x = PositionalEncoding(sequence_length=mots, d_model=d_model)(x)
x = entree
#
#x = LayerNormalization(epsilon=1e-6)(x)
#x = keras_nlp.layers.TransformerEncoder(intermediate_dim=head_size, num_heads=num_heads, activation="gelu")(x)
#
x = Flatten()(x); x = Dropout(0.50)(x)
x = Dense(256, activation='relu')(x); x = Dropout(0.50)(x)
x = Dense(128, activation='relu')(x); x = Dropout(0.50)(x)
x = Dense( 64, activation='relu')(x); x = Dropout(0.50)(x)
#
#x = Moe(d=128, Expertises=4)(x)
#
#x = Dense(128, use_bias=False)(x); x = Dropout(0.2)(x)
#x = Dense(128, activation='gelu', use_bias=False)(x)#, kernel_regularizer=l2(0.001), use_bias=False)(x)
#x = Dropout(0.3)(x)
#
x = Dense(3)(x)#, kernel_regularizer=l2(0.001), use_bias=False)(x)

#	-------------------------
#
model = Model(entree, x)
model.compile(optimizer=Adam(learning_rate=1e-5), loss=custom_confident_loss) #'mse'
model.summary()

############################ Entrainnement #########################

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

history = model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_test,Y_test),
	#shuffle=True, callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

plt.plot(history.history['loss'],     label='Perte d\'entraînement')
plt.plot(history.history['val_loss'], label='Perte de validation')
plt.legend()
plt.show()

############################    Tester     #########################

y_pred_train = model.predict(X_train)

Y_pred = model.predict(X_test)

print(Y_test)
print(Y_pred)

Y_test          = list(w for w,_   in Y_test)
predictions     = list(y for y,h,c in Y_pred)
investissements = list(h for y,h,c in Y_pred)
assurance       = list(h for y,h,c in Y_pred)

Y_train               = list(w for w,_ in Y_train)
predictions_train     = list(y for y,h,c in y_pred_train)
investissements_train = list(h for y,h,c in y_pred_train)
assurance_train       = list(h for y,h,c in y_pred_train)

plt.plot(Y_test,          label='Valeurs réelles',  color='blue')
plt.plot(predictions,     label='Prédictions',      color='red')
plt.plot(investissements, label='% Investissement', color='green')
plt.plot(assurance_train, label='% Investissement', color='green')

#	Flèches
for t in range(LIGNES_TEST):
	pred       = predictions[t]
	point_prix = Y_test     [t]
	if pred >= 0.0: plt.plot([t, t+1], [point_prix, point_prix + 0.03], 'g')
	else:           plt.plot([t, t+1], [point_prix, point_prix - 0.03], 'r')
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
	invest_pourcent += logistique(h)
	#
	p1   = (0 if t==LIGNES_TEST-1 else prixs[t+1])
	p0   = prixs[ t ]
	#
	ch = p1/p0 - 1
	#
	print("%+3.3f | %+3.3f(%+4.3f) %+4.3f(%+4.3f) | %+7.4f$ %6s%% | %s %s" % (
		w,
		#
		tanh(y),y,
		logistique(h),h,
		#
		prixs[t],
		(str(ch*100)[:5] if t != LIGNES_TEST-1 else '?'),
		#
		dates[t],
		(f'\033[92m +$ \033[0m' if signe(w) == signe(y) else '\033[91m -$ \033[0m'),
	))

print(f"Investissement moyen {round(100*invest_pourcent/LIGNES_TEST,3)} %")

moyenne = lambda x: (0 if len(list(x))==0 else sum(list(x))/len(list(x)))
eq1     = lambda x: (1 if signe(x)==1 else 0)

print("\n === Train === ")
T = len(X_train)
#
val_moy =   moyenne([int(signe(predictions_train[i])==signe(Y_train[i]))                              for i in range(T)])
dst_moy = 1-moyenne([abs((predictions_train[i]-signe(Y_train[i]))/2)                                  for i in range(T)])
inv_bon = 1-moyenne([abs(eq1(Y_train[i]*predictions_train[i]) - logistique(investissements_train[i])) for i in range(T)])
inv_tot =   moyenne([logistique(investissements_train[i])                                             for i in range(T)])
#
print(f"(y) Validité moyenne       : {'%7s' % str(round(100*val_moy,4))}%")
print(f"(y) Distance moyenne       : {'%7s' % str(round(100*dst_moy,4))}%")
print(f"(h) Invest % bonne moyenne : {'%7s' % str(round(100*inv_bon,4))}%")
print(f"(h) Investissement moyen   : {'%7s' % str(round(100*inv_tot,4))}%")
print(" ============= \n")

##################################################################################
############################          Gains      #################################
##################################################################################

fig, ax = plt.subplots(2,2)

h_max = max(np_logistique(investissements))

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
		u0 += u0 * L *       y  * (p1/p0-1) * h / h_max
		u1 += u1 * L *       y  * (p1/p0-1) * 1
		u2 += u2 * L * signe(y) * (p1/p0-1) * h / h_max
		u3 += u3 * L * signe(y) * (p1/p0-1) * 1
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