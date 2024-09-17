import matplotlib.pyplot as plt

from données import btcusdt_15m
df, Close, la_Date = btcusdt_15m()

from random import random, seed
seed(0)

rnd = random

T = 100

s=10
l = [s:=(s+rnd()-.5) for _ in range(T)]
l = [rnd() + rnd()**2 + rnd()**6*3 for _ in range(T)]

l = list(df['Close'][0:20])

T = len(l)

def ema(l,K):
	e=[l[0]]
	for a in l[1:]:e+=[e[-1]*(1-1/K) + a/K]
	return e

norme = lambda lst: [0.01*((e-min(lst))/(max(lst)-min(lst))-0.5) for e in lst]

#	Informations : prixs, volume, macd, rsi ... xI fois
#	Expertises   : l[i]/ema(K)[i-n] - 1     ... xE fois    (voire même ema(K)[i]/ema(K)[i-n] - 1, ou K=1 est le plus interessant)
#	Intervalles  : 

#	Courbes = IxE en tout. Car chaque infos aura autant d'expertises (par soucis de simplicité d'implémentation)
#	donc on a au finale I*E*(len paramettres nécessaires par expertises (K et N = 2))
#						I*E*2 paramtres pour les entrées
#	[Selection Naturelle] on fait la selection de ces paramettres qui ont un nombre finis d'états.
#		Ex : n dans {-1,-2,-3,-4}, K dans {1,2,3,4...100}, interv dans {1,2,3..64}
#
#	Selections naturelle (cloner les modèle et les entrainner en parrallèle):
#	P-meilleur -> (P*enfants).fit(echopes=20) -> P-meilleurs
#		Soit la selection est haplogroupique, soit autosomale. Soit un mixte.
#
#	... voire pour cloner les modèles
#

#	Modèle : X-> Conv1D (? séparé ou des noyeaux ensembre ?) -> FFN -> 3 ou 2 

_l_ = [
	(ema(l, K=1), -1),
	(ema(l, K=3), -1), (ema(l, K=3), -2),
	(ema(l, K=5), -1), (ema(l, K=5), -2), (ema(l, K=5), -3), 
	(ema(l, K=9), -1), (ema(l, K=9), -2), (ema(l, K=9), -3), (ema(l, K=9), -4),
]

fig,ax = plt.subplots(len(_l_))

for k,(e,n) in enumerate(_l_):
	ax[k].plot(norme(l))
	ax[k].plot(norme(e))
	for i in range(T):
		if i+n>=0:
			ax[k].plot(
				[i,i],
				[0, l[i]/e[i+n]-1],
				('g' if l[i]/e[i+n]-1>0 else 'r')
			)
		else:
			pass
	#ax[k].plot([0]*(-n) + [l[i]/e[i+n]-1 for i in range(T) if i+n>=0])
	ax[k].grid(True)

plt.show()