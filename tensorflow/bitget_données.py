#
#	Utilisation :
#		python3 bitget_données.py 15m 1000 bitget_btcusdt.csv
#
#


#	Principe : On va avoire une liste de valeurs qu'on veut (les heures ex : 12janvier2024 12h00, 12janvier2024 13h00, 12janvier2024 14h00 ...)
#	Puis bitget autorise de recuperer 100 heure en une fois.
#	Donc on va separer la liste en 100, et faire len(liste)/100 requettes. De=depart_bloque_liste, A=fin_bloque_liste

import time
import requests
from datetime import datetime

unix_ms_vers_date = lambda ums: datetime.fromtimestamp(
	int(ums/1000)
).strftime('%Y-%m-%d %H:%M:%S')

unix_ms_vers_date_sans_secondes = lambda ums: datetime.fromtimestamp(
	int(ums/1000)
).strftime('%Y-%m-%d %H:%M')

ARONDIRE_AU_MODULO_SUPERIEUR = lambda x,mod: (x if x%mod==0 else x-(x%mod)+mod)

millisecondes_a_secondes = lambda t: int(t/1000)
secondes_a_millisecondes = lambda t: t*1000
heures_a_millisecondes   = lambda t: t*60*60*1000

#https://www.bitget.com/api-doc/contract/market/Get-History-Candle-Data

requette_bitget = lambda de, a, SYMBOLE, d: eval(
	requests.get(
		f"https://api.bitget.com/api/mix/v1/market/history-candles?symbol={SYMBOLE}_UMCBL&granularity={d}&startTime={de}&endTime={a}&productType=usdt-futures"
	).text
)

HEURES_PAR_REQUETTE = 100

def DONNEES_BITGET(__HEURES, d):
	assert d in ('1D', '1H', '1m', '15m')
	print(f"L'intervalle choisie est : {d}")
	print(f"Demande de {__HEURES} elements de {d}")
	#
	HEURES = ARONDIRE_AU_MODULO_SUPERIEUR(__HEURES, HEURES_PAR_REQUETTE)
	#
	correspondance_millisecondes = {
		'1D'  : 24*60*60*1000,
		'1H'  :    60*60*1000,
		'15m' :    15*60*1000,
		'5m'  :     5*60*1000,
		'3m'  :     3*60*1000,
		'1m'  :     1*60*1000
	}
	#
	la = time.time()
	la = int(la) - (int(la)%3600) # (19h05m12s -> 19h00m00s  car  60*60=3600s)
	la = secondes_a_millisecondes(la)
	heures_voulues = [
		int(la - correspondance_millisecondes[d]*(HEURES - 1 - i))
		for i in range(HEURES)
	]
	print(f"Depart : {unix_ms_vers_date(heures_voulues[0])} ||| Fin : {unix_ms_vers_date(heures_voulues[-1])}")

	donnees_BTCUSDT = []

	REQUETTES = int(len(heures_voulues) / HEURES_PAR_REQUETTE)
	print(f"Extraction de {len(heures_voulues)} {d} depuis api.bitget.com ...")
	#
	depart = time.time()
	for i in range(REQUETTES):
		#	-- de à --
		de = heures_voulues[ i   *HEURES_PAR_REQUETTE  ]
		a  = heures_voulues[(i+1)*HEURES_PAR_REQUETTE-1] #-1 car le prochain repètera le meme
		
		#	---- Requette https ----
		paquet_heures_btc = requette_bitget(de, a, "BTCUSDT", d)
		assert len(paquet_heures_btc) == 100
		#	Bitget nomme 18h l'information 18h->19h qui ouvre a 18h et ferme a 19h
		#	Moi j'appèle ça 19h : de 18h->19h
		#	donc on ajoute 3600 secondes a toutes les lignes
		paquet_heures_btc = [[float(ums)+secondes_a_millisecondes(3600), o,h,l,c,vB,vU]for ums,o,h,l,c,vB,vU in paquet_heures_btc]
		donnees_BTCUSDT += paquet_heures_btc
		
		#	-- print( Status et Temps restant) --
		pourcent = i*HEURES_PAR_REQUETTE/len(heures_voulues)
		str_de = unix_ms_vers_date_sans_secondes(float(de))
		str__a = unix_ms_vers_date_sans_secondes(float( a))
		mins_ = int((time.time()-depart          )/pourcent*(1-pourcent)/60 if pourcent!=0 else 0)
		secs_ = int((time.time()-depart-(mins_-1))/pourcent*(1-pourcent)    if pourcent!=0 else 0) % 60
		mins_ = '%2s' % str(mins_)
		secs_ = '%2s' % str(secs_)
		#
		p = '%4s' % str(round(float(pourcent*100),1))
		#
		print(f"[{p}%], len(paquet)={len(paquet_heures_btc)}, {mins_}mins {secs_}sec len(donnees)={'%6s'%str(len(donnees_BTCUSDT))}  {str_de} -> {str__a}")

	print(f"HEURES VOULUES = {len(heures_voulues)}, len(donnees_BTCUSDT)={len(donnees_BTCUSDT)}")

	return donnees_BTCUSDT[-__HEURES:]

#	Ancien site : https://www.CryptoDataDownload.com

def faire_un_csv(donnees_BTCUSDT):
	csv = """Unix,Date,Symbol,Open,High,Low,Close,Volume BTC,Volume USDT,tradecount\n"""

	for ums,o,h,l,c,vB,vU in donnees_BTCUSDT[::-1]:
		date = unix_ms_vers_date(float(ums))
		csv += f'0,{date},BTCUSDT,{o},{h},{l},{c},{vB},{vU},0\n'

	return csv.strip('\n')

def ecrire_le_csv(temporalitée, elements, nom_csv):
	txt = faire_un_csv(
		DONNEES_BITGET(
			elements,
			d=temporalitée
		)
	)
	with open(nom_csv, 'w') as co:
		co.write(txt)

if __name__ == "__main__":
	from sys import argv
	#
	temporalitée, elements, nom_csv = argv[1], int(argv[2]), argv[3]
	#
	assert temporalitée in ('1D', '1H', '15m', '5m', '3m', '1m')
	#
	if temporalitée == '1H' and elements > 28000:
		print("\033[91mAttention : Bitget a des erreurs dans ces données plus ou moins avant 28000 !\033[0m")
	#
	ecrire_le_csv(temporalitée, elements, nom_csv)