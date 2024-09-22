import numpy as np
import pandas as pd

def bitget_btcusdt_1H():
	colonnes = "Unix,Date,Symbol,Open,High,Low,Close,Volume,Volume Base Asset,tradecount"
	fichier  = '/home/vadim/Bureau/Vecteur-V0.1/4a1a1a/BTCUSDT.csv'
	date     = 'Date'
	close    = 'Close'
	#
	df = pd.read_csv(fichier).iloc[::-1].reset_index(drop=True) #car ce fichier est du plus jeune au plus vieux
	print(df)
	#
	DEPART = 1
	#
	#	Transformations
	df['Close_change'] = (df['Close']/df['Close'].shift(+1) - 1) * 100
	#
	return df[DEPART:], close, date

def bitget_btcusdt_15m():
	colonnes = "Unix,Date,Symbol,Open,High,Low,Close,Volume,Volume Base Asset,tradecount"
	fichier  = '/home/vadim/Bureau/tensorflow/bitget_btcusdt_15m.csv'
	date     = 'Date'
	close    = 'Close'
	#
	#	Inverser le fichier
	#
	#
	df = pd.read_csv(fichier).iloc[::-1].reset_index(drop=True) #car ce fichier est du plus jeune au plus vieux
	print(df)
	#
	for i in range(1, len(df)-1):
		if df['Close'].iloc[i-1] == df['Close'].iloc[i]:
			if df['Close'].iloc[i] == df['Close'].iloc[i+1]:
				raise Exception(f"Trop de mêmes valeurs : {list(df['Close'].iloc[i-1:i+3])}")
			else:
				df.loc[i, 'Close'] = (df['Close'].iloc[i-1] + df['Close'].iloc[i+1])/2
	#
	#
	DEPART = 1
	#
	#	Transformations
	df['Close_change'] = (df['Close']/df['Close'].shift(+1) - 1) * 100
	df['Volume_change'] = (df['Volume BTC']/df['Volume BTC'].shift(+1) - 1)
	#
	df['ema12'] = df['Close'].ewm(com=12).mean()
	df['ema26'] = df['Close'].ewm(com=26).mean()
	df['_macd'] = df['ema12']-df['ema26']
	df['_macd_ema9'] = df['_macd'].ewm(com=9).mean()
	df['macd'] = df['_macd'] - df['_macd_ema9']
	df['macd_change'] = (df['macd']/df['macd'].shift(+1) - 1)
	#
	df.replace([-np.inf, np.inf], np.nan, inplace=True)
	df = df.dropna().reset_index()
	#
	#	========== Anti Zero ==========
	
	#
	return df[DEPART:], close, date

def binance_btcusdt_15m(verbose=False):
	colonnes = "OUnix,CUnix,ODate,CDate,Symbol,Open,High,Low,Close,qaV,trades,btcVol,usdtVol"
	fichier  = '/home/vadim/Bureau/tensorflow/binance_btcusdt_15m.csv'
	date     = 'Date'
	close    = 'Close'
	
	#	Inverser le fichier
	df = pd.read_csv(fichier)
	print(df)
	
	#	Si des valeurs se répètent
	for i in range(1, len(df)-1):
		if df['Close'].iloc[i-1] == df['Close'].iloc[i]:
			if verbose: print(f"\033[93mDeux mêmes valeurs    (Odata={df['ODate'][i]}): {list(df['Close'].iloc[i-2:i+2])}\033[0m")
		if df['Close'].iloc[i-1] == df['Close'].iloc[i]:
			if df['Close'].iloc[i] == df['Close'].iloc[i+1]:
				if verbose: print(f"\033[91mTrop de mêmes valeurs (Odata={df['ODate'][i]}): {list(df['Close'].iloc[i-2:i+6])}\033[0m")
				plus = 1
				while not df['Close'].iloc[i+plus] != df['Close'].iloc[i]:
					plus += 1
				for j in range(1+plus):
					df.loc[i+j, 'Close'] = df['Close'].iloc[i-1] + (1+j)*(df['Close'].iloc[i+plus+1]-df['Close'].iloc[i-1])/(plus+1+1)
				if verbose: print(f"Nouvelle maj                                       {list(df['Close'].iloc[i-2:i+6])}")
			else:
				df.loc[i, 'Close'] = (df['Close'].iloc[i-1] + df['Close'].iloc[i+1])/2
	#
	#
	DEPART = 1
	#
	#	Transformations
	df['ema12_Close'] = df['Close'].ewm(com=12).mean()
	df['ema26_Close'] = df['Close'].ewm(com=26).mean()
	df['_macd_Close'] = df['ema12_Close']-df['ema26_Close']
	df['_macd_ema9_Close'] = df['_macd_Close'].ewm(com=9).mean()
	df['macd_Close'] = df['_macd_Close'] - df['_macd_ema9_Close']
	#
	#df.replace([-np.inf, np.inf], np.nan, inplace=True)
	#df = df.dropna().reset_index()
	
	#
	return df[DEPART:].reset_index(), close, date

def eurousdt():
	colonnes = "Time,Open,High,Low,Close,Volume"
	fichier  = '/home/vadim/Bureau/tensorflow/EURUSD_H1.csv'
	date     = 'Time'
	close    = 'Close'
	#
	df = pd.read_csv(fichier)
	print(df)
	#
	DEPART = 1
	#
	#	Transformations
	df['Close_change'] = (df['Close']/df['Close'].shift(+1) - 1) * 1000
	df['Volume_change'] = (df['Volume']/df['Volume'].shift(+1) - 1)
	#
	return df[DEPART:], close, date

def CAC_40_Données_Historiques():
	colonnes = "Date,Dernier,Ouv., Plus Haut,Plus Bas,Vol.,Variation %"
	fichier  = '/home/vadim/Bureau/tensorflow/CAC_40_Données_Historiques.csv'
	date     = 'Date'
	close    = 'Dernier'
	#
	df = pd.read_csv(fichier).iloc[::-1].reset_index(drop=True)
	print(df)
	#
	DEPART = 1
	#
	#	Transformations
	df['Close_change'] = (df['Dernier']/df['Dernier'].shift(+1) - 1) * 100
	#
	return df[DEPART:], close, date