import numpy as np
import pandas as pd

def btcusdt_1H():
	colonnes = "Unix,Date,Symbol,Open,High,Low,Close,Volume,Volume Base Asset,tradecount"
	fichier  = '/home/vadim/Bureau/Vecteur-V0.1/4a1a1a/BTCUSDT.csv'
	date     = 'Date'
	close    = 'Close'
	#
	df = pd.read_csv(fichier)[::-1] #car ce fichier est du plus jeune au plus vieux
	print(df)
	#
	DEPART = 1
	#
	#	Transformations
	df['Close_change'] = (df['Close']/df['Close'].shift(+1) - 1) * 100
	#
	return df[DEPART:], close, date

def btcusdt_15m():
	colonnes = "Unix,Date,Symbol,Open,High,Low,Close,Volume,Volume Base Asset,tradecount"
	fichier  = '/home/vadim/Bureau/tensorflow/bitget_btcusdt_15m.csv'
	date     = 'Date'
	close    = 'Close'
	#
	df = pd.read_csv(fichier)[::-1] #car ce fichier est du plus jeune au plus vieux
	print(df)
	#
	DEPART = 1
	#
	#	Transformations
	df['Close_change'] = (df['Close']/df['Close'].shift(+1) - 1) * 100
	#
	return df[DEPART:], close, date

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
	df = pd.read_csv(fichier)[::-1]
	print(df)
	#
	DEPART = 1
	#
	#	Transformations
	df['Close_change'] = (df['Dernier']/df['Dernier'].shift(+1) - 1) * 100
	#
	return df[DEPART:], close, date