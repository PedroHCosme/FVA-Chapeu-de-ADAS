# -*- coding: utf-8 -*-
# Carrega os logs e prepara os sinais para identificacao.
# Usado por identifica.py e melhora.py.
import glob, os
import numpy as np


def carrega(tmax=np.inf, ignora=(), vmin=0.10):
	"""devolve lista de (nome, t, v, vd, u, ok) com v COM SINAL e ok = amostras validas"""
	ensaios = []
	for csv in sorted(glob.glob(os.path.join('logs', '*', 'car.csv'))):
		nome = os.path.basename(os.path.dirname(csv))
		if any(s in csv for s in ignora):
			continue
		d = np.genfromtxt(csv, delimiter=',', names=True)
		d = d[d['t'] <= tmax]
		if len(d) < 30:
			continue
		t, u, x, y = d['t'], d['u'], d['x'], d['y']
		# a coluna v do log e |v|*marcha: reconstroi o sinal a partir da posicao
		passo = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
		v = np.r_[0.0, passo * np.sign(np.diff(x))]
		vd = np.gradient(v, t)

		ok = (np.abs(v) > vmin) & (t > 0.15) & (t < t[-1] - 0.05)
		# depois da inversao de sentido a marcha fica errada e o modelo nao vale
		vs = np.convolve(v, np.ones(5) / 5, mode='same')
		inv = np.nonzero((vs <= -0.03) & (np.maximum.accumulate(vs) > 0.03))[0]
		if len(inv):
			ok &= t < t[inv[0]]
		if ok.sum() < 30:
			continue
		ensaios.append((nome, t, v, vd, u, ok))
	return ensaios


def empilha(ensaios):
	"""concatena so as amostras validas de todos os ensaios"""
	sel = [(t[ok], v[ok], vd[ok], u[ok]) for _, t, v, vd, u, ok in ensaios]
	return [np.concatenate(c) for c in zip(*sel)]
