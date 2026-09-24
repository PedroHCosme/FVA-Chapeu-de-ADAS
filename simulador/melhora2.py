# -*- coding: utf-8 -*-
# Duas hipoteses para o residuo que sobrou:
#   H1: e ruido da derivada numerica (nao e erro de modelo)
#   H2: e atraso do atuador/filtro (e erro de modelo, tem conserto)
import numpy as np
from scipy.signal import savgol_filter
from dados import carrega, empilha

ens = carrega(tmax=8.0, ignora=('20260901_200913',))


def ajusta(vd_, v_, u_):
	A = np.c_[np.where(u_ > 0, u_, 0.), np.where(u_ < 0, u_, 0.), -v_ * np.abs(v_), -np.sign(v_)]
	p = np.linalg.lstsq(A, vd_, rcond=None)[0]
	return p, (vd_ - A @ p).std()


t, v, vd, u = empilha(ens)
p0, r0 = ajusta(vd, v, u)
print(f'H0  derivada bruta (np.gradient)      rms = {r0:.4f} m/s2')

# ---- H1: derivada suavizada (Savitzky-Golay) --------------------------------
sel = []
for nome, te, ve, vde, ue, ok in ens:
	vds = savgol_filter(ve, 11, 3, deriv=1, delta=te[1] - te[0])
	sel.append((ve[ok], vds[ok], ue[ok]))
V, VD, U = [np.concatenate(c) for c in zip(*sel)]
p1, r1 = ajusta(VD, V, U)
print(f'H1  derivada suavizada (savgol)       rms = {r1:.4f} m/s2   '
	  f'({100*(1-r1/r0):+.0f}%)   K+={p1[0]:.3f} a0={p1[3]:.3f}')

# ---- H2: atraso de 1a ordem no atuador --------------------------------------
print('\nH2  atraso de atuador: u_ef segue u com constante de tempo Ta')
melhor = None
for Ta in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
	sel = []
	for nome, te, ve, vde, ue, ok in ens:
		uf = np.empty_like(ue)
		uf[0] = ue[0]
		for i in range(len(ue) - 1):
			dt = te[i + 1] - te[i]
			# forma exata do filtro de 1a ordem: estavel para qualquer Ta < dt
			al = 1.0 if Ta <= 0 else 1.0 - np.exp(-dt / Ta)
			uf[i + 1] = uf[i] + (ue[i + 1] - uf[i]) * al
		vds = savgol_filter(ve, 11, 3, deriv=1, delta=te[1] - te[0])
		sel.append((ve[ok], vds[ok], uf[ok]))
	V, VD, U = [np.concatenate(c) for c in zip(*sel)]
	p2, r2 = ajusta(VD, V, U)
	print(f'    Ta={Ta:4.2f}s   rms={r2:.4f}   K+={p2[0]:.3f} K-={p2[1]:.3f} '
		  f'c={p2[2]:.5f} a0={p2[3]:.3f}')
	if melhor is None or r2 < melhor[1]:
		melhor = (Ta, r2, p2)
print(f'\n  melhor: Ta={melhor[0]:.2f}s  rms={melhor[1]:.4f}  '
	  f'({100*(1-melhor[1]/r1):+.0f}% sobre H1)')

# ---- H3: o residuo nas trocas e so a derivada atravessando o degrau? --------
sel = []
for nome, te, ve, vde, ue, ok in ens:
	salto = np.abs(np.r_[0, np.diff(ue)]) > 1e-9
	perto = np.convolve(salto.astype(float), np.ones(7), mode='same') > 0
	vds = savgol_filter(ve, 11, 3, deriv=1, delta=te[1] - te[0])
	s = ok & ~perto
	sel.append((ve[s], vds[s], ue[s]))
V, VD, U = [np.concatenate(c) for c in zip(*sel)]
p3, r3 = ajusta(VD, V, U)
print(f'\nH3  savgol + descartando +-3 amostras das trocas de u')
print(f'    rms = {r3:.4f} m/s2  ({100*(1-r3/r1):+.0f}% sobre H1)   '
	  f'K+={p3[0]:.3f} K-={p3[1]:.3f} c={p3[2]:.5f} a0={p3[3]:.3f}')
