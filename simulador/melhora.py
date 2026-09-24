# -*- coding: utf-8 -*-
# 1) Onde ainda ha erro no modelo atual (analise de residuo)
# 2) SINDy caseiro (STLSQ): monta uma biblioteca de termos candidatos
#    e deixa os dados escolherem quais sobrevivem.
# uso: python melhora.py
import numpy as np
from dados import carrega, empilha

IGNORA = ('20260901_200913',)   # ensaio em pista inclinada
ensaios = carrega(tmax=8.0, ignora=IGNORA)
t, v, vd, u = empilha(ensaios)

# ---------------------------------------------------------------- modelo atual
A = np.c_[np.where(u > 0, u, 0.), np.where(u < 0, u, 0.), -v * np.abs(v), -np.sign(v)]
p = np.linalg.lstsq(A, vd, rcond=None)[0]
r = vd - A @ p
print('=' * 74)
print(f'MODELO ATUAL  K+={p[0]:.3f} K-={p[1]:.3f} c={p[2]:.5f} a0={p[3]:.3f}')
print(f'residuo: rms={r.std():.4f}  max={np.abs(r).max():.4f} m/s2')

# ---------------------------------------------- o residuo tem estrutura?
print('\n' + '=' * 74)
print('O RESIDUO E RUIDO OU E SINAL? (se tiver padrao, falta termo no modelo)')
print('=' * 74)
print(f'{"faixa":>22} {"n":>5} {"media do residuo":>18} {"rms":>8}')
for lo, hi in [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 7.0)]:
	s = (np.abs(v) >= lo) & (np.abs(v) < hi)
	if s.sum() > 10:
		print(f'  |v| em [{lo:.1f},{hi:.1f})      {s.sum():5d} {r[s].mean():+18.4f} {r[s].std():8.4f}')
for uu in np.unique(np.round(u, 2)):
	s = np.round(u, 2) == uu
	if s.sum() > 10:
		print(f'  u = {uu:+.2f}            {s.sum():5d} {r[s].mean():+18.4f} {r[s].std():8.4f}')
# residuo logo apos a troca de comando -> atraso de atuador/filtro
salto = np.abs(np.r_[0, np.diff(u)]) > 1e-9
jan = np.convolve(salto.astype(float), np.ones(6), mode='same') > 0
print(f'  ate 0,3s apos trocar u  {jan.sum():5d} {r[jan].mean():+18.4f} {r[jan].std():8.4f}')
print(f'  longe da troca de u     {(~jan).sum():5d} {r[~jan].mean():+18.4f} {r[~jan].std():8.4f}')
# a troca foi pra acelerar (u a favor de v) ou pra frear (u contra v)?
idx = np.flatnonzero(salto)
tipo_troca = np.zeros_like(v)
tipo_troca[idx] = np.where(np.sign(u[idx]) == np.sign(v[idx - 1]), 1, -1)
tipo_prop, ultimo = np.zeros_like(v), 0.0
for i in range(len(v)):
	if tipo_troca[i] != 0:
		ultimo = tipo_troca[i]
	tipo_prop[i] = ultimo if jan[i] else 0.0
for tp, rotulo in [(1, 'troca -> acelerando'), (-1, 'troca -> freando')]:
	s = jan & (tipo_prop == tp)
	if s.sum() > 5:
		print(f'  {rotulo:22s}  {s.sum():5d} {r[s].mean():+18.4f} {r[s].std():8.4f}')
# a0 de cada ensaio separadamente
print('\n  a0 ajustado por ensaio (o atrito e igual em toda a pista?)')
for nome, te, ve, vde, ue, ok in ensaios:
	s = ok & (ue == 0)
	if s.sum() > 20:
		a0e = -(vde[s] + p[2] * ve[s] * np.abs(ve[s])).mean() / np.sign(ve[s]).mean()
		print(f'    {nome}: a0={a0e:.3f}')

# ------------------------------------------------------------ SINDy (STLSQ)
print('\n' + '=' * 74)
print('SINDy CASEIRO: biblioteca de candidatos + limiar iterativo')
print('=' * 74)
sv = np.sign(v)
biblioteca = {
	'u+':        np.where(u > 0, u, 0.),
	'u-':        np.where(u < 0, u, 0.),
	'sign(v)':   sv,
	'v':         v,
	'v|v|':      v * np.abs(v),
	'v^3':       v ** 3,
	'u*v':       u * v,
	'u*|v|':     u * np.abs(v),
	'u^2*sign':  u ** 2 * sv,
	'1':         np.ones_like(v),
	'sqrt|v|':   np.sqrt(np.abs(v)) * sv,
}
nomes = list(biblioteca)
T = np.c_[tuple(biblioteca.values())]
# normaliza as colunas: sem isso o limiar unico penaliza termos de escala grande
esc = np.linalg.norm(T, axis=0)
Tn = T / esc

xi = np.linalg.lstsq(Tn, vd, rcond=None)[0]
for it in range(12):                       # STLSQ: zera o que for pequeno e reajusta
	pequeno = np.abs(xi) < 0.05 * np.abs(xi).max()
	if not pequeno.any():
		break
	xi[pequeno] = 0.0
	viv = ~pequeno
	xi[viv] = np.linalg.lstsq(Tn[:, viv], vd, rcond=None)[0]

coef = xi / esc
rs = vd - T @ coef
print('  termos que sobreviveram:')
for nome, cf in zip(nomes, coef):
	if cf != 0.0:
		print(f'    {nome:>10s}  {cf:+.5f}')
print(f'  residuo: rms={rs.std():.4f}  ({100*(1-rs.std()/r.std()):+.1f}% vs modelo atual)')

# ---------------------------------- vale a pena cada termo extra? (validacao cruzada)
print('\n' + '=' * 74)
print('CADA TERMO EXTRA SE PAGA? (treina em 5 ensaios, testa no 6o)')
print('=' * 74)
candidatos = {
	'base (4 termos)': ['u+', 'u-', 'sign(v)', 'v|v|'],
	'+ v':             ['u+', 'u-', 'sign(v)', 'v|v|', 'v'],
	'+ u*v':           ['u+', 'u-', 'sign(v)', 'v|v|', 'u*v'],
	'+ v e u*v':       ['u+', 'u-', 'sign(v)', 'v|v|', 'v', 'u*v'],
	'so u e sign(v)':  ['u+', 'u-', 'sign(v)'],
}
for rotulo, termos in candidatos.items():
	err = []
	for k in range(len(ensaios)):
		tr = [e for j, e in enumerate(ensaios) if j != k]
		tt_, vv_, vd_, uu_ = empilha(tr)
		te_, ve_, vde_, ue_ = empilha([ensaios[k]])
		M = lambda vv, uu: np.c_[tuple(
			{'u+': np.where(uu > 0, uu, 0.), 'u-': np.where(uu < 0, uu, 0.),
			 'sign(v)': -np.sign(vv), 'v|v|': -vv * np.abs(vv), 'v': -vv,
			 'u*v': uu * vv}[n] for n in termos)]
		q = np.linalg.lstsq(M(vv_, uu_), vd_, rcond=None)[0]
		err.append((vde_ - M(ve_, ue_) @ q).std())
	print(f'  {rotulo:18s} erro medio fora da amostra = {np.mean(err):.4f} m/s2')
