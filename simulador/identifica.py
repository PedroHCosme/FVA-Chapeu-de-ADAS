# -*- coding: utf-8 -*-
# Memoria de calculo da identificacao: ajusta o modelo longitudinal
#     v_ponto = K(u)*u - c*v*|v| - a0*sign(v)
# usando TODOS os logs de uma vez, e mostra a conta passo a passo.
# uso: python identifica.py [--tmax=8]
import sys, glob, os, webbrowser
import numpy as np

tmax = next((float(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--tmax=')), np.inf)
# ensaios a excluir do ajuste (ex.: feitos em outra pista/condicao)
ignora = [s for a in sys.argv[1:] if a.startswith('--ignora=') for s in a.split('=')[1].split(',')]

T, V, VD, U, ENSAIOS = [], [], [], [], []
# acumuladores do ajuste EM ETAPAS (secao 7.4): coast (u=0) isola c,a0; motor (u!=0) isola K
CV, CVD, DV, DVD, DU = [], [], [], [], []
print('=' * 78)
print('ENSAIOS USADOS')
print('=' * 78)
for csv in sorted(glob.glob(os.path.join('logs', '*', 'car.csv'))):
	if any(s in csv for s in ignora):
		print(f'{os.path.basename(os.path.dirname(csv)):16s}  IGNORADO')
		continue
	d = np.genfromtxt(csv, delimiter=',', names=True)
	d = d[d['t'] <= tmax]
	if len(d) < 30:
		continue
	t, u, x, y = d['t'], d['u'], d['x'], d['y']
	# velocidade COM SINAL, reconstruida da posicao (a coluna v do log e |v|*marcha)
	passo = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
	v = np.r_[0.0, passo * np.sign(np.diff(x))]
	vd = np.gradient(v, t)
	# descarta bordas e amostras paradas (atrito estatico nao entra no modelo)
	ok = (np.abs(v) > 0.10) & (t > 0.15) & (t < t[-1] - 0.05)
	# depois que o carro inverte o sentido o modelo do simulador nao vale mais:
	# a marcha continua em +1 e a compensacao de atrito passa a empurrar junto
	vs = np.convolve(v, np.ones(5) / 5, mode='same')
	inv = np.nonzero((vs <= -0.03) & (np.maximum.accumulate(vs) > 0.03))[0]
	if len(inv):
		ok &= t < t[inv[0]]
		print(f'  (corta em t={t[inv[0]]:.2f}s: carro inverteu o sentido)')
	if ok.sum() < 30:
		continue
	T += [t[ok]]; V += [v[ok]]; VD += [vd[ok]]; U += [u[ok]]
	# separa por condicao para o ajuste em etapas (secao 7.4)
	coast = ok & (u == 0) & (np.abs(v) > 0.3)          # so roda-livre: isola c e a0
	drive = ok & (u != 0)                              # so com motor: isola K
	for j in np.nonzero(np.diff(u))[0]:                # tira +-3 amostras das trocas de u
		drive[max(0, j - 3):j + 4] = False
	CV += [v[coast]]; CVD += [vd[coast]]
	DV += [v[drive]]; DVD += [vd[drive]]; DU += [u[drive]]
	ENSAIOS.append((os.path.basename(os.path.dirname(csv)), t, v, u, ok))
	print(f'{os.path.basename(os.path.dirname(csv)):16s}  {ok.sum():4d} amostras uteis  '
		  f'u em {np.unique(np.round(u[ok],2))}  v de {v[ok].min():+.2f} a {v[ok].max():+.2f} m/s')

t, v, vd, u = np.concatenate(T), np.concatenate(V), np.concatenate(VD), np.concatenate(U)

# ==========================================================================
# AJUSTE EM ETAPAS (isola um parametro por vez, secao 2/3 e 7.4)
#   Etapa 1: c e a0 SO dos coastdowns (u=0), onde o motor sai da conta.
#   Etapa 2: K+/K- SO dos trechos com motor (u!=0), com c e a0 ja fixos.
# Fitar tudo num lstsq unico deixa c e K se compensarem e enviesa o c (secao 7.4).
# ==========================================================================
cv, cvd = np.concatenate(CV), np.concatenate(CVD)
(c, a0), *_ = np.linalg.lstsq(np.c_[-cv * np.abs(cv), -np.sign(cv)], cvd, rcond=None)

dv, dvd, du = np.concatenate(DV), np.concatenate(DVD), np.concatenate(DU)
r = dvd + c * dv * np.abs(dv) + a0 * np.sign(dv)          # sobra que o motor deve explicar: r = K*u
(Kp, Km), *_ = np.linalg.lstsq(np.c_[np.where(du > 0, du, 0.0), np.where(du < 0, du, 0.0)], r, rcond=None)

# residuo do modelo montado, em todas as amostras
A = np.c_[np.where(u > 0, u, 0.0), np.where(u < 0, u, 0.0), -v * np.abs(v), -np.sign(v)]
res = vd - A @ np.array([Kp, Km, c, a0])

print('\n' + '=' * 78)
print('MODELO AJUSTADO (em etapas)   v_ponto = K(u)*u - c*v|v| - a0*sign(v)')
print('=' * 78)
print(f'  K+ (acelerando) = {Kp:6.3f}   [m/s2 por unidade de u]   (de {len(du[du>0])} amostras com motor)')
print(f'  K- (freando)    = {Km:6.3f}                             (de {len(du[du<0])} amostras com motor)')
print(f'  c  (arrasto)    = {c:7.5f}  [1/m]  -> a 2 m/s vale {c*4:.4f} m/s2   (de {len(cv)} amostras de coast)')
print(f'  a0 (atrito seco)= {a0:6.3f}   [m/s2]  -> f = a0/g = {a0/9.81:.4f}')
print(f'  erro rms = {res.std():.4f} m/s2   (amostras: {len(vd)})')

# comparacao: o lstsq global (tudo de uma vez) enviesa o c -- ver secao 7.4
pg, *_ = np.linalg.lstsq(A, vd, rcond=None)
print(f'  [comparacao] lstsq global daria c={pg[2]:.5f} a0={pg[3]:.4f} '
	  f'(enviesado pelos trechos com motor; nao usar -- secao 7.4)')

print('\n' + '=' * 78)
print('CONFERENCIA: o modelo reproduz a aceleracao media de cada patamar?')
print('=' * 78)
print(f'{"u":>6} {"faixa de v":>16} {"medido":>9} {"modelo":>9} {"erro":>8}')
for uu in np.unique(np.round(u, 2)):
	s = np.round(u, 2) == uu
	if s.sum() < 10:
		continue
	prev = (Kp if uu > 0 else Km) * uu - c * (v[s] * np.abs(v[s])).mean() - a0 * np.sign(v[s]).mean()
	print(f'{uu:+6.2f} {v[s].min():+7.2f}..{v[s].max():+6.2f} '
		  f'{vd[s].mean():+9.3f} {prev:+9.3f} {vd[s].mean()-prev:+8.3f}')

# ==========================================================================
# VALIDACAO: simula o modelo com o mesmo u do ensaio e compara com o medido.
# Esse e o teste que vale: o ajuste acerta a DERIVADA amostra a amostra;
# aqui o erro se acumula ao longo de todo o ensaio, sem chance de esconder.
# ==========================================================================
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def simula(t, u, v0):
	"""integra v_ponto = K(u)*u - c*v|v| - a0*tanh(v/0.05) por Euler"""
	v = np.empty_like(t)
	v[0] = v0
	for i in range(len(t) - 1):
		K = Kp if u[i] > 0 else (Km if u[i] < 0 else 0.0)
		# tanh no lugar de sign: evita chattering numerico perto de v=0
		vdot = K * u[i] - c * v[i] * abs(v[i]) - a0 * np.tanh(v[i] / 0.05)
		v[i + 1] = v[i] + vdot * (t[i + 1] - t[i])
	return v

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n = len(ENSAIOS)
fig = make_subplots(rows=n, cols=1, shared_xaxes=False, vertical_spacing=0.04,
					subplot_titles=[e[0] for e in ENSAIOS])
figm, axs = plt.subplots((n + 1) // 2, 2, figsize=(12, 2.6 * ((n + 1) // 2)))
axs = np.atleast_1d(axs).ravel()
print('\n' + '=' * 78)
print('VALIDACAO (simula o modelo do inicio ao fim do ensaio)')
print('=' * 78)
for k, (nome, tt, vv, uu, ok) in enumerate(ENSAIOS, start=1):
	m = slice(np.argmax(ok), np.nonzero(ok)[0][-1] + 1)   # do 1o ao ultimo ponto valido
	ts, vs_med, us = tt[m], vv[m], uu[m]
	vs_sim = simula(ts, us, vs_med[0])
	err = vs_sim - vs_med
	fig.add_trace(go.Scatter(x=ts, y=vs_med, mode='lines', line=dict(width=3, color='#1f77b4'),
							 name='medido', showlegend=(k == 1)), row=k, col=1)
	fig.add_trace(go.Scatter(x=ts, y=vs_sim, mode='lines', line=dict(width=2, color='crimson', dash='dash'),
							 name='modelo', showlegend=(k == 1)), row=k, col=1)
	fig.update_yaxes(title_text='v [m/s]', row=k, col=1)
	print(f'  {nome:16s}  erro rms={err.std():.4f} m/s   erro max={np.abs(err).max():.4f} m/s'
		  f'   ({100*np.abs(err).max()/max(np.abs(vs_med).max(),1e-9):.1f}% do fundo de escala)')

	ax = axs[k - 1]
	ax.plot(ts, vs_med, lw=2.5, color='#1f77b4', label='medido')
	ax.plot(ts, vs_sim, lw=1.6, color='crimson', ls='--', label='modelo')
	for lim in np.nonzero(np.diff(us))[0]:          # onde o comando muda
		ax.axvline(ts[lim + 1], color='gray', lw=.7, ls=':')
	ax.set_title(f'{nome}   u={np.unique(np.round(us,2))}   rms={err.std():.3f} m/s', fontsize=9)
	ax.set_xlabel('t [s]', fontsize=8); ax.set_ylabel('v [m/s]', fontsize=8)
	ax.grid(alpha=.3); ax.tick_params(labelsize=8)
	if k == 1:
		ax.legend(fontsize=8)

fig.update_xaxes(title_text='t [s]', row=n, col=1)
fig.update_layout(height=260 * n, hovermode='x unified',
				  title=f'Validacao do modelo  |  K+={Kp:.3f}  K-={Km:.3f}  c={c:.5f}  a0={a0:.3f}')
saida = os.path.join('logs', 'validacao.html')
fig.write_html(saida)

for ax in axs[n:]:                                   # apaga painel sobrando
	ax.axis('off')
figm.suptitle(f'Validacao em malha aberta   |   K+={Kp:.3f}  K-={Km:.3f}  '
			  f'c={c:.5f}  a0={a0:.3f}', fontsize=11)
figm.tight_layout()
png = os.path.join('logs', 'validacao.png')
figm.savefig(png, dpi=140)
print('\n->', saida)
print('->', png)

