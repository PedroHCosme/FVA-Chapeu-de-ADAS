# -*- coding: utf-8 -*-
# Analise interativa do log longitudinal.
# Segmenta o log por trechos de comando u constante e ajusta cada trecho:
#   u != 0  -> reta  (inclinacao = aceleracao efetiva -> ganho K = dv/dt / u)
#   u == 0  -> exponencial livre v = v0*exp(-t/tau)  (arrasto)
# uso: python analise_log.py [logs/AAAAMMDD_HHMMSS]   (sem argumento: log mais recente)
import sys, os, glob, webbrowser
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CORES = ['crimson', 'seagreen', 'darkorange', 'mediumpurple', 'teal', 'brown']

args = [a for a in sys.argv[1:] if not a.startswith('--')]
tmax = next((float(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--tmax=')), np.inf)

if args:
	pasta = args[0] if os.path.isdir(args[0]) else os.path.join('logs', args[0])
else:
	pasta = os.path.dirname(max(glob.glob(os.path.join('logs', '*', 'car.csv')), key=os.path.getmtime))

d = np.genfromtxt(os.path.join(pasta, 'car.csv'), delimiter=',', names=True)
d = d[d['t'] <= tmax]        # --tmax=8 descarta o final invalido do ensaio
t, v, a, u, x, y = d['t'], d['v'], d['a'], d['u'], d['x'], d['y']

# velocidade com sinal verdadeiro, a partir do deslocamento (o log traz |v|*marcha)
passo = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
sentido = np.sign(np.diff(x))                                 # projeta no eixo do movimento
v_real = np.r_[0.0, passo * sentido]

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
					subplot_titles=('comando u', 'velocidade v', 'aceleracao a'))
fig.add_trace(go.Scatter(x=t, y=u, mode='lines', line_shape='hv', name='u [m/s2]'), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=v, mode='lines+markers', marker_size=4, name='v logado [m/s]'), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=v_real, mode='lines', line=dict(width=1, dash='dot', color='gray'),
						 name='v da posicao (com sinal)'), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=a, mode='lines', name='a logada [m/s2]'), row=3, col=1)

# --- segmenta por u constante ---
bordas = np.r_[0, np.nonzero(np.diff(u))[0] + 1, len(t)]
print(f'log: {pasta}   dt={np.diff(t).mean():.3f}s   {len(t)} amostras\n')

# pre-passada: arrasto (so existe se houver um trecho com u=0 longo o bastante)
b_arrasto = 0.0
for i0, i1 in zip(bordas[:-1], bordas[1:]):
	if u[i0] == 0.0 and i1 - i0 >= 20 and np.abs(v_real[i0 + 3:i1]).min() > 1e-3:
		b_arrasto = -np.polyfit(t[i0 + 3:i1], np.log(np.abs(v_real[i0 + 3:i1])), 1)[0]

resumo = []
for k, (i0, i1) in enumerate(zip(bordas[:-1], bordas[1:])):
	if i1 - i0 < 8:                     # trecho curto demais para ajustar
		continue
	cor = CORES[k % len(CORES)]
	m = slice(i0 + 3, i1)               # descarta o transitorio do filtro
	# ajusta sempre na velocidade COM SINAL: a coluna v do log e magnitude*marcha
	tt, vv, uu = t[m], v_real[m], float(u[i0])
	fig.add_vrect(x0=t[i0], x1=t[i1 - 1], fillcolor=cor, opacity=0.06, line_width=0)

	if uu == 0.0:                       # trecho livre: ajuste exponencial
		vv = np.abs(vv)
		b = -np.polyfit(tt, np.log(np.clip(vv, 1e-6, None)), 1)[0]
		tau = 1.0 / b
		fig.add_trace(go.Scatter(x=tt, y=vv[0] * np.exp(-b * (tt - tt[0])), mode='lines',
								 line=dict(color=cor, dash='dash'),
								 name=f'u=0: tau={tau:.1f}s'), row=2, col=1)
		# marca 36,8% de v0 (leitura grafica de tau na resposta livre)
		alvo = 0.368 * vv[0]
		if vv.min() <= alvo <= vv.max():
			tc = float(np.interp(alvo, vv[::-1], tt[::-1]))
			fig.add_trace(go.Scatter(x=[tc], y=[alvo], mode='markers+text', marker=dict(size=11, symbol='x', color=cor),
									 text=[f'  36,8% -> tau={tc-tt[0]:.2f}s'], textposition='middle right',
									 name='36,8%'), row=2, col=1)
		else:
			fig.add_hline(y=alvo, line=dict(color=cor, dash='dot', width=1), row=2, col=1,
						  annotation_text=f'36,8% de v0 = {alvo:.2f} m/s (nao cruzado)', annotation_position='top left')
		# a mesma descida serve a dois modelos; o residuo diz qual descreve melhor
		pl = np.polyfit(tt, vv, 1)
		r_exp = np.abs(vv - vv[0] * np.exp(-b * (tt - tt[0]))).max()
		r_ret = np.abs(vv - np.polyval(pl, tt)).max()
		melhor = 'atrito constante' if r_ret < r_exp else 'arrasto proporcional a v'
		resumo.append(f'  u= 0.00  t=[{t[i0]:.2f},{t[i1-1]:.2f}]  '
					  f'exp: tau={tau:.1f}s (res={r_exp:.4f})  |  '
					  f'reta: a={pl[0]:+.3f} m/s2 (res={r_ret:.4f})  ->  {melhor}')
	else:                               # trecho com comando: inclinacao = aceleracao
		# usa so a parte em que o carro realmente responde; se ele satura
		# (velocidade de regime), o plato achataria a reta e mataria o ganho
		vd = np.gradient(vv, tt)
		val = np.abs(vd) > 0.2 * np.abs(vd).max()
		if val.sum() < 5:
			val = np.ones_like(vv, bool)
		p = np.polyfit(tt[val], vv[val], 1)
		K = (p[0] + b_arrasto * vv[val].mean()) / uu   # desconta o arrasto ja identificado
		fig.add_trace(go.Scatter(x=tt[val], y=np.polyval(p, tt[val]), mode='lines',
								 line=dict(color=cor, dash='dash'),
								 name=f'u={uu:+.2f}: dv/dt={p[0]:+.3f}'), row=2, col=1)
		nota = ''
		if val.sum() < len(vv) - 3:
			t_sat = tt[val][-1]
			nota = f'  [saturou em t={t_sat:.2f}s, v={vv[val][-1]:+.3f} -> {vv[-1]:+.3f} m/s]'
			fig.add_trace(go.Scatter(x=[t_sat], y=[vv[val][-1]], mode='markers',
									 marker=dict(size=10, symbol='circle-open', color=cor, line_width=2),
									 name='fim do trecho util'), row=2, col=1)
		resumo.append(f'  u={uu:+.2f}  t=[{t[i0]:.2f},{t[i1-1]:.2f}]  dv/dt={p[0]:+.3f} m/s2  ->  K={K:.3f}{nota}')

		# trecho longo: quebra em janelas para ver se o ganho cai com a velocidade
		if val.sum() >= 40:
			for w in np.array_split(np.nonzero(val)[0], 4):
				sl = np.polyfit(tt[w], vv[w], 1)[0]
				resumo.append(f'       janela t=[{tt[w][0]:5.2f},{tt[w][-1]:5.2f}]  '
							  f'v_med={vv[w].mean():5.2f}  dv/dt={sl:+.3f}')

print('\n'.join(resumo))

# --- ajuste global v_ponto = K*u - b*v, com K separado por sinal de u ---
vd = np.gradient(v, t)
for s, nome in ((u > 0, 'acelerando'), (u < 0, 'freando')):
	if s.sum() > 10:
		A = np.c_[u[s], -v[s]]
		K, b = np.linalg.lstsq(A, vd[s], rcond=None)[0]
		print(f'  ajuste {nome:10s}: K={K:.3f}  b={b:.3f} 1/s')

# --- onde o carro inverteu o sentido ---
vs = np.convolve(v_real, np.ones(5) / 5, mode='same')          # suaviza para nao pegar ruido de repouso
inv = np.nonzero((vs <= -0.03) & (np.maximum.accumulate(vs) > 0.03))[0]
if len(inv):
	ti = t[inv[0]]
	fig.add_vline(x=ti, line=dict(color='red', width=2),
				  annotation_text='carro inverte o sentido', annotation_position='top')
	print(f'\n  [!] o carro inverte o sentido em t={ti:.2f}s, mas o v logado continua positivo (marcha nao trocada)')

fig.update_yaxes(title_text='u [m/s2]', row=1, col=1)
fig.update_yaxes(title_text='v [m/s]', row=2, col=1)
fig.update_yaxes(title_text='a [m/s2]', row=3, col=1)
fig.update_xaxes(title_text='t [s]', row=3, col=1)
fig.update_layout(height=880, hovermode='x unified', title=f'Resposta longitudinal - {os.path.basename(pasta)}')

saida = os.path.join(pasta, 'analise.html')
fig.write_html(saida)
print('\n->', saida)
webbrowser.open('file://' + os.path.abspath(saida))
