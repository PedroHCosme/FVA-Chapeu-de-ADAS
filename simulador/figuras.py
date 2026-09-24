# -*- coding: utf-8 -*-
# Gera as figuras do IDENTIFICACAO.md. Roda de dentro de simulador/.
#   python figuras.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle, Circle

SAIDA = 'figuras'
os.makedirs(SAIDA, exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.grid': True,
					 'grid.alpha': .3, 'axes.titlesize': 10})

K, C, A0 = 1.00, 0.00425, 0.095


def log(nome):
	return np.genfromtxt(os.path.join('logs', nome, 'car.csv'), delimiter=',', names=True)


def salva(fig, nome):
	fig.tight_layout()
	p = os.path.join(SAIDA, nome)
	fig.savefig(p)
	plt.close(fig)
	print('  ', p)


# ---------------------------------------------------------------- 01 forcas
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.add_patch(Rectangle((3.2, 1.6), 3.6, 1.2, fc='#cfd8dc', ec='#455a64', lw=1.5))
for cx in (4.0, 6.0):
	ax.add_patch(Circle((cx, 1.6), .32, fc='#37474f'))
setas = [
	(6.8, 2.2, 2.2, 0, '#2e7d32', 'Fx  força de tração\n(o que o motor faz)'),
	(3.2, 2.5, -1.7, 0, '#c62828', 'F_aero = ½ρCdAf·v²\n(cresce com v²)'),
	(3.2, 1.9, -1.2, 0, '#ef6c00', 'R = f·mg\n(atrito, constante)'),
	(5.0, 1.5, 0, -1.1, '#1565c0', 'peso  mg'),
]
for x, y, dx, dy, cor, txt in setas:
	ax.add_patch(FancyArrow(x, y, dx, dy, width=.05, head_width=.18, head_length=.25,
							fc=cor, ec=cor, length_includes_head=True))
	ax.text(x + dx + (.15 if dx > 0 else (-.15 if dx < 0 else .2)), y + dy - (.3 if dy < 0 else 0),
			txt, color=cor, fontsize=8.5, va='center',
			ha='left' if dx > 0 or dy < 0 else 'right')
ax.annotate('', xy=(8.6, 1.0), xytext=(6.9, 1.0), arrowprops=dict(arrowstyle='->', lw=1.2))
ax.text(7.75, .72, 'sentido do movimento (v > 0)', ha='center', fontsize=8.5)
ax.set_xlim(0, 10.2); ax.set_ylim(0, 3.6); ax.axis('off')
ax.set_title('As forças da equação  m·v̇ = Fx − F_aero − F_grad − R', fontsize=10)
salva(fig, '01_forcas.png')

# ------------------------------------------------------- 02 peso de cada termo
v = np.linspace(0, 6, 200)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(v, K * 1.0 * np.ones_like(v), lw=2.2, color='#2e7d32', label='K·u  com u=1  (motor)')
ax.plot(v, C * v ** 2, lw=2.2, color='#c62828', label='c·v²  (arrasto)')
ax.plot(v, A0 * np.ones_like(v), lw=2.2, color='#ef6c00', label='a₀  (atrito seco)')
ax.axvspan(0, 2, color='gray', alpha=.12)
ax.text(1.0, .55, 'faixa de\noperação', ha='center', fontsize=8.5, color='#555')
ax.set_xlabel('velocidade v [m/s]'); ax.set_ylabel('contribuição [m/s²]')
ax.set_title('Quanto cada termo pesa, em m/s², conforme a velocidade')
ax.legend(fontsize=8.5); ax.set_ylim(0, 1.1)
salva(fig, '02_termos.png')

# ------------------------------------------------- 03 anatomia de um ensaio
d = log('20260915_231556')
t, vv, uu = d['t'], d['v'], d['u']
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True,
							 gridspec_kw={'height_ratios': [1, 2]})
a1.step(t, uu, where='post', color='#455a64', lw=1.8)
a1.set_ylabel('u [m/s²]')
a2.plot(t, vv, lw=2, color='#1565c0')
a2.axhline(0, color='k', lw=.8)
for x0, x1, cor, txt in [(0, 3.0, '#2e7d32', 'acelera\n(mede K₊)'),
						 (3.0, 4.4, '#c62828', 'freia\n(mede K₋)'),
						 (4.4, 7.2, '#6a1b9a', 'acelera de ré'),
						 (7.2, 10, '#ef6c00', 'roda-livre de ré\n(mede a₀)')]:
	for a in (a1, a2):
		a.axvspan(x0, x1, color=cor, alpha=.10)
	a2.text((x0 + x1) / 2, -1.45, txt, ha='center', fontsize=8, color=cor)
a2.set_ylim(-1.8, 1.5); a2.set_ylabel('v [m/s]'); a2.set_xlabel('t [s]')
a1.set_title('Anatomia de um ensaio: cada trecho mede um parâmetro diferente')
salva(fig, '03_ensaio.png')

# --------------------------------------------------- 04 o que e ler dv/dt
d = log('20260915_203340')
t, vv = d['t'], d['v']
m = (t >= .3) & (t <= 1.3)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(t[t < 2.2], vv[t < 2.2], 'o-', ms=3, lw=1.2, color='#1565c0', label='amostras do log')
p = np.polyfit(t[m], vv[m], 1)
ax.plot(t[m], np.polyval(p, t[m]), '--', lw=2, color='crimson',
		label=f'reta ajustada: inclinação = {p[0]:.3f} m/s²')
i0, i1 = np.argmin(abs(t - .5)), np.argmin(abs(t - 1.0))
ax.plot([t[i0], t[i1], t[i1]], [vv[i0], vv[i0], vv[i1]], color='#ef6c00', lw=1.5)
ax.text(.75, vv[i0] - .09, 'Δt = 0,50 s', ha='center', color='#ef6c00', fontsize=8.5)
ax.text(t[i1] + .04, (vv[i0] + vv[i1]) / 2, 'Δv = 0,44 m/s', color='#ef6c00', fontsize=8.5)
ax.set_xlabel('t [s]'); ax.set_ylabel('v [m/s]')
ax.set_title('v̇ é a inclinação da curva:  Δv/Δt = 0,44/0,50 = 0,88 m/s²')
ax.legend(fontsize=8.5, loc='upper left')
salva(fig, '04_inclinacao.png')

# ------------------------------------------ 05 separando c com duas janelas
d = log('20260915_203340')
t, vv = d['t'], d['v']
jan, vm, sl = [], [], []
for lo, hi in [(.25, 2.15), (2.2, 4.1), (4.15, 6.05), (6.1, 8.0)]:
	m = (t >= lo) & (t <= hi)
	vm.append(vv[m].mean()); sl.append(np.polyfit(t[m], vv[m], 1)[0]); jan.append((lo, hi))
vm, sl = np.array(vm), np.array(sl)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
a1.plot(t[t <= 8], vv[t <= 8], lw=1.5, color='#1565c0')
for (lo, hi), s, m_ in zip(jan, sl, vm):
	a1.axvspan(lo, hi, alpha=.12, color='crimson')
	a1.text((lo + hi) / 2, .5, f'{s:.3f}', ha='center', fontsize=8, color='crimson')
a1.set_xlabel('t [s]'); a1.set_ylabel('v [m/s]')
a1.set_title('mesmo u=1, quatro janelas de velocidade\n(números = inclinação medida)')
a2.plot(vm, sl, 'o', ms=8, color='#1565c0', label='medido')
g = np.linspace(.5, 6.5, 100)
pl = np.polyfit(vm, sl, 1)
q = np.linalg.lstsq(np.c_[np.ones(4), -vm ** 2], sl, rcond=None)[0]
a2.plot(g, np.polyval(pl, g), '--', color='#ef6c00', label='arrasto linear  (K−b·v)')
a2.plot(g, q[0] - q[1] * g ** 2, '-', color='crimson', label='arrasto quadrático  (K−c·v²)')
a2.set_xlabel('velocidade média da janela [m/s]'); a2.set_ylabel('v̇ medido [m/s²]')
a2.set_title('a queda de v̇ com v revela o arrasto'); a2.legend(fontsize=8)
salva(fig, '05_separa_c.png')

# ------------------------------------------------------- 06 o polo fantasma
d = log('20260915_201815')
t, vv, uu = d['t'], d['v'], d['u']
m = (uu == 0) & (t > 3.1)
tc, vc = t[m], vv[m]
b = -np.polyfit(tc, np.log(vc), 1)[0]
pl = np.polyfit(tc, vc, 1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
a1.plot(tc, vc, 'o', ms=4, color='#1565c0', label='medido (u=0)')
a1.plot(tc, vc[0] * np.exp(-b * (tc - tc[0])), '-', lw=2, color='crimson',
		label=f'exponencial  τ={1/b:.1f}s')
a1.plot(tc, np.polyval(pl, tc), '--', lw=2, color='#2e7d32',
		label=f'reta  a={pl[0]:.3f} m/s²')
a1.set_xlabel('t [s]'); a1.set_ylabel('v [m/s]')
a1.set_title('Na faixa estreita do ensaio,\nos dois modelos são idênticos')
a1.legend(fontsize=8)
vg = np.linspace(0, 7, 100)
# as duas curvas sao ancoradas no MESMO ponto medido a 1 m/s
K1 = sl[0] + b * vm[0]                    # 1a ordem: v_ponto = K1 - b*v
K2 = sl[0] + C * vm[0] ** 2               # real:     v_ponto = K2 - c*v^2
a2.plot(vg, K1 - b * vg, '-', lw=2, color='crimson', label='1ª ordem: v̇ = K − b·v')
a2.plot(vg, K2 - C * vg ** 2, '--', lw=2, color='#2e7d32', label='real: v̇ = K − c·v²')
a2.plot(vm, sl, 'o', ms=8, color='#1565c0', label='medido')
prev = K1 - b * vm[-1]
a2.annotate('', xy=(vm[-1], sl[-1]), xytext=(vm[-1], prev),
			arrowprops=dict(arrowstyle='<->', color='k', lw=1.3))
a2.text(vm[-1] - .25, (sl[-1] + prev) / 2, f'1ª ordem previa {prev:.2f}\nmedido {sl[-1]:.2f}',
		ha='right', fontsize=8.5)
a2.set_xlabel('v [m/s]'); a2.set_ylabel('v̇ [m/s²]')
a2.set_title('Extrapolando para 6 m/s, eles discordam\ne o dado escolhe o vencedor')
a2.legend(fontsize=8)
salva(fig, '06_polo_fantasma.png')

# --------------------------------------------- 07 coast frente x re (atrito)
df, dr = log('20260915_201815'), log('20260915_231556')
mf = (df['u'] == 0) & (df['t'] > 3.1)
mr = (dr['u'] == 0) & (dr['t'] > 7.2)
fig, ax = plt.subplots(figsize=(7.0, 3.6))
tf, vf = df['t'][mf] - df['t'][mf][0], df['v'][mf]
tr, vr = dr['t'][mr] - dr['t'][mr][0], dr['v'][mr]
ax.plot(tf, vf, 'o-', ms=3, color='#1565c0', label=f'para frente: {np.polyfit(tf,vf,1)[0]:+.3f} m/s²')
ax.plot(tr, vr, 'o-', ms=3, color='#c62828', label=f'de ré:       {np.polyfit(tr,vr,1)[0]:+.3f} m/s²')
ax.axhline(0, color='k', lw=.8)
ax.set_xlabel('tempo desde o início da roda-livre [s]'); ax.set_ylabel('v [m/s]')
ax.set_title('Roda-livre nos dois sentidos: as inclinações são simétricas\n'
			 '→ é atrito (0,093 m/s²), não rampa (0,0006 m/s²)')
ax.legend(fontsize=8.5)
salva(fig, '07_frente_re.png')

# ------------------------------------------------------------- 08 o sinal de v
da, db = log('20260915_202657'), log('20260915_231556')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
for ax, d_, tit in [(a1, da, 'ANTES: sinal vem da marcha'), (a2, db, 'DEPOIS: sinal vem das rodas')]:
	t_, v_, x_ = d_['t'], d_['v'], d_['x']
	vp = np.r_[0, np.hypot(np.diff(x_), np.diff(d_['y'])) / np.diff(t_) * np.sign(np.diff(x_))]
	ax.plot(t_, v_, lw=2, color='#1565c0', label='v do log')
	ax.plot(t_, vp, '--', lw=1.4, color='crimson', label='v real (da posição)')
	ax.axhline(0, color='k', lw=.8)
	ax.set_title(tit); ax.set_xlabel('t [s]'); ax.legend(fontsize=8)
a1.set_ylabel('v [m/s]')
a1.annotate('carro anda de ré,\nlog diz positivo', xy=(6, .12), xytext=(5.2, .75),
			arrowprops=dict(arrowstyle='->', color='crimson'), fontsize=8.5, color='crimson')
salva(fig, '08_sinal_v.png')

print('ok')
