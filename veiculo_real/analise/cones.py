# -*- coding: utf-8 -*-
"""Experimentos de segmentacao dos cones (vermelho = lado direito, verde = lado esquerdo) nos quadros gravados.

Rodar de veiculo_real/analise:
    python cones.py amostra <quadro.png>          # H, S, V e L, a, b em pontos escolhidos (para ajustar limiares)
    python cones.py roda <pasta_quadros> <saida>  # segmenta todos os quadros, escreve o resumo e a folha de overlays

Sem OpenCV (o venv do PC nao tem): so numpy + scipy.ndimage. Mesma geometria de camera do main.py.
"""
import glob
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi

# camera (iguais ao main.py)
ALTURA_CAMERA, FOV_H, LINHA_HORIZONTE, W, H = 0.255, 60.0, 100, 640, 480
FOCO = (W / 2.0) / np.tan(np.deg2rad(FOV_H) / 2.0)   # pixels

########################################
# CORES
########################################
def carrega(caminho):
	return (plt.imread(caminho)[..., :3] * 255).astype(np.uint8)

def para_hsv(img):
	return rgb_to_hsv(img / 255.0) * [360.0, 100.0, 100.0]   # H graus, S e V em %

def para_lab(img):
	# sRGB -> Lab (D65). a* > 0 avermelhado, a* < 0 esverdeado, 0 = cinza: o piso e as paredes ficam em ~0 qualquer que seja a luz
	c = img / 255.0
	lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
	M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
	xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
	f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
	return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

########################################
# SEGMENTACAO: cada metodo devolve (mascara_vermelha, mascara_verde)
########################################
def segmenta_hsv(img, p):
	hsv = para_hsv(img)
	h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
	vermelho = ((h <= p['h_verm_max']) | (h >= p['h_verm_min'])) & (s >= p['s_verm']) & (v >= p['v_min'])
	verde = (h >= p['h_verde'][0]) & (h <= p['h_verde'][1]) & (s >= p['s_verde']) & (v >= p['v_min'])
	return vermelho, verde

def segmenta_lab(img, p):
	lab = para_lab(img)
	L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
	ang = np.degrees(np.arctan2(b, a))   # vermelho do cone ~ 40 graus; madeira laranja ~ 55-65
	vermelho = (a >= p['a_verm']) & (ang >= p['ang_verm'][0]) & (ang <= p['ang_verm'][1]) & (L >= p['L_min'])
	verde = (a <= -p['a_verde']) & (L >= p['L_min'])
	return vermelho, verde

PARAMETROS = {
	'hsv': dict(h_verm_max=12, h_verm_min=345, s_verm=45, h_verde=(135, 195), s_verde=28, v_min=18),
	'lab': dict(a_verm=20, ang_verm=(8, 48), a_verde=9, L_min=18),   # cone vermelho a 27 / coluna marrom a 15 / porta L 12 (amostra)
}
METODOS = {'hsv': segmenta_hsv, 'lab': segmenta_lab}

########################################
# CONEXAS -> CANDIDATOS A CONE
########################################
def limpa(mascara):
	# abre (tira ruido de 1-2 px) e fecha (une a faixa clara do cone, o reflexo, ao corpo)
	m = ndi.binary_opening(mascara, structure=np.ones((2, 2)))
	return ndi.binary_closing(m, structure=np.ones((3, 3)))

def cortes_entre_cones(coluna, vale_max=0.6, largura_min=3):
	# colunas onde cortar uma mancha que junta cones vizinhos. coluna[x] = pixels da mascara na coluna x (um cone e um triangulo:
	# um pico). Entre dois cones lado a lado o perfil cai a ~50% do menor pico (as laterais se cruzam na meia altura)
	c = np.convolve(coluna.astype(float), np.ones(3) / 3.0, mode='same')
	picos = [x for x in range(1, len(c) - 1) if c[x] >= c[x - 1] and c[x] > c[x + 1] and c[x] >= 0.4 * c.max()]
	fundidos = []
	for p in picos:
		if fundidos and p - fundidos[-1] < largura_min:
			fundidos[-1] = p if c[p] > c[fundidos[-1]] else fundidos[-1]
		else:
			fundidos.append(p)
	cortes = []
	for p, q in zip(fundidos[:-1], fundidos[1:]):
		v = p + int(np.argmin(c[p:q + 1]))
		if c[v] <= vale_max * min(c[p], c[q]) and v - (cortes[-1] if cortes else 0) >= largura_min:
			cortes.append(v)
	return cortes

def pe_do_cone(sub, largura_rel=0.85):
	# o cone e mais largo na base; o reflexo no piso so afina para baixo. O pe e a ultima linha ainda quase tao larga quanto a maior
	larguras = sub.sum(1)
	return int(np.nonzero(larguras >= largura_rel * larguras.max())[0].max())

def componentes(mascara, cor, p_forma):
	rotulos, n = ndi.label(limpa(mascara))
	achados = []
	for i, caixa in enumerate(ndi.find_objects(rotulos), start=1):
		ys, xs = caixa
		mancha = rotulos[caixa] == i
		limites = [0] + (cortes_entre_cones(mancha.sum(0)) if p_forma['separa'] else []) + [mancha.shape[1]]
		for a, b in zip(limites[:-1], limites[1:]):
			sub = mancha[:, a:b]
			linhas = np.nonzero(sub.any(1))[0]
			if len(linhas) == 0:
				continue
			sub = sub[linhas[0]:linhas[-1] + 1]
			topo = ys.start + int(linhas[0])
			if p_forma['pe'] and sub.shape[0] >= 8:
				sub = sub[:pe_do_cone(sub) + 1]
			alt, larg = sub.shape
			colunas = np.nonzero(sub.any(0))[0]
			x0 = xs.start + a + int(colunas[0])
			area = int(sub.sum())
			if area < p_forma['area_min'] or topo + alt < LINHA_HORIZONTE + p_forma['abaixo_horizonte']:
				continue
			larg = int(colunas[-1] - colunas[0] + 1)
			if not p_forma['razao'][0] <= alt / larg <= p_forma['razao'][1] or area / (alt * larg) < p_forma['preench_min']:
				continue
			achados.append(dict(cor=cor, x0=x0, x1=x0 + larg, y0=topo, y1=topo + alt, cx=x0 + larg / 2.0, alt=alt, larg=larg, area=area))
	return achados

FORMA = dict(area_min=14, abaixo_horizonte=3, razao=(0.9, 3.0), preench_min=0.38, separa=True,
             pe=False)   # pe=True (pe pela linha mais larga) encurta as caixas, perde cones pequenos e nao melhora o teste de altura (sel: 333 contra 362 cones)

def detecta(img, metodo):
	verm, verde = METODOS[metodo](img, PARAMETROS[metodo])
	return componentes(verm, 'vermelho', FORMA) + componentes(verde, 'verde', FORMA)

########################################
# DISTANCIA
########################################
def distancia_pelo_pe(v_pe):
	# raio do pixel (coluna do centro, linha v_pe) ate o plano do chao; devolve a distancia a frente da lente [m]
	inclinacao = np.arctan((H / 2.0 - LINHA_HORIZONTE) / FOCO)
	dz = -np.sin(inclinacao) - (v_pe - H / 2.0) / FOCO * np.cos(inclinacao)
	dx = np.cos(inclinacao) - (v_pe - H / 2.0) / FOCO * np.sin(inclinacao)
	return np.where(dz < -1e-6, ALTURA_CAMERA / -dz * dx, np.nan)

def no_chao(u, v):
	# pixel (coluna, linha) -> (x a frente da lente, y a esquerda) em metros, no plano do chao. Semelhanca de triangulos
	# (Thales) do raio da camera ate o plano z = -altura; mesma geometria do main.py
	inclinacao = np.arctan((H / 2.0 - LINHA_HORIZONTE) / FOCO)
	a, b = (np.asarray(u, float) - W / 2.0) / FOCO, (np.asarray(v, float) - H / 2.0) / FOCO
	dx, dy, dz = np.cos(inclinacao) - b * np.sin(inclinacao), -a, -np.sin(inclinacao) - b * np.cos(inclinacao)
	t = np.where(dz < -1e-6, ALTURA_CAMERA / -dz, np.nan)
	return t * dx, t * dy

def mapa_dos_cones(img, z_max=7.0, altura_min=10):
	# cones detectados -> posicao no chao pelo pe da caixa. Ignora os de menos de altura_min px (distantes: 1 px de erro = metros)
	pontos = []
	for d in detecta(img, 'lab'):
		x, y = no_chao(d['cx'], d['y1'])
		if d['alt'] >= altura_min and np.isfinite(x) and 0.2 < x < z_max:
			pontos.append((d['cor'], float(x), float(y), d))
	return pontos

def pares(pontos, janela=0.35):
	# suas linhas vermelhas: cada cone verde com o vermelho de profundidade x mais proxima (a menos de 'janela' m)
	verdes = sorted(p for p in pontos if p[0] == 'verde')
	vermelhos = [p for p in pontos if p[0] == 'vermelho']
	usados, saida = set(), []
	for _, xv, yv, _ in verdes:
		cand = [(abs(xr - xv), i) for i, (_, xr, yr, _) in enumerate(vermelhos) if i not in usados and abs(xr - xv) < janela and yr < yv]
		if cand:
			i = min(cand)[1]
			usados.add(i)
			saida.append((xv, yv, vermelhos[i][1], vermelhos[i][2]))
	return saida   # (x_verde, y_verde, x_vermelho, y_vermelho)

def lateral(cx, distancia):
	# posicao lateral [m], esquerda positiva (mesmo sinal do main.py)
	return -(cx - W / 2.0) / FOCO * distancia

########################################
# COMANDOS
########################################
def amostra(caminho):
	img = carrega(caminho)
	hsv, lab = para_hsv(img), para_lab(img)
	pontos = [('cone verde grande', 280, 150), ('cone verde 2', 304, 130), ('cone vermelho grande', 585, 145),
	          ('porta madeira', 40, 100), ('coluna marrom', 305, 60), ('piso', 200, 300), ('parede', 90, 60),
	          ('jeans', 560, 60), ('fita amarela', 415, 330)]
	print(f"{'':22s} {'H':>4s} {'S':>4s} {'V':>4s}   {'L':>4s} {'a':>5s} {'b':>5s} {'ang':>5s}")
	for nome, x, y in pontos:
		hh, ll = np.median(hsv[y - 3:y + 4, x - 3:x + 4].reshape(-1, 3), 0), np.median(lab[y - 3:y + 4, x - 3:x + 4].reshape(-1, 3), 0)
		print(f"{nome:22s} {hh[0]:4.0f} {hh[1]:4.0f} {hh[2]:4.0f}   {ll[0]:4.0f} {ll[1]:5.1f} {ll[2]:5.1f} {np.degrees(np.arctan2(ll[2], ll[1])):5.0f}")

def roda(pasta, saida):
	arquivos = sorted(glob.glob(pasta + '/*.png'))
	resumo = {}
	todos = {}
	for metodo in METODOS:
		t0 = time.perf_counter()
		todos[metodo] = [detecta(carrega(f), metodo) for f in arquivos]
		ms = 1000 * (time.perf_counter() - t0) / len(arquivos)
		n = np.array([[sum(d['cor'] == c for d in q) for c in ('vermelho', 'verde')] for q in todos[metodo]])
		resumo[metodo] = (ms, n)
		print(f"{metodo}: {ms:.0f} ms/quadro (so a segmentacao+conexas, PC); cones por quadro: vermelho {n[:, 0].mean():.1f}, verde {n[:, 1].mean():.1f}; "
		      f"quadros sem nenhum: {(n.sum(1) == 0).sum()}/{len(arquivos)}")
	np.save(saida + '_deteccoes.npy', todos, allow_pickle=True)
	folha(arquivos, todos, saida)
	return arquivos, todos

def desenha(ax, img, deteccoes, titulo):
	ax.imshow(img)
	ax.axhline(LINHA_HORIZONTE, color='c', lw=0.5, ls=':')
	for d in deteccoes:
		cor = 'r' if d['cor'] == 'vermelho' else 'lime'
		ax.add_patch(Rectangle((d['x0'], d['y0']), d['larg'], d['alt'], fill=False, ec=cor, lw=1.4))
	ax.set_title(titulo, fontsize=8)
	ax.axis('off')

def escolhe_quadros(todos, n=8):
	# so quadros em que TODOS os metodos acham ao menos um cone (a olho eu confiro na folha), espalhados no tempo e
	# misturando poucos e muitos cones. Sem isto a folha cai em quadros de parede, mao ou jeans
	k = len(next(iter(todos.values())))
	candidatos = [i for i in range(k) if all(len(todos[m][i]) >= 1 for m in todos)]
	if len(candidatos) <= n:
		return candidatos
	dif = np.array([sum(len(todos[m][i]) for m in todos) for i in candidatos])
	grupos = np.array_split(np.array(candidatos), n)   # um por faixa de tempo: o de contagem mediana da faixa
	return [int(g[np.argsort(dif[np.searchsorted(candidatos, g)])[len(g) // 2]]) for g in grupos]

def folha(arquivos, todos, saida, n=8):
	indices = escolhe_quadros(todos, n)
	print('quadros da folha:', [arquivos[i][-14:-4] for i in indices])
	fig, ax = plt.subplots(len(METODOS), len(indices), figsize=(3.4 * len(indices), 3.0 * len(METODOS)), squeeze=False)
	for r, metodo in enumerate(METODOS):
		for c, i in enumerate(indices):
			desenha(ax[r, c], carrega(arquivos[i]), todos[metodo][i], f"{metodo} {arquivos[i][-14:-4]}  ({len(todos[metodo][i])})")
	plt.tight_layout()
	plt.savefig(saida + '_folha.png', dpi=70)

def altura_do_cone(pasta, saida):
	# Teste da distancia pelo tamanho: o pe do cone (linha de baixo da caixa) da a distancia Z pela geometria do chao;
	# se a caixa e boa, H = alt_px * Z / foco e a mesma para todos os cones (eles tem o mesmo tamanho).
	# A dispersao de H e o erro da distancia por tamanho; pontos fora da faixa sao caixas ruins (fragmento, fusao com a sombra)
	arquivos = sorted(glob.glob(pasta + '/*.png'))
	fig, ax = plt.subplots(1, len(METODOS), figsize=(6.5 * len(METODOS), 4.6), squeeze=False)
	for c, metodo in enumerate(METODOS):
		Z, Hc, cor = [], [], []
		for f in arquivos:
			for d in detecta(carrega(f), metodo):
				z = float(distancia_pelo_pe(np.array(d['y1'], float)))
				if np.isfinite(z) and z > 0:
					Z.append(z)
					Hc.append(d['alt'] * z / FOCO)
					cor.append('r' if d['cor'] == 'vermelho' else 'g')
		Z, Hc = np.array(Z), np.array(Hc)
		med = np.median(Hc)
		mad = np.median(np.abs(Hc - med))
		perto = Z < 2.5   # longe a caixa tem poucos pixels: 1 px de erro = muitos cm
		print(f"{metodo}: {len(Z)} cones; H mediana {100 * med:.1f} cm, MAD {100 * mad:.1f} cm ({100 * mad / med:.0f}%); "
		      f"dentro de +-30% da mediana: {np.mean(np.abs(Hc / med - 1) < 0.3) * 100:.0f}% (todos), {np.mean(np.abs(Hc[perto] / med - 1) < 0.3) * 100:.0f}% (Z < 2,5 m)")
		a = ax[0, c]
		a.scatter(Z, 100 * Hc, c=cor, s=10, alpha=0.6)
		a.axhline(100 * med, color='k', lw=0.8)
		a.set_xlabel('distancia pelo pe do cone Z [m]')
		a.set_ylabel('altura estimada H [cm]')
		a.set_title(f"{metodo}: H mediana {100 * med:.1f} cm, MAD {100 * mad:.1f} cm")
		a.set_ylim(0, 60)
	plt.tight_layout()
	plt.savefig(saida, dpi=80)

def folhas_de_rotulo(saida, por_folha, *caminhos):
	# pre-anotacao para conferir a olho: faixa dos cones (linhas 70-290) ampliada, caixas do Lab numeradas, grade a cada 40 px
	for k in range(0, len(caminhos), por_folha):
		grupo = caminhos[k:k + por_folha]
		fig, ax = plt.subplots(len(grupo), 1, figsize=(16, 4.4 * len(grupo)), squeeze=False)
		for a, caminho in zip(ax[:, 0], grupo):
			img = carrega(caminho)
			achados = sorted(detecta(img, 'lab'), key=lambda d: d['cx'])
			a.imshow(img)
			for n, d in enumerate(achados):
				a.add_patch(Rectangle((d['x0'], d['y0']), d['larg'], d['alt'], fill=False, ec='r' if d['cor'] == 'vermelho' else 'lime', lw=1.2))
				a.text(d['x0'], d['y0'] - 2, str(n), color='yellow', fontsize=8, ha='left', va='bottom')
			a.set_ylim(290, 70)
			a.set_xlim(0, 640)
			a.set_xticks(range(0, 641, 40))
			a.set_yticks(range(80, 290, 40))
			a.grid(color='w', alpha=0.25, lw=0.5)
			a.set_title(caminho.split('/')[-1] + f"  ({len(achados)} caixas)", fontsize=9)
		plt.tight_layout()
		plt.savefig(f"{saida}_{k // por_folha}.png", dpi=62)
		plt.close(fig)

def teste_do_mapa(pasta, saida, quadros_vista):
	# 1) largura do corredor em cada par (a linha vermelha do esboco, em metros) contra a profundidade: tem que ser ~constante
	# 2) vista de cima de alguns quadros: cones, ligacoes, ponto medio
	arquivos = sorted(glob.glob(pasta + '/*.png'))
	larguras, prof = [], []
	for f in arquivos:
		for xv, yv, xr, yr in pares(mapa_dos_cones(carrega(f))):
			larguras.append(yv - yr)
			prof.append((xv + xr) / 2.0)
	larguras, prof = np.array(larguras), np.array(prof)
	med = np.median(larguras)
	print(f"{len(larguras)} pares em {len(arquivos)} quadros; largura do corredor mediana {100 * med:.0f} cm, MAD {100 * np.median(np.abs(larguras - med)):.0f} cm; "
	      f"dentro de +-20%: {100 * np.mean(np.abs(larguras / med - 1) < 0.2):.0f}%")
	for lo, hi in [(0.2, 1.5), (1.5, 3), (3, 5), (5, 7)]:
		m = (prof >= lo) & (prof < hi)
		if m.sum():
			print(f"   profundidade {lo}-{hi} m: {m.sum()} pares, largura mediana {100 * np.median(larguras[m]):.0f} cm, MAD {100 * np.median(np.abs(larguras[m] - np.median(larguras[m]))):.0f} cm")
	fig, ax = plt.subplots(1, 1 + len(quadros_vista), figsize=(4.2 * (1 + len(quadros_vista)), 5.2))
	ax[0].scatter(prof, 100 * larguras, s=10)
	ax[0].axhline(100 * med, color='k', lw=0.8)
	ax[0].set_xlabel('profundidade x do par [m]')
	ax[0].set_ylabel('largura do corredor [cm]')
	ax[0].set_ylim(0, 400)
	ax[0].set_title('largura por par (tem que ser ~constante)', fontsize=9)
	for a, q in zip(ax[1:], quadros_vista):
		pts = mapa_dos_cones(carrega(q))
		for cor, x, y, _ in pts:
			a.plot(y, x, 'o', color='r' if cor == 'vermelho' else 'g', ms=6)
		for xv, yv, xr, yr in pares(pts):
			a.plot([yv, yr], [xv, xr], '-', color='gray', lw=0.8)
			a.plot((yv + yr) / 2.0, (xv + xr) / 2.0, 'k.', ms=5)
		a.plot(0, 0, 'b^')
		a.set_xlim(2.5, -2.5)   # esquerda a esquerda
		a.set_ylim(0, 7)
		a.set_aspect('equal')
		a.grid(alpha=0.3)
		a.set_title(q.split('/')[-1][:-4].replace('quadro_', ''), fontsize=9)
		a.set_xlabel('y [m] (esquerda +)')
	plt.tight_layout()
	plt.savefig(saida, dpi=75)

def detalhe(saida, *caminhos):
	# um quadro por linha, um metodo por coluna, em tamanho legivel
	fig, ax = plt.subplots(len(caminhos), len(METODOS), figsize=(6.4 * len(METODOS), 4.8 * len(caminhos)), squeeze=False)
	for r, caminho in enumerate(caminhos):
		img = carrega(caminho)
		for c, metodo in enumerate(METODOS):
			d = detecta(img, metodo)
			desenha(ax[r, c], img, d, f"{metodo} {caminho[-14:-4]}: {sum(x['cor'] == 'vermelho' for x in d)} vermelhos, {sum(x['cor'] == 'verde' for x in d)} verdes")
	plt.tight_layout()
	plt.savefig(saida, dpi=75)

if __name__ == '__main__':
	if sys.argv[1] == 'amostra':
		amostra(sys.argv[2])
	elif sys.argv[1] == 'mapa':
		teste_do_mapa(sys.argv[2], sys.argv[3], sys.argv[4:])
	elif sys.argv[1] == 'rotula':
		folhas_de_rotulo(sys.argv[2], 3, *sys.argv[3:])
	elif sys.argv[1] == 'altura':
		altura_do_cone(sys.argv[2], sys.argv[3])
	elif sys.argv[1] == 'detalhe':
		detalhe(sys.argv[2], *sys.argv[3:])
	else:
		roda(sys.argv[2], sys.argv[3])
