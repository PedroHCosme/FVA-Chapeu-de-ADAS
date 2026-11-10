# -*- coding: utf-8 -*-
"""Compara os filtros de cor do main.py nos quadros gravados no carro (GRAVA_QUADROS).

    ../../simulador/.venv/Scripts/python.exe compara_filtros.py <pasta> [<pasta> ...]      (de veiculo_real/analise)

<pasta> = pasta de uma corrida, ou uma pasta acima dela: acha todo quadro_*.png dentro.
Cada quadro e medido do zero (sem previsao): os quadros ficam 0,5 s distantes, a previsao nao ajudaria aqui.
Imprime, por filtro: % de quadros com medida, saltos de e e psi entre quadros seguidos e tempo de processamento.
Grava compara_filtros.csv (por quadro) e compara_filtros.png (quadros em que os filtros mais discordam) na primeira pasta.
Sem verdade de campo: salto pequeno e medida presente sao indicios, a figura e que confirma.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analisa as A

# (nome, FILTRO_RELATIVO, AMARELO_REL_MIN)
FILTROS = [('saturacao', False, None), ('relativo 12', True, 12), ('relativo 16', True, 16), ('relativo 24', True, 24)]
SALTO_E = 0.05            # m -- salto grande de e entre quadros seguidos
SALTO_PSI = np.deg2rad(10.0)   # rad -- salto grande de psi
N_FIGURA = 12             # quadros na figura


def le_quadro(caminho):
	# PNG gravado em BGR pelo cv2: o matplotlib le como se fosse RGB, entao os canais ja vem na ordem do main.py
	return (plt.imread(caminho)[..., :3] * 255).astype(np.uint8)


def acha_quadros(pastas):
	quadros = []
	for pasta in pastas:
		quadros += sorted(Path(pasta).rglob('quadro_*.png'))
	return quadros


def mede_com_filtro(m, img, relativo, limiar):
	# (medida ou None, pixels amarelos, pixels usados, ms da mascara, ms da medida inteira, desenho)
	m.FILTRO_RELATIVO = relativo
	if limiar is not None:
		m.AMARELO_REL_MIN = limiar
	m._estado.clear()
	t0 = time.perf_counter()
	m.mascara_amarela(img[int(img.shape[0] * m.FAIXA_VISAO[0]):int(img.shape[0] * m.FAIXA_VISAO[1])])
	t1 = time.perf_counter()
	medida = m.mede_fita(img, None)
	t2 = time.perf_counter()
	if medida is not None and abs(medida[0]) > m.E_MAX_FITA:
		medida = None
	uu, _, perto, _ = m._estado.get('desenho', (np.array([]), None, None, None))
	usados = int(np.count_nonzero(perto)) if perto is not None and medida is not None else 0
	return medida, len(uu), usados, 1e3 * (t1 - t0), 1e3 * (t2 - t1), m._estado.get('desenho')


def resume(linhas):
	# linhas de um filtro, em ordem de tempo
	com = [l for l in linhas if l['medida'] is not None]
	saltos = [(abs(a['medida'][0] - b['medida'][0]), abs(a['medida'][1] - b['medida'][1]))
	          for a, b in zip(linhas, linhas[1:])
	          if a['medida'] is not None and b['medida'] is not None and a['seq'] == b['seq']]
	saltos = np.array(saltos).reshape(-1, 2)
	return dict(pct=100 * len(com) / len(linhas),
	            usados=np.median([l['usados'] / max(l['amarelos'], 1) for l in com]) * 100 if com else 0.0,
	            de=100 * np.median(saltos[:, 0]) if len(saltos) else np.nan,
	            dpsi=np.degrees(np.median(saltos[:, 1])) if len(saltos) else np.nan,
	            grandes=int(np.sum((saltos[:, 0] > SALTO_E) | (saltos[:, 1] > SALTO_PSI))),
	            pares=len(saltos),
	            ms_mascara=np.median([l['ms_mascara'] for l in linhas]),
	            ms_total=np.median([l['ms_total'] for l in linhas]))


def discordancia(por_filtro, i):
	# 1 se so alguns filtros mediram; senao a maior diferenca de e entre eles [m]
	medidas = [por_filtro[nome][i]['medida'] for nome, _, _ in FILTROS]
	if any(x is None for x in medidas) and any(x is not None for x in medidas):
		return 1.0
	if all(x is None for x in medidas):
		return 0.0
	return float(np.ptp([x[0] for x in medidas]))


def desenha(m, img, desenho, medida):
	vis = img.copy()
	if desenho is not None:
		uu, vv, perto, circ = desenho
		vis[vv, uu] = (0, 0, 255)
		if circ is not None and medida is not None:
			vis[vv[perto], uu[perto]] = (0, 255, 0)
			m._desenha_fita(vis, circ, np.arange(0.3, m.X_MAX_VISAO, 0.05))
	return vis[..., ::-1]   # para RGB


def figura(m, quadros, por_filtro, saida):
	piores = sorted(range(len(quadros)), key=lambda i: -discordancia(por_filtro, i))[:N_FIGURA]
	piores = sorted(piores)
	fig, ax = plt.subplots(len(piores), len(FILTROS) + 1, figsize=(2.6 * (len(FILTROS) + 1), 2.0 * len(piores)), squeeze=False)
	for lin, i in enumerate(piores):
		img = le_quadro(quadros[i])
		ax[lin, 0].imshow(img[..., ::-1])
		ax[lin, 0].set_ylabel(quadros[i].stem[7:], fontsize=7)
		for col, (nome, _, _) in enumerate(FILTROS):
			linha = por_filtro[nome][i]
			ax[lin, col + 1].imshow(desenha(m, img, linha['desenho'], linha['medida']))
			med = linha['medida']
			texto = 'sem medida' if med is None else f"e {100 * med[0]:+.1f} cm, psi {np.degrees(med[1]):+.1f}"
			ax[lin, col + 1].set_title(f"{nome}: {texto}" if lin == 0 else texto, fontsize=7)
	ax[0, 0].set_title('quadro cru', fontsize=7)
	for a in ax.ravel():
		a.set_xticks([])
		a.set_yticks([])
	fig.tight_layout(pad=0.3)
	fig.savefig(saida, dpi=80)
	plt.close(fig)


def grava_csv(quadros, por_filtro, saida):
	with open(saida, 'w', newline='', encoding='utf-8') as f:
		w = csv.writer(f)
		w.writerow(['quadro', 'filtro', 'e_cm', 'psi_graus', 'amarelos', 'usados', 'ms_total'])
		for i, q in enumerate(quadros):
			for nome, _, _ in FILTROS:
				l = por_filtro[nome][i]
				med = l['medida']
				w.writerow([q.relative_to(q.parents[2]) if len(q.parents) > 2 else q, nome,
				            '' if med is None else f'{100 * med[0]:.2f}', '' if med is None else f'{np.degrees(med[1]):.2f}',
				            l['amarelos'], l['usados'], f"{l['ms_total']:.1f}"])


def compara(quadros, pasta_saida):
	m = A.carrega_main()
	por_filtro = {nome: [] for nome, _, _ in FILTROS}
	for q in quadros:
		img = le_quadro(q)
		for nome, relativo, limiar in FILTROS:
			medida, amarelos, usados, ms_m, ms_t, desenho = mede_com_filtro(m, img, relativo, limiar)
			por_filtro[nome].append(dict(seq=q.parent, medida=medida, amarelos=amarelos, usados=usados, ms_mascara=ms_m, ms_total=ms_t, desenho=desenho))
	print(f'{len(quadros)} quadros')
	print(f"{'filtro':<14}{'com medida':>11}{'pixels usados':>15}{'salto e':>10}{'salto psi':>11}{'saltos grandes':>16}{'mascara':>10}{'total':>9}")
	resumo = {}
	for nome, _, _ in FILTROS:
		r = resumo[nome] = resume(por_filtro[nome])
		print(f"{nome:<14}{r['pct']:>10.0f}%{r['usados']:>14.0f}%{r['de']:>7.1f} cm{r['dpsi']:>7.1f} grau{r['grandes']:>9d} de {r['pares']:<4d}{r['ms_mascara']:>7.1f} ms{r['ms_total']:>6.1f} ms")
	grava_csv(quadros, por_filtro, Path(pasta_saida) / 'compara_filtros.csv')
	figura(m, quadros, por_filtro, Path(pasta_saida) / 'compara_filtros.png')
	print('figura e tabela em', pasta_saida)
	return resumo


if __name__ == '__main__':
	if len(sys.argv) < 2:
		sys.exit(__doc__)
	achados = acha_quadros(sys.argv[1:])
	if not achados:
		sys.exit(f'Nenhum quadro_*.png em {sys.argv[1:]}')
	compara(achados, sys.argv[1])
