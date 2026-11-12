# -*- coding: utf-8 -*-
"""O GUIA = 'cones' do main.py nos quadros reais gravados (sem o carro).  Rodar de veiculo_real/analise:
    python cones_reais.py <pasta_com_quadros_png> [saida.png]
Mede (e, psi_e, largura) do corredor em cada quadro, o tempo por quadro e desenha o resultado de alguns. Sem verdade de campo: o que da para
conferir sozinho e a largura do corredor (tem que ser ~constante) e se e / psi_e variam de forma continua de um quadro para o outro.
"""
import glob
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import testa_cones as TC


def carrega_bgr(caminho):
	return (plt.imread(caminho)[..., :3] * 255).astype(np.uint8)[..., ::-1].copy()   # o carro entrega BGR


def main(pasta, saida):
	m = TC.carrega()
	arquivos = sorted(glob.glob(pasta + '/*.png'))
	linhas, tempos = [], []
	for f in arquivos:
		img = carrega_bgr(f)
		m._estado.clear()
		t0 = time.perf_counter()
		medida = m.mede_cones(img, None)
		tempos.append(1000 * (time.perf_counter() - t0))
		cones, arco = m._estado['desenho_cones']
		x, y, esq = m._cones_no_chao(cones)
		largura = None
		if medida is not None:
			aj = m.ajusta_corredor(x, y, esq, 0.0)   # esq = e verde (o lado de cada cor e descoberto)
			largura = m._arco_do_meio(aj)[1]
		linhas.append((f.split('/')[-1].replace('quadro_', '')[:-4], len(cones), len(x), medida, largura))
	ok = [l for l in linhas if l[3] is not None]
	print(f"{len(ok)}/{len(linhas)} quadros com medida; tempo por quadro: mediana {np.median(tempos):.0f} ms, maximo {max(tempos):.0f} ms (PC, numpy+scipy)")
	larguras = np.array([l[4] for l in ok if l[4] is not None])
	if len(larguras):
		print(f"largura do corredor: mediana {100 * np.median(larguras):.0f} cm, MAD {100 * np.median(np.abs(larguras - np.median(larguras))):.0f} cm ({len(larguras)} quadros com as duas fileiras)")
	for nome, n_det, n_chao, medida, largura in linhas:
		texto = 'sem medida' if medida is None else f"e {100 * medida[0]:+6.1f} cm  psi {np.degrees(medida[1]):+6.1f}  c {medida[2]:+.2f}"
		print(f"  {nome}: {n_det:2d} cones ({n_chao:2d} ao alcance)  {texto}" + ('' if largura is None else f"  largura {100 * largura:.0f} cm"))
	if saida:
		escolhidos = np.linspace(0, len(arquivos) - 1, min(8, len(arquivos))).astype(int)
		fig, ax = plt.subplots(2, 4, figsize=(18, 7))
		for a, i in zip(ax.ravel(), escolhidos):
			img = carrega_bgr(arquivos[i])
			m._estado.clear()
			medida = m.mede_cones(img, None)
			cones, arco = m._estado['desenho_cones']
			vis = img.copy()
			m._desenha_cones(vis, cones, arco, np.arange(0.3, m.X_MAX_CONES, 0.05))
			a.imshow(vis[..., ::-1])
			a.set_title(linhas[i][0] + ('  sem medida' if medida is None else f"  e {100 * medida[0]:+.0f} cm  psi {np.degrees(medida[1]):+.0f}"), fontsize=9)
			a.axis('off')
		plt.tight_layout()
		plt.savefig(saida, dpi=70)


if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
