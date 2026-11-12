# -*- coding: utf-8 -*-
"""Confere o GUIA = 'cones' do main.py sem o carro.  Rodar de veiculo_real/analise:  python testa_cones.py

Mundo sintetico: cones (triangulos verde a esquerda, vermelho a direita) de 17 cm, corredor de 1,2 m, um cone a cada 0,5 m, vistos pela
camera pinhole de geometria conhecida (a mesma do testa_stanley.py). Usa o mesmo mede_cones / processa_quadro / direcao do main.py.
1. leitura estatica: (e, psi_e) do CENTRO do corredor com erro pequeno, em varias posicoes e angulos; com um cone faltando, com um
   falso positivo e com so uma fileira visivel
2. malha fechada: bicicleta + camera a 15 Hz com 1 quadro de atraso + servo de 1a ordem; corredor reto e em curva (raio 4 m no centro)
"""
import types

import numpy as np
import testa_stanley as T

T.ALT, T.XC = 0.255, 0.175
T.GX, T.GY, T.CHAO = T.chao_dos_pixels()
ALTURA_CONE, LARGURA_CONE, LARGURA_CORREDOR, PASSO = 0.17, 0.12, 1.2, 0.5
BGR_VERDE, BGR_VERMELHO, BGR_PAREDE = (85, 105, 30), (45, 45, 140), (215, 215, 210)


def corredor_reto(comprimento=30.0, x0=-1.0):
	xs = np.arange(x0, comprimento, PASSO)
	return [('verde', x, LARGURA_CORREDOR / 2) for x in xs] + [('vermelho', x, -LARGURA_CORREDOR / 2) for x in xs]


def corredor_em_curva(R=4.0, x_reta=3.0, angulo_max=np.pi):
	# reta ate x = x_reta; depois arco para a esquerda, centro em (x_reta, R). Fileira esquerda no raio R - meia largura
	cones = [(c, x, y) for c, x, y in corredor_reto(x_reta, -1.0)]
	meia = LARGURA_CORREDOR / 2
	for cor, raio in (('verde', R - meia), ('vermelho', R + meia)):
		for s in np.arange(PASSO * np.sqrt(R / raio), angulo_max * raio, PASSO):
			ang = s / raio
			cones.append((cor, x_reta + raio * np.sin(ang), R - raio * np.cos(ang)))
	return cones


def renderiza(cones, xf, yf, th, rng=None):
	# imagem BGR do eixo dianteiro (xf, yf, th): parede acima do horizonte, piso com ruido, cones do longe para o perto
	img = np.repeat(T.PISO[..., :1], 3, axis=2).astype(float)   # piso neutro (o ruido do PISO e independente por canal: vira cor)
	img[:T.LH] = BGR_PAREDE
	vistos = []
	for cor, X, Y in cones:
		dx, dy = np.cos(th) * (X - xf) + np.sin(th) * (Y - yf), -np.sin(th) * (X - xf) + np.cos(th) * (Y - yf)
		if dx < 0.3 or dx > 12:
			continue
		p = np.array([dx - T.XC, dy, -T.ALT])
		profundidade = p @ T.Z_C
		u, v = T.W / 2 + T.F * (p @ T.X_C) / profundidade, T.H / 2 + T.F * (p @ T.Y_C) / profundidade
		vistos.append((profundidade, cor, u, v))
	for profundidade, cor, u, v in sorted(vistos, reverse=True):
		alt, larg = T.F * ALTURA_CONE / profundidade, T.F * LARGURA_CONE / profundidade
		for linha in range(int(max(v - alt, 0)), int(min(v, T.H - 1)) + 1):
			meia = larg / 2 * (linha - (v - alt)) / alt
			esq, dir = max(int(u - meia), 0), min(int(u + meia) + 1, T.W)   # sem isto um cone fora do campo a esquerda vira fatia negativa
			if dir > esq:
				img[linha, esq:dir] = BGR_VERDE if cor == 'verde' else BGR_VERMELHO
	ruido = (rng or np.random.default_rng(0)).normal(0, 2, img.shape[:2])
	return np.clip(img + ruido[..., None], 0, 255).astype(np.uint8)


def carrega():
	m = T.carrega()
	m.GUIA = 'cones'
	m.K_STANLEY, m.K_SOFT, m.TAXA_RAMPA_V, m.V_CURVA, m.V_GUARDA = 2.33, 0.2, 0.9, 1.0, 1.6
	m.ADAS_ON, m.GRAVA_QUADROS, m.BUSCA_ARCO, m.BUSCA_RE = False, False, False, False
	m._estado.clear()
	return m


def verdade(xf, yf, th, cones_x=None):
	# (e, psi) do centro de um corredor reto em y = 0 no frame do eixo dianteiro
	return -yf * np.cos(th) - 0.0, -th


def teste_leitura_estatica():
	m = carrega()
	cones = corredor_reto()
	casos = [(0.0, 0.0), (0.10, 0.0), (-0.12, 0.0), (0.0, 0.2), (0.06, -0.15), (-0.08, 0.25)]   # (y do eixo, theta): centro em y = 0
	pior_e = pior_psi = 0.0
	for y, th in casos:
		m._estado.clear()
		medida = m.mede_cones(renderiza(cones, 0.0, y, th), None)
		assert medida is not None, f'nao mediu em y {y}, th {th}'
		e_real, psi_real = -y / np.cos(th), -th   # y do centro no frame do carro, onde ele cruza x = 0, e sua inclinacao
		pior_e, pior_psi = max(pior_e, abs(medida[0] - e_real)), max(pior_psi, abs(medida[1] - psi_real))
		assert abs(medida[0] - e_real) < 0.05 and abs(medida[1] - psi_real) < np.deg2rad(3.0), \
			f'y {y}, th {th}: leu ({medida[0]:.3f}, {np.degrees(medida[1]):.1f} graus), real ({e_real:.3f}, {np.degrees(psi_real):.1f} graus)'
	furado = [c for i, c in enumerate(cones) if i not in (3, 4, 20)]   # sem 3 cones
	e, psi, _ = m.mede_cones(renderiza(furado, 0.0, 0.1, 0.0), None)
	assert abs(e + 0.1) < 0.03 and abs(psi) < np.deg2rad(2.5), f'cones faltando: leu ({e:.3f}, {np.degrees(psi):.1f})'
	falso = cones + [('verde', 2.0, -0.2), ('vermelho', 1.5, 0.3)]   # cones de cor trocada no meio do corredor
	e, psi, _ = m.mede_cones(renderiza(falso, 0.0, 0.1, 0.0), None)
	assert abs(e + 0.1) < 0.05 and abs(psi) < np.deg2rad(3.5), f'falsos positivos: leu ({e:.3f}, {np.degrees(psi):.1f})'
	trocado = [('vermelho' if c == 'verde' else 'verde', x, y) for c, x, y in cones]   # o lado de cada cor nao e fixo: verde a direita
	for y, th in casos[:3]:
		m._estado.clear()
		e, psi, _ = m.mede_cones(renderiza(trocado, 0.0, y, th), None)
		assert abs(e + y / np.cos(th)) < 0.05 and abs(psi + th) < np.deg2rad(3.0), f'cores trocadas, y {y}, th {th}: leu ({e:.3f}, {np.degrees(psi):.1f})'
		assert m._estado['verde_esquerda'] is False
	m._estado.clear()
	m._estado['largura'] = LARGURA_CORREDOR
	so_esquerda = [c for c in cones if c[0] == 'verde']
	e, psi, _ = m.mede_cones(renderiza(so_esquerda, 0.0, 0.1, 0.0), None)
	assert abs(e + 0.1) < 0.05 and abs(psi) < np.deg2rad(3.5), f'so a fileira esquerda: leu ({e:.3f}, {np.degrees(psi):.1f})'
	print(f'1. leitura estatica: pior erro em 6 posicoes: e {100 * pior_e:.1f} cm, psi_e {np.degrees(pior_psi):.1f} graus; com 3 cones faltando, '
	      'com 2 cones de cor trocada e com so a fileira esquerda (largura da ultima medida) continua certo')


def malha(cones, erro_do_centro, T_TOTAL=14.0, DT=0.02, TAU_SERVO=0.10, v=1.0, x=0.0, y=-0.2, th=np.deg2rad(10), FPS=15):
	# a lei do main.py (processa_quadro + direcao) dirigindo a bicicleta. erro_do_centro(xf, yf) = distancia do eixo ao centro
	m = carrega()
	m._estado['linha'] = (0.0, 0.0, 0.0, 0.0)
	car = types.SimpleNamespace(t=0.0, v=v, st=0.0)
	st, quadro, hist, passo_quadro = 0.0, None, [], max(int(round(1.0 / (FPS * DT))), 1)
	for k in range(int(T_TOTAL / DT)):
		car.t = k * DT
		xf, yf = x + T.L * np.cos(th), y + T.L * np.sin(th)
		if k % passo_quadro == 0:   # quadro novo; processa o anterior (1 quadro de atraso)
			if quadro is not None:
				m.processa_quadro(*quadro)
			quadro = (renderiza(cones, xf, yf, th), car.t)
		car.st = float(np.clip(m.direcao(car), -0.349, 0.349))
		st += DT / TAU_SERVO * (car.st - st)
		x, y, th = x + v * np.cos(th) * DT, y + v * np.sin(th) * DT, th + v / T.L * np.tan(st) * DT
		hist.append((car.t, erro_do_centro(x + T.L * np.cos(th), y + T.L * np.sin(th)), m._estado.get('sem_fita', 0.0)))
	return np.array(hist), m


def teste_malha():
	print('2. malha fechada (sai 20 cm fora e 10 graus torto, 1 m/s):')
	resultados = {}
	hist, m = malha(corredor_reto(), lambda xf, yf: yf)
	resultados['reta'] = (np.abs(hist[hist[:, 0] > 6, 1]).max(), hist[:, 2].max(), m.D_SEM_FITA)

	R, x_reta = 4.0, 3.0

	def erro_curva(xf, yf):
		return yf if xf < x_reta else R - np.hypot(xf - x_reta, yf - R)   # positivo para dentro da curva

	hist, m = malha(corredor_em_curva(R, x_reta), erro_curva, T_TOTAL=12.0)
	apos = hist[(hist[:, 0] > 6.0)]
	resultados['curva R 4 m'] = (np.abs(apos[:, 1]).max(), hist[:, 2].max(), m.D_SEM_FITA)
	limites = {'reta': 0.04, 'curva R 4 m': 0.12}
	for nome, (e_max, sem, d_max) in resultados.items():
		print(f'   {nome:12s}: |e| depois de 6 s <= {100 * e_max:.1f} cm, maior trecho sem medida {100 * sem:.0f} cm')
		assert e_max < limites[nome], f'{nome}: |e| {100 * e_max:.1f} cm'
		assert sem < d_max, f'{nome}: andou {sem:.2f} m sem medida (o carro pararia)'


def teste_main_com_cones():
	# o main.py inteiro (GUIA = 'cones') com o Car falso do testa_stanley: corrida, lateral.csv e o video com as caixas dos cones
	import contextlib
	import io
	import os
	from pathlib import Path
	render_original = T.render
	T.render = lambda *a, **k: renderiza(corredor_reto(), 0.0, -0.03, 0.0)   # eixo 3 cm a direita do centro: e = +3 cm
	T._CarFalso.publicados.clear()
	try:
		with contextlib.redirect_stdout(io.StringIO()):
			ns, pasta = T.roda_main(GUIA="'cones'", ACELERADOR_MAX='5.0')
	finally:
		T.render = render_original
	lat = np.loadtxt(os.path.join(pasta, 'lateral.csv'), delimiter=',', skiprows=1)
	assert abs(lat[-1, 2] - 0.03) < 0.02 and lat[-1, 6] > 0, f"leu e {lat[-1, 2]:.3f} m, st {lat[-1, 6]:.3f} rad (esperado e +3 cm e esterco para a esquerda)"
	assert T._CarFalso.publicados, 'a visao nao publicou quadros no video'
	vis = T._CarFalso.publicados[-1]
	caixas_verdes, caixas_vermelhas = ((vis == (0, 255, 0)).all(-1)).sum(), ((vis == (0, 0, 255)).all(-1)).sum()
	arco = ((vis == (255, 0, 255)).all(-1)).sum()
	assert caixas_verdes > 50 and caixas_vermelhas > 50 and arco > 30, f'video sem caixas dos cones ou sem o centro: {caixas_verdes}, {caixas_vermelhas}, {arco}'
	print(f"3. main.py com cones e Car falso: leu e {100 * lat[-1, 2]:+.1f} cm, esterca para a esquerda; o video mostra as caixas dos cones "
	      f"(verde {caixas_verdes} px, vermelho {caixas_vermelhas} px) e o centro do corredor ({arco} px)")


if __name__ == '__main__':
	teste_leitura_estatica()
	teste_malha()
	teste_main_com_cones()
	print('OK')
