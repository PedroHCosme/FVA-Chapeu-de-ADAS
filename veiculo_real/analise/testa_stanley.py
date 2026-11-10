# -*- coding: utf-8 -*-
"""Confere o Stanley do main.py sem o carro.  Rodar de veiculo_real/analise:  python testa_stanley.py

1. camera sintetica (pinhole de geometria conhecida): homografia pela altura; (e, psi_e) lidos de imagens renderizadas,
   com fita inteira e picotada, e com uma caixa amarela no chao
2. foto real do corredor (29/10): a visao acha a fita
3. malha fechada: bicicleta cinematica + camera a 30 Hz com 1 quadro de atraso + servo de 1a ordem; reta e curva,
   fita inteira e picotada
4. main.py rodando como no carro, com um Car falso: corrida, teto do acelerador, video, fita perdida com encoder zerado
"""
import os
import re
import sys
import tempfile
import time
import types
import importlib.util
from pathlib import Path

import numpy as np
import analisa as A

REAL = Path(__file__).resolve().parents[1]
FOTO = REAL / 'tools' / 'experimentos' / 'roxo' / '20261029_172239' / 'foto.png'

# camera sintetica: 12 cm de altura, 5 cm atras do eixo dianteiro, FOV 60 graus, horizonte na linha 100
W, H, ALT, XC, LH = 640, 480, 0.12, -0.05, 100
F = (W / 2) / np.tan(np.deg2rad(30))
P = np.arctan((H / 2 - LH) / F)
Z_C, X_C, Y_C = np.array([np.cos(P), 0, -np.sin(P)]), np.array([0, -1.0, 0]), np.array([-np.sin(P), 0, -np.cos(P)])
L, PICOTE = 0.36, (0.10, 0.15)   # entre-eixos; fita picotada: pedacos de 10 cm, vaos de 15 cm
T_IMG = (H, W, 3)


def carrega():
	m = A.carrega_main()
	m.ALTURA_CAMERA, m.X_CAMERA, m.FOV_HORIZONTAL, m.LINHA_HORIZONTE, m.SINAL_DIRECAO = ALT, XC, 60.0, LH, 1.0
	m.H_CHAO = m.homografia_pinhole()
	m.K_STANLEY, m.V_REF = 2.0, 1.0   # os limites abaixo valem para K 2 e 1 m/s, qualquer que seja a sintonia do main.py
	m.K_SOFT, m.BUSCA_ARCO, m.BUSCA_RE = 0.3, False, False
	return m


def projeta(x, y):
	p = np.array([x - XC, y, -ALT])
	return W / 2 + F * (p @ X_C) / (p @ Z_C), H / 2 + F * (p @ Y_C) / (p @ Z_C)


def chao_dos_pixels():
	# (x, y) no chao de cada pixel, no frame do eixo dianteiro, e se o pixel ve o chao (abaixo do horizonte)
	uu, vv = np.meshgrid(np.arange(W), np.arange(H))
	d = Z_C[:, None, None] + ((uu - W / 2) / F) * X_C[:, None, None] + ((vv - H / 2) / F) * Y_C[:, None, None]
	s = ALT / -np.minimum(d[2], -1e-9)
	return XC + s * d[0], s * d[1], d[2] < 0


GX, GY, CHAO = chao_dos_pixels()
PISO = np.clip(np.random.default_rng(0).normal(120, 8, (H, W, 3)), 0, 255).astype(np.uint8)


def render(dist, s=None, picote=None, extra=None):
	# imagem BGR: fita amarela de 2,5 cm onde |dist| < 1,25 cm; picote = (pedaco, vao) ao longo de s; extra = outra mancha amarela
	fita = CHAO & (np.abs(dist) < 0.0125)
	if picote is not None:
		fita &= np.mod(s, sum(picote)) < picote[0]
	if extra is not None:
		fita |= CHAO & extra
	img = PISO.copy()
	img[fita] = (20, 210, 225)
	return img


def fita_reta(e, psi):
	# distancia ate a fita e posicao ao longo dela, no frame do carro: fita passa a e do eixo dianteiro, aponta psi
	return (GY - e) * np.cos(psi) - GX * np.sin(psi), GX * np.cos(psi) + (GY - e) * np.sin(psi)


def teste_camera_sintetica():
	m = carrega()
	erro = max(np.hypot(*np.subtract(m.no_chao(*projeta(x, y)), (x, y))) for x in (0.2, 0.5, 1.5) for y in (-0.3, 0, 0.3))
	assert erro < 1e-9, f'homografia pela altura erra {erro} m'
	casos = [(0.0, 0.0), (0.06, 0.0), (-0.05, 0.0), (0.0, 0.25), (0.04, -0.2), (-0.06, 0.3)]
	for picote in (None, PICOTE):
		for e, psi in casos:
			em, pm, _ = m.mede_fita(render(*fita_reta(e, psi), picote=picote), None)
			assert abs(em - e) < 0.01 and abs(pm - psi) < np.deg2rad(2), f'picote {picote}: real ({e}, {psi}) lido ({em:.3f}, {pm:.3f})'
	for psi in np.deg2rad([0.0, 10.0, 15.0]):   # arco de R 2 m pelo eixo dianteiro: sem vies (a parabola errava psi_e em ~3 graus)
		wx, wy = GX * np.cos(psi) + GY * np.sin(psi), -GX * np.sin(psi) + GY * np.cos(psi)
		em, pm, cm = m.mede_fita(render(np.hypot(wx, wy - 2.0) - 2.0), None)
		assert abs(em) < 0.003 and abs(pm - psi) < np.deg2rad(0.5) and abs(cm - 0.25) < 0.05, \
			f'arco, psi {np.degrees(psi):.0f}: lido ({em:.3f}, {np.degrees(pm):.2f} graus, c {cm:.3f})'
	caixa = (GX > 0.5) & (GX < 0.8) & (GY > 0.15) & (GY < 0.45)   # caixa amarela de 30 x 30 cm a esquerda
	em, pm, _ = m.mede_fita(render(*fita_reta(-0.05, 0.0), extra=caixa), None)
	assert abs(em + 0.05) < 0.01 and abs(pm) < np.deg2rad(2), f'com a caixa leu ({em:.3f}, {pm:.3f})'
	print('1. camera sintetica: homografia exata; e e psi_e com erro < 1 cm e < 2 graus, com fita inteira, picotada '
	      f'({100 * PICOTE[0]:.0f} cm de fita, {100 * PICOTE[1]:.0f} de vao) e com uma caixa amarela no chao; '
	      'arco de R 2 m sem vies (< 0,3 cm e < 0,5 grau)')


def teste_filtro_relativo():
	# fita lavada pelo reflexo, com os valores medidos nas capturas de 03/11, sobre o chao medido: so o filtro relativo enxerga
	m = carrega()
	dist, s = fita_reta(0.04, 0.1)
	img = np.clip(np.random.default_rng(1).normal(0, 3, (H, W, 3)) + (165, 164, 159), 0, 255).astype(np.uint8)
	img[CHAO & (np.abs(dist) < 0.0125)] = (202, 221, 223)
	assert m.mede_fita(img, None) is None, 'o filtro de saturacao nao deveria enxergar a fita lavada'
	m.FILTRO_RELATIVO = True
	e, psi, _ = m.mede_fita(img, None)
	assert abs(e - 0.04) < 0.01 and abs(psi - 0.1) < np.deg2rad(2), f'relativo leu ({e:.3f}, {psi:.3f})'
	print(f'1b. fita com reflexo: saturacao nao ve; relativo le e {100 * e:+.1f} cm, psi {np.degrees(psi):+.1f} graus')


def teste_foto_real():
	if not FOTO.exists():
		print(f'2. foto real: PULADO ({FOTO} nao existe)')
		return
	import matplotlib.pyplot as plt
	rgb = (plt.imread(FOTO)[..., :3] * 255).astype(np.uint8)
	m = carrega()
	medida = m.mede_fita(rgb[..., ::-1].copy(), None)   # o carro entrega BGR
	assert medida is not None, 'nao achou a fita na foto real'
	e, psi, _ = medida
	assert -20 < np.degrees(psi) < -5 and abs(e) < 0.3, f'e {e:.3f} m, psi_e {np.degrees(psi):.1f} graus (a fita aponta para a direita)'
	print(f'2. foto real: fita achada, psi_e = {np.degrees(psi):+.1f} graus (o e depende da altura real da camera)')


def malha(pista, picote=None, T=10.0, DT=0.02, TAU_SERVO=0.10, v=1.0, x=0.0, y=-0.12, th=np.deg2rad(10)):
	# a lei do main.py (processa_quadro + direcao) dirigindo a bicicleta; pista 'reta' (y = 0) ou 'curva' (circulo R = 2 m)
	m = carrega()
	m._estado.clear()
	m._estado['linha'] = (0.0, 0.0, 0.0, 0.0)
	car = types.SimpleNamespace(t=0.0, v=v, st=0.0)
	st, quadro, hist = 0.0, None, []
	for k in range(int(T / DT)):
		car.t = k * DT
		xf, yf = x + L * np.cos(th), y + L * np.sin(th)
		if k == 0 or int(car.t * 30) != int((car.t - DT) * 30):   # quadro novo a 30 Hz; processa o anterior (1 quadro de atraso)
			if quadro is not None:
				m.processa_quadro(*quadro)
			wx, wy = xf + GX * np.cos(th) - GY * np.sin(th), yf + GX * np.sin(th) + GY * np.cos(th)
			if pista == 'reta':
				dist, s = wy, wx
			else:
				dist, s = np.hypot(wx, wy - 2.0) - 2.0, 2.0 * np.arctan2(wx, 2.0 - wy)
			quadro = (render(dist, s, picote), car.t)
		car.st = float(np.clip(m.direcao(car), -0.349, 0.349))
		st += DT / TAU_SERVO * (car.st - st)
		x, y, th = x + v * np.cos(th) * DT, y + v * np.sin(th) * DT, th + v / L * np.tan(st) * DT
		xf, yf = x + L * np.cos(th), y + L * np.sin(th)
		hist.append((abs(yf) if pista == 'reta' else abs(2.0 - np.hypot(xf, yf - 2.0)), m._estado.get('sem_fita', 0.0)))
	h = np.array(hist)
	return np.abs(h[int(5 / DT):, 0]).max(), h[:, 1].max(), m.D_SEM_FITA


def teste_malha():
	limites = {('reta', None): 0.01, ('reta', PICOTE): 0.01, ('reta', (0.10, 0.40)): 0.015,
	           ('curva', None): 0.01, ('curva', PICOTE): 0.01, ('curva', (0.20, 0.60)): 0.03}   # valores de 31/10 (arco) + folga
	print('3. malha fechada (sai 12 cm fora e 10 graus torto, 1 m/s):')
	for (pista, picote), lim in limites.items():
		e_max, sem_fita, d_max = malha(pista, picote)
		nome = 'inteira' if picote is None else f'picotada {100 * picote[0]:.0f}/{100 * picote[1]:.0f} cm'
		print(f'   {pista:5s} fita {nome:22s}: |e| depois de 5 s <= {100 * e_max:.1f} cm, maior trecho sem fita {100 * sem_fita:.0f} cm')
		assert e_max < lim, f'{pista} {nome}: |e| {100 * e_max:.1f} cm'
		assert sem_fita < d_max, f'{pista} {nome}: andou {sem_fita:.2f} m sem fita (o carro pararia)'


class _CarFalso:
	# o minimo do fva_car.Car que o main.py usa; parado ou andando reto sobre a fita sintetica
	encoder_ok = True
	fita_ate = 1e9   # depois deste t a camera nao ve mais a fita
	publicados = []

	def __init__(self, parameters):
		self.parameters, self.logfile = parameters, tempfile.mkdtemp(prefix='testa_stanley_')
		self.cam = types.SimpleNamespace(start_stream=lambda fps=10: None, publish_frame=_CarFalso.publicados.append)
		self.t, self.dt, self.v, self.a, self.u, self.st, self.vref, self.w, self.th = 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		self.p, self.traj = (0.0, 0.0), []
		self.v_filt = types.SimpleNamespace(n=20)
		self.atuador = types.SimpleNamespace(set_trim=lambda steer=0.0, throttle=0.0: None, max_throttle=0.3, th_pwm=0.0)
		self.bz = types.SimpleNamespace(beep=lambda *a, **k: None)
		_CarFalso.ultimo = self
		self.img = render(*fita_reta(0.03, 0.0))   # fita 3 cm a esquerda, reta

	def get_image(self, gray=False):
		return self.img if self.t < self.fita_ate else PISO

	def start_mission(self):
		self.traj = [0]

	def step(self):
		time.sleep(0.005)   # da tempo a thread de visao de processar quadros, como no carro
		self.t += self.dt
		self.v = min(self.v + max(self.u, 0) * self.dt, 1.0) if self.encoder_ok else 0.0
		return True

	def set_steer(self, st):
		self.st = float(np.clip(st, -0.349, 0.349))

	def set_u(self, u):
		self.u = u

	def get_distance(self):
		return 4.0, False

	def save(self):
		Path(self.logfile, 'car.csv').write_text('t\n0\n')

	def close(self):
		pass


def roda_main(**troca):
	# executa o main.py como no carro (__main__), com o Car falso e constantes trocadas; devolve (namespace, pasta do log)
	filtro = importlib.util.spec_from_file_location('fva_car.filter', REAL / 'fva_car' / 'filter.py')
	pacote = types.ModuleType('fva_car')
	pacote.filter = importlib.util.module_from_spec(filtro)
	filtro.loader.exec_module(pacote.filter)
	carros = []
	pacote.Car = lambda par: carros.append(_CarFalso(par)) or carros[-1]
	# cv2 de mentira: o venv do PC nao tem OpenCV; o desenho do video e numpy, so o texto usaria cv2
	def imwrite_falso(caminho, img, parametros=None):
		np.save(caminho, img[:1, :1])
	cv2_falso = types.SimpleNamespace(FONT_HERSHEY_SIMPLEX=0, putText=lambda *a: None, IMWRITE_PNG_COMPRESSION=16, imwrite=imwrite_falso)
	sys.modules['fva_car'], sys.modules['fva_car.filter'], sys.modules['cv2'] = pacote, pacote.filter, cv2_falso
	src = (REAL / 'main.py').read_text(encoding='utf-8')
	troca = dict(dict(TEMPO_CORRIDA='1.0', ALTURA_CAMERA=str(ALT), X_CAMERA=str(XC), LINHA_HORIZONTE=str(LH), SINAL_DIRECAO='1.0'), **troca)
	for nome, valor in troca.items():
		src, n = re.subn(rf'(?m)^{nome}(\s*)=\s*[^#\n]*', rf'{nome}\g<1>= {valor}  ', src)
		assert n == 1, f'{nome} nao encontrado no main.py'
	ns = {'__name__': '__main__'}
	try:
		exec(compile(src, str(REAL / 'main.py'), 'exec'), ns)
	except SystemExit:
		pass
	finally:
		for k in ('fva_car', 'fva_car.filter', 'cv2'):
			sys.modules.pop(k, None)
	return ns, carros[-1].logfile


def teste_modos():
	import io
	import contextlib
	saida = io.StringIO()
	with contextlib.redirect_stdout(saida):
		ns, pasta = roda_main(ADAS_ON='True', ACELERADOR_MAX='5.0')
		assert {'car.csv', 'sonar.csv', 'lateral.csv'} <= set(os.listdir(pasta)), 'corrida normal nao gravou os logs'
		gravados = sorted(Path(pasta, 'quadros').glob('quadro_*.png*'))
		assert 2 <= len(gravados) <= 3, f'esperava 2 ou 3 quadros em 1 s a 2 Hz, gravou {len(gravados)}'
		lat = np.loadtxt(os.path.join(pasta, 'lateral.csv'), delimiter=',', skiprows=1)
		assert abs(lat[-1, 2] - 0.03) < 0.01 and lat[-1, 6] > 0, f"le a fita a +3 cm: e {lat[-1, 2]:.3f} m, st {lat[-1, 6]:.3f} rad"
		assert abs(_CarFalso.ultimo.atuador.max_throttle - np.deg2rad(ns['ACELERADOR_MAX'])) < 1e-9, 'teto do acelerador nao aplicado'
		assert _CarFalso.publicados, 'a visao nao publicou quadros no video'
		vis = _CarFalso.publicados[-1]
		assert vis.shape == T_IMG and ((vis == (0, 255, 0)).all(-1)).sum() > 100 and ((vis == (255, 0, 255)).all(-1)).sum() > 50, 'video sem os pixels usados ou sem a fita ajustada'
		_CarFalso.encoder_ok, _CarFalso.fita_ate = False, 0.5   # encoder zerado e fita some: so o tempo pode parar o carro
		roda_main(TEMPO_CORRIDA='2.0', T_SEM_FITA='0.5', BUSCA_ARCO='False', BUSCA_RE='False')
		_CarFalso.encoder_ok, _CarFalso.fita_ate = True, 1e9
		assert _CarFalso.ultimo.u == -1.0, 'fita perdida com encoder zerado nao freou'
	print("4. main.py com Car falso: corrida grava os logs, le a fita a +3 cm e esterca para a esquerda; teto do acelerador aplicado; "
	      "video publica os pixels usados e a fita; fita perdida freia mesmo com o encoder zerado")


if __name__ == '__main__':
	teste_camera_sintetica()
	teste_filtro_relativo()
	teste_foto_real()
	teste_malha()
	teste_modos()
	print('OK')
