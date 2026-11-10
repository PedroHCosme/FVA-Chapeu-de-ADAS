# -*- coding: utf-8 -*-
"""Confere a busca da fita do main.py sem o carro.  Rodar de veiculo_real/analise:  python testa_busca.py

Bicicleta + camera sintetica com a geometria real + atraso de direcao medido (0,26 s) + acelerador integrado como no servos.py,
com marcha a frente e re (a troca de marcha "gasta" o tempo real dela). Usa as mesmas funcoes do main.py: processa_quadro,
direcao e control_func. Cenarios: fita some na curva fechada, apagao da camera, fita acaba, encoder zerado, chaves desligadas.
"""
from collections import deque
import types

import numpy as np
import testa_stanley as T

T.ALT, T.XC = 0.255, 0.175
T.GX, T.GY, T.CHAO = T.chao_dos_pixels()
DT, FPS = 0.03, 15
TROCA_RE, TROCA_FRENTE = 3.8, 2.5   # s -- quanto set_reverse e set_forward travam o laco
V_POR_RAD = np.degrees(1.0) * 0.2   # planta: 0,2 m/s por grau de acelerador
GANHO_U = {1: 0.2016, -1: 0.605}    # rad/s por u, como no servos.py


class CarroSim:
	# o que o main.py usa do Car, com planta simples
	def __init__(self, x, y, th, encoder_ok):
		self.x, self.y, self.th_real = x, y, th
		self.t, self.dt, self.v_real, self.st, self.u, self.vref, self.marcha = 0.0, DT, 0.0, 0.0, 0.0, 0.0, 1
		self.encoder_ok = encoder_ok
		self.v_filt = types.SimpleNamespace(n=20)
		self.atuador = types.SimpleNamespace(th_pwm=0.0, max_throttle=np.deg2rad(18.0))
		self.bz = types.SimpleNamespace(beep=lambda *a, **k: None)
		self.st_real, self.fila = 0.0, deque([0.0] * int(0.16 / DT))
		self.dist_re, self.dist_frente = 0.0, 0.0

	@property
	def v(self):
		return self.v_real if self.encoder_ok else 0.0

	def set_steer(self, st):
		self.st = float(np.clip(st, -0.349, 0.349))

	def set_u(self, u):
		self.u = float(np.clip(u, -1.0, 1.0))

	def _trava(self, segundos, marcha):
		self.v_real = 0.0
		self.atuador.th_pwm = 0.0
		self.marcha = marcha
		self.t += segundos

	def set_reverse(self):
		self._trava(TROCA_RE, -1)

	def set_forward(self):
		self._trava(TROCA_FRENTE, 1)

	def anda(self):
		self.t += DT
		a = self.atuador
		if self.marcha > 0:
			a.th_pwm = float(np.clip(a.th_pwm + GANHO_U[1] * self.u * DT, 0.0, a.max_throttle))
		else:
			a.th_pwm = float(np.clip(a.th_pwm - GANHO_U[-1] * self.u * DT, -np.deg2rad(95.0), 0.0))
		alvo = a.th_pwm * V_POR_RAD
		if abs(alvo) >= abs(self.v_real) or abs(alvo) > 0.01:
			self.v_real += (alvo - self.v_real) * DT / 0.5
		else:
			self.v_real -= np.sign(self.v_real) * min(0.6 * DT, abs(self.v_real))   # sem freio: desacelera por atrito
		self.fila.append(self.st)
		self.st_real += DT / 0.10 * (self.fila.popleft() - self.st_real)
		v = self.v_real
		self.x += v * np.cos(self.th_real) * DT
		self.y += v * np.sin(self.th_real) * DT
		self.th_real += v / T.L * np.tan(self.st_real) * DT
		if v > 0:
			self.dist_frente += v * DT
		else:
			self.dist_re -= v * DT


def carrega(**troca):
	m = T.carrega()
	m.K_STANLEY, m.K_SOFT, m.TAXA_RAMPA_V, m.V_CURVA, m.V_GUARDA = 2.33, 0.2, 0.9, 1.0, 1.6
	m.ADAS_ON, m.GRAVA_QUADROS, m.FILTRO_RELATIVO = False, False, False
	m.BUSCA_ARCO, m.BUSCA_RE = True, True
	m._estado.clear()
	m._vdes_filt.reset(0.0)
	for nome, valor in troca.items():
		setattr(m, nome, valor)
	return m


def roda(fita, cego=None, TT=40.0, encoder_ok=True, v_ref=1.0, **troca):
	# fita(wx, wy, xf, yf) -> distancia ate a fita em cada pixel do chao; cego(t) -> camera sem fita
	m = carrega(V_REF=v_ref, **troca)
	car = CarroSim(-T.L, 0.0, 0.0, encoder_ok)
	m._estado['linha'] = (0.0, 0.0, 0.0, 0.0)
	modos, prox_quadro, quadro = [], 0.0, None
	saida = []
	while car.t < TT:
		if car.t >= prox_quadro:
			prox_quadro = car.t + 1.0 / FPS
			if quadro is not None:
				m.processa_quadro(*quadro)
			xf, yf = car.x + T.L * np.cos(car.th_real), car.y + T.L * np.sin(car.th_real)
			wx = xf + T.GX * np.cos(car.th_real) - T.GY * np.sin(car.th_real)
			wy = yf + T.GX * np.sin(car.th_real) + T.GY * np.cos(car.th_real)
			img = T.PISO if cego is not None and cego(car.t) else T.render(fita(wx, wy))
			quadro = (img, car.t)
		car.set_steer(m.direcao(car))
		m.control_func(car)
		modo = m._estado.get('busca')
		if not modos or modos[-1] != modo:
			modos.append(modo)
		xf, yf = car.x + T.L * np.cos(car.th_real), car.y + T.L * np.sin(car.th_real)
		saida.append((car.t, car.x, car.y, car.v_real, car.st, float(fita(np.array(xf), np.array(yf)))))
		car.anda()
		car.dt = DT
	return dict(modos=modos, log=np.array(saida), car=car, m=m)


def reta_ate(fim):
	# fita reta em y = 0 ate x = fim; depois nao existe
	return lambda wx, wy, *_: np.where(wx < fim, wy, 9.0)


def curva_fechada(wx, wy, *_):
	# reta ate x = 3, depois arco de raio 1,7 m para a esquerda (abaixo de ~1,5 m a fita nem cabe na imagem em regime)
	R = 1.7
	arco = np.hypot(wx - 3.0, wy - R) - R
	return np.where((wx < 3.0) & (wy < R), wy, arco)


def reta(wx, wy, *_):
	return wy


def resumo(nome, r):
	L = r['log']
	print(f"{nome}: etapas {[x or 'segue' for x in r['modos']]}; frente {r['car'].dist_frente:.1f} m, re {r['car'].dist_re:.1f} m, "
	      f"|v| max {np.abs(L[:, 3]).max():.2f}, v final {L[-1, 3]:+.2f}, |e| final {100 * abs(L[-1, 5]):.0f} cm", flush=True)


def teste_chaves_desligadas():
	r = roda(reta, cego=lambda t: 5 < t < 8, TT=12.0, BUSCA_ARCO=False, BUSCA_RE=False)
	resumo('chaves desligadas, apagao', r)
	assert r['modos'] == [None], 'com as chaves desligadas nao pode haver busca'
	assert r['car'].dist_re == 0.0 and abs(r['log'][-1, 5]) < 0.1, 'deveria so frear e seguir quando a fita volta'


def teste_arco_curva():
	r = roda(curva_fechada, TT=25.0, v_ref=1.3)
	resumo('curva R 1,7 m a 1,3 m/s', r)
	sem = roda(curva_fechada, TT=25.0, v_ref=1.3, BUSCA_ARCO=False, BUSCA_RE=False)
	resumo('   a mesma, sem busca', sem)
	assert 'arco' in r['modos'], 'a curva deveria fazer perder a fita a 1,3 m/s'
	assert r['modos'][-1] is None, f"nao terminou seguindo a fita: {r['modos']}"
	assert abs(r['log'][-1, 5]) < 0.1, f"terminou longe da fita: {100 * r['log'][-1, 5]:.0f} cm"
	assert r['car'].dist_frente > 1.5 * sem['car'].dist_frente, 'a busca deveria andar mais na fita do que so parar'


def teste_re_apagao():
	# so re: a camera apaga, o carro para, recua pelo caminho e reencontra quando a camera volta
	r = roda(reta, cego=lambda t: 5 < t < 16, TT=28.0, v_ref=0.6, BUSCA_ARCO=False)
	resumo('so re, apagao 11 s', r)
	assert 're' in r['modos'], 'deveria ter dado re'
	assert r['modos'][-1] is None and r['car'].dist_re > 0.2, f"nao reencontrou de re: {r['modos']}"
	assert abs(r['log'][-1, 5]) < 0.1, 'terminou longe da fita'


def teste_fita_acaba():
	r = roda(reta_ate(5.0), TT=70.0, v_ref=0.6)
	resumo('fita acaba', r)
	assert r['modos'][-1] == 'desistiu', f"deveria desistir: {r['modos']}"
	assert abs(r['log'][-1, 3]) < 0.02, 'desistiu mas nao parou'
	assert r['m']._estado['buscas'] > r['m'].MAX_BUSCAS, 'desistiu antes de esgotar as buscas'


def teste_encoder_zerado():
	r = roda(reta, cego=lambda t: t > 4, TT=16.0, v_ref=0.6, encoder_ok=False)
	resumo('encoder zerado, fita some', r)
	assert 're' not in r['modos'], 'sem encoder nao ha trilha: nao pode dar re as cegas'
	assert r['modos'][-1] == 'desistiu' and abs(r['log'][-1, 3]) < 0.05, f"deveria parar: {r['modos']}, v {r['log'][-1, 3]:.2f}"


if __name__ == '__main__':
	teste_chaves_desligadas()
	teste_re_apagao()
	teste_encoder_zerado()
	teste_arco_curva()
	teste_fita_acaba()
	print('OK')
