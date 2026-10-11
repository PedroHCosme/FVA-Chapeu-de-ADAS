# -*- coding: utf-8 -*-
########################################
# Disciplina: Topicos em Engenharia de Controle e Automacao IV (ENG075):
# Fundamentos de Veiculos Autonomos - 2026/2
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automacao
# DELT - Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
from fva_car import Car
from fva_car.filter import AlphaFilter, MovingAverage
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import os

########################################
# ENSAIO: troque SO esta linha entre as corridas (este e o unico arquivo copiado para o carro); chaves em ROTEIRO.
# Toda corrida grava car.csv e sonar.csv (leitura do ultrassom). Analise: analise/ROTEIRO.md.
ENSAIO = 'avanco_rt'

V_REF          = 1.3   # m/s -- alvo das rampas
TAXA_RAMPA_V   = 0.9    # m/s^2 -- inclinacao da rampa de referencia
PISTA_M        = 500   # m -- pista disponivel; o ensaio se recusa a rodar se passar de PISTA_M - 1
ATRASO_PARTIDA = 1.0    # s -- tempo ate o carro sair do lugar (zona morta), so para estimar a distancia
V_CORTE        = 0.7    # m/s -- 'u_degrau' para de acelerar aqui...
PWM_CORTE      = 8.0    # graus -- ...ou quando o acelerador estimado chega aqui
T_SONAR        = 10.0   # s -- 'sonar': tempo da caixa em cada distancia
V_GUARDA       = 1.35   # m/s -- guarda de VELMAX: acima disto u = -1 (None desliga). Escolhida em analise/sintonia.py

# perfil de referencia: lista de (t [s], alvo [m/s]) ou (t, alvo, taxa [m/s^2]); None = rampa ate V_REF
ROTEIRO = {
	'avanco'     : dict(projeto='avanco',    ts=13.0, perfil=None),   # lei escolhida (P + D + feedforward), rampa ate V_REF, ~15 m
	'avanco_ff'  : dict(projeto='avanco_ff', ts=13.0, perfil=None),   # a mesma malha com o feedforward completo (K_FF, TAU_FF)
	'avanco_rt'  : dict(projeto='avanco_rt', ts=35.0, perfil=None),   # a mesma rampa com a lei re-sintonizada (analise/sintonia.py)
	'frenagem'   : dict(projeto='avanco',    ts=20.0, perfil=[(0, 1.0), (8, 0.5), (13, 0.0, 0.25)]),   # referencia DECRESCENTE, ~11 m: 1,0 > 0,5 rapido > 0 devagar
	'frenagem_ff': dict(projeto='avanco_ff', ts=20.0, perfil=[(0, 1.0), (8, 0.5), (13, 0.0, 0.25)]),
	'frenagem_rt': dict(projeto='avanco_rt', ts=20.0, perfil=[(0, 1.0), (8, 0.5), (13, 0.0, 0.25)]),

	# subida: parte ~5 m antes da rampa, V_REF que voce editar (1,0 ou 1,3). 17 s (a 1,3 m/s ~20 m: 5 de plano + rampa + topo; a 1,0 ~15 m).
	# Fora do ajuste da planta (o disturbio enviesaria o K); a duracao diferente da do 'avanco' e o que separa os dois na analise.
	'subida_ff'  : dict(projeto='avanco_ff', ts=17.0, perfil=None, subida=True, pista=24.0),
	'subida_rt'  : dict(projeto='avanco_rt', ts=17.0, perfil=None, subida=True, pista=24.0),

	# parada: cruzeiro em V_REF ate t = 8 s (~8 m a 1,3 m/s) e CORTA (u = -1, vref 0: a mesma acao da emergencia do ADAS);
	# mede a distancia e o tempo de parada. 7 s depois para (ts). Segurem o carro. Lei 'avanco_rt'.
	'parada'     : dict(projeto='avanco_rt', ts=15.0, perfil=None, parada=8.0),

	'sonar'      : dict(ts=72.0, sonar=[0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 0.0]),   # carro PARADO, alguem segura uma caixa nestas distancias [m] (0 = sem caixa); instrucoes no terminal
	'u_degrau'   : dict(ts=30.0, aberto=[0.5, 0.8, 1.0, 1.0]),                 # malha ABERTA: niveis de u a partir do repouso (zona morta e atraso)
	'patamares'  : dict(projeto='avanco',    ts=16.0,                          # degraus de referencia 0,6 > 0,9 > 1,1 > 0,8 > 0,5 m/s (separa ganho G de zona morta)
	                    perfil=[(0, 0.6), (3.5, 0.9), (6.5, 1.1), (9.5, 0.8), (12, 0.5), (13.5, 0.0)]),
}
_E = ROTEIRO[ENSAIO]

########################################
# SUPERVISOR ADAS (ultrassom): fica entre a referencia e a lei de controle (_adas, mais abaixo).
# Bloco entre as marcas: calibrado com o ensaio 'sonar' por  python analise/calibra_adas.py <sonar.csv> --aplica
# >>> ADAS (calibra_adas.py)
ADAS_ON        = True   # True = supervisor ligado
SONAR_VIES     = 0.00   # m -- leitura menos distancia real (mediana de 0,5 a 2 m)
SONAR_RUIDO    = 0.005  # m -- desvio das leituras [20260930_222919]
SONAR_ALCANCE  = 2.900  # m -- maior distancia com >= 90% de leituras validas e erro < 0,10 m [20260930_222919]
# <<< ADAS
A_FREIO        = 0.5    # m/s^2 -- desaceleracao so por atrito. Medido (analise/parada.py): 0,52-0,59 a 0,8 m/s, 0,67 a 1,34 m/s
T_REACAO       = 0.7    # s -- do corte (u = -1) ate a velocidade cair 5%. Medido: 0,40-0,55 s a 0,8 m/s, 0,43 s a 1,34 m/s (+ folga)
D_PARADA       = 0.1   # m -- folga entre o para-choque e o obstaculo, parado
A_PLAN         = 0.30   # m/s^2 -- a camada suave planeja frear com isto (< A_FREIO: sobra margem)
D_LIBERA       = 1.5    # m -- parado por obstaculo, so volta a andar quando a distancia passar disto (ou livre)
TH_GAP         = 1.0    # s -- seguindo um lider em movimento: distancia desejada = D_PARADA + TH_GAP * v
K_ACC          = 0.4    # 1/s -- vdes <= v_lider + K_ACC * (d - distancia desejada)
JANELA_AP      = 0.6    # s -- janela da reta d(t) que da a velocidade de aproximacao (ultrassom a 10 Hz: ~6 leituras)
RESET_AO_FREAR = True   # zera filtros e estados da lei enquanto freia (retomada sem estado velho); testa_adas.py compara

########################################
# GANHOS. Planta (bateria > 90%): u -> servos.py integra no acelerador (9,46 graus/s por unidade de u) -> zona morta e ganho
# do ESC -> atraso tau ~0,6-0,75 s -> car.v (media de 30 amostras, ~0,42 s). K = dv/dt por unidade de u ~1,95-2,9.
#   KP [u por m/s de erro]  ganho proporcional; a malha tem velocidade ~KP*K rad/s (validado no carro: 0,55-0,6)
#   TD [s]                  tempo derivativo (0 desliga); derivada filtrada com TD/5
#   TAU_VDES [s]            filtro de 1a ordem da referencia
#   K_FF [m/s^2 por u]      o K que o feedforward supoe: 4 no 'avanco' (acima do K real); 2,9 nos demais, o MAIOR K medido,
#                           para o FF nunca empurrar mais que o carro precisa
#   TAU_FF [s]              atraso da planta que o feedforward compensa (0 desliga); 0,61 s: ajuste de 01/10
_PROJETOS = {
	'avanco'   : dict(KP=0.2, TD=0.6, TAU_VDES=0.5, K_FF=4.0, TAU_FF=0.0),
	'avanco_ff': dict(KP=0.2, TD=0.6, TAU_VDES=0.5, K_FF=2.9, TAU_FF=0.61),
	'avanco_rt': dict(KP=0.225, TD=0.6, TAU_VDES=0.2, K_FF=2.9, TAU_FF=0.61),   # analise/sintonia.py, ponto #5, V_GUARDA 1,35 (02/10)
}
PROJETO = _E.get('projeto', 'avanco')
KP, TD, TAU_VDES, K_FF, TAU_FF = (_PROJETOS[PROJETO][k] for k in ('KP', 'TD', 'TAU_VDES', 'K_FF', 'TAU_FF'))

_estado = {}                          # memoria do controlador e dos ensaios entre chamadas
_vdes_filt = AlphaFilter(alpha=1.0)   # alpha recalculado a cada chamada com car.dt

########################################
# LEI DE CONTROLE (u = aceleracao pedida ao car.set_u, saturada em +-1)
#
#   u = KP*e + KP*TD*de/dt + (dvdes/dt + TAU_FF*d2vdes/dt2) / K_FF      e = MA30(vdes) - v
#
# Como o servos.py integra u, no acelerador esta MESMA lei e um PI + feedforward estatico:
# Ki = 9,46*KP, Kp = 9,46*KP*TD e 9,46*vdes/K_FF graus (analise/prova_integrador.py). Malha tipo 1:
# erro zero em regime para referencia constante. O feedforward fica fora da malha: K_FF e TAU_FF
# nao mexem nas margens.
def controlador_longitudinal(car, vdes_bruto):

	# referencia filtrada; a derivada dela alimenta o feedforward
	if car.dt > 0:
		_vdes_filt.alpha = np.clip(car.dt / TAU_VDES, 0.0, 1.0)
	vdes_anterior = _vdes_filt.value
	vdes = _vdes_filt.filter(vdes_bruto)
	dvdes_dt = (vdes_bruto - vdes_anterior) / TAU_VDES

	# erro contra a referencia atrasada pela MESMA media movel do car.v: o P nao reage ao atraso do sensor
	vdes_ma = _estado.setdefault('vdes_ma', MovingAverage(n=car.v_filt.n))
	erro = vdes_ma.filter(vdes) - car.v

	# derivada do erro filtrada com TD/5 (devolve a fase que a media movel tira)
	derro_dt = 0.0
	if TD > 0.0 and car.dt > 0:
		b = car.dt / (TD / 5.0 + car.dt)
		derro_dt = _estado.get('derro', 0.0)
		derro_dt += b * ((erro - _estado.get('erro_ant', erro)) / car.dt - derro_dt)
		_estado['derro'], _estado['erro_ant'] = derro_dt, erro

	# segunda derivada da referencia, so quando o feedforward compensa o atraso da planta
	d2vdes_dt2 = 0.0
	if TAU_FF > 0.0 and car.dt > 0:
		d2vdes_dt2 = (dvdes_dt - _estado.get('dvdes_ant', dvdes_dt)) / car.dt
		_estado['dvdes_ant'] = dvdes_dt

	u = KP * erro + KP * TD * derro_dt + (dvdes_dt + TAU_FF * d2vdes_dt2) / K_FF
	# guarda de VELMAX: acima de 1,5 o car.py zera u e, como o servos.py integra u, o acelerador CONGELA
	# (no topo de uma subida a 1,3 m/s o carro ficaria preso a ~1,8 m/s). Tira acelerador antes disso.
	if V_GUARDA is not None and car.v > V_GUARDA:
		u = -1.0
	return float(np.clip(u, -1.0, 1.0))   # u < 0 baixa o acelerador (o carro so desacelera por atrito)

########################################
# SUPERVISOR ADAS
#   emergencia: d < d_gatilho(v) -> u = -1 e referencia 0, travado ate d > D_LIBERA (histerese); freando, a lei e zerada
#   camada suave: vdes <= sqrt(2*A_PLAN*(d - D_PARADA)) e vdes <= V_CRUZ (a maior v cujo gatilho cabe no alcance do sensor);
#     com lider que anda (v_lider = v - v_ap > 0,2): vdes <= v_lider + K_ACC*(d - D_PARADA - TH_GAP*v)
#   v_ap = velocidade de aproximacao (reta d(t)); o gatilho usa max(v, v_ap). Ultrassom sem eco (valid = False) = caminho livre.
def _d_gatilho(v):
	return v * T_REACAO + v * v / (2.0 * A_FREIO) + D_PARADA

def _v_cruzeiro(alcance):
	return A_FREIO * (-T_REACAO + np.sqrt(T_REACAO ** 2 + 2.0 * (alcance - D_PARADA) / A_FREIO))

V_CRUZ = _v_cruzeiro(SONAR_ALCANCE)

def _zera_lei():
	# filtros da referencia e estados da derivada voltam a zero: a retomada nao parte de estado velho
	_vdes_filt.reset(0.0)
	if 'vdes_ma' in _estado:
		_estado['vdes_ma'].reset(0.0)
	for k in ('derro', 'erro_ant', 'dvdes_ant'):
		_estado.pop(k, None)

def _adas(car, vdes):
	# devolve (vdes, frear); o sensor chega em _estado['sonar'] = (d, valid) a cada ciclo
	s = _estado.setdefault('adas', dict(buf=[], hist=[], frear=False, v_out=0.0, ult=(-1e9, 0.0)))
	d_lido, valido = _estado.get('sonar', (4.0, False))
	if valido:
		d = d_lido - SONAR_VIES
		s['ult'] = (car.t, d)
	elif car.t - s['ult'][0] < 1.0 and s['ult'][1] < 1.0:
		d = s['ult'][1]                       # sumiu o eco colado no obstaculo: segura a ultima leitura
	else:
		d = np.inf
	# mediana de ~0,35 s (3 leituras do sensor a 10 Hz): rejeita eco espurio
	s['hist'] = [(t, x) for t, x in s['hist'] if car.t - t < JANELA_AP] + ([(car.t, d)] if np.isfinite(d) else [])
	s['buf'] = [(t, x) for t, x in s['buf'] if car.t - t < 0.35] + [(car.t, d)]
	d = float(np.median([x for _, x in s['buf']])) - 2.0 * SONAR_RUIDO
	v = max(car.v, 0.0)
	# velocidade de aproximacao = -inclinacao da reta d(t) nas ultimas JANELA_AP s (estatico: ~v; lider se afastando: menor)
	v_ap, v_lider = 0.0, None
	h = s['hist']
	if len(h) >= 4 and h[-1][0] - h[0][0] >= 0.5 * JANELA_AP:
		tt, dd = np.array(h).T
		v_ap = -float(np.polyfit(tt - tt[-1], dd, 1)[0])
		v_lider = max(v - v_ap, 0.0)         # velocidade do lider (obstaculo que anda)
	v_g = max(v, v_ap)                         # o gatilho vale para o mais rapido dos dois: nunca mais tarde que so com car.v
	if not s['frear'] and v > 0.05 and d < _d_gatilho(v_g):
		s['frear'] = True
	elif s['frear'] and d > D_LIBERA and d > _d_gatilho(v) + 0.3:
		s['frear'] = False
	if s['frear']:
		s['v_out'] = 0.0
		_estado['adas_cod'] = 2
		return 0.0, True
	alvo = min(vdes, V_CRUZ, float(np.sqrt(2.0 * A_PLAN * max(d - D_PARADA, 0.0))) if np.isfinite(d) else vdes)
	if np.isfinite(d) and v_lider is not None and v_lider > 0.2:   # seguindo um lider que anda: distancia cresce com a velocidade
		alvo = min(alvo, v_lider + K_ACC * (d - D_PARADA - TH_GAP * v))
	s['v_out'] = min(alvo, s['v_out'] + TAXA_RAMPA_V * max(car.dt, 0.0))   # sobe devagar, desce na hora
	_estado['adas_cod'] = 1 if s['v_out'] < vdes - 1e-3 else 0
	return s['v_out'], False

########################################
# referencia e ensaios
def _pontos(perfil, taxa):
	# vertices (t, v) do perfil: cada alvo e alcancado com a taxa dada, ate o proximo comecar
	pts = [(0.0, 0.0)]
	for t_i, alvo, *r in perfil:
		v_i = float(np.interp(t_i, *zip(*pts)))
		pts = [p for p in pts if p[0] < t_i] + [(t_i, v_i), (t_i + abs(alvo - v_i) / (r[0] if r else taxa), alvo)]
	return pts

_PONTOS = _pontos(_E['perfil'], TAXA_RAMPA_V) if _E.get('perfil') else None

def _vdes(tau):
	# referencia tau segundos depois da primeira chamada do controle
	if 'parada' in _E and tau >= _E['parada']:
		return 0.0
	if _PONTOS is None:
		return min(V_REF, TAXA_RAMPA_V * tau)
	return float(np.interp(tau, *zip(*_PONTOS)))

def _confere_pista():
	# distancia aproximada = integral da referencia atrasada pela partida; recusa o que nao cabe na pista
	pista = _E.get('pista', PISTA_M)   # a subida tem pista propria (plano + rampa + topo)
	t = np.arange(0.0, _E['ts'], 0.01)
	dist = 0.01 * sum(_vdes(x - ATRASO_PARTIDA) for x in t if x > ATRASO_PARTIDA)
	if dist > pista - 1.0:
		raise ValueError(f"ensaio '{ENSAIO}' anda ~{dist:.1f} m, pista de {pista} m")
	return dist

def _u_degrau(car, s):
	# cada nivel de u a partir do repouso ate v >= V_CORTE (ou acelerador estimado >= PWM_CORTE); depois
	# u = -1 ate o acelerador zerar (u = 0 SEGURARIA a velocidade: o servos.py integra u) e espera parar
	k_pwm = np.rad2deg(0.4 * 0.08 * 5.16)   # graus/s de acelerador por unidade de u
	dt = car.dt if 0.0 < car.dt < 0.5 else 0.0
	s.setdefault('fase', 'espera'); s.setdefault('i', 0); s.setdefault('pwm', 0.0)
	s['tf'] = s.get('tf', 0.0) + dt
	def vai(f):
		s['fase'], s['tf'] = f, 0.0
	niveis = _E['aberto']
	if s['fase'] == 'espera':
		if s['tf'] > 1.0 and abs(car.v) < 0.05 or s['tf'] > 4.0:
			vai('sobe' if s['i'] < len(niveis) else 'fim')
		return 0.0
	if s['fase'] == 'sobe':
		u = min(niveis[s['i']], 1.0)
		s['pwm'] = min(s['pwm'] + k_pwm * u * dt, 25.0)
		if car.v >= V_CORTE or s['pwm'] >= PWM_CORTE or s['tf'] > 6.0:
			vai('desce')
		return u
	if s['fase'] == 'desce':
		s['pwm'] = max(s['pwm'] - k_pwm * dt, 0.0)
		if s['pwm'] <= 0.0 and s['tf'] > 0.3:
			s['i'] += 1
			vai('espera')
		return -1.0
	return 0.0

def _sonar(car, s):
	# carro parado (u = -1 mantem o acelerador em 0); a cada T_SONAR s a caixa muda de distancia
	t0 = s.setdefault('t0', car.t)
	k = min(int((car.t - t0) // T_SONAR), len(_E['sonar']) - 1)
	s['alvo'] = _E['sonar'][k]
	car.vref = 0.0
	car.set_u(-1.0)
	if __name__ == '__main__' and s.get('k') != k:   # so no carro (a analise reexecuta este arquivo)
		s['k'] = k
		msg = f"caixa a {s['alvo']:.1f} m do sensor" if s['alvo'] else "TIRE a caixa (nada a frente ate 4 m)"
		print(f"SONAR: {msg} por {T_SONAR:.0f} s", flush=True)

def _salva_sonar(car, linhas):
	# sonar.csv ao lado do car.csv: t [s], d [m], valid (0 = sem eco por 0,3 s), alvo [m] ('sonar'; nan nos outros),
	# adas (0 livre, 1 supervisor limitando a referencia, 2 emergencia)
	np.savetxt(os.path.join(car.logfile, 'sonar.csv'), np.array(linhas, float).reshape(-1, 5), delimiter=',',
	           header='t,d,valid,alvo,adas', comments='', fmt='%.4f')

########################################
# controle de velocidade
def control_func(car):
	if 'aberto' in _E:
		car.vref = 0.0
		car.set_u(_u_degrau(car, _estado))
		return
	if 'sonar' in _E:
		_sonar(car, _estado)
		return
	t0 = _estado.setdefault('t0', car.t)   # a referencia comeca na primeira chamada
	vdes = _vdes(car.t - t0)
	frear = 'parada' in _E and car.t - t0 >= _E['parada']   # 'parada': depois do corte, u = -1 direto (nao a lei)
	if ADAS_ON and not frear:
		vdes, frear = _adas(car, vdes)
		if frear and RESET_AO_FREAR:
			_zera_lei()
	car.vref = vdes                        # so para o log e a GUI
	u = controlador_longitudinal(car, vdes)   # roda sempre: os filtros nao ficam velhos
	car.set_u(-1.0 if frear else u)

########################################
# thread de visao
def vision_func(car, vision_data, stop_event):

	W, H = car.cam.get_resolution()

	while not stop_event.is_set():

		# pega imagem
		frame = car.get_image(gray=True)

		# detecta aruco
		frame, point = car.cam.detect_aruco(
			frame,
			aruco_id=23
		)

		# disponibiliza imagem para o main
		vision_data["frame"] = frame

		if point is None:
			continue

		# esterçamento aponta para o aruco
		cx = point[0] - W/2

		vision_data["refste"] = -np.deg2rad(20.0*cx/(W/2))

########################################
# main
########################################
if __name__ == "__main__":

	parameters = {
				'ts'					: _E['ts'], # tempo da execucao (vem do ROTEIRO)
				'save'					: True,		# salva dados da trajetoria
				'logfile'				: 'logs/',	# log file
				'camera'				: False,	# habilitar camera e thread de visao
				'us_buzzer'				: True,		# aviso sonoro para objetos proximos
				'initial_position'		: [0, 0, np.deg2rad(0)]	# (x, y, theta) configuracao inicial
			}

	if 'aberto' in _E:
		print(f"ENSAIO '{ENSAIO}' (malha aberta, u = {_E['aberto']}), ts = {_E['ts']} s", flush=True)
	elif 'sonar' in _E:
		print(f"ENSAIO '{ENSAIO}' (carro PARADO; caixa a {_E['sonar']} m, {T_SONAR:.0f} s cada), ts = {_E['ts']} s", flush=True)
	else:
		print(f"ENSAIO '{ENSAIO}': lei '{PROJETO}', ~{_confere_pista():.1f} m de {_E.get('pista', PISTA_M):.0f} m, ts = {_E['ts']} s", flush=True)

	car = Car(parameters)

	vision_data = {"refste": 0.0, "frame": None}
	sonar_log = []   # (t, d, valid, alvo) de cada ciclo -> sonar.csv

	stop_event = threading.Event()
	thread_vision = None

	try:
		car.start_mission()

		# inicia visao somente se solicitada
		if parameters['camera']:
			thread_vision = threading.Thread(
												target=vision_func,
												args=(car, vision_data, stop_event),
												daemon=True
											)
			thread_vision.start()

		if parameters['camera']:
			plt.ion()
			plt.figure(1)

		t_plot = time.monotonic()

		# controle fica na thread principal
		while car.t < parameters['ts']:

			# atualiza sensores
			if not car.step():
				break

			# direcao
			car.set_steer(vision_data["refste"])

			# ultrassom: com ADAS_ON o supervisor (_adas) freia; desligado, SEGUREM o carro
			dist, valid = car.get_distance()
			_estado['sonar'] = (dist, valid)
			control_func(car)
			sonar_log.append((car.t, dist, int(valid), _estado.get('alvo', np.nan), _estado.get('adas_cod', 0)))
			

			# telemetria para plots remotos
			print(
				f"DATA,"
				f"{car.t:.2f},"
				f"{car.p[0]:.2f},"
				f"{car.p[1]:.2f},"
				f"{car.v:.2f},"
				f"{car.vref:.2f},"
				f"{car.a:.2f},"
				f"{car.u:.2f},"
				f"{car.w:.2f},"
				f"{car.th:.2f}",
				flush=True
			)

			# atualiza grafico aproximadamente 1 Hz
			if time.monotonic() - t_plot >= 1.0:

				if parameters['camera']:
					frame = vision_data["frame"]

					if frame is not None:
						plt.cla()
						plt.imshow(frame, cmap='gray')
						plt.pause(0.001)

				t_plot = time.monotonic()

		# salva dados
		if parameters['save']:
			car.save()
			_salva_sonar(car, sonar_log)

	finally:
		# termina a thread de visao
		stop_event.set()

		if thread_vision is not None:
			thread_vision.join(timeout=1.0)

		car.close()

	print('Terminou...')
