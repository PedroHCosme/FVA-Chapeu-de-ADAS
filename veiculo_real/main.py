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
# CARRO REAL: controlador de velocidade PD + feedforward com inversao de atraso, e supervisor ADAS (FCW/AEB pelo ultrassom).
# Este e o unico arquivo copiado para o carro; cada corrida grava car.csv e sonar.csv.
V_REF          = 1.3    # m/s -- velocidade de cruzeiro
TAXA_RAMPA_V   = 0.9    # m/s^2 -- a referencia sobe ate V_REF com esta inclinacao (o ADAS solta o freio com a mesma)
TEMPO_CORRIDA  = 35.0   # s -- duracao da corrida (~45 m a 1,3 m/s: confira a pista)
V_GUARDA       = 1.35   # m/s -- guarda de VELMAX: acima disto u = -1 (None desliga). Escolhida em analise/sintonia.py

########################################
# ENSAIOS LATERAIS: caracterizam a direcao para o Stanley. None = corrida normal (direcao pela visao).
#   'trim'    : delta = 0 em linha reta. Desvio lateral medido com trena no chao -> ajusta TRIM_DIRECAO.
#   'circulo' : delta = DELTA_ENSAIO constante. Diametro medido com trena -> delta real = arctan(L/R);
#               depois de N voltas o th do log deve ter girado N*360 graus.
#   'degrau'  : reto por T_ASSENTA s, depois patamares +DELTA, 0, -DELTA, 0 de T_PATAMAR s. Atraso = st -> w_imu.
ENSAIO_LATERAL = None
V_ENSAIO       = 0.5    # m/s -- velocidade constante do ensaio (no lugar de V_REF)
DELTA_ENSAIO   = 20.0   # graus -- esterçamento do circulo e do degrau; o lado (esquerda/direita) anote vendo o carro
TRIM_DIRECAO   = 0.0    # graus -- somado a todo comando de direcao (servos.set_trim), tambem na corrida normal
T_ASSENTA      = 4.0    # s -- reto ate a velocidade assentar, antes do primeiro degrau
T_PATAMAR      = 3.0    # s -- duracao de cada patamar do degrau (a 15-20 graus e 0,5 m/s: ~60-80 graus de giro)
TEMPO_ENSAIO   = {'trim': 20.0, 'circulo': 40.0, 'degrau': T_ASSENTA + 4 * T_PATAMAR + 1.0}   # s

########################################
# GANHOS. Planta (bateria > 90%): u -> servos.py integra no acelerador (9,46 graus/s por unidade de u)
# -> zona morta e ganho do ESC -> atraso tau ~0,6-0,75 s -> car.v (media de 30 amostras, ~0,42 s). K = dv/dt por unidade de u ~1,95-2,9.
# Sintonia: analise/sintonia.py, ponto #5 da fronteira de Pareto.
KP       = 0.225   # [u por m/s de erro] ganho proporcional; a malha tem velocidade ~KP*K rad/s (validado no carro: 0,55-0,6)
TD       = 0.6     # [s] tempo derivativo (0 desliga); derivada filtrada com TD/5
TAU_VDES = 0.2     # [s] filtro de 1a ordem da referencia
K_FF     = 2.9     # [m/s^2 por u] o K que o feedforward supoe: o MAIOR K medido, para o FF nunca empurrar mais que o carro precisa
TAU_FF   = 0.61    # [s] atraso da planta que o feedforward compensa (0 desliga); ajuste de 01/10

########################################
# SUPERVISOR ADAS (ultrassom): fica entre a referencia e a lei de controle (_adas, mais abaixo).
ADAS_ON        = True   # False = sem supervisor (SEGUREM o carro)

# sensor: calibrado com o carro parado e uma caixa a distancias conhecidas (python analise/calibra_adas.py <sonar.csv> imprime estes valores)
SONAR_VIES     = 0.00   # m -- leitura menos distancia real (mediana de 0,5 a 2 m)
SONAR_RUIDO    = 0.005  # m -- desvio das leituras [20260930_222919]
SONAR_ALCANCE  = 2.900  # m -- maior distancia com >= 90% de leituras validas e erro < 0,10 m [20260930_222919]

# freio e folgas
A_FREIO        = 0.5    # m/s^2 -- desaceleracao so por atrito. MEDIDO (analise/parada.py, 12 paradas): >= 0,62 de 0,3 a 1,3 m/s; 0,5 e o piso
T_REACAO       = 0.7    # s -- do corte (u = -1) ate a velocidade cair 5%. MEDIDO: 0,43-0,98 s, mediana 0,65; com A_FREIO preve as 12 paradas com folga minima de 0,12 m
D_PARADA       = 0.1    # m -- folga entre o para-choque e o obstaculo, parado
A_PLAN         = 0.30   # m/s^2 -- a camada suave planeja frear com isto (< A_FREIO: sobra margem)
D_LIBERA       = 1.5    # m -- parado por obstaculo, so volta a andar quando a distancia passar disto (ou livre)
TH_GAP         = 1.0    # s -- seguindo um lider em movimento: distancia desejada = D_PARADA + TH_GAP * v
K_ACC          = 0.4    # 1/s -- vdes <= v_lider + K_ACC * (d - distancia desejada)
JANELA_AP      = 0.6    # s -- janela da reta d(t) que da a velocidade de aproximacao (ultrassom a 10 Hz: ~6 leituras)

########################################
# LEI DE CONTROLE (u = aceleracao pedida ao car.set_u, saturada em +-1)
#
#   u = KP*e + KP*TD*de/dt + (dvdes/dt + TAU_FF*d2vdes/dt2) / K_FF      e = MA30(vdes) - v
#
# Como o servos.py integra u, no acelerador esta MESMA lei e um PI + feedforward estatico:
# Ki = 9,46*KP, Kp = 9,46*KP*TD e 9,46*vdes/K_FF graus (analise/prova_integrador.py). Malha tipo 1:
# erro zero em regime para referencia constante. O feedforward fica fora da malha: K_FF e TAU_FF
# nao mexem nas margens.
_estado = {}                          # memoria do controlador entre chamadas
_vdes_filt = AlphaFilter(alpha=1.0)   # alpha recalculado a cada chamada com car.dt

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

def _salva_sonar(car, linhas):
	# sonar.csv ao lado do car.csv: t [s], d [m], valid (0 = sem eco por 0,3 s), alvo [m] (sempre nan),
	# adas (0 livre, 1 supervisor limitando a referencia, 2 emergencia)
	np.savetxt(os.path.join(car.logfile, 'sonar.csv'), np.array(linhas, float).reshape(-1, 5), delimiter=',',
	           header='t,d,valid,alvo,adas', comments='', fmt='%.4f')

########################################
# esterçamento [rad] do ensaio lateral; o tempo conta da primeira chamada, como a rampa de velocidade
def direcao_ensaio(car):
	t = car.t - _estado.setdefault('t0', car.t)
	delta = np.deg2rad(DELTA_ENSAIO)
	if ENSAIO_LATERAL == 'circulo':
		return delta
	if ENSAIO_LATERAL == 'degrau' and t >= T_ASSENTA:
		patamar = int((t - T_ASSENTA) // T_PATAMAR)
		return (delta, 0.0, -delta, 0.0)[min(patamar, 3)]
	return 0.0

########################################
# controle de velocidade
def control_func(car):
	t0 = _estado.setdefault('t0', car.t)   # a referencia comeca na primeira chamada
	v_alvo = V_ENSAIO if ENSAIO_LATERAL else V_REF
	vdes = min(v_alvo, TAXA_RAMPA_V * (car.t - t0))
	frear = False
	if ADAS_ON:
		vdes, frear = _adas(car, vdes)
		if frear:
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
				'ts'					: TEMPO_ENSAIO[ENSAIO_LATERAL] if ENSAIO_LATERAL else TEMPO_CORRIDA, # tempo da execucao
				'save'					: True,		# salva dados da trajetoria
				'logfile'				: 'logs/',	# log file
				'camera'				: False,	# habilitar camera e thread de visao
				'us_buzzer'				: True,		# aviso sonoro para objetos proximos
				'initial_position'		: [0, 0, np.deg2rad(0)]	# (x, y, theta) configuracao inicial
			}

	print(f"PD + FF (KP {KP}, TD {TD}, TAU_VDES {TAU_VDES}, K_FF {K_FF}, TAU_FF {TAU_FF}), V_REF {V_REF} m/s, "
	      f"ADAS {'ligado' if ADAS_ON else 'DESLIGADO'}, ts = {parameters['ts']:.0f} s", flush=True)
	if ENSAIO_LATERAL:
		print(f"ENSAIO LATERAL '{ENSAIO_LATERAL}': v {V_ENSAIO} m/s, delta {DELTA_ENSAIO} graus, trim {TRIM_DIRECAO} graus", flush=True)

	car = Car(parameters)
	car.atuador.set_trim(steer=np.deg2rad(TRIM_DIRECAO))

	vision_data = {"refste": 0.0, "frame": None}
	sonar_log = []   # (t, d, valid, alvo, adas) de cada ciclo -> sonar.csv

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
			car.set_steer(direcao_ensaio(car) if ENSAIO_LATERAL else vision_data["refste"])

			# ultrassom: com ADAS_ON o supervisor (_adas) freia; desligado, SEGUREM o carro
			dist, valid = car.get_distance()
			_estado['sonar'] = (dist, valid)
			control_func(car)
			sonar_log.append((car.t, dist, int(valid), np.nan, _estado.get('adas_cod', 0)))

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
