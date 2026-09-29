# -*- coding: utf-8 -*-
########################################
# Disciplina: Topicos em Engenharia de Controle e Automacao IV (ENG075): 
# Fundamentos de Veiculos Autonomos - 2026/2
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automacao
# DELT - Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
# -*- coding: utf-8 -*-
from fva_car import Car
from fva_car.filter import AlphaFilter, MovingAverage
import numpy as np
import matplotlib.pyplot as plt
import threading
import time

########################################
# longitudinal controller for the REAL car. simulador/main.py's controller
# does not transfer: here servos.py INTEGRATES u into the throttle PWM, so
# u=0 holds the speed instead of coasting. Plant fitted on the logs of
# 2026-09-24 (veiculo_real/dados*): u -> PWM integrator -> ESC deadband and
# gain G -> lag tau -> car.v through a 30-sample moving average (~0.42 s at
# the real ~29 ms loop). Designed for a (near) fully charged battery:
# K = dv/dt per unit u ~1.6, tau ~0.25 s. A discharged battery lowers K
# (0.9 seen with a half-empty one) -- the loop just gets slower.
########################################
K_NOM        = 1.6     # m/s^2 per unit u -- charged battery (fitted on runs 002634..003330)
V_REF        = 1       # m/s -- cruise speed target
TAXA_RAMPA_V = 0.9     # m/s^2 -- ramp-up rate for vdes

########################################
# TEST SCRIPT. To switch test runs change ONLY this line (everything lives in
# this file on purpose: it is the one file that is copied to the car).
#   'base'      controller of the chapter (FF + P) on a ramp to 1 m/s.
#   'patamares' same controller, reference in steps 0.6 > 0.9 > 1.1 > 0.8 > 0.5 m/s:
#               separates gain G from deadband DB in the analysis.
#   'u_degrau'  OPEN LOOP: constant u = 0.3, 0.5, 0.8 from rest, one after the other
#               (measures deadband DB and delay d from the start time, t = DB/(9.46 u) + d).
#   'avanco'    same as 'base' plus the derivative term (see _PROJETOS).
#   'v_alta'    'base' cruising at 1.3 m/s.
#   'auto'      the preset that analisa.py printed (paste it into _PROJETOS['auto']).
# Then: python analise/analisa.py   (on the PC, after copying the new log folder).
ENSAIO = 'base'

PISTA_M        = 12.0   # m of straight track available
ATRASO_PARTIDA = 1.0    # s -- the car needs about this long to start moving (deadband)
V_CORTE        = 0.7    # m/s -- 'u_degrau' stops accelerating here (measured; MA30 lags ~0.4 s and the
                        #        throttle keeps rising meanwhile, so the real peak is ~1.1 m/s)
PWM_CORTE      = 8.0    # deg -- ... or when the estimated throttle gets here (v_ss ~ 1 m/s), whichever first

# vref profile = list of (t [s], target [m/s]); vdes slews toward the target at
# TAXA_RAMPA_V. None = ramp to V_REF and hold (the validated 'base' reference).
ROTEIRO = {
	'base'     : dict(projeto='base',   perfil=None, ts=12.0),
	'avanco'   : dict(projeto='avanco', perfil=None, ts=12.0),
	'auto'     : dict(projeto='auto',   perfil=None, ts=12.0),
	'v_alta'   : dict(projeto='base',   perfil=None, ts=10.0, vref=1.3),
	'patamares': dict(projeto='base',   ts=14.5,
	                  perfil=[(0, 0.6), (3.5, 0.9), (6.5, 1.1), (9.5, 0.8), (12, 0.5), (13.5, 0.0)]),
	'u_degrau' : dict(aberto=[0.3, 0.5, 0.8], ts=16.0),
}
_E = ROTEIRO[ENSAIO]
V_REF = _E.get('vref', V_REF)

# Tuning presets (names are used by ROTEIRO):
#   'base'   P + reference feedforward. The design of the chapter: phase margin
#            67 deg nominal, 56 deg at K=2.0 tau=0.4.
#   'avanco' the same law plus a derivative (lead) term on the error. Found in
#            simulation only (matlab-control/): phase margin 68 deg at K=2.0
#            tau=0.4, faster settling. NOT tested on the car yet.
#   'auto'   starts equal to 'base'; analisa.py prints the line to paste here.
# TD_DERIV = 0 makes the law identical to 'base' for any KP_MOD / TAU_VDES.
_PROJETOS = {
	'base'  : dict(KP=0.6, TD=0.0, TAU_VDES=0.3),
	'avanco': dict(KP=0.8, TD=0.6, TAU_VDES=0.5),
	'auto'  : dict(KP=0.6, TD=0.0, TAU_VDES=0.3),
}
PROJETO      = _E.get('projeto', 'base')
K_NOM        = _PROJETOS[PROJETO].get('K_NOM', K_NOM)  # m/s^2 per unit u
KP_MOD       = _PROJETOS[PROJETO]['KP']        # rad/s -- P gain
TD_DERIV     = _PROJETOS[PROJETO]['TD']        # s -- derivative time on the error (0 = off)
TAU_VDES     = _PROJETOS[PROJETO]['TAU_VDES']  # s -- low-pass filter on the reference

def _pontos(perfil, taxa):
	# breakpoints (t, v) of the reference: it slews to each target at `taxa` and is
	# cut when the next step starts
	pts = [(0.0, 0.0)]
	for t_i, alvo in perfil:
		v_i = float(np.interp(t_i, *zip(*pts)))
		pts = [p for p in pts if p[0] < t_i] + [(t_i, v_i), (t_i + abs(alvo - v_i) / taxa, alvo)]
	return pts

_PONTOS = _pontos(_E['perfil'], TAXA_RAMPA_V) if _E.get('perfil') else None

def _vdes(tau):
	# reference at tau seconds after the first control call
	if _PONTOS is None:
		return min(V_REF, TAXA_RAMPA_V * tau)
	return float(np.interp(tau, *zip(*_PONTOS)))

def _confere_pista():
	# rough distance travelled = integral of the reference delayed by the startup;
	# refuse to run a profile that does not fit the track
	t = np.arange(0.0, _E['ts'], 0.01)
	v = np.array([_vdes(x - ATRASO_PARTIDA) if x > ATRASO_PARTIDA else 0.0 for x in t])
	dist = float(v.sum() * 0.01)
	if dist > PISTA_M - 1.0:   # 1 m left for the stop
		raise ValueError(f"ensaio '{ENSAIO}' anda ~{dist:.1f} m, pista de {PISTA_M} m")
	return dist

# open-loop test state lives in _estado too
def _u_aberto(car, s):
	# u constant per level from rest until v >= V_CORTE (or pwm >= PWM_CORTE), then u = -1 until the throttle
	# is back to zero (u = 0 would HOLD the speed here: servos.py integrates u),
	# then wait for the car to stop before the next level. Never leaves forward gear.
	k_pwm = np.rad2deg(0.4 * 0.08 * 5.16)   # deg/s of throttle per unit u (servos.py, car.py)
	dt = car.dt if 0.0 < car.dt < 0.5 else 0.0
	s.setdefault('fase', 'espera'); s.setdefault('i', 0); s.setdefault('pwm', 0.0)
	s['tf'] = s.get('tf', 0.0) + dt
	def vai(f):
		s['fase'], s['tf'] = f, 0.0
	niveis = _E['aberto']
	if s['fase'] == 'espera':
		if s['tf'] > 1.0 and abs(car.v) < 0.05 or s['tf'] > 4.0:
			if s['i'] < len(niveis):
				vai('sobe')
			else:
				vai('fim')
		return 0.0
	if s['fase'] == 'sobe':
		u = niveis[s['i']]
		s['pwm'] = min(s['pwm'] + k_pwm * u * dt, 25.0)
		if car.v >= V_CORTE or s['pwm'] >= PWM_CORTE or s['tf'] > 6.0:
			vai('desce')
		return u
	if s['fase'] == 'desce':
		s['pwm'] = max(s['pwm'] - k_pwm * dt, 0.0)
		if s['pwm'] <= 0.0 and s['tf'] > 0.3:   # servos.py clips the throttle at 0
			s['i'] += 1
			vai('espera')
		return -1.0
	return 0.0

# controller memory between calls (ramp start, reference moving average,
# derivative state)
_estado = {}
_vdes_filt = AlphaFilter(alpha=1.0)   # alpha set every call from car.dt

########################################
# CONTROL LAW (what every closed-loop ENSAIO runs; only the gains change)
#
#   u = u_p + u_d + u_ff          then clipped to [-1, 1]
#
#   u_p  = KP / K_NOM * e                    proportional on the tracking error
#   u_d  = KP * TD / K_NOM * de/dt           derivative (lead) on the error, TD = 0 turns it off
#   u_ff = (dvdes/dt) / K_NOM                feedforward of the (filtered) reference rate
#
#   e    = MA30(vdes) - v                    vdes = TAU_VDES low-pass of the raw ramp/steps
#                                            v    = car.v, already a 30-sample moving average
#
# Type: reference feedforward + P (+ filtered D), NO integral, 2 degrees of freedom (the
# reference enters both through the feedforward and through the error). The plant already
# integrates u into the throttle PWM (servos.py), so the loop is type 1 and needs no
# controller integrator for zero steady-state error to a constant reference. The
# ESC deadband, battery sag and slopes all enter AFTER that integrator.
# Loop stability depends only on C = KP / K_NOM (and TD); K_NOM also sets the feedforward
# scale: a K_NOM below the real K makes both the loop and the feedforward stronger.
def controlador_longitudinal(car, vdes_bruto):

	# ---- reference: low-pass of the raw reference (its own derivative is the feedforward)
	if car.dt > 0:
		_vdes_filt.alpha = np.clip(car.dt / TAU_VDES, 0.0, 1.0)
	vdes_anterior = _vdes_filt.value
	vdes = _vdes_filt.filter(vdes_bruto)
	dvdes_dt = (vdes_bruto - vdes_anterior) / TAU_VDES   # [m/s^2]

	# ---- tracking error, against the reference delayed by the SAME moving average car.v
	# goes through (car.py v_filt): P reacts to real tracking error, not to the sensor delay
	vdes_ma = _estado.setdefault('vdes_ma', MovingAverage(n=car.v_filt.n))
	erro = vdes_ma.filter(vdes) - car.v                  # [m/s]

	# ---- derivative of the error, low-passed with TD/5 to keep the noise down. It gives
	# back the phase the moving average takes away, so KP can be higher. 0 when TD = 0.
	derro_dt = 0.0                                       # [m/s^2]
	if TD_DERIV > 0.0 and car.dt > 0:
		b = car.dt / (TD_DERIV / 5.0 + car.dt)
		derro_dt = _estado.get('derro', 0.0)
		derro_dt += b * ((erro - _estado.get('erro_ant', erro)) / car.dt - derro_dt)
		_estado['derro'], _estado['erro_ant'] = derro_dt, erro

	# ---- the three terms, each one already in units of u
	u_p  = KP_MOD * erro / K_NOM
	u_d  = KP_MOD * TD_DERIV * derro_dt / K_NOM
	u_ff = dvdes_dt / K_NOM

	# ---- the law is only the sum. u < 0 is how the throttle comes back down (coasting,
	# not braking). No friction feedforward: friction is inside the PWM->speed curve here.
	u = u_p + u_d + u_ff
	return float(np.clip(u, -1.0, 1.0))

########################################
# controle de velocidade
def control_func(car):
	# open-loop test: vref is logged as 0, u is the test signal
	if 'aberto' in _E:
		car.vref = 0.0
		car.set_u(_u_aberto(car, _estado))
		return
	# the reference starts at the first control call: here car.t already counts
	# start_mission()'s 1 s sleep (in the sim, t starts at ~0 in the loop)
	t0 = _estado.setdefault('t0', car.t)
	vdes = _vdes(car.t - t0)
	car.vref = vdes   # only for the log and the GUI's vref trace
	car.set_u(controlador_longitudinal(car, vdes))

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
	else:
		print(f"ENSAIO '{ENSAIO}': projeto '{PROJETO}', ~{_confere_pista():.1f} m de {PISTA_M} m, ts = {_E['ts']} s", flush=True)

	car = Car(parameters)

	vision_data = {"refste": 0.0, "frame": None}

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

			# ultrassom
			dist, valid = car.get_distance()

			if (not valid) or (dist < 0.20):
				print(f"Colisao: distance {dist:.2f} [m]")
				car.set_vel(0.0)
			else:
				control_func(car)

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

	finally:
		# termina a thread de visao
		stop_event.set()

		if thread_vision is not None:
			thread_vision.join(timeout=1.0)

		car.close()

	print('Terminou...')
