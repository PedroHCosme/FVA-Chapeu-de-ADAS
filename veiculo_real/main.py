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
# K_NOM is the plant gain the controller ASSUMES (v_dot = K * u); the real K is what
# analise/analisa.py prints as "K medido". What sets the loop speed is KP_eff = KP * K / K_NOM.
# Runs of 2026-09-25: real K ~ 0.95-1.09 and plant lag tau ~ 0.7 s (not K 1.6 / tau 0.25), and the
# 'avanco' preset (KP 0.8) worked well with K_NOM = 1.6: KP_eff = 0.55. K_NOM ABOVE the real K is
# the safe side (slower loop, no overshoot). Simulated on the fitted plant (tau 0.715, DB 3.95),
# overshoot of the 'avanco' law:
#     K real   K_NOM 1.6 (KP_eff)    K_NOM = K real    K_NOM ~1.4 x K real
#     0.9-1.09   0-1.2%  (0.55)        13-16%            (not needed)
#     1.4        8.5%    (0.70)        15%               K_NOM 2.0 -> 0.8%
#     1.6        13.8%   (0.80)        13.8%             K_NOM 2.3 -> 0.3%
#     2.0        24.7%   (1.00)        11%               K_NOM 2.6 -> 1.1%
# So: keep 1.6 while the measured K stays <= ~1.2. If it comes back HIGHER (fresh battery, K >=
# ~1.4 as on 2026-09-24), raise K_NOM to ~1.4 x K (aim at KP_eff ~ 0.55), NOT to K itself: with
# the slow plant K_NOM = K leaves KP_eff = 0.8, too fast (13-16% overshoot). analisa.py simulates
# the candidates and prints the choice. A higher K_NOM also weakens the feedforward (u_ff = rate / K_NOM).
# The simulation under-predicts the real 'avanco' overshoot by ~3 points (sim 1.2%, car 4-6%).
V_REF        = 1.3      # m/s -- cruise speed target
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
#   'frenagem'  'avanco' following a DECREASING reference: holds 1.0 m/s long enough for the slow
#               car to get there and settle, then steps down 1.0 > 0.6 > (up) 1.0 > 0.6 slowly
#               > 0.2 > 0 with different slew rates. Tests what the professor calls braking.
#   'freio_esc' EXPLORATORY, hardware risk: does the ESC brake actively when the throttle goes
#               BELOW neutral in forward gear? See _freio_esc. Not run on the car yet.
# Then: python analise/analisa.py   (on the PC, after copying the new log folder).
ENSAIO = 'avanco'   # 'base', 'patamares', 'u_degrau', 'avanco', 'v_alta', 'auto', 'frenagem', 'freio_esc'

PISTA_M        = 100000000   # m of straight track available
ATRASO_PARTIDA = 1.0    # s -- the car needs about this long to start moving (deadband)
V_CORTE        = 0.7    # m/s -- 'u_degrau' stops accelerating here (measured; MA30 lags ~0.4 s and the
                        #        throttle keeps rising meanwhile, so the real peak is ~1.1 m/s)
PWM_CORTE      = 8.0    # deg -- ... or when the estimated throttle gets here (v_ss ~ 1 m/s), whichever first

V_FREIO        = 0.8    # m/s -- speed the 'freio_esc' test brakes from
T_SOBE         = 10.0   # s -- 'freio_esc': time allowed to reach V_FREIO from rest (the car is slow: ~1.5 s
                        #      of deadband + ~4 s to rise, see the straight-line runs of 2026-09-25)
T_PULSO        = 0.8    # s -- 'freio_esc': longest time the throttle stays below neutral (released earlier at v < 0.15)

# vref profile = list of (t [s], target [m/s]) or (t, target, slew rate [m/s^2]); vdes slews toward the
# target at TAXA_RAMPA_V unless the step gives its own rate. None = ramp to V_REF and hold (the validated
# 'base' reference). Steps are spaced >= 6-7 s: the car needs ~4 s to settle after a step (tau ~0.7 s + MA30).
ROTEIRO = {
	'base'     : dict(projeto='base',   perfil=None, ts=30000.0),
	'avanco'   : dict(projeto='avanco', perfil=None, ts=40.0),
	'auto'     : dict(projeto='auto',   perfil=None, ts=3000000.0),
	'v_alta'   : dict(projeto='base',   perfil=None, ts=30.0, vref=2.0),
	'patamares': dict(projeto='base',   ts=90.5,
	                  perfil=[(0, 0.6), (3.5, 0.9), (6.5, 1.1), (9.5, 0.8), (12, 0.5), (13.5, 0.0)]),
	'u_degrau' : dict(aberto=[0.5, 0.8, 1.3, 1.8], ts=30.0),
	# reach 1.0 (t = 0-13: ~5 s to get there + settling), step down -0.4 fast, back up, step down -0.4 at only
	# 0.25 m/s^2, -0.4 fast to 0.2, stop. ~32 m.
	'frenagem' : dict(projeto='avanco', ts=50.0,
	                  perfil=[(0, 1.0), (13, 0.6), (20, 1.0), (27, 0.6, 0.25), (34, 0.2), (41, 0.0)]),
	# throttle pulse below neutral (deg) applied after the throttle is drained; 0 = plain coasting control
	'freio_esc': dict(projeto='avanco', freio=[0, 6, 12, 18], ts=80.0),
}
_E = ROTEIRO[ENSAIO]
V_REF = _E.get('vref', V_REF)

# Tuning presets (names are used by ROTEIRO):
#   'base'   P + reference feedforward. The design of the chapter: phase margin
#            67 deg nominal, 56 deg at K=2.0 tau=0.4.
#   'avanco' the same law plus a derivative (lead) term on the error. Found in
#            simulation only (matlab-control/): phase margin 68 deg at K=2.0
#            tau=0.4, faster settling. Tested on the car 2026-09-25 (straight line
#            and ramps): overshoot 4-6%, RMS error 0.02 m/s. This is the chosen one.
#   'auto'   starts equal to 'base'; analisa.py prints the line to paste here.
# TD_DERIV = 0 makes the law identical to 'base' for any KP_MOD / TAU_VDES.
_PROJETOS = {
	'base'  : dict(KP=0.6, TD=0.0, TAU_VDES=0.3),
	'avanco': dict(KP=0.8, TD=0.6, TAU_VDES=0.5),
	'auto'  : dict(KP=0.5, TD=0.8, TAU_VDES=0.3, K_NOM=1.09),
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
	for t_i, alvo, *r in perfil:   # optional 3rd item: slew rate of this step
		v_i = float(np.interp(t_i, *zip(*pts)))
		pts = [p for p in pts if p[0] < t_i] + [(t_i, v_i), (t_i + abs(alvo - v_i) / (r[0] if r else taxa), alvo)]
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
	if 'freio' in _E:   # per level: climb to V_FREIO, coast while the throttle drains, stop
		return len(_E['freio']) * (V_FREIO * (T_SOBE - 2.0) + 3.0)
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
		u = min(niveis[s['i']], 1.0)   # set_u clips at +-1: the pwm estimate below must see the same u
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
# EXPLORATORY 'freio_esc' -- NOT run on the car yet; hardware risk, run it with someone holding the car.
# Question: servos.py never commands the throttle below neutral in forward gear (th_pwm >= 0), so the
# car only COASTS. _backward() taps the ESC below neutral four times, the usual brake/reverse
# sequence of hobby ESCs, so a pulse below neutral may brake. servos._set_pwm adds trim_throttle
# (rad) to the command and its forward clip allows 18 deg below neutral, so the test shifts the trim
# for a short pulse while the throttle is at 0. Per level in _E['freio'] (deg below neutral):
#   sobe   avanco controller to V_FREIO from rest (T_SOBE s)
#   drena  u = -1 until the estimated throttle is 0 (the car coasts)
#   pulso  trim = -level for at most T_PULSO s, released early at v < 0.15 (an ESC can go into
#          REVERSE if the pulse outlasts the stop; v is signed, v < 0 would show it)
#   espera trim = 0, wait until the car is stopped
# The log has no trim column, so vref carries the marker: -0.01 while draining, -max(level/10, 0.02)
# during the pulse (level 0 = plain coasting control), 0 while waiting. analisa.py reads it.
def _aplica_trim(car, graus):
	a = getattr(car, 'atuador', None)
	if a is not None:
		a.trim_throttle = -np.deg2rad(graus)

def _freio_esc(car, s):
	k_pwm = np.rad2deg(0.4 * 0.08 * 5.16)   # deg/s of throttle per unit u (servos.py, car.py)
	dt = car.dt if 0.0 < car.dt < 0.5 else 0.0
	s.setdefault('fase', 'sobe'); s.setdefault('i', 0); s.setdefault('pwm', 0.0)
	s['tf'] = s.get('tf', 0.0) + dt
	s['pwm'] = min(max(s['pwm'] + k_pwm * car.u * dt, 0.0), 18.08)   # car.u = what was applied last cycle
	niveis = _E['freio']
	nivel = niveis[min(s['i'], len(niveis) - 1)]
	def vai(f):
		s['fase'], s['tf'] = f, 0.0
		if f == 'sobe':   # the controller starts from rest again
			for k in ('vdes_ma', 'derro', 'erro_ant'):
				s.pop(k, None)
			_vdes_filt.reset(0.0)
	trim, u = 0.0, 0.0
	car.vref = 0.0
	if s['fase'] == 'sobe':
		car.vref = min(V_FREIO, TAXA_RAMPA_V * s['tf'])
		u = controlador_longitudinal(car, car.vref)
		if s['tf'] > T_SOBE:
			vai('drena')
	elif s['fase'] == 'drena':
		u, car.vref = -1.0, -0.01
		if (s['pwm'] <= 0.0 and s['tf'] > 0.3) or s['tf'] > 4.0:
			vai('pulso')
	elif s['fase'] == 'pulso':
		trim, car.vref = nivel, -max(nivel / 10.0, 0.02)
		if s['tf'] > T_PULSO or (s['tf'] > 0.1 and car.v < 0.15):
			vai('espera')
			trim, car.vref = 0.0, 0.0
	elif s['fase'] == 'espera':
		s['parado'] = s.get('parado', 0.0) + dt if abs(car.v) < 0.05 else 0.0
		if s['parado'] >= 1.0 or s['tf'] > 8.0:
			s['i'] += 1
			s['parado'] = 0.0
			vai('sobe' if s['i'] < len(niveis) else 'fim')
	_aplica_trim(car, trim)
	car.set_u(float(np.clip(u, -1.0, 1.0)))

########################################
# controle de velocidade
def control_func(car):
	# open-loop test: vref is logged as 0, u is the test signal
	if 'aberto' in _E:
		car.vref = 0.0
		car.set_u(_u_aberto(car, _estado))
		return
	if 'freio' in _E:
		_freio_esc(car, _estado)
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
	elif 'freio' in _E:
		print(f"ENSAIO '{ENSAIO}' (EXPLORATORIO: pulsos abaixo do neutro {_E['freio']} graus, ~{_confere_pista():.0f} m), ts = {_E['ts']} s", flush=True)
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

			#if (not valid) or (dist < 0.20):
				#print(f"Colisao: distance {dist:.2f} [m]")
				#car.set_vel(0.0)
			if True:
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

		_aplica_trim(car, 0.0)   # never leave the 'freio_esc' pulse applied
		car.close()

	print('Terminou...')
