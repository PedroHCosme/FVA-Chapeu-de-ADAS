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
KP_MOD       = 0.6     # rad/s -- P gain: phase margin 67 deg nominal, 56 deg at K=2.0 tau=0.4
TAU_VDES     = 0.3     # s -- low-pass filter on the reference

# controller memory between calls (ramp start, reference moving average)
_estado = {}
_vdes_filt = AlphaFilter(alpha=1.0)   # alpha set every call from car.dt

def controlador_longitudinal(car, vdes_bruto):
	# reference low-pass; the filter's own derivative is the feedforward
	if car.dt > 0:
		_vdes_filt.alpha = np.clip(car.dt / TAU_VDES, 0.0, 1.0)
	vdes_anterior = _vdes_filt.value
	vdes = _vdes_filt.filter(vdes_bruto)
	feedforward_dvdes = (vdes_bruto - vdes_anterior) / TAU_VDES

	# compare against the reference delayed by the SAME moving average car.v
	# goes through (car.py v_filt), so P reacts to real tracking error, not to
	# the sensor delay -- cuts the overshoot in simulation
	vdes_ma = _estado.setdefault('vdes_ma', MovingAverage(n=car.v_filt.n))
	erro = vdes_ma.filter(vdes) - car.v

	# P + reference feedforward, divided by the nominal gain. No integral: the
	# throttle integrator already gives zero steady-state error (battery sag,
	# slopes and the ESC deadband all enter after it). No friction feedforward:
	# friction is inside the PWM->speed curve here.
	u = (KP_MOD * erro + feedforward_dvdes) / K_NOM

	# u<0 is how the throttle comes back down (coasting, not braking)
	return float(np.clip(u, -1.0, 1.0))

########################################
# controle de velocidade
def control_func(car):
	# the ramp starts at the first control call: here car.t already counts
	# start_mission()'s 1 s sleep (in the sim, t starts at ~0 in the loop)
	t0 = _estado.setdefault('t0', car.t)
	vdes = min(V_REF, TAXA_RAMPA_V * (car.t - t0))
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
		'ts'                   : 20.0,
		'save'                 : True,
		'logfile'              : 'logs/',
		'camera'               : False,
		'ultrasonic_steering'  : False,
		'us_buzzer'            : False,
		'initial_position'     : [0, 0, np.deg2rad(0)]
	}

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
				f"{car.t:.3f},"
				f"{car.p[0]:.3f},"
				f"{car.p[1]:.3f},"
				f"{car.v:.3f},"
				f"{car.vref:.3f},"
				f"{car.a:.3f},"
				f"{car.u:.3f},"
				f"{car.w:.3f},"
				f"{car.th:.3f}",
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
