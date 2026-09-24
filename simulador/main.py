# -*- coding: utf-8 -*-
# Disciplina: Tópicos em Engenharia de Controle e Automação IV (ENG075):
# Fundamentos de Veículos Autônomos - 2026/1
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automação
# DELT – Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
from fva_car import Car
from fva_car.filter import AlphaFilter
import numpy as np
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6, 8)

# ==========================================================================
# SIMULATION
# ==========================================================================
parameters = {
    'ts': 30.0,       # simulation time [s]
    'save': True,
    'logfile': 'logs/',
    'beep': True,
}

# ==========================================================================
# IDENTIFIED MODEL -- v_dot = K*u - c*v|v| - a0*sign(v)  (IDENTIFICACAO.md)
# ==========================================================================
K_MOD  = 1.00      # gain: fraction of the command that becomes real acceleration
A0_MOD = 0.096     # dry friction [m/s^2]
C_MOD  = 0.00425   # quadratic drag v|v| [1/m]
EPS_V  = 0.05      # smooths the feedforward's sign(v) near v=0

# ==========================================================================
# LONGITUDINAL CONTROLLER -- cruise control
# ==========================================================================
V_REF        = 1      # m/s -- cruise speed target
TAXA_RAMPA_V = 0.9     # m/s^2 -- ramp-up rate for vdes

# PI: Kp and Ki, no Kd -- derivative would amplify the measurement
# noise around command changes
# left out on purpose. Tt is  the back-calculation anti-windup tracking time
# constant (Astrom)
KP_MOD       = 4.0      # rad/s -- proportional gain 
KI_MOD       = 2.67     # 1/s^2 -- integral gain, Ti=Kp/Ki=1.5s
TT_MOD       = 0.5     # s -- back-calculation anti-windup, Tt=sqrt(Ti*Td)
TAU_VDES     = 0.3     # s -- low-pass filter on the reference 
TAU_U        = 0.10    # s -- low-pass filter on u, AFTER the clip, BEFORE the actuator.
                        # Attacks high-frequency noise (v measurement amplified by Kp/Ki)
                        # without delaying the error the controller reacts to -- filtering
                        # v instead of u would slow the reaction to a real disturbance
                        # (the ramp). Chosen as an overshoot-vs-chatter trade-off.

# Controller memory, kept alive between calls (control_func() is called
# fresh every loop iteration -- car.step(); control_func(car); ... -- and a
# plain function has no memory of its own between calls, so this lives at
# module scope instead of as a local variable):
#   I             -- the integrator's current value. I[k+1] = I[k] + dt*(...)
#                    is a recursion: without last step's I there is no
#                    integral action, just a one-shot dt*Ki*erro every call.
#   u_prev        -- u actually applied to the actuator on the PREVIOUS step.
#   u_bruto_prev  -- u_bruto (pre-clip) the control law asked for on the
#                    PREVIOUS step.
# u_prev and u_bruto_prev feed this step's anti-windup back-calculation,
# which by definition compares against what happened last step.
_estado_completo = {'I': 0.0, 'u_prev': 0.0, 'u_bruto_prev': 0.0}
_vdes_filt = AlphaFilter(alpha=1.0)   # alpha set on first use from car.dt (see below)
_u_filt = AlphaFilter(alpha=1.0)      # same, filters u after it's already clipped


def controlador_longitudinal(car, vdes_bruto):
    # Each term of the control law is its own variable, in the order it
    # appears in the equation (CONTROLE_LONGITUDINAL.md sections 6 and 13):
    #   a_cmd   = Kp*erro + I + feedforward_dvdes
    #   u_bruto = (a_cmd + compensa_atrito + compensa_arrasto) / K
    #   u_clip  = clip(u_bruto, 0, 1)
    #   u       = low_pass(u_clip, TAU_U)   -- only this reaches the actuator
    #   I[k+1]  = I[k] + dt*(Ki*erro + anti_windup_back_calc)
    #
    # DISCRETIZATION: every continuous-time dynamic in this function (the
    # integrator, both low-pass filters) is discretized with forward Euler,
    # y[k+1] = y[k] + dt*f(y[k]), using car.dt -- the ACTUAL step measured by
    # CoppeliaSim each call, not a fixed constant (it isn't perfectly
    # constant, and hardcoding one would silently desync the model from the
    # sim, cf. CLAUDE.md "armadilhas ja aprendidas"). Three places it shows
    # up, marked "Euler" below: the integrator a few lines down, and both
    # AlphaFilter.alpha assignments (alpha=dt/tau IS the Euler step of the
    # filter's ODE, tau*dy/dt=x-y -- see the comment there for the algebra).

    # --- reference: low-pass filtered, and its derivative comes for free
    # from the filter's own dynamics (tau*dy/dt=x-y) without differentiating
    # a noisy signal -- tells the controller the reference is rising BEFORE
    # the error shows up (this is what zeroes the startup overshoot) ---
    if car.dt > 0:
        # Euler step of tau*dy/dt=x-y: y[k+1]=y[k]+dt*(x[k]-y[k])/tau =
        # y[k]+alpha*(x[k]-y[k]), alpha=dt/tau -- exactly what AlphaFilter.filter()
        # computes (filter.py: a*x+(1-a)*y_prev = y_prev+a*(x-y_prev)).
        _vdes_filt.alpha = np.clip(car.dt / TAU_VDES, 0.0, 1.0)
    vdes_anterior = _vdes_filt.value
    vdes = _vdes_filt.filter(vdes_bruto)
    # Same ODE, dy/dt = (x-y)/tau, evaluated instead of Euler-stepped: this
    # IS the filter's derivative, reused as feedforward instead of numerically
    # differentiating vdes_bruto (which would amplify any noise in it).
    feedforward_dvdes = (vdes_bruto - vdes_anterior) / TAU_VDES if TAU_VDES > 0 else 0.0

    # --- velocity error ---
    erro = vdes - car.v

    # --- anti-windup (back-calculation): when u saturated on the previous
    # step, (u_prev - u_bruto_prev) != 0 and pulls the integrator back --
    # without this it would keep accumulating a command the actuator can't
    # deliver ---
    st = _estado_completo
    anti_windup_back_calc = (st['u_prev'] - st['u_bruto_prev']) / TT_MOD

    # --- integrator (anti-windup already added in). Euler step of
    # dI/dt = Ki*erro + anti_windup: I[k+1] = I[k] + dt*dI/dt ---
    st['I'] += car.dt * (KI_MOD * erro + anti_windup_back_calc)

    # --- P + I + reference feedforward = desired acceleration ---
    termo_proporcional = KP_MOD * erro
    termo_integral = st['I']
    a_cmd = termo_proporcional + termo_integral + feedforward_dvdes

    # --- model feedforward: adds back what v_dot = K*u - ... subtracts
    # (model inversion, IDENTIFICACAO.md) ---
    compensa_atrito = A0_MOD * np.tanh(car.v / EPS_V)     # dry friction a0*sign(v), smoothed near v=0
    compensa_arrasto = C_MOD * car.v * abs(car.v)         # quadratic drag c*v|v|
    u_bruto = (a_cmd + compensa_atrito + compensa_arrasto) / K_MOD

    # --- saturate: no active brake, u only ranges 0 to 1 ---
    u_clip = float(np.clip(u_bruto, 0.0, 1.0))

    # --- low-pass filter on the already-saturated u: smooths only the
    # command that reaches the motor, without delaying the error (v still
    # goes in raw above) ---
    if car.dt > 0:
        # Euler step, same as the vdes filter above: alpha=dt/tau.
        _u_filt.alpha = np.clip(car.dt / TAU_U, 0.0, 1.0)
    u = _u_filt.filter(u_clip)

    # next step's anti-windup compares against the u ACTUALLY applied (post-filter)
    st['u_prev'] = u
    st['u_bruto_prev'] = u_bruto
    return u


########################################
# velocity control thread
########################################
def control_func(car):
    # cruise: holds V_REF indefinitely, no position target. vdes ramps up
    # (TAXA_RAMPA_V) to V_REF. Downhill is out of scope (no active brake,
    # coast only) but needs no position guard: u=clip(u_bruto,0,1) already
    # saturates at 0 once v>vdes, so the controller coasts on its own once
    # it goes downhill.
    car.set_steer(0.0)
    vdes = min(V_REF, TAXA_RAMPA_V * car.t)
    car.set_u(controlador_longitudinal(car, vdes))


########################################
# vision thread
########################################
def vision_func(car):

    # grab image
    image = car.get_image(gray=False)

    # ultrasonic
    dist, _ = car.get_distance()
    #print(f'Ultrasonic distance: {dist:.1f}')

    return image


########################################
# main program
########################################
if __name__ == "__main__":

    plt.figure(1)
    plt.ion()

    # open communication with the car
    car = Car(parameters)

    try:
        # start the simulation
        car.start_mission()

        # main loop
        while car.t <= parameters['ts']:

            # read sensors
            car.step()

            # control function
            control_func(car)

            # vision function
            image = vision_func(car)

            ########################################
            # plot
            plt.subplot(311)
            plt.cla()
            #plt.gca().imshow(image, cmap='gray')
            plt.axis('off')
            plt.title(f'Telemetry at t={car.t:.1f}s')

            t = [traj['t'] for traj in car.traj]
            v = [traj['v'] for traj in car.traj]
            u = [traj['u'] for traj in car.traj]

            plt.subplot(312)
            plt.cla()
            plt.plot(t, v, color='g', linestyle='-', label='v')
            plt.axhline(y=V_REF, color='r', linestyle='--')
            plt.ylabel('v[m/s]')
            plt.xlabel('t[s]')
            plt.legend()

            plt.subplot(313)
            plt.cla()
            plt.plot(t, u, color='b', linestyle='-', label='u')
            plt.ylabel('u')
            plt.xlabel('t[s]')
            plt.legend()

            plt.show()
            plt.pause(0.01)

    finally:
        # save in finally: an interrupted run (crash, Ctrl+C, window closed)
        # still keeps the data collected up to that point
        try:
            if parameters['save']:
                car.save()
        except Exception as e:
            # don't let a save failure hide the original error or block the close
            print(f'\033[31mfailed to save: {e}\033[0m', flush=True)
        car.close()
