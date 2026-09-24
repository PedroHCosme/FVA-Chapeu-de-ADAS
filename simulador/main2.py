# -- coding: utf-8 --
# Disciplina: Tópicos em Engenharia de Controle e Automação IV (ENG075): 
# Fundamentos de Veículos Autônomos - 2026/1
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automação
# DELT – Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
from fva_car import Car as cp
import numpy as np
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)

# Globais
parameters = {  
                'ts'        : 20.0,             # tempo da simulacao
                'save'      : True,
                'logfile'   : 'logs/',
            }
    
########################################
# thread de controle de velocidade (PI Ajuste Fino para Rampa)
########################################
def control_func(car):
    
    # 1. Direção mantida reta
    car.set_steer(0)

    # 2. Referência de velocidade
    v_ref = 0.5

    # 3. Leitura da velocidade atual
    v_atual = car.traj[-1]['v'] if len(car.traj) > 0 else 0.0

    # 4. Ganhos Reajustados para resposta rápida e sem overshoot
    Kp = 3.8    # Resposta rápida na entrada da rampa
    Ki = 1.2    # Sem overshoot na largada e boa recuperação
    Kb = 0.5    # Anti-windup equilibrado
    u_min, u_max = 0.0, 1.0

    # 5. Inicialização dos estados
    if not hasattr(car, 'e_integral'):
        car.e_integral = 0.0
        car.last_t = car.t

    # 6. Passo de tempo dt
    dt = car.t - car.last_t
    car.last_t = car.t

    # 7. Cálculo do Erro
    e = v_ref - v_atual

    # 8. Cálculo da saída sem saturação
    u_nosat = Kp * e + Ki * car.e_integral

    # 9. Saturação do Atuador
    u_sat = np.clip(u_nosat, u_min, u_max)

    # 10. Atualização da Integral com Back-Calculation
    if dt > 0:
        car.e_integral += (e + Kb * (u_sat - u_nosat)) * dt

    # Envia o sinal ao carrinho
    car.set_u(u_sat)

########################################
# thread de visão
########################################
def vision_func(car):
    image = car.get_image(gray=False)
    dist, _ = car.get_distance()
    return image
                
########################################
# main program
########################################
if __name__ == "_main_":
    
    plt.figure(1)
    plt.ion()
    
    # cria comunicação com o carrinho
    car = cp.Car(parameters)
    
    try:
        # começa a simulação
        car.start_mission()

        # main loop
        while car.t <= parameters['ts']:
            
            # lê sensores
            car.step()
            
            # funcao de controle
            control_func(car)
            
            ########################################
            # plota resposta
            plt.subplot(212)
            plt.cla()
            t = [traj['t'] for traj in car.traj]
            v = [traj['v'] for traj in car.traj]
            
            plt.plot(t, v, label='v [medida]', color='tab:blue')
            plt.axhline(y=0.5, color='r', linestyle='--', label='v_ref')
            plt.ylabel('v [m/s]')
            plt.xlabel('t [s]')
            plt.title('Controle PI Ajustado - Rejeição Rápida de Rampa')
            plt.legend(loc='upper right')
            plt.grid(True)
            
            plt.show()
            plt.pause(0.01)

        # salva
        if parameters['save']:
            car.save()
            
    finally:
        car.close()