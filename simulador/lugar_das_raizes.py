# -*- coding: utf-8 -*-
# Lugar das raizes da malha de velocidade do controlador longitudinal.
#
# Depois que o feedforward (K,c,a0) cancela a nao-linearidade conhecida, a
# planta compensada e' um integrador puro: v_ponto = a_cmd  ->  P(s) = 1/s.
# O controlador e' um PI: C(s) = Kp*(1 + 1/(Ti*s)), com Ti=Kp/Ki fixo em 1,5s
# (a razao que o usuario escolheu ao dobrar Kp e Ki juntos).
# O atraso de 1 amostra (Td=0,05s, IDENTIFICACAO.md) entra como e^{-s*Td},
# aproximado por Pade de 1a ordem pra virar polinomio (raiz nao lida com
# atraso puro): e^{-s*Td} ~= (1 - s*Td/2)/(1 + s*Td/2).
#
# Malha aberta:  L(s) = C(s)*P(s)*Delay(s) = Kp*(Ti*s+1)/(Ti*s^2) * (1-sTd/2)/(1+sTd/2)
# Malha fechada: 1 + L(s) = 0  ->  polinomio caracteristico cubico em s (por causa
# do polo extra que a aproximacao de Pade da ao atraso).
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Td = 0.05      # atraso de 1 amostra [s], IDENTIFICACAO.md
TI_TA = 1.5    # Ti = Kp/Ki, razao fixada pelo usuario (era 3/2, virou 6/4, mesma 1,5s)


def raizes_malha_fechada(Kp, Ti=TI_TA, Td=Td):
    # (Ti*Td/2)*s^3 + [Ti - Kp*(Ti*Td/2)]*s^2 + Kp*(Ti - Td/2)*s + Kp = 0
    a3 = Ti * Td / 2.0
    a2 = Ti - Kp * (Ti * Td / 2.0)
    a1 = Kp * (Ti - Td / 2.0)
    a0 = Kp
    return np.roots([a3, a2, a1, a0])


# --- varre Kp e acha onde alguma raiz cruza pro semiplano direito ---
Kps = np.linspace(0.2, 200.0, 20000)
kp_critico = None
for Kp in Kps:
    r = raizes_malha_fechada(Kp)
    if np.any(r.real > 0):
        kp_critico = Kp
        break

if kp_critico:
    print(f'Kp critico (instabilidade) ~= {kp_critico:.2f} rad/s')
else:
    print('SEM cruzamento pro semiplano direito ate Kp=200 -- o teto de banda')
    print('nao vem de instabilidade da malha linearizada, ver nota no script.')

# --- pontos de interesse: Kp=3 (validado, log 20260922_230516), Kp=6 (chatter) ---
for Kp in [3.0, 4.0, 6.0]:
    r = raizes_malha_fechada(Kp)
    print(f'Kp={Kp:.1f}: polos = {np.round(r, 3)}')

# --- lugar das raizes ---
Kps_plot = np.geomspace(0.2, 20.0, 800)
todas = np.array([raizes_malha_fechada(k) for k in Kps_plot])  # shape (N,3)

plt.figure(figsize=(7, 6))
for i in range(3):
    plt.plot(todas[:, i].real, todas[:, i].imag, '.', markersize=2, color='C0')

for Kp, cor, rotulo in [(3.0, 'green', 'Kp=3 (validado)'), (6.0, 'red', 'Kp=6 (chatter)')]:
    r = raizes_malha_fechada(Kp)
    plt.plot(r.real, r.imag, 'o', color=cor, label=rotulo, markersize=8)

plt.axvline(0, color='k', linewidth=0.8)
plt.axhline(0, color='k', linewidth=0.8)
plt.xlabel('Re(s)')
plt.ylabel('Im(s)')
titulo_kp = f'Kp critico ~= {kp_critico:.2f} rad/s' if kp_critico else 'sem instabilidade ate Kp=200'
plt.title(f'Lugar das raizes (Ti={TI_TA}s, Td={Td}s, Pade 1a ordem)\n{titulo_kp}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figuras/lugar_das_raizes.png', dpi=130)
print('salvo em figuras/lugar_das_raizes.png')
