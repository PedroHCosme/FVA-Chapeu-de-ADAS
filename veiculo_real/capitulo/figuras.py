# -*- coding: utf-8 -*-
"""
Figuras e numeros do capitulo "Do simulador ao carro real" (controle_carro_real.tex).

Tudo o que o texto afirma sai daqui: triagem dos logs, reconstrucao do PWM,
ajuste da planta, validacoes, pericia do controlador, simulador de malha
fechada e projeto do controlador novo. Rodar com o venv do simulador, de
qualquer pasta:

    simulador/.venv/Scripts/python.exe veiculo_real/capitulo/figuras.py

Le os logs de veiculo_real/dados2_sem_feedforward/roxo, grava os PDFs em
veiculo_real/capitulo/figs/ e imprime os numeros usados nas tabelas.
"""
import sys
import importlib.util
from pathlib import Path
from collections import deque
from unittest.mock import MagicMock

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')   # console do Windows (cp1252) nao imprime tau
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
REAL = AQUI.parent
LOGS = REAL / 'dados2_sem_feedforward' / 'roxo'
FIGS = AQUI / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)

# paleta de referencia da skill dataviz (modo claro, validada) e semantica fixa:
# preto = medido no carro | azul = modelo real / projeto escolhido
# laranja = heranca do simulador | cinza = alternativa descartada | rampa = familia ordenada
AZUL, LARANJA = '#2a78d6', '#eb6834'
TINTA, TINTA2, MUDO = '#0b0b0b', '#52514e', '#898781'
GRADE, EIXO = '#e1e0d9', '#c3c2b7'
RAMPA = ['#86b6ef', '#2a78d6', '#104281']
plt.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.edgecolor': EIXO, 'axes.linewidth': 0.6, 'axes.labelcolor': TINTA2, 'axes.titlecolor': TINTA,
    'xtick.color': MUDO, 'ytick.color': MUDO, 'text.color': TINTA,
    'axes.grid': True, 'grid.color': GRADE, 'grid.linewidth': 0.5, 'grid.linestyle': '-',
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.5, 'legend.frameon': False,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
LARG = 6.3  # largura util da pagina [pol]


def salva(fig, nome):
    fig.savefig(FIGS / f'{nome}.pdf')
    plt.close(fig)


def br(x, casas=1):
    """Numero com virgula decimal, como no texto."""
    return f'{x:.{casas}f}'.replace('.', ',')


def figura(linhas, colunas, altura, largura=1.0, **kw):
    return plt.subplots(linhas, colunas, figsize=(LARG * largura, altura), layout='constrained', **kw)


def secao(titulo):
    print(f'\n=== {titulo} ===')


# ---------------------------------------------------------------------------
# 1. O que o codigo diz (car.py CAR, servos.py)
# ---------------------------------------------------------------------------
MASS, RW, GAIN_FWD = 5.16, 0.08, 0.4
K_PWM = GAIN_FWD * RW * MASS                           # rad/s de PWM por unidade de u
TH_MAX = np.deg2rad((1.5 + 8.9370) / 0.092294 - 95.0)  # limite do PWM acima do neutro
N_MA = 30                                              # car.py: v_filt = MovingAverage(n=30)
V_MAX = 1.5                                            # car.py: protecao |v| > VELMAX -> u = 0


def carrega(nome):
    return np.genfromtxt(LOGS / nome / 'car.csv', delimiter=',', names=True)


TODOS = sorted(p.parent.name for p in LOGS.glob('*/car.csv'))
BOA = ['20260924_002634', '20260924_002751', '20260924_003330']
DESC = ['20260924_003552', '20260924_003904', '20260924_004516', '20260924_004851']
CLASSE = {
    '20260924_000844': 'codigo do professor', '20260924_002158': 'manuseado',
    '20260924_002236': 'manuseado', '20260924_002318': 'manuseado',
    '20260924_002506': 'parado', '20260924_003019': 'parada ultrassom',
    '20260924_003157': 'parada ultrassom', '20260924_003238': 'parada ultrassom',
}
hora = lambda n: f'{n[9:11]}:{n[11:13]}'


# ---------------------------------------------------------------------------
# 2. Modelo da planta real: integrador do PWM -> zona morta + ganho -> lag -> MA30
# ---------------------------------------------------------------------------
def pwm(t, u):
    """Integra u como a thread actuator do servos.py. A linha k do log guarda o u
    aplicado entre t[k-1] e t[k] (save_traj roda antes do controle)."""
    th = np.zeros_like(t)
    for k in range(1, len(t)):
        th[k] = np.clip(th[k - 1] + K_PWM * u[k] * (t[k] - t[k - 1]), 0.0, TH_MAX)
    return th


def planta(t, u, G, DB, tau, d):
    """Simulacao em malha aberta: recebe so o u registrado e devolve o v que o log mostraria."""
    th = pwm(t, u)
    v = np.zeros_like(t)
    for k in range(1, len(t)):
        vss = G * max(0.0, np.rad2deg(th[k - 1]) - DB)
        v[k] = vss + (v[k - 1] - vss) * np.exp(-(t[k] - t[k - 1]) / tau)
    v_enc = np.interp(t - d, t, v, left=0.0)
    return np.convolve(np.concatenate([np.zeros(N_MA - 1), v_enc]), np.ones(N_MA) / N_MA, mode='valid')


P0, LO, HI = [0.1, 2.0, 0.3, 0.05], [0.01, 0.0, 0.01, 0.0], [0.5, 8.0, 3.0, 0.5]


def ajusta(nomes, d_fixo=None):
    """Minimos quadrados no erro de SAIDA (trajetoria inteira), um (G, DB, tau, d) para todos os nomes."""
    dados = [carrega(n) for n in nomes]
    if d_fixo is None:
        res = lambda p: np.concatenate([planta(x['t'], x['u'], *p) - x['v'] for x in dados])
        return least_squares(res, P0, bounds=(LO, HI)).x
    res = lambda p: np.concatenate([planta(x['t'], x['u'], *p, d_fixo) - x['v'] for x in dados])
    return np.append(least_squares(res, P0[:3], bounds=(LO[:3], HI[:3])).x, d_fixo)


rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
K_de = lambda G: G * np.rad2deg(1) * K_PWM
G_de = lambda K: K / (np.rad2deg(1) * K_PWM)


# ---------------------------------------------------------------------------
# 3. Lei de controle do simulador/main.py, repassada sobre os logs (pericia)
# ---------------------------------------------------------------------------
def lei_simulador(t, v, vref, I=True, ff_modelo=True, ff_ref=True, filtro_u=True):
    """Iteracao j usa v[j] e grava u na linha j+1. Na parada do ultrassom (vref volta a 0
    depois de a rampa comecar) roda set_vel(0) no lugar: o controlador fica congelado."""
    K, A0, C, EPS = 1.0, 0.096, 0.00425, 0.05
    KP, KI, TT, TV, TU = 4.0, 2.67, 0.5, 0.3, 0.10
    vf = uf = Ii = up = ubp = 0.0
    out = np.full(len(t), np.nan)
    comecou = False
    for j in range(1, len(t) - 1):
        dt, vdes = t[j] - t[j - 1], vref[j + 1]
        comecou |= vdes > 0
        if comecou and vdes == 0:
            continue
        vp = vf
        if dt > 0:
            vf += np.clip(dt / TV, 0, 1) * (vdes - vf)
        ff = (vdes - vp) / TV if ff_ref else 0.0
        e = vf - v[j]
        if I:
            Ii += dt * (KI * e + (up - ubp) / TT)
        atrito = (A0 * np.tanh(v[j] / EPS) + C * v[j] * abs(v[j])) if ff_modelo else 0.0
        ub = (KP * e + Ii + ff + atrito) / K
        uc = float(np.clip(ub, 0.0, 1.0))
        if filtro_u:
            uf = uf + np.clip(dt / TU, 0, 1) * (uc - uf) if dt > 0 else uf
        else:
            uf = uc
        up, ubp = uf, ub
        out[j + 1] = 0.0 if abs(v[j]) > V_MAX else uf
    return out


# ---------------------------------------------------------------------------
# 4. Simulador de malha fechada do carro real
# ---------------------------------------------------------------------------
DT_NOM = 0.0288
_dts = np.concatenate([np.diff(carrega(n)['t'])[1:] for n in TODOS])
PASSOS = np.random.default_rng(0).choice(_dts, size=3000)   # jitter real do loop


def malha(planta_p, controlador, v_ref=1.0, ts=20.0, queda=None, desce=1.0):
    """Mesma ordem do carro: planta anda com o u anterior -> mede (MA30) -> controla -> set_u."""
    G, DB, tau = planta_p
    n = int(ts / DT_NOM)
    t, vm, ul, vr = (np.zeros(n) for _ in range(4))
    th = v = u = 0.0
    buf = deque([0.0] * N_MA, maxlen=N_MA)
    ctrl = controlador(v_ref)
    for k in range(1, n):
        h = 1.0 if k == 1 else PASSOS[k]          # start_mission dorme ~1 s
        t[k] = t[k - 1] + h
        g = G if (queda is None or t[k] < queda[0]) else G * queda[1]
        th = np.clip(th + K_PWM * u * h, 0.0, TH_MAX)
        vss = g * max(0.0, np.rad2deg(th) - DB)
        v = vss + (v - vss) * np.exp(-h / (tau * (desce if v > vss else 1.0)))
        buf.append(v)
        vm[k], vr[k], ul[k] = np.mean(buf), v, u
        u = ctrl(t[k], h, vm[k])
        u = 0.0 if abs(vm[k]) > V_MAX else float(np.clip(u, -1.0, 1.0))
    return t, vm, ul, vr


def antigo(v_ref):
    """O controlador que rodou no carro (simulador/main.py, completo)."""
    s = {'vf': 0.0, 'uf': 0.0, 'I': 0.0, 'up': 0.0, 'ubp': 0.0, 't0': None}

    def f(t, dt, v):
        if s['t0'] is None:
            s['t0'] = t
        vdes = min(v_ref, 0.9 * (t - s['t0']))
        vp = s['vf']
        s['vf'] += np.clip(dt / 0.3, 0, 1) * (vdes - s['vf'])
        e = s['vf'] - v
        s['I'] += dt * (2.67 * e + (s['up'] - s['ubp']) / 0.5)
        ub = 4.0 * e + s['I'] + (vdes - vp) / 0.3 + 0.096 * np.tanh(v / 0.05) + 0.00425 * v * abs(v)
        s['uf'] += np.clip(dt / 0.1, 0, 1) * (float(np.clip(ub, 0, 1)) - s['uf'])
        s['up'], s['ubp'] = s['uf'], ub
        return s['uf']
    return f


def novo(KP=0.6, K_NOM=1.6, casado=True, TAXA=0.9, TV=0.3):
    """Controlador do carro real: P + FF da referencia, referencia casada com a MA30, u em +-1."""
    def factory(v_ref):
        s = {'vf': 0.0, 't0': None, 'buf': deque([0.0] * N_MA, maxlen=N_MA)}

        def f(t, dt, v):
            if s['t0'] is None:
                s['t0'] = t
            vdes = min(v_ref, TAXA * (t - s['t0']))
            vp = s['vf']
            if dt > 0:
                s['vf'] += np.clip(dt / TV, 0, 1) * (vdes - s['vf'])
            ff = (vdes - vp) / TV
            r = s['vf']
            if casado:
                s['buf'].append(r)
                r = float(np.mean(s['buf']))
            f.erro.append(r - v)
            return (KP * (r - v) + ff) / K_NOM
        f.erro = [0.0]
        factory.ultimo = f
        return f
    return factory


def metricas(t, vm, ul, vr, v_ref):
    fora = np.abs(vr - v_ref) > 0.05 * v_ref
    return {'over': 100 * (vr.max() / v_ref - 1), 'acomoda': t[np.nonzero(fora)[0][-1]] if fora.any() else 0.0,
            'erro': vr[-100:].mean() - v_ref}


def margem(KP, K_NOM, K, tau, extra=0.03, T=DT_NOM):
    w = np.logspace(-2, 1.5, 60000)
    s = 1j * w
    MA = (1 - np.exp(-s * N_MA * T)) / (N_MA * (1 - np.exp(-s * T)))
    L = (KP / K_NOM) * K * MA * np.exp(-extra * s) / (s * (tau * s + 1))
    i = np.argmax(np.abs(L) < 1)
    return w[i], 180 + np.rad2deg(np.unwrap(np.angle(L))[i])


# ===========================================================================
# RODA TUDO
# ===========================================================================
secao('1. triagem dos logs')
print('ensaio   hora   max|v|  u_min  u_max  vref_max  std(a_x)  classe')
for n in TODOS:
    x = carrega(n)
    cl = CLASSE.get(n, 'bateria boa' if n in BOA else 'descarregando')
    print(f'{n[9:]}  {hora(n)}  {np.abs(x["v"]).max():5.2f}  {x["u"].min():+5.2f}  {x["u"].max():5.2f}'
          f'  {x["vref"].max():5.2f}   {np.std(x["a_x"]):6.3f}   {cl}')
x = carrega('20260924_002506')
print(f'002506: u > 0,5 durante {np.sum(np.diff(x["t"])[x["u"][1:] > 0.5]):.1f} s com v = 0')
print(f'dt do loop: mediana {np.median(_dts)*1e3:.1f} ms, p95 {np.percentile(_dts, 95)*1e3:.1f} ms (nominal 20 ms)')
print(f'atraso de grupo da MA30: {(N_MA-1)/2*np.median(_dts):.3f} s (a 20 ms seria {(N_MA-1)/2*0.02:.3f} s)')
print(f'K_PWM = {K_PWM:.5f} rad/s por u = {np.rad2deg(K_PWM):.2f} graus/s; TH_MAX = {np.rad2deg(TH_MAX):.2f} graus')

fig, axs = figura(1, 3, 2.1, sharey=True)
for ax, n, txt in zip(axs, ['20260924_002506', '20260924_002158', '20260924_003330'],
                      ['parado', 'manuseado', 'andando']):
    x = carrega(n)
    ax.plot(x['t'], x['a_x'], color=TINTA, lw=0.7)
    ax.set_title(f'{n[9:]}: {txt}\ndesvio de a_x = {br(np.std(x["a_x"]), 2)} m/s²')
    ax.set_xlabel('t [s]')
axs[0].set_ylabel('a_x da IMU [m/s²]')
salva(fig, 'triagem_imu')

fig, ax = figura(1, 1, 1.9, 0.6)
ax.hist(_dts * 1e3, bins=np.arange(15, 62, 1), color=AZUL, rwidth=0.85)
ax.axvline(20, color=MUDO, ls='--', lw=1)
ax.axvline(np.median(_dts) * 1e3, color=TINTA, lw=1)
ax.text(20.5, ax.get_ylim()[1] * 0.9, 'nominal\n20 ms', color=TINTA2, fontsize=7, va='top')
ax.text(np.median(_dts) * 1e3 + 0.7, ax.get_ylim()[1] * 0.9, f'mediana\n{br(np.median(_dts)*1e3)} ms',
        color=TINTA2, fontsize=7, va='top')
ax.set_xlabel('dt do loop principal [ms]')
ax.set_ylabel('amostras')
salva(fig, 'dt_loop')

secao('2. reconstrucao do PWM (003330)')
x = carrega('20260924_003330')
th = pwm(x['t'], x['u'])
fig, axs = figura(3, 1, 4.0, 0.8, sharex=True)
axs[0].plot(x['t'], x['u'], color=TINTA)
axs[0].set_title('comando registrado', loc='left')
axs[0].set_ylabel('u')
axs[1].plot(x['t'], np.rad2deg(th), color=AZUL)
axs[1].axhline(np.rad2deg(TH_MAX), color=MUDO, ls='--', lw=1)
axs[1].text(0.1, np.rad2deg(TH_MAX) - 1.5, 'limite 18,1°', color=TINTA2, fontsize=7, va='top')
axs[1].set_title('PWM reconstruído (acima do neutro)', loc='left')
axs[1].set_ylabel('θ [°]')
axs[2].plot(x['t'], x['v'], color=TINTA)
axs[2].set_title('velocidade registrada', loc='left')
axs[2].set_ylabel('v [m/s]')
axs[2].set_xlabel('t [s]')
axs[2].set_xlim(0, 8)
salva(fig, 'pwm_reconstruido')
print(f'PWM congelado em {np.rad2deg(th[-1]):.1f} graus = {th[-1]/TH_MAX*100:.0f}% do limite')

secao('3. ajuste por ensaio (erro de saida)')
PAR = {}
for n in BOA + DESC:
    PAR[n] = ajusta([n])
    x = carrega(n)
    G, DB, tau, d = PAR[n]
    print(f'{n[9:]} {hora(n)}  G={G:.4f}  DB={DB:4.2f}  tau={tau:.3f}  d={d:.3f}  K={K_de(G):.2f}'
          f'  rmse={rmse(planta(x["t"], x["u"], *PAR[n]) - x["v"]):.3f}')

# ruido do encoder bruto, estimado da propria MA30: v[k]-v[k-1] = (x[k]-x[k-30])/30
for n in ['20260924_002634', '20260924_003330', '20260924_004516']:
    x = carrega(n)
    seg = x['t'] > 8.0
    print(f'{n[9:]}: ruido estimado do encoder bruto = {np.std(np.diff(x["v"][seg])) * N_MA / np.sqrt(2):.3f} m/s')

fig, axs = figura(2, 3, 3.8, sharex=True, sharey=True)
for ax, n in zip(axs.flat, ['20260924_002634', '20260924_002751', '20260924_003330',
                            '20260924_003904', '20260924_004516', '20260924_004851']):
    x = carrega(n)
    vp = planta(x['t'], x['u'], *PAR[n])
    ax.plot(x['t'], x['v'], color=TINTA, lw=2.0, label='medido')
    ax.plot(x['t'], vp, color=AZUL, lw=1.1, ls='--', label='modelo')
    ax.set_title(f'{n[9:]} ({hora(n)})\nK = {br(K_de(PAR[n][0]), 2)}, rmse = {br(rmse(vp - x["v"]), 3)} m/s')
for ax in axs[1]:
    ax.set_xlabel('t [s]')
for ax in axs[:, 0]:
    ax.set_ylabel('v [m/s]')
axs[0, 0].legend(loc='lower right')
salva(fig, 'ajustes')

secao('4. u = 0 segura a velocidade? (inclinacao depois de t = 6 s)')
for n in BOA + DESC[1:]:
    x = carrega(n)
    seg = (x['t'] > 6.0) & (np.abs(x['u']) < 1e-9)
    vm = x['v'][seg].mean()
    print(f'{n[9:]}  v~{vm:.2f}  medido dv/dt={np.polyfit(x["t"][seg], x["v"][seg], 1)[0]:+.4f}'
          f'  modelo do simulador: {-(0.096 + 0.00425*vm*vm):+.3f} m/s^2')

x = carrega('20260924_003330')
v_sim = np.zeros_like(x['t'])
for k in range(1, len(x['t'])):
    h, vk = x['t'][k] - x['t'][k - 1], v_sim[k - 1]
    v_sim[k] = max(0.0, vk + h * (x['u'][k] - 0.096 * np.sign(vk) - 0.00425 * vk * abs(vk)))
fig, axs = figura(2, 1, 3.7, 0.85, sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
axs[0].plot(x['t'], x['v'], color=TINTA, lw=2.2, label='carro real (medido)')
axs[0].plot(x['t'], planta(x['t'], x['u'], *PAR['20260924_003330']), color=AZUL, ls='--', lw=1.2,
            label='modelo real identificado')
axs[0].plot(x['t'], v_sim, color=LARANJA, label='modelo do simulador, mesmo u')
axs[0].plot(x['t'], x['vref'], color=MUDO, ls=':', lw=1.2, label='vref')
axs[0].set_ylabel('v [m/s]')
axs[0].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
axs[1].plot(x['t'], x['u'], color=TINTA)
axs[1].set_ylabel('u')
axs[1].set_xlabel('t [s]')
salva(fig, 'mesmo_u')

secao('5. PWM descendo: 003552 (u < 0 vem da parada do ultrassom)')
x = carrega('20260924_003552')
th = pwm(x['t'], x['u'])
print(f'003552: ajuste proprio G={PAR["20260924_003552"][0]:.4f} DB={PAR["20260924_003552"][1]:.2f} '
      f'tau={PAR["20260924_003552"][2]:.3f} d={PAR["20260924_003552"][3]:.3f}; u_min = {x["u"].min():.2f}; '
      f'PWM minimo depois do pico = {np.rad2deg(th[x["t"] > 14].min()):.1f} graus')
def planta_assim(t, u, G, DB, tau_sobe, tau_desce, d):
    """Como planta(), mas com constante de tempo diferente quando o carro precisa desacelerar."""
    th = pwm(t, u)
    v = np.zeros_like(t)
    for k in range(1, len(t)):
        vss = G * max(0.0, np.rad2deg(th[k - 1]) - DB)
        tau = tau_sobe if vss >= v[k - 1] else tau_desce
        v[k] = vss + (v[k - 1] - vss) * np.exp(-(t[k] - t[k - 1]) / tau)
    v_enc = np.interp(t - d, t, v, left=0.0)
    return np.convolve(np.concatenate([np.zeros(N_MA - 1), v_enc]), np.ones(N_MA) / N_MA, mode='valid')


q = least_squares(lambda p: planta_assim(x['t'], x['u'], *p) - x['v'], [0.1, 2.0, 0.4, 2.0, 0.02],
                  bounds=([0.01, 0, 0.01, 0.01, 0], [0.5, 8, 3, 30, 0.5])).x
print(f'003552 com tau separado: tau_sobe={q[2]:.3f} s, tau_desce={q[3]:.3f} s, '
      f'rmse={rmse(planta_assim(x["t"], x["u"], *q) - x["v"]):.3f} m/s')

fig, axs = figura(3, 1, 4.3, 0.8, sharex=True)
axs[0].plot(x['t'], x['v'], color=TINTA, lw=2.0, label='medido')
axs[0].plot(x['t'], planta(x['t'], x['u'], *PAR['20260924_003552']), color=AZUL, ls='--', lw=1.2, label='modelo')
axs[0].set_title('velocidade', loc='left')
axs[0].set_ylabel('v [m/s]')
axs[0].legend(loc='upper left')
axs[1].plot(x['t'], x['u'], color=TINTA)
axs[1].fill_between(x['t'], x['u'], 0, where=x['u'] < 0, color=AZUL, alpha=0.25, lw=0)
axs[1].text(16.9, -0.45, 'u < 0', color=TINTA2, fontsize=7.5)
axs[1].axhline(0, color=EIXO, lw=0.8)
axs[1].set_title('comando registrado (u < 0 vem da parada do ultrassom)', loc='left')
axs[1].set_ylabel('u')
axs[2].plot(x['t'], np.rad2deg(th), color=AZUL)
axs[2].set_title('PWM reconstruído', loc='left')
axs[2].set_ylabel('θ [°]')
axs[2].set_xlabel('t [s]')
axs[2].set_xlim(10, 20)
salva(fig, 'pwm_desce')

secao('6. deriva da bateria e perfil de custo tau x d')
nomes = BOA + DESC
minutos = [int(n[9:11]) * 60 + int(n[11:13]) for n in nomes]
fig, axs = figura(2, 1, 3.0, 0.6, sharex=True)
axs[0].plot(minutos, [K_de(PAR[n][0]) for n in nomes], 'o-', color=TINTA, ms=4, lw=1)
axs[0].set_ylabel('K [m/s² por u]')
axs[1].plot(minutos, [PAR[n][2] for n in nomes], 'o-', color=TINTA, ms=4, lw=1)
axs[1].set_ylabel('τ [s]')
axs[1].set_xticks(minutos[::2])
axs[1].set_xticklabels([hora(n) for n in nomes][::2])
axs[1].set_xlabel('hora do ensaio (24/09)')
axs[0].text(minutos[1], K_de(PAR[nomes[1]][0]) - 0.25, 'bateria boa', color=TINTA2, fontsize=7)
axs[0].text(minutos[4] - 3, K_de(PAR[nomes[4]][0]) + 0.12, 'descarregando', color=TINTA2, fontsize=7)
salva(fig, 'bateria')

ds = [0.0, 0.1, 0.2, 0.3, 0.4]
perfil = [ajusta(BOA, d_fixo=d) for d in ds]
custo = [rmse(np.concatenate([planta(carrega(n)['t'], carrega(n)['u'], *p) - carrega(n)['v'] for n in BOA]))
         for p in perfil]
for d, p, c in zip(ds, perfil, custo):
    print(f'd fixo = {d:.1f} s -> G={p[0]:.4f} DB={p[1]:.2f} tau={p[2]:.3f}  rmse conjunto = {c:.4f}')
fig, axs = figura(1, 2, 1.9, 0.8)
axs[0].plot(ds, custo, 'o-', color=AZUL, ms=4)
axs[0].set_xlabel('atraso puro d fixado [s]')
axs[0].set_ylabel('rmse do ajuste [m/s]')
axs[0].set_ylim(0, max(custo) * 1.4)
axs[1].plot(ds, [p[2] for p in perfil], 'o-', color=AZUL, ms=4)
axs[1].set_xlabel('atraso puro d fixado [s]')
axs[1].set_ylabel('τ ajustado [s]')
salva(fig, 'perfil_tau_d')

secao('7. validacao cega (deixa um de fora), por bateria')
for grupo, nomes in [('bateria boa', BOA), ('descarregando', DESC)]:
    for fora in nomes:
        p = ajusta([n for n in nomes if n != fora])
        x = carrega(fora)
        print(f'{grupo:14s} preve {fora[9:]}: rmse = {rmse(planta(x["t"], x["u"], *p) - x["v"]):.3f} m/s')
p = ajusta(BOA + DESC[1:])
x = carrega('20260924_003552')
print(f'misturando as baterias e prevendo 003552: rmse = {rmse(planta(x["t"], x["u"], *p) - x["v"]):.3f} m/s')

secao('8. pericia: qual lei de controle rodou em cada ensaio')
for n in ['20260924_002158', '20260924_002318', '20260924_002506'] + BOA + DESC:
    x = carrega(n)
    ok = ~np.isnan(lei_simulador(x['t'], x['v'], x['vref']))
    e1 = rmse((lei_simulador(x['t'], x['v'], x['vref']) - x['u'])[ok])
    e2 = rmse((lei_simulador(x['t'], x['v'], x['vref'], ff_ref=False) - x['u'])[ok])
    print(f'{n[9:]}  completo: {e1:.4f}   sem FF da referencia: {e2:.4f}')
x = carrega('20260924_004851')
fig, ax = figura(1, 1, 2.4, 0.75)
ax.plot(x['t'], x['u'], color=TINTA, lw=2.4, label='u registrado')
ax.plot(x['t'], lei_simulador(x['t'], x['v'], x['vref']), color=MUDO, lw=1.2, label='lei completa (errada)')
ax.plot(x['t'], lei_simulador(x['t'], x['v'], x['vref'], ff_ref=False), color=AZUL, ls='--', lw=1.2,
        label='lei sem FF da referência')
ax.set_xlim(0.8, 4.0)
ax.set_xlabel('t [s]')
ax.set_ylabel('u')
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
salva(fig, 'pericia')

secao('9. simulador de malha fechada: controlador antigo contra os logs')
VALID = [('20260924_003330', 1.0), ('20260924_003904', 1.0), ('20260924_004516', 0.5)]
fig, axs = figura(1, 3, 2.0, sharey=True)
for ax, (n, vr) in zip(axs, VALID):
    x = carrega(n)
    G, DB, tau, _ = PAR[n]
    t, vm, ul, _ = malha((G, DB, tau), antigo, v_ref=vr)
    sat_sim = np.sum(np.diff(t)[ul[1:] > 0.99])
    sat_log = np.sum(np.diff(x['t'])[x['u'][1:] > 0.99])
    print(f'{n[9:]}: v final sim {vm[-100:].mean():.2f} x log {x["v"][-100:].mean():.2f} m/s | '
          f'u=1 sim {sat_sim:.2f} s x log {sat_log:.2f} s')
    ax.plot(x['t'], x['v'], color=TINTA, lw=2.2, label='carro real')
    ax.plot(t, vm, color=AZUL, lw=1.2, ls='--', label='simulador')
    ax.set_title(f'{n[9:]} (vref = {br(vr)} m/s)')
    ax.set_xlabel('t [s]')
    ax.set_xlim(0, 20)
axs[0].set_ylabel('v medido [m/s]')
axs[0].legend(loc='lower right')
salva(fig, 'validacao_malha')

secao('10. projeto: fase da MA30, margens, respostas')
w = np.logspace(-1, np.log10(6), 400)   # abaixo do 1o zero da media: 2*pi/(N*T) = 7,3 rad/s
s = 1j * w
MA = (1 - np.exp(-s * N_MA * DT_NOM)) / (N_MA * (1 - np.exp(-s * DT_NOM)))
for wx in (1, 2, 4):
    i = np.argmin(np.abs(w - wx))
    print(f'fase a {wx} rad/s: MA30 {np.rad2deg(np.angle(MA[i])):+.0f} graus, atraso de 0,05 s {np.rad2deg(-0.05*wx):+.0f} graus')
fig, ax = figura(1, 1, 2.1, 0.6)
ax.semilogx(w, np.rad2deg(np.unwrap(np.angle(MA))), color=AZUL, label='média de 30 amostras (carro)')
ax.semilogx(w, np.rad2deg(-0.05 * w), color=LARANJA, label='atraso de 0,05 s (simulador)')
ax.set_xlabel('ω [rad/s]')
ax.set_ylabel('fase [°]')
ax.set_ylim(-160, 5)
ax.legend(loc='lower left')
salva(fig, 'fase_ma')

NOM = (G_de(1.6), 3.42, 0.251)
CASOS = {'nominal (K 1,6, τ 0,25)': (1.6, 0.251), 'K 2,0': (2.0, 0.251), 'K 1,3': (1.3, 0.251),
         'τ 0,40': (1.6, 0.40), 'K 2,0 e τ 0,40': (2.0, 0.40)}
print('margem de fase [graus]  KP | ' + ' | '.join(CASOS))
for KP in (0.4, 0.6, 0.8, 1.0, 1.2):
    print(f'  {KP:.1f} | ' + ' | '.join(f'{margem(KP, 1.6, K, tau)[1]:5.1f}' for K, tau in CASOS.values()))

# os ganhos do simulador na planta real: PI (Kp=4, Ki=2,67) + filtro de 0,1 s em u, sem dividir por K
wv = np.logspace(-2, 1.5, 60000)
sv = 1j * wv
MAv = (1 - np.exp(-sv * N_MA * DT_NOM)) / (N_MA * (1 - np.exp(-sv * DT_NOM)))
for K, tau in [(1.6, 0.251), (1.06, 0.749)]:
    L = (4.0 + 2.67 / sv) / (0.1 * sv + 1) * K * MAv * np.exp(-0.03 * sv) / (sv * (tau * sv + 1))
    i = np.argmax(np.abs(L) < 1)
    print(f'ganhos do simulador na planta real (K={K}, tau={tau}): cruzamento {wv[i]:.2f} rad/s, '
          f'margem de fase {180 + np.rad2deg(np.unwrap(np.angle(L))[i]):+.0f} graus')

wl = np.logspace(-2, np.log10(6), 800)
sl = 1j * wl
MAl = (1 - np.exp(-sl * N_MA * DT_NOM)) / (N_MA * (1 - np.exp(-sl * DT_NOM)))
L_sem = 0.6 / (sl * (0.251 * sl + 1))
L_com = L_sem * MAl * np.exp(-0.03 * sl)
wc, pm = margem(0.6, 1.6, 1.6, 0.251)
fig, axs = figura(2, 1, 3.4, 0.75, sharex=True)
axs[0].semilogx(wl, 20 * np.log10(np.abs(L_sem)), color=MUDO, label='sem a média móvel')
axs[0].semilogx(wl, 20 * np.log10(np.abs(L_com)), color=AZUL, label='com a média móvel')
axs[0].axhline(0, color=TINTA2, lw=0.8)
axs[0].set_ylabel('|L| [dB]')
axs[0].set_ylim(-40, 40)
axs[0].legend(loc='upper right')
axs[1].semilogx(wl, np.rad2deg(np.unwrap(np.angle(L_sem))), color=MUDO)
axs[1].semilogx(wl, np.rad2deg(np.unwrap(np.angle(L_com))), color=AZUL)
axs[1].axhline(-180, color=TINTA2, lw=0.8)
axs[1].axvline(wc, color=MUDO, ls='--', lw=0.8)
axs[1].annotate('', xy=(wc, -180), xytext=(wc, -180 + pm), arrowprops=dict(arrowstyle='<->', color=TINTA, lw=0.8))
axs[1].text(wc * 1.12, -180 + pm / 2, f'margem\n{pm:.0f}°', fontsize=7, va='center')
axs[1].set_ylabel('fase de L [°]')
axs[1].set_ylim(-320, -60)
axs[1].set_xlabel('ω [rad/s]')
salva(fig, 'bode_L')

kps = np.linspace(0.3, 1.5, 25)
fig, ax = figura(1, 1, 2.3, 0.6)
for (rot, (K, tau)), cor in zip([('K 1,3', (1.3, 0.251)), ('nominal', (1.6, 0.251)), ('K 2,0 e τ 0,40', (2.0, 0.40))], RAMPA):
    pms = [margem(kp, 1.6, K, tau)[1] for kp in kps]
    ax.plot(kps, pms, color=cor)
    ax.text(kps[-1] + 0.02, pms[-1], rot, color=TINTA2, fontsize=7, va='center')
ax.axhline(55, color=MUDO, ls='--', lw=1)
ax.axvline(0.6, color=TINTA, lw=0.8)
ax.text(0.62, 22, 'KP = 0,6', fontsize=7)
ax.text(1.22, 56.5, 'critério 55°', color=TINTA2, fontsize=7)
ax.set_xlim(0.3, 1.85)
ax.set_xlabel('KP [rad/s]')
ax.set_ylabel('margem de fase [°]')
salva(fig, 'margem_kp')

print('resposta no tempo (pior entre vref 1,0 e 0,5; descida 2x mais lenta; queda de 20% no ganho em t=10 s)')
for KP in (0.6, 0.8, 1.0):
    for casado in (True, False):
        for rot, (K, tau) in CASOS.items():
            pl = (G_de(K), 3.42, tau)
            m = [metricas(*malha(pl, novo(KP, 1.6, casado), v_ref=vr, desce=2.0), vr) for vr in (1.0, 0.5)]
            t, vm, ul, vrr = malha(pl, novo(KP, 1.6, casado), v_ref=1.0, ts=25.0, queda=(10.0, 0.8), desce=2.0)
            fora = (t >= 10) & (np.abs(vrr - 1) > 0.05)
            volta = (t[np.nonzero(fora)[0][-1]] - 10) if fora.any() else 0.0
            print(f'  KP={KP} casada={casado!s:5s} {rot:24s} overshoot={max(q["over"] for q in m):5.1f}% '
                  f'acomoda={max(q["acomoda"] for q in m):4.1f} s  erro={max(abs(q["erro"]) for q in m):.4f}  volta={volta:.1f} s')

print('controlador escolhido nas plantas da bateria que descarregava (ajustes proprios, descida 2x mais lenta)')
for n in ['20260924_003904', '20260924_004516']:
    for vr in (1.0, 0.5):
        m = metricas(*malha(tuple(PAR[n][:3]), novo(), v_ref=vr, desce=2.0), vr)
        print(f'  {n[9:]} (K={K_de(PAR[n][0]):.2f}) vref={vr}: overshoot={max(0, m["over"]):.1f}%  acomoda={m["acomoda"]:.1f} s')

def com_vies(b, **kw):
    """Controlador novo + um vies constante em u (como a antiga compensacao de atrito)."""
    base = novo(**kw)

    def factory(v_ref):
        f = base(v_ref)
        return lambda t, dt, v: f(t, dt, v) + b
    return factory


t, vm, ul, vr = malha(NOM, com_vies(0.1), v_ref=1.0, ts=25.0)
print(f'vies b = 0,1 somado em u: erro de regime simulado = {1.0 - vr[-100:].mean():+.3f} m/s '
      f'(formula -b*K_NOM/K_P = {-0.1 * 1.6 / 0.6:+.3f})')

fig, axs = figura(2, 1, 3.3, 0.8, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
for KP, cor in zip((0.6, 0.8, 1.0), RAMPA):
    t, vm, ul, vr = malha(NOM, novo(KP), v_ref=1.0, ts=12.0, desce=2.0)
    axs[0].plot(t, vr, color=cor, label=f'KP = {br(KP)}: sobressinal {br(100*(vr.max()-1))}%')
    axs[1].plot(t, ul, color=cor)
axs[0].axhline(1.0, color=MUDO, ls='--', lw=1)
axs[0].set_ylabel('v real [m/s]')
axs[0].legend(loc='lower right')
axs[1].axhline(0, color=EIXO, lw=0.8)
axs[1].set_ylabel('u')
axs[1].set_xlabel('t [s]')
salva(fig, 'resposta_kp')

fig, axs = figura(2, 1, 3.4, 0.8, sharex=True)
for casado, cor, rot in [(False, MUDO, 'erro = vdes − v medido'), (True, AZUL, 'erro = MA30(vdes) − v medido')]:
    fab = novo(0.6, 1.6, casado)
    t, vm, ul, vr = malha(NOM, fab, v_ref=1.0, ts=12.0, desce=2.0)
    erro = np.array(fab.ultimo.erro)
    print(f'referencia casada={casado}: sobressinal {100*(vr.max()-1):.1f}%, erro visto pelo P: '
          f'pico {erro.max():.2f} m/s, minimo {erro.min():+.3f} m/s')
    axs[0].plot(t, vr, color=cor, label=f'{rot}: sobressinal {br(100*(vr.max()-1))}%')
    axs[1].plot(t[:len(erro)], erro, color=cor)
axs[0].axhline(1.0, color=MUDO, ls='--', lw=1)
axs[0].set_ylabel('v real [m/s]')
axs[0].legend(loc='lower right')
axs[1].axhline(0, color=EIXO, lw=0.8)
axs[1].set_ylabel('erro que o P vê [m/s]')
axs[1].set_xlabel('t [s]')
salva(fig, 'referencia_casada')

fig, ax = figura(1, 1, 2.3, 0.7)
for K, cor in zip((1.3, 1.6, 2.0), RAMPA):
    t, vm, ul, vr = malha((G_de(K), 3.42, 0.251), novo(), v_ref=1.0, ts=12.0, desce=2.0)
    ax.plot(t, vr, color=cor, label=f'K = {br(K)}: sobressinal {br(max(0, 100*(vr.max()-1)))}%')
ax.axhline(1.0, color=MUDO, ls='--', lw=1)
ax.set_xlabel('t [s]')
ax.set_ylabel('v real [m/s]')
ax.legend(loc='lower right')
salva(fig, 'robustez')

t0, vm0, ul0, vr0 = malha(NOM, antigo, v_ref=1.0, ts=15.0, desce=2.0)
t1, vm1, ul1, vr1 = malha(NOM, novo(), v_ref=1.0, ts=15.0, desce=2.0)
fig, axs = figura(2, 1, 3.7, 0.85, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
axs[0].plot(t0, vr0, color=LARANJA, label='antigo (simulador/main.py)')
axs[0].plot(t1, vr1, color=AZUL, label='novo (P + FF da referência)')
axs[0].plot(t1, vm1, color=AZUL, lw=0.9, ls='--', label='novo: v medido (MA30)')
axs[0].axhline(1.0, color=MUDO, ls=':', lw=1.2)
axs[0].set_ylabel('v [m/s]')
axs[0].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
axs[1].plot(t0, ul0, color=LARANJA)
axs[1].plot(t1, ul1, color=AZUL)
axs[1].axhline(0, color=EIXO, lw=0.8)
axs[1].set_ylabel('u')
axs[1].set_xlabel('t [s]')
salva(fig, 'antes_depois')


secao('11. o controlador gravado em veiculo_real/main.py e o do projeto')
def _carrega_main():
    sys.path.insert(0, str(REAL))
    try:
        while True:
            for m in [m for m in sys.modules if m == 'fva_car' or m.startswith('fva_car.')]:
                del sys.modules[m]
            try:
                spec = importlib.util.spec_from_file_location('main_real', REAL / 'main.py')
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
            except ModuleNotFoundError as e:
                if e.name.split('.')[0] == 'fva_car':
                    raise
                sys.modules[e.name] = MagicMock()   # bibliotecas de hardware ausentes no PC
    finally:
        sys.path.pop(0)


mod = _carrega_main()


class CarroFalso:
    v_filt = type('F', (), {'n': N_MA})()
    t = dt = v = vref = u = 0.0

    def set_u(self, u):
        self.u = float(np.clip(u, -1.0, 1.0))


def do_main(v_ref):
    mod.V_REF = v_ref
    mod._estado.clear()
    mod._vdes_filt.reset(0.0)
    car = CarroFalso()

    def f(t, dt, v):
        car.t, car.dt, car.v = t, dt, v
        mod.control_func(car)
        return car.u
    return f


for vr in (1.0, 0.5):
    a = malha(NOM, do_main, v_ref=vr)
    b = malha(NOM, novo(mod.KP_MOD, mod.K_NOM), v_ref=vr)
    m = metricas(*a, vr)
    print(f'main.py (KP={mod.KP_MOD}, K_NOM={mod.K_NOM}) vref={vr}: max|u_main - u_projeto| = '
          f'{np.max(np.abs(a[2] - b[2])):.1e}; overshoot={m["over"]:.1f}%  acomoda={m["acomoda"]:.1f} s  erro={m["erro"]:+.4f}')
print('\nfiguras em', FIGS)
