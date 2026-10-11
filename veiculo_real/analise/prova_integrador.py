# -*- coding: utf-8 -*-
"""Prova de que o servos.py INTEGRA o u no acelerador (PWM do ESC).

    python analise/prova_integrador.py          (a partir de veiculo_real/, com o venv do projeto)

Parte A (sem carro): roda o car.py e o servos.py da disciplina SEM MODIFICACAO, com o hardware
trocado por um ServoKit falso que anota cada angulo escrito no canal do ESC. Manda u pelo
Car.set_u de verdade e confere: u constante -> acelerador em rampa; u = 0 -> acelerador parado;
limites 0 e 18,08 graus; |v| > 1,5 -> u trocado por 0; escrita direta no canal e sobrescrita.
Parte B (logs do carro, arvore ensaios/): o mesmo u = 0 segura 1,69 m/s ou deixa o carro parar,
conforme a historia de u; em cruzeiro u medio = 0; e qual sinal explica v: u ou a integral de u.
Saida: numeros na tela, analise/saida/prova_integrador.png e .json (dados dos graficos).
"""
import json, sys, threading, time, types
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from scipy.optimize import least_squares

REAL = Path(__file__).resolve().parents[1]
RAIZ = REAL.parent
SAIDA = REAL / 'analise' / 'saida'
sys.path.insert(0, str(REAL / 'analise'))
from analisa import carrega, preve, fit_pct, K_PWM, TH_MAX   # noqa: E402


# ---------------- Parte A: o codigo da disciplina rodando no PC ---------------------------
ESCRITAS = []                       # (t, canal, angulo em graus) de cada escrita no PCA9685
_T0 = time.monotonic()


class _Canal:
    def __init__(self, i):
        self.i, self._a = i, None

    @property
    def angle(self):
        return self._a

    @angle.setter
    def angle(self, a):
        self._a = float(a)
        ESCRITAS.append((time.monotonic() - _T0, self.i, float(a)))


class ServoKitFalso:
    def __init__(self, channels=16):
        self.servo = [_Canal(i) for i in range(channels)]


def carrega_fva_car():
    """fva_car da disciplina, intacto; so as bibliotecas de hardware viram falsas."""
    kit = types.ModuleType('adafruit_servokit')
    kit.ServoKit = ServoKitFalso
    sys.modules['adafruit_servokit'] = kit
    for m in ('serial', 'serial.tools', 'serial.tools.list_ports', 'smbus2'):
        sys.modules[m] = MagicMock()
    sys.path.insert(0, str(REAL))
    from fva_car import car, servos
    return car, servos


# (duracao [s], u pedido, v que o carro "le") -- v so importa para a protecao de VELMAX
ROTEIRO_A = [(1.0, 0.0, 0.0), (2.0, 0.5, 0.0), (2.0, 0.0, 0.0), (1.5, -1.0, 0.0), (1.0, 0.0, 0.0),
             (2.5, 1.0, 0.0), (1.5, 0.0, 0.0), (1.5, -1.0, 1.6), (1.0, -1.0, 0.0)]


def parte_a():
    car, servos = carrega_fva_car()
    s = servos.Servos()                                   # o construtor ja dispara a thread actuator
    carro = types.SimpleNamespace(emergencia=False, v=0.0, u=0.0, atuador=s)   # o que Car.set_u usa
    amostras = []                                         # (t, u pedido, u aplicado, v)
    fases = []
    for dur, u, v in ROTEIRO_A:
        t_ini = time.monotonic() - _T0
        while time.monotonic() - _T0 < t_ini + dur:
            carro.v = v
            car.Car.set_u(carro, u)                       # o metodo de verdade do car.py
            amostras.append((time.monotonic() - _T0, u, float(carro.u), v))
            time.sleep(0.0288)                            # periodo do loop do main.py no carro
        fases.append((t_ini, t_ini + dur, u, v))

    # escrita direta no canal do ESC (tentativa de "nao chamar" o integrador)
    th_antes = np.rad2deg(s.th_pwm)
    t_escr = time.monotonic() - _T0
    s.kit.servo[servos.SERVO_THROTTLE].angle = 130.0
    time.sleep(0.1)
    depois = [a for (tt, c, a) in ESCRITAS if c == servos.SERVO_THROTTLE and tt > t_escr]
    s.close()

    zero = np.rad2deg(servos.ZERO_THROTTLE_ANGLE)
    e = np.array([(tt, a - zero) for (tt, c, a) in ESCRITAS if c == servos.SERVO_THROTTLE and tt < t_escr])
    am = np.array(amostras)
    dts = np.diff(e[:, 0])
    th_de = lambda tt: np.interp(tt, e[:, 0], e[:, 1])

    r = dict(periodo_thread_ms=1e3 * float(np.median(dts)), fases=[])
    print('PARTE A - car.py + servos.py da disciplina, hardware falso')
    print(f'  thread actuator escreve no canal do ESC a cada {r["periodo_thread_ms"]:.1f} ms (dt nominal 30 ms)')
    for (a, b, u, v) in fases:
        th_a, th_b = th_de(a + 0.1), th_de(b - 0.05)
        taxa = (th_b - th_a) / (b - 0.05 - a - 0.1)
        u_apl = am[(am[:, 0] > a + 0.05) & (am[:, 0] < b), 2]
        r['fases'].append(dict(t=[a, b], u=u, v=v, u_aplicado=float(np.median(u_apl)), th_ini=float(th_de(a)),
                               th_fim=float(th_b), taxa_graus_s=float(taxa)))
        print(f'  {a:5.1f}-{b:5.1f} s  u={u:+.1f} v={v:.1f} -> u aplicado {np.median(u_apl):+.1f}: acelerador '
              f'{th_de(a):5.2f} -> {th_b:5.2f} graus ({taxa:+5.2f} graus/s)')
    # passo de integracao por ciclo (independe do relogio do PC): K_PWM * u * 0,03 s
    f05 = [f for f in r['fases'] if f['u'] == 0.5][0]
    m = (e[:, 0] > f05['t'][0] + 0.2) & (e[:, 0] < f05['t'][1] - 0.1)
    passo = float(np.median(np.diff(e[m, 1])))
    r['passo_por_ciclo_u05'] = passo
    print(f'  passo por ciclo com u=0,5: {passo:.4f} graus (esperado K_PWM*0,5*0,03 = {K_PWM * 0.5 * 0.03:.4f})')
    print(f'  escrita direta de 130 graus no canal do ESC: proxima escrita da thread = {depois[1] - zero:.2f} graus '
          f'acima do neutro (acelerador integrado era {th_antes:.2f}) -> sobrescrita em ~{r["periodo_thread_ms"]:.0f} ms')
    r['escrita_direta'] = dict(escrito=130.0 - zero, sobrescrito_por=depois[1] - zero, th_integrado=th_antes)

    # o que tem que valer (falha alto se o servos.py mudar)
    taxa = {(f['u'], f['v']): f['taxa_graus_s'] for f in r['fases']}
    th_ini = {(f['u'], f['v']): f['th_ini'] for f in r['fases']}
    assert abs(passo - K_PWM * 0.5 * 0.03) < 1e-3, 'u nao vira taxa do acelerador'
    assert taxa[(0.5, 0.0)] > 1.0, 'u constante deveria fazer o acelerador subir'
    assert all(abs(f['taxa_graus_s']) < 0.05 for f in r['fases'] if f['u'] == 0.0 and f['th_ini'] > 1.0), \
        'u = 0 deveria SEGURAR o acelerador'
    assert min(e[:, 1]) >= -1e-9 and max(e[:, 1]) <= TH_MAX + 1e-6, 'limites 0 / 18,08 graus'
    assert abs(taxa[(-1.0, 1.6)]) < 0.05 and th_ini[(-1.0, 1.6)] > 1.0, '|v| > 1,5 deveria travar o acelerador'
    assert abs(depois[1] - zero - th_antes) < 0.5, 'escrita direta deveria ser sobrescrita'
    print('  todas as verificacoes passaram')
    r['serie'] = dict(t=e[:, 0].round(3).tolist(), th=e[:, 1].round(3).tolist(),
                      tu=am[:, 0].round(3).tolist(), u=am[:, 2].round(3).tolist(), u_pedido=am[:, 1].tolist())
    return r


# ---------------- Parte B: logs do carro real --------------------------------------------
def log(id_carro):
    d = id_carro[:8]
    return carrega(next((RAIZ / 'ensaios' / 'real' / 'roxo').glob(f'*/{d}T{id_carro[9:]}__*.csv')))


def trecho(x, t0, t1):
    k = (x['t'] >= t0) & (x['t'] <= t1)
    return {c: x[c][k] for c in ('t', 'v', 'u', 'th', 'vref')}


def ajusta(xs, entrada):
    """Um modelo para o dia: v = MA30(atraso(G_i*max(0, entrada - limiar))), ganho G_i por corrida (a
    bateria muda), limiar e atraso tau comuns. Devolve o fit % de cada corrida (melhor de 3 chutes)."""
    ys = [x['vg'][x['tg'] <= x['tmax']] for x in xs]
    ss = [x[entrada][x['tg'] <= x['tmax']] for x in xs]
    n = len(xs)
    def res(p):
        return np.concatenate([preve(s, x['Ts'], p[2 + i], p[0], p[1], 0.1) - y
                               for i, (x, s, y) in enumerate(zip(xs, ss, ys))])
    if entrada == 'thg':
        lo, hi, chutes = [0.0, 0.05] + [0.01] * n, [10.0, 2.0] + [1.0] * n, ([2.0, 0.5, 0.2], [4.0, 0.7, 0.3])
    else:
        lo, hi, chutes = [-1.0, 0.05] + [0.01] * n, [1.0, 2.0] + [20.0] * n, ([0.0, 0.5, 2.0], [-0.5, 0.5, 1.0],
                                                                              [0.2, 0.3, 5.0])
    melhor = None
    for c in chutes:
        p = least_squares(res, np.clip([c[0], c[1]] + [c[2]] * n, lo, hi), bounds=(lo, hi)).x
        r = res(p)
        if melhor is None or np.sum(r ** 2) < melhor[0]:
            melhor = (np.sum(r ** 2), p)
    p = melhor[1]
    fits, i0 = [], 0
    for y in ys:
        r = res(p)[i0:i0 + len(y)]
        fits.append(fit_pct(y, y + r))
        i0 += len(y)
    return fits, p


# corridas de 01/10 e o fim do trecho em que o carro andou livre (depois disso foi pego / segurado)
CORRIDAS_B = [('20260930_211005', 29.5, 'malha fechada, K_NOM 1,6, trava de VELMAX'),
              ('20260930_212048', 30.5, 'malha fechada, K_NOM 1,6, trava de VELMAX'),
              ('20260930_214209', 40.0, 'malha fechada, K_NOM 4,0'),
              ('20260930_220658', 30.0, 'malha aberta, degraus de u')]


def parte_b():
    r = {}
    print('\nPARTE B - logs do carro roxo (o acelerador e reconstruido: o log nao tem o PWM)')
    # 211005: u fica em 0 desde que v passa de 1,5 (car.py:449) ate o carro ser pego (mesma regra do .m)
    a = log('20260930_211005')
    k0 = int(np.argmax((a['u'] == 0) & (a['v'] > 1.5)))
    k1 = k0 + int(np.argmax(a['u'][k0:] != 0)) - 1
    k = slice(k0, k1 + 1)
    ta = a['t'][k0]
    print(f'  211005: u = {np.unique(a["u"][k])} durante {a["t"][k1] - ta:.1f} s; '
          f'v = {a["v"][k].mean():.2f} m/s (min {a["v"][k].min():.2f}, max {a["v"][k].max():.2f}); '
          f'acelerador {a["th"][k].mean():.2f} graus')
    # 220658: primeira vez que u volta a 0 depois do u = -1 que zera o acelerador
    b = log('20260930_220658')
    j0 = int(np.nonzero((b['u'][:-1] == -1) & (b['u'][1:] == 0))[0][0]) + 1
    tb = b['t'][j0]
    t_para = b['t'][j0 + int(np.argmax(b['v'][j0:] < 0.05))] - tb
    print(f'  220658: u = 0 a partir de t = {tb:.2f} s com acelerador {b["th"][j0]:.2f} graus; v cai de '
          f'{b["v"][j0]:.2f} para < 0,05 m/s em {t_para:.2f} s')
    r['mesmo_u0'] = dict(trava=dict(id='20260930_211005', dur=float(a['t'][k1] - ta),
                                    v_med=float(a['v'][k].mean()), th=float(a['th'][k].mean())),
                         para=dict(id='20260930_220658', v0=float(b['v'][j0]), t_parada=float(t_para)))
    # series alinhadas no instante em que u fica 0
    r['serie_u0'] = {}
    for nome, x, t0 in (('trava', a, ta), ('para', b, tb)):
        tr = trecho(x, t0 - 2.0, t0 + 10.0)
        r['serie_u0'][nome] = dict(t=(tr['t'] - t0).round(3).tolist(), v=tr['v'].round(3).tolist(),
                                   u=tr['u'].round(3).tolist(), th=tr['th'].round(3).tolist())

    c = log('20260930_214209')
    w = (c['t'] > 10) & (c['t'] < 39)
    print(f'  214209 cruzeiro (10-39 s): v = {c["v"][w].mean():.3f} m/s com u medio = {c["u"][w].mean():+.4f} '
          f'(|u| medio {np.abs(c["u"][w]).mean():.4f}); acelerador {c["th"][w].mean():.2f} graus')
    r['cruzeiro'] = dict(id='20260930_214209', v=float(c['v'][w].mean()), u_med=float(c['u'][w].mean()),
                         u_abs=float(np.abs(c['u'][w]).mean()), th=float(c['th'][w].mean()),
                         t=c['t'].round(3).tolist(), v_s=c['v'].round(3).tolist(), vref=c['vref'].round(3).tolist(),
                         u_s=c['u'].round(4).tolist(), th_s=c['th'].round(3).tolist())

    print('  quem explica v? UM modelo para as 4 corridas de 01/10: v = MA30(atraso(G_i*max(0, entrada - limiar)))')
    xs = []
    for idc, tmax, _ in CORRIDAS_B:
        x = log(idc)
        x['tmax'] = tmax
        xs.append(x)
    f_int, p_int = ajusta(xs, 'thg')
    f_u, p_u = ajusta(xs, 'ug')
    print(f'    entrada = integral de u: limiar {p_int[0]:.2f} graus, tau {p_int[1]:.2f} s | '
          f'entrada = u: limiar {p_u[0]:+.2f}, tau {p_u[1]:.2f} s')
    r['ajustes'] = []
    for (idc, _, desc), fi, fu in zip(CORRIDAS_B, f_int, f_u):
        r['ajustes'].append(dict(id=idc, desc=desc, fit_integral=fi, fit_u=fu))
        print(f'    {idc} ({desc}): integral de u -> fit {fi:5.1f}%   |   u -> fit {fu:6.1f}%')
    r['ajuste_params'] = dict(integral=list(map(float, p_int)), u=list(map(float, p_u)))
    return r


if __name__ == '__main__':
    SAIDA.mkdir(parents=True, exist_ok=True)
    ra = parte_a()
    rb = parte_b()
    (SAIDA / 'prova_integrador.json').write_text(json.dumps(dict(a=ra, b=rb)), encoding='utf-8')

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 1, figsize=(10, 12), sharex=False)
    s = ra['serie']
    ax[0].step(s['tu'], s['u'], where='post'); ax[0].set_ylabel('u aplicado')
    ax[0].set_title('A: car.py + servos.py da disciplina no PC (hardware falso)', loc='left', fontsize=9)
    ax[1].plot(s['t'], s['th']); ax[1].set_ylabel('acelerador escrito\nno ESC [graus]')
    for nome, cor in (('trava', 'C3'), ('para', 'C0')):
        q = rb['serie_u0'][nome]
        ax[2].plot(q['t'], q['u'], cor); ax[3].plot(q['t'], q['th'], cor); ax[4].plot(q['t'], q['v'], cor, label=nome)
    ax[2].set_ylabel('u'); ax[3].set_ylabel('acelerador\nreconstruido [graus]'); ax[4].set_ylabel('v [m/s]')
    ax[2].set_title('B: carro real, u = 0 a partir de t = 0 (211005 trava / 220658 para)', loc='left', fontsize=9)
    ax[4].legend(); ax[4].set_xlabel('t [s]')
    fig.tight_layout(); fig.savefig(SAIDA / 'prova_integrador.png', dpi=90)
    print(f'\nfigura: {SAIDA / "prova_integrador.png"}')
