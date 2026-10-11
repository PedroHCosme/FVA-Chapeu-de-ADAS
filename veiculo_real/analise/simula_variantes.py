# -*- coding: utf-8 -*-
"""Simulacao OFFLINE (nao toca no main.py): lei atual x variantes (integral extra, anti-windup, FF estatico).
Rodar de veiculo_real/:  ../simulador/.venv/Scripts/python.exe analise/simula_variantes.py   (~2 min)"""
import sys, io, contextlib
from collections import deque
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import analisa as A
K_PWM, TH_MAX, N_MA = A.K_PWM, A.TH_MAX, A.N_MA
MOD = A.carrega_main('avanco')
KP, TD, TAU = 0.8, 0.6, 0.5          # preset avanco


# ------------------------------------------------------------------ planta
class Planta:
    """u -> pwm (integra, clip) -> v_ss = G max(0, pwm-DB) -> lag tau -> atraso d -> MA30 (+ruido).
    Descida: a desaceleracao nao passa de a_coast (carro so 'coasta'; 0.45 = max observado em 212026)."""
    def __init__(s, K, DB=3.5, tau=0.72, d=0.08, a_coast=0.45, seed=0, sig=0.004):
        s.G, s.DB, s.tau, s.d, s.ac = A.G_DE(K), DB, tau, d, a_coast
        s.pwm = s.vr = 0.0
        s.th, s.hv = [0.0], [0.0]
        s.buf = [0.0] * N_MA
        s.rng = np.random.default_rng(seed); s.sig = sig

    def passo(s, t, h, u, dist=0.0):
        s.pwm = min(max(s.pwm + K_PWM * u * h, 0.0), TH_MAX)
        vss = s.G * max(0.0, s.pwm - s.DB)
        a = max((vss - s.vr) / s.tau, -s.ac) + dist
        s.vr = max(s.vr + a * h, 0.0)
        s.th.append(t); s.hv.append(s.vr)
        s.buf = s.buf[1:] + [float(np.interp(t - s.d, s.th, s.hv)) + s.rng.normal(0, s.sig)]
        return float(np.mean(s.buf))


# ------------------------------------------------------------------ referencia
def pontos(perfil):
    return MOD._pontos(perfil, 0.9)

def ref_de(pts, tau):
    return float(np.interp(tau, *zip(*pts)))


# ------------------------------------------------------------------ controladores
class Modulo:
    """A lei EXATA do main.py (controlador_longitudinal), com K_NOM/preset trocados."""
    def __init__(s, KN, pts, ff=True):
        MOD.KP, MOD.TD, MOD.TAU_VDES, MOD.K_FF, MOD.TAU_FF = KP / KN, TD, TAU, KN, 0.0   # KN = o antigo K_NOM
        MOD._PONTOS = pts
        MOD._estado.clear(); MOD._vdes_filt.reset(0.0)
        s.car, s.ff, s.t0 = A.CarroFalso(), ff, None
        s.car.atuador.trim_throttle = 0.0
    def step(s, t, h, v, pwm):
        c = s.car
        c.t, c.dt, c.v = t, h, v
        s.t0 = t if s.t0 is None else s.t0
        MOD.control_func(c)
        return c._u_raw
# o CarroFalso zera u se |v|>1.5 e clipa; queremos o u bruto: capturamos via set_u
def _set_u(self, u):
    self._u_raw = float(u)
    self.u = 0.0 if abs(self.v) > 1.5 else float(np.clip(u, -1.0, 1.0))
A.CarroFalso.set_u = _set_u


class LeiU:
    """Mesma lei no espaco de u, escrita aqui (para variar): FF opcional, integral extra opcional."""
    def __init__(s, KN, pts, KFF=None, Ti=None, aw=True, gate=False):
        s.KN, s.KFF, s.Ti, s.aw, s.pts, s.gate = KN, KFF if KFF else KN, Ti, aw, pts, gate
        s.vf, s.buf, s.der, s.eant, s.I, s.t0, s.e = 0.0, deque([0.0] * N_MA, maxlen=N_MA), 0.0, None, 0.0, None, 0.0
    def step(s, t, h, v, pwm):
        s.t0 = t if s.t0 is None else s.t0
        raw = ref_de(s.pts, t - s.t0)
        a = np.clip(h / TAU, 0, 1)
        ant = s.vf; s.vf = a * raw + (1 - a) * s.vf
        dvd = (raw - ant) / TAU
        s.buf.append(s.vf); e = float(np.mean(s.buf)) - v; s.e = e
        b = h / (TD / 5 + h)
        s.der += b * ((e - (e if s.eant is None else s.eant)) / h - s.der); s.eant = e
        u_i = 0.0
        u = KP * e / s.KN + KP * TD * s.der / s.KN + (dvd / s.KFF if s.KFF < 1e8 else 0.0)
        if s.Ti:
            u_try = u + KP / s.KN / s.Ti * s.I
            sat = (pwm >= TH_MAX - 1e-6 and u_try > 0) or (pwm <= 1e-6 and u_try < 0) or abs(v) > 1.5 \
                or (abs(u_try) >= 1 and np.sign(u_try) == np.sign(e)) or (s.gate and (abs(e) > 0.1 or abs(dvd) > 0.1))
            if not (s.aw and sat):
                s.I += e * h
            u = u + KP / s.KN / s.Ti * s.I
        return u


class LeiTH:
    """Mesma lei no espaco do ACELERADOR (forma de PI posicional):
       th_cmd = DBn + vf/Gn + (K_PWM/KN) KP (TD lp(e) + int e);  u = (th_cmd - pwm)/(K_PWM h) clipado.
       Gn = KN/K_PWM. DBn = 0 reproduz a lei atual (menos efeito do clip)."""
    def __init__(s, KN, pts, DBn=0.0, aw=True):
        s.KN, s.DBn, s.aw, s.pts = KN, DBn, aw, pts
        s.vf, s.buf, s.lp, s.I, s.t0 = 0.0, deque([0.0] * N_MA, maxlen=N_MA), 0.0, 0.0, None
    def step(s, t, h, v, pwm):
        s.t0 = t if s.t0 is None else s.t0
        raw = ref_de(s.pts, t - s.t0)
        a = np.clip(h / TAU, 0, 1)
        s.vf = a * raw + (1 - a) * s.vf
        s.buf.append(s.vf); e = float(np.mean(s.buf)) - v
        s.lp += h / (TD / 5 + h) * (e - s.lp)
        k = K_PWM / s.KN
        base = (s.DBn if s.vf > 0.02 else 0.0) + k * s.vf + k * KP * TD * s.lp
        cmd = base + s.I
        sat = (cmd >= TH_MAX and e > 0) or (cmd <= 0 and e < 0) or abs(v) > 1.5
        if not (s.aw and sat):
            s.I += k * KP * e * h
        cmd = base + s.I
        return (cmd - pwm) / (K_PWM * h)


# ------------------------------------------------------------------ simulacao
def roda(ctl, pl, ts, dist_t=None, dist=0.0, seed=1):
    rng = np.random.default_rng(seed)
    t, u_app, v, pwm = 0.0, 0.0, 0.0, 0.0
    L = []
    while t < ts:
        h = 0.0288 + abs(rng.normal(0, 0.003))
        t += h
        v = pl.passo(t, h, u_app, dist if (dist_t is not None and t >= dist_t) else 0.0)
        u_cmd = ctl.step(t, h, v, pl.pwm)
        u_app = 0.0 if abs(v) > 1.5 else float(np.clip(u_cmd, -1.0, 1.0))
        L.append((t, v, u_app, pl.pwm))
    return np.array(L)


def seg(L, a, b):
    return L[(L[:, 0] >= a) & (L[:, 0] < b)]

def metr_degrau(L, vr=1.3):
    t, v = L[:, 0], L[:, 1]
    fora = np.abs(v - vr) > 0.02 * vr
    ts = t[fora][-1] if fora.any() else 0.0
    m = seg(L, 8, 1e9)
    return dict(over=(v.max() / vr - 1) * 100, tas=ts if ts < t[-1] - 1 else np.nan,
                rms=float(np.sqrt(np.mean((m[:, 1] - vr) ** 2))), trap=float((v > 1.5).sum() * 0.0288), vfim=float(v[-1]))

def metr_desce(L):
    t, v = L[:, 0], L[:, 1]
    p = seg(L, 22.5, 24.0)
    d = seg(L, 14, 24)
    fora = np.abs(d[:, 1] - 0.5) > 0.05
    tas = d[:, 0][fora][-1] - 14 if fora.any() else 0.0
    f = seg(L, 24, 1e9)
    par = f[:, 0][f[:, 1] < 0.05]
    c = d[:, 0][d[:, 1] < 0.9]
    lag = (c[0] - 14.44) if len(c) else np.nan
    return dict(lag=lag, err_plato=float(p[:, 1].mean() - 0.5), tas=tas, undershoot=float(0.5 - d[:, 1].min()),
                tpara=(par[0] - 24) if len(par) else np.nan, vfim=float(v[-1]))

def metr_dist(L, vr=1.3):
    d = seg(L, 15, 1e9)
    fim = seg(L, d[-1, 0] - 3, 1e9)
    fora = np.abs(d[:, 1] - vr) > 0.03
    return dict(dip=float(vr - d[:, 1].min()), err_reg=float(fim[:, 1].mean() - vr),
                trec=(d[:, 0][fora][-1] - 15) if fora.any() else 0.0, pwm_fim=float(fim[:, 3].mean()))


VARIANTES = {
    'A0 atual K_NOM 4.0':        lambda p, K: Modulo(4.0, p),
    'A1 atual K_NOM 1.6':        lambda p, K: Modulo(1.6, p),
    'A2 atual sem FF':           lambda p, K: LeiU(4.0, p, KFF=1e9),
    'A3 FF com K real (oraculo)': lambda p, K: LeiU(4.0, p, KFF=K),
    'B1 +I Ti=8 c/ AW (ingenuo)': lambda p, K: LeiU(4.0, p, Ti=8.0, aw=True),
    'B4 +I Ti=8 AW + gate':      lambda p, K: LeiU(4.0, p, Ti=8.0, aw=True, gate=True),
    'B5 +I Ti=4 AW + gate':      lambda p, K: LeiU(4.0, p, Ti=4.0, aw=True, gate=True),
    'C0 forma th, sem DB':       lambda p, K: LeiTH(4.0, p, DBn=0.0),
    'C1 forma th, FF c/ DB 3.5': lambda p, K: LeiTH(4.0, p, DBn=3.5),
}


def valida_igualdade():
    """LeiU (sem integral) tem que dar o MESMO u que o main.py, sobre a mesma entrada v."""
    pts = pontos([(0, 1.3)])
    pl = Planta(1.6, seed=3)
    m = Modulo(4.0, pts)
    u_app, t, L = 0.0, 0.0, []
    rng = np.random.default_rng(5)
    own = LeiU(4.0, pts)
    dmax = 0.0
    for _ in range(1200):
        h = 0.0288 + abs(rng.normal(0, 0.003)); t += h
        v = pl.passo(t, h, u_app)
        um, uo = m.step(t, h, v, pl.pwm), own.step(t, h, v, pl.pwm)
        dmax = max(dmax, abs(um - uo))
        u_app = 0.0 if abs(v) > 1.5 else float(np.clip(um, -1, 1))
    return dmax


if __name__ == '__main__':
    print(f"validacao: max|u_main - u_LeiU| = {valida_igualdade():.2e}  (deve ser ~0)")
    Ks = [0.9, 1.09, 1.6, 2.0, 2.9, 3.5]
    for nome_cen, perfil, ts, f, chaves in (
            ('DEGRAU 0->1.3 m/s (rampa .9): overshoot %, rms(t>8) m/s, s em v>1.5', [(0, 1.3)], 30, metr_degrau, ('over', 'rms', 'trap')),
            ('REF. DECRESCENTE 1.3->0.5 (t=14) ->0 (t=24, .25 m/s2): atraso (s) ao cruzar 0.9, erro no patamar 0.5, s p/ parar', [(0, 1.3), (14, 0.5), (24, 0.0, 0.25)], 40, metr_desce, ('lag', 'err_plato', 'tpara')),
            ('DISTURBIO -0.3 m/s2 em t=15: queda de v, erro final, s p/ recuperar', [(0, 1.3)], 40, metr_dist, ('dip', 'err_reg', 'trec'))):
        print('\n==', nome_cen)
        print(' ' * 27 + ''.join(f"{'K=' + str(K):>22s}" for K in Ks))
        pts = pontos(perfil)
        for nv, mk in VARIANTES.items():
            cel = []
            for K in Ks:
                L = roda(mk(pts, K), Planta(K, seed=0), ts, dist_t=15 if f is metr_dist else None, dist=-0.3)
                m = f(L)
                cel.append('/'.join(f"{m[c]:.2f}" for c in chaves))
            print(f"{nv:27s}" + ''.join(f"{c:>22s}" for c in cel))
