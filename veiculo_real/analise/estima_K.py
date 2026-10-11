# -*- coding: utf-8 -*-
"""Estimar o ganho K da planta ONLINE (offline, nao toca no main.py). Estimador + 4 experimentos:
  E1 simulacao: K_hat(t) em malha fechada (lei atual) para K, DB, tau, ruido diferentes
  E2 simulacao: FF adaptativo (KFF = K_hat) na referencia decrescente, contra a lei atual e o FF 'oraculo'
  E3 logs reais: o estimador rodando sobre car.csv, contra o K ajustado em lote (analisa.py)
Rodar de veiculo_real/:  ../simulador/.venv/Scripts/python.exe analise/estima_K.py   (~3 min)

Modelo do estimador (a MESMA media movel de 30 amostras em v e no acelerador estimado pm):
    v_ma = -tau * dv_ma/dt + G * pm_ma - G*DB          (planta 1a ordem + zona morta, valida acima da DB)
  => regressao linear  y = v,  phi = [pm, -dv/dt, -1],  theta = [G, tau, G*DB];  K = G * K_PWM.
RLS com prior (centro K=2, tau=.7, DB=3.5) e SEM esquecimento (planta constante dentro de uma corrida).
"""
import sys, io, glob, contextlib
from collections import deque
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from acervo import caminho
with contextlib.redirect_stdout(io.StringIO()):
    import simula_variantes as S
A = S.A
K_PWM, TH_MAX, N_MA = S.K_PWM, S.TH_MAX, S.N_MA


class EstK:
    def __init__(s, K0=2.0, tau0=0.7, DB0=3.5, sG=0.15, sT=0.4, sC=1.0, R=0.03 ** 2, atraso=4, janela=8,
                 t_espera=1.5, v_min=0.25, K_lim=(0.8, 4.5), sK_max=0.4):
        G0 = K0 / K_PWM
        s.th = np.array([G0, tau0, G0 * DB0])
        s.P = np.diag([sG ** 2, sT ** 2, sC ** 2])
        s.R, s.at, s.jn = R, atraso, janela
        s.t_espera, s.v_min, s.K_lim, s.sK_max = t_espera, v_min, K_lim, sK_max
        s.T, s.V, s.PM = [], [], []
        s.pbuf = deque([0.0] * N_MA, maxlen=N_MA)
        s.t_move, s.n, s.t_pronto = None, 0, None

    def update(s, t, v, pwm, exc=True):
        s.pbuf.append(pwm)
        s.T.append(t); s.V.append(v); s.PM.append(float(np.mean(s.pbuf)))
        if s.t_move is None and v > 0.05:
            s.t_move = t
        k = len(s.T) - 1
        if k < s.jn or s.t_move is None or t - s.t_move < s.t_espera or not exc or abs(v) > 1.45:   # 1.45: o car.py zera u acima de 1.5
            return
        j = k - s.at   # regressores no instante central da diferenca (derivada centrada, sem olhar o futuro de verdade)
        vdot = (s.V[k] - s.V[k - s.jn]) / (s.T[k] - s.T[k - s.jn])
        vj, pmj = s.V[j], s.PM[j]
        G, tau, c = s.th
        if vj < s.v_min or pmj < c / max(G, 1e-3) + 0.3:   # parado/devagar ou coasting (abaixo da DB): equacao nao vale
            return
        phi = np.array([pmj, -vdot, -1.0])
        Pp = s.P @ phi
        g = Pp / (s.R + phi @ Pp)
        s.th = s.th + g * (vj - phi @ s.th)
        s.th[0] = np.clip(s.th[0], 0.05 / K_PWM, 6.0 / K_PWM); s.th[1] = np.clip(s.th[1], 0.1, 1.6)
        s.P = s.P - np.outer(g, Pp)
        s.n += 1
        if s.t_pronto is None and s.sK() < s.sK_max:
            s.t_pronto = t

    def K(s):
        return float(s.th[0] * K_PWM)

    def sK(s):
        return float(np.sqrt(max(s.P[0, 0], 0.0)) * K_PWM)

    def kff(s, padrao):
        """K para o feedforward: estimado se ja ha confianca, senao o padrao (K_NOM)."""
        return float(np.clip(s.K(), *s.K_lim)) if s.t_pronto is not None else padrao


class LeiUAdapt(S.LeiU):
    """Lei atual com KFF vindo do estimador (laco continua com K_NOM fixo)."""
    def __init__(s, KN, pts, est):
        super().__init__(KN, pts)
        s.est = est
        s.kk = []
        s.gated, s.lim_e = True, 0.08
    def step(s, t, h, v, pwm):
        s.est.update(t, v, pwm, exc=(not s.gated) or abs(s.e) > s.lim_e)
        s.KFF = s.est.kff(s.KN)
        s.kk.append((t, s.est.K(), s.KFF, s.est.sK()))
        return super().step(t, h, v, pwm)


def roda_adapt(K, pts, ts, DB=3.5, tau=0.72, sig=0.004, seed=0, d=0.08, **kw):
    est = EstK(**kw)
    ctl = LeiUAdapt(4.0, pts, est)
    L = S.roda(ctl, S.Planta(K, DB=DB, tau=tau, sig=sig, seed=seed, d=d), ts)
    return L, np.array(ctl.kk), est


def em(kk, t):
    i = np.searchsorted(kk[:, 0], t)
    return kk[min(i, len(kk) - 1)]


def e1():
    print('\n== E1  K_hat em malha fechada (degrau 0->1.3, depois patamar). Celula: K_hat(t=6)/K_hat(t=12)/K_hat(t=30); "pronto" = t em que sigma_K < 0.4')
    pts = S.pontos([(0, 1.3)])
    casos = [('base  DB 3.5 tau .72', {}), ('DB 1.5', dict(DB=1.5)), ('DB 5.5', dict(DB=5.5)), ('tau 0.3', dict(tau=0.3)),
             ('ruido 3x (0.012)', dict(sig=0.012)), ('sem atraso (d=0)', dict(d=0.0))]
    Ks = [1.09, 1.6, 2.0, 2.9, 3.5]
    print(' ' * 24 + ''.join(f"{'K=' + str(K):>24s}" for K in Ks))
    for nome, kw in casos:
        cel, pr = [], []
        for K in Ks:
            L, kk, est = roda_adapt(K, pts, 30, **kw)
            cel.append('/'.join(f"{em(kk, t)[1]:.2f}" for t in (6, 12, 30)))
            pr.append(est.t_pronto)
        print(f"{nome:24s}" + ''.join(f"{c:>24s}" for c in cel))
        print(f"{'   pronto em (s)':24s}" + ''.join(f"{('-' if p is None else f'{p:.1f}'):>24s}" for p in pr))


def e2():
    print('\n== E2  Referencia decrescente 1.3->0.5 (t=14)->0 (t=24): atraso(s)/erro patamar/tempo p/ parar | e overshoot do degrau')
    pts = S.pontos([(0, 1.3), (14, 0.5), (24, 0.0, 0.25)])
    Ks = [0.9, 1.09, 1.6, 2.0, 2.9, 3.5]
    print(' ' * 22 + ''.join(f"{'K=' + str(K):>24s}" for K in Ks))
    for nome in ('A0 atual (FF K_NOM 4)', 'A3 oraculo (FF K real)', 'A5 adaptativo (FF K_hat)'):
        cel = []
        for K in Ks:
            if nome.startswith('A0'):
                L = S.roda(S.Modulo(4.0, pts), S.Planta(K, seed=0), 40)
            elif nome.startswith('A3'):
                L = S.roda(S.LeiU(4.0, pts, KFF=K), S.Planta(K, seed=0), 40)
            else:
                L, _, _ = roda_adapt(K, pts, 40)
            m = S.metr_desce(L)
            ov = (L[:, 1][(L[:, 0] < 14)].max() / 1.3 - 1) * 100
            cel.append(f"{m['lag']:.2f}/{m['err_plato']:+.2f}/{m['tpara']:.1f} | {ov:.1f}%")
        print(f"{nome:22s}" + ''.join(f"{c:>24s}" for c in cel))
    # K muda no meio da corrida? (bateria): estimador sem esquecimento nao acompanha -> mostra o limite
    print('\n   (planta constante dentro da corrida; K_hat reinicia a cada corrida)')


def k_lote(x, tmax=1e9, DB=A.DB_REF, d=A.D_REF):
    from scipy.optimize import least_squares
    m = x['tg'] <= tmax
    f = lambda p: (A.preve(x['thg'], x['Ts'], p[0], DB, p[1], d) - x['vg'])[m]
    r = least_squares(f, [0.2, 0.7], bounds=([0.02, 0.1], [1.0, 2.0]))
    return A.K_DE(r.x[0]), r.x[1], A.rmse(r.fun)


def e3():
    print('\n== E3  Logs reais: K_hat(t) do estimador vs K ajustado em lote (analisa.py). K_hat em t=6/12/20/fim ; pronto em')
    grupos = [('25/09 planta lenta (K lote ~1.09)', ['20260925_212928', '20260925_213354', '20260925_213455']),
              ('30/09 bateria cheia, K_NOM 1.6 (K lote 2.84 em 211005)', ['20260930_211005', '20260930_212048']),
              ('30/09 bateria cheia, K_NOM 3.5-4.0', ['20260930_213516', '20260930_214209'])]   # ids do carro (relogio do carro)
    for nome, ids_ in grupos:
        print(' ', nome)
        for p in ids_:
            x = A.carrega(caminho(p))
            est = EstK()
            ctl = S.LeiU(4.0, list(zip(x['t'] - x['t'][0], x['vref'])))   # so para reconstruir o erro e (excitacao)
            kk = []
            for k in range(2, len(x['t'])):
                ctl.step(x['t'][k], x['t'][k] - x['t'][k - 1], x['v'][k], 0.0)
                est.update(x['t'][k], x['v'][k], x['th'][k], exc=abs(ctl.e) > 0.08)
                kk.append((x['t'][k], est.K(), est.sK()))
            kk = np.array(kk)
            tmax = 19.5 if p.endswith('213516') else 1e9   # o carro foi parado a mao em ~20 s
            Kl, tl, er = k_lote(x, tmax)
            vals = [f"{em(kk, t)[1]:.2f}" for t in (6, 12, 20)] + [f"{kk[-1, 1]:.2f}"]
            print(f"    {p}: K_hat ={' / '.join(vals)}   sigma_K {kk[-1, 2]:.2f}   | K em lote (G,tau livres, DB 2.5): {Kl:.2f} (tau {tl:.2f})   pronto em {'-' if est.t_pronto is None else f'{est.t_pronto:.1f}s'}   n={est.n}")


if __name__ == '__main__':
    import sys as _s
    sel = _s.argv[1:] or ['e1', 'e2', 'e3']
    for nome in sel:
        globals()[nome]()
