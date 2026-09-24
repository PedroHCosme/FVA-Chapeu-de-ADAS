# -*- coding: utf-8 -*-
"""O log do colega ajuda o ajuste GLOBAL? Ajusta sem e com o log dele,
com o resto identico, e compara parametros + validacao (mesma logica do
identifica.py: v com sinal da posicao, atrito com tanh)."""
import glob, os
import numpy as np

import sys
IGNORA_SEMPRE = ["20260901", "20260917_074529"]   # cena anomala + log bruto da queda
# alvo do experimento "sem x com": passe --alvo=20260917_altavel (default) ou 20260917_colega
COLEGA = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--alvo=")), "20260917_altavel")


def carrega(ignora):
    T, V, VD, U, ENS = [], [], [], [], []
    for csv in sorted(glob.glob(os.path.join("logs", "*", "car.csv"))):
        if any(s in csv for s in ignora):
            continue
        d = np.genfromtxt(csv, delimiter=",", names=True)
        if len(d) < 30:
            continue
        t, u, x, y = d["t"], d["u"], d["x"], d["y"]
        passo = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
        v = np.r_[0.0, passo * np.sign(np.diff(x))]
        vd = np.gradient(v, t)
        ok = (np.abs(v) > 0.10) & (t > 0.15) & (t < t[-1] - 0.05)
        vs = np.convolve(v, np.ones(5) / 5, mode="same")
        inv = np.nonzero((vs <= -0.03) & (np.maximum.accumulate(vs) > 0.03))[0]
        if len(inv):
            ok &= t < t[inv[0]]
        if ok.sum() < 30:
            continue
        T += [t[ok]]; V += [v[ok]]; VD += [vd[ok]]; U += [u[ok]]
        ENS.append((os.path.basename(os.path.dirname(csv)), t, v, u, ok))
    return np.concatenate(T), np.concatenate(V), np.concatenate(VD), np.concatenate(U), ENS


def ajusta(v, vd, u):
    A = np.c_[np.where(u > 0, u, 0.0), np.where(u < 0, u, 0.0),
              -v * np.abs(v), -np.sign(v)]
    p, *_ = np.linalg.lstsq(A, vd, rcond=None)
    return p  # Kp, Km, c, a0


def simula(t, u, v0, p):
    Kp, Km, c, a0 = p
    v = np.empty_like(t); v[0] = v0
    for i in range(len(t) - 1):
        K = Kp if u[i] > 0 else (Km if u[i] < 0 else 0.0)
        vdot = K * u[i] - c * v[i] * abs(v[i]) - a0 * np.tanh(v[i] / 0.05)
        v[i + 1] = v[i] + vdot * (t[i + 1] - t[i])
    return v


def valida_rms(ENS, p):
    out = {}
    for nome, tt, vv, uu, ok in ENS:
        m = slice(np.argmax(ok), np.nonzero(ok)[0][-1] + 1)
        vs = simula(tt[m], uu[m], vv[m][0], p)
        out[nome] = (vs - vv[m]).std()
    return out


# --- ajuste A: sem o colega --------------------------------------------------
vA = carrega(IGNORA_SEMPRE + [COLEGA])
pA = ajusta(vA[1], vA[2], vA[3])
# --- ajuste B: com o colega --------------------------------------------------
vB = carrega(IGNORA_SEMPRE)
pB = ajusta(vB[1], vB[2], vB[3])

nomes = ["K+", "K-", "c", "a0"]
print(f"ALVO = {COLEGA}")
print("PARAMETROS         " + "".join(f"{n:>10}" for n in nomes))
print("sem alvo        " + "".join(f"{x:>10.5f}" for x in pA))
print("com alvo        " + "".join(f"{x:>10.5f}" for x in pB))
print("variacao %      " + "".join(f"{100*(b-a)/a:>9.1f}%" for a, b in zip(pA, pB)))

rA = valida_rms(vA[4], pA)   # validacao nos logs comuns, params A
rB = valida_rms(vA[4], pB)   # validacao nos MESMOS logs comuns, params B
print("\nVALIDACAO (erro rms m/s) nos logs comuns — params sem x com alvo")
print(f"{'ensaio':<18}{'sem':>10}{'com':>10}{'delta':>10}")
sa = sb = 0.0
for nome in rA:
    print(f"{nome:<18}{rA[nome]:>10.4f}{rB[nome]:>10.4f}{rB[nome]-rA[nome]:>10.4f}")
    sa += rA[nome]; sb += rB[nome]
n = len(rA)
print(f"{'MEDIA':<18}{sa/n:>10.4f}{sb/n:>10.4f}{(sb-sa)/n:>10.4f}")

# validacao no proprio log do colega, com cada conjunto de params
rc = valida_rms([e for e in vB[4] if COLEGA in e[0]], pA)
rc2 = valida_rms([e for e in vB[4] if COLEGA in e[0]], pB)
print(f"\nno log do colega   params-sem={list(rc.values())[0]:.4f}   params-com={list(rc2.values())[0]:.4f} m/s")
