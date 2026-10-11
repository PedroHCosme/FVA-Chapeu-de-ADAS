# -*- coding: utf-8 -*-
"""Caracteriza o ultrassom a partir do sonar.csv que o main.py grava ao lado do car.csv (t, d, valid, alvo).

    python analise/sonar.py ..\\ensaios\\_entrada\\roxo\\<id>     pasta do log (ou o sonar.csv, ou varias)
    python analise/sonar.py --teste                              autoteste com dados sinteticos

Ensaio 'sonar' (carro parado, caixa a 0,3 ... 4 m e depois nada): por distancia, % de leituras validas,
mediana, ruido (desvio) e erro (mediana - distancia da caixa); os 3 s iniciais de cada janela sao descartados
(a caixa esta sendo posta). Outras corridas: so % validas e a menor distancia lida (ex.: chegando na parede).
Saida: tabela na tela e analise/saida/sonar_<id>.png.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAIDA = Path(__file__).resolve().parent / 'saida'
DESCARTA = 3.0   # s no inicio de cada janela da caixa


def carrega(p):
    a = np.genfromtxt(p, delimiter=',', names=True)
    return {c: np.asarray(a[c], float) for c in ('t', 'd', 'valid', 'alvo')}


def tabela(x):
    """Uma linha por distancia da caixa: (alvo, n, % validas, mediana, desvio, erro). Linhas so com validas."""
    t, d, ok, alvo = x['t'], x['d'], x['valid'] > 0.5, x['alvo']
    linhas = []
    for a in dict.fromkeys(alvo[~np.isnan(alvo)]):          # na ordem do ensaio
        j = np.nonzero(alvo == a)[0]
        j = j[t[j] >= t[j[0]] + DESCARTA]
        v = d[j][ok[j]]
        med = float(np.median(v)) if len(v) else np.nan
        linhas.append((float(a), len(j), 100 * float(ok[j].mean()), med,
                       float(np.std(v)) if len(v) else np.nan, med - a if a and len(v) else np.nan))
    return linhas


def relatorio(p):
    x = carrega(p)
    nome = Path(p).parent.name
    if np.isnan(x['alvo']).all():   # corrida normal: sem caixa programada
        ok = x['valid'] > 0.5
        i = int(np.argmin(np.where(ok, x['d'], np.inf))) if ok.any() else None
        print(f"{nome}: {100 * ok.mean():.0f}% validas" + (f", menor distancia {x['d'][i]:.2f} m em t = {x['t'][i]:.1f} s" if i is not None else ''))
        return
    print(f"{nome}: ensaio 'sonar'  (descarta {DESCARTA:.0f} s no inicio de cada janela)")
    print("   caixa [m]    n   validas   mediana [m]   ruido (desvio) [m]   erro [m]")
    f = lambda v, fmt: '-' if np.isnan(v) else format(v, fmt)
    for a, n, pv, med, sd, err in tabela(x):
        print(f"   {'nada' if a == 0 else f'{a:.1f}':>8s}  {n:5d}   {pv:5.0f}%     {f(med, '8.3f'):>8s}        {f(sd, '8.3f'):>8s}          {f(err, '+.3f')}")
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    ok = x['valid'] > 0.5
    ax.plot(x['t'], np.where(np.isnan(x['alvo']), np.nan, x['alvo']), 'k--', lw=1, label='caixa (0 = nada)')
    ax.plot(x['t'][ok], x['d'][ok], '.', ms=3, color='#2A78D6', label='leitura valida')
    ax.plot(x['t'][~ok], x['d'][~ok], 'x', ms=4, color='#EB6834', label='invalida (sem eco por 0,3 s)')
    ax.set_xlabel('t [s]'); ax.set_ylabel('distancia [m]'); ax.grid(alpha=.15); ax.legend(fontsize=9)
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA / f'sonar_{nome}.png', dpi=130)
    print(f"   figura: {SAIDA / f'sonar_{nome}.png'}")


def teste():
    """Sensor sintetico: vies +0,02 m, ruido 0,01 m, a 4 m metade das leituras some; sem caixa, nenhuma."""
    rng = np.random.default_rng(0)
    alvos, T, dt = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 0.0], 10.0, 0.0288
    t = np.arange(0, len(alvos) * T, dt)
    alvo = np.array(alvos)[(t // T).astype(int)]
    d = alvo + 0.02 + rng.normal(0, 0.01, len(t))
    ok = np.where(alvo == 4.0, rng.random(len(t)) < 0.5, alvo > 0)
    tab = {a: r for a, *r in tabela(dict(t=t, d=np.where(ok, d, 4.0), valid=ok.astype(float), alvo=alvo))}
    assert abs(tab[1.0][4] - 0.02) < 0.005 and abs(tab[1.0][3] - 0.01) < 0.003, tab[1.0]
    assert 40 < tab[4.0][1] < 60 and tab[0.0][1] == 0, (tab[4.0], tab[0.0])
    print('OK')


if __name__ == '__main__':
    if sys.argv[1:] == ['--teste']:
        teste()
    for c in map(Path, sys.argv[1:]):
        if c.name == '--teste':
            continue
        for p in ([c] if c.is_file() else sorted(c.rglob('sonar.csv'))):
            relatorio(p)
