# -*- coding: utf-8 -*-
"""Mede a parada do ensaio 'parada' (cruzeiro, corte u = -1, ate parar) e confere as constantes do supervisor ADAS.

    ../simulador/.venv/Scripts/python.exe analise/parada.py <pasta do log, ou car.csv, ou varias>    (de veiculo_real/)
    ../simulador/.venv/Scripts/python.exe analise/parada.py --teste

Por corrida: velocidade no corte v0, tempo e distancia ate parar (v < 0,05 m/s), T_REACAO medido (do corte ate v cair 5%
de v0: inclui o atraso da media movel do car.v, ~0,2 s), desaceleracao media depois disso e a distancia que o main.py
(A_FREIO, T_REACAO, D_PARADA) preveria. Distancia: coluna x do car.csv (odometro). Se medido > previsto, o ADAS freia tarde:
suba T_REACAO / baixe A_FREIO no main.py.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def mede(t, x, v, u, v_min_corte=0.9):
    """dict da parada que comeca no 1o u <= -0.99 com v >= v_min_corte; None se nao achar ou se o log termina antes de parar."""
    j = np.nonzero((u <= -0.99) & (v >= v_min_corte))[0]
    if not len(j):
        return None
    i = int(j[0])
    v0 = float(v[i])
    k = int(np.argmax(v[i:] < 0.05)) + i
    if v[k] >= 0.05:
        return None
    k95 = int(np.argmax(v[i:] < 0.95 * v0)) + i
    return dict(v0=v0, t_para=float(t[k] - t[i]), dist=float(x[k] - x[i]), T=float(t[k95] - t[i]),
                A=float((0.95 * v0 - 0.05) / (t[k] - t[k95])))


def teste():
    t = np.arange(0, 12, 0.03)
    v = np.where(t < 6.0, 1.3, np.clip(1.3 - 0.5 * (t - 6.6), 0, 1.3))   # 0,6 s parado no acelerador, depois 0,5 m/s2
    v = np.where((t >= 6.0) & (t < 6.6), 1.3, v)
    x = np.cumsum(v) * 0.03
    u = np.where(t >= 6.0, -1.0, 0.2)
    m = mede(t, x, v, u)
    assert abs(m['v0'] - 1.3) < 1e-9 and abs(m['A'] - 0.5) < 0.05 and 0.55 < m['T'] < 0.95, m
    assert abs(m['dist'] - (1.3 * 0.6 + 1.3 ** 2 / 1.0)) < 0.1, m   # v T + v^2/(2 A)
    assert mede(t[:200], x[:200], v[:200], u[:200]) is None            # log termina antes de parar
    print('OK')


def main(args):
    if args == ['--teste']:
        return teste()
    import analisa as A
    mod = A.carrega_main('parada')
    fim = []
    for c in map(Path, args):
        for p in ([c] if c.is_file() else sorted(c.rglob('car.csv')) or sorted(c.glob('*.csv'))):
            a = np.genfromtxt(p, delimiter=',', names=True)
            m = mede(a['t'], a['x'], a['v'], a['u'])
            nome = p.parent.name if p.name == 'car.csv' else p.name[:40]
            if m is None:
                print(f"{nome}: sem parada (nao ha u = -1 com v >= 0,9 ou o log acaba antes de parar)")
                continue
            prev = m['v0'] * mod.T_REACAO + m['v0'] ** 2 / (2 * mod.A_FREIO)
            print(f"{nome}: corte a {m['v0']:.2f} m/s  parou em {m['t_para']:.2f} s e {m['dist']:.2f} m | T medido {m['T']:.2f} s, "
                  f"desaceleracao {m['A']:.2f} m/s2 | main.py preve {prev:.2f} m ({'OK, conservador' if prev >= m['dist'] else 'FREIA TARDE: medido > previsto'})")
            fim.append(m)
    if len(fim) > 1:
        print(f"media de {len(fim)}: v0 {np.mean([m['v0'] for m in fim]):.2f}, {np.mean([m['dist'] for m in fim]):.2f} m, "
              f"T {np.mean([m['T'] for m in fim]):.2f} s, A {np.mean([m['A'] for m in fim]):.2f} m/s2  "
              f"(main.py: T_REACAO {mod.T_REACAO}, A_FREIO {mod.A_FREIO})")


if __name__ == '__main__':
    main(sys.argv[1:])
