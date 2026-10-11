# -*- coding: utf-8 -*-
"""Autoteste do supervisor ADAS do main.py na simulacao (planta P1/P2 da sintonia.py + ultrassom simulado a 10 Hz,
vies +0,02 m, ruido 0,01 m, sem eco alem de 4 m). Termina em OK.

    ../simulador/.venv/Scripts/python.exe analise/testa_adas.py          (de veiculo_real/)
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sintonia as T

A = T.A
VIES, RUIDO = 0.02, 0.01


def roda(pl, v_ref, parede, tira_em=None, ts=40.0, alcance=3.0, adas=True, v_parede=0.0, para_em=1e9, eco=None):
    """Carro rumo a uma parede em x = parede [m] (sai em t = tira_em; anda a v_parede [m/s] ate t = para_em: lider).
    Retorna dict com a folga final/minima, v max, o pico de v depois que o obstaculo sai, e as series."""
    mod = A.carrega_main()
    mod.V_REF, mod.ADAS_ON, mod.SONAR_VIES, mod.SONAR_RUIDO, mod.SONAR_ALCANCE = v_ref, adas, VIES, RUIDO, alcance
    mod.V_CRUZ = mod._v_cruzeiro(alcance)
    A.zera(mod)
    planta = A.Planta(pl['K'], DB=pl['DB'], tau=pl['tau'], d=pl['d'], sig=0.016)
    rng, car = np.random.default_rng(0), A.CarroFalso()
    t = x = u = 0.0
    t_leitura, d_lido, ult = -1.0, 4.0, -9.0
    log = []
    while t < ts:
        h = T.TS
        t += h
        v = planta.passo(t, h, u)
        x += planta.vr * h
        pos = parede + v_parede * min(t, para_em)
        d_real = pos - x if tira_em is None or t < tira_em else np.inf
        if eco and eco[0] <= t < eco[1]:                          # eco espurio: obstaculo fantasma a eco[2] m so nessa janela
            d_real = eco[2]
        if t - t_leitura >= 0.1:                                  # ultrassom a 10 Hz; fora de 4 m nao volta eco
            t_leitura = t
            if 0.02 < d_real <= 4.0:
                d_lido, ult = d_real + VIES + rng.normal(0, RUIDO), t
        valid = t - ult < 0.3
        car.t, car.dt, car.v = t, h, v
        mod._estado['sonar'] = (min(d_lido, 4.0) if valid else 4.0, valid)
        mod.control_func(car)
        u = car.u
        log.append((t, x, planta.vr, mod._estado.get('adas_cod', 0), pos))
    t_, x_, v_, c_, p_ = np.array(log).T
    pico = float(v_[t_ > tira_em].max()) if tira_em is not None else 0.0
    return dict(folga=p_[-1] - x_[-1], x_min=float(np.min(p_ - x_)), vmax=float(v_.max()), pico_pos=pico, x=x_, v=v_, c=c_, t=t_)


def teste():
    # 1. parede fixa: para antes dela em todas as velocidades e plantas
    for nome, pl in T.PLANTAS.items():
        for v_ref in (0.6, 1.0, 1.3):
            r = roda(pl, v_ref, parede=14.0)
            print(f"{nome} v_ref {v_ref}: vmax {r['vmax']:.2f}, parou a {r['folga']:.2f} m da parede (minimo {r['x_min']:.2f} m)")
            assert r['x_min'] > 0.15, f'{nome} v_ref {v_ref}: chegou a {r["x_min"]:.2f} m da parede'
            assert r['v'][-1] < 0.02, f'{nome} v_ref {v_ref}: nao parou'
            assert r['folga'] < 3.0, f'{nome} v_ref {v_ref}: parou longe demais ({r["folga"]:.1f} m)'
    # 2. pista livre: o supervisor nao interfere (o teto de cruzeiro com alcance de 4 m e 1,43 m/s: nao corta 1,3)
    r = roda(T.PLANTAS['P2'], 1.3, parede=1e9, ts=15.0, alcance=4.0)
    assert (r['c'] == 0).all() and r['vmax'] > 1.2, 'o supervisor atuou em pista livre'
    print(f"pista livre: vmax {r['vmax']:.2f}, supervisor inativo")
    # 3. obstaculo que sai (parede some em t = 12 s, carro parado antes): volta a andar
    r = roda(T.PLANTAS['P1'], 1.0, parede=12.0, tira_em=16.0, ts=40.0)
    assert r['v'][-1] > 0.5, 'nao voltou a andar quando o obstaculo saiu'
    print(f"obstaculo saiu: voltou a {r['v'][-1]:.2f} m/s")
    # 4. lider que anda a 0,6 m/s (parte a 8 m): o carro (v_ref 1,3) o segue, sem encostar; distancia > D_PARADA
    for nome, pl in T.PLANTAS.items():
        r = roda(pl, 1.3, parede=8.0, v_parede=0.6, ts=60.0)
        gap_fim = r['folga']
        print(f"{nome} seguindo lider a 0,6 m/s: v final {r['v'][-1]:.2f}, distancia final {gap_fim:.2f} m, minima {r['x_min']:.2f} m")
        assert r['x_min'] > 0.3 and 0.35 < r['v'][-1] < 0.8, f'{nome}: nao seguiu o lider (v {r["v"][-1]:.2f}, minima {r["x_min"]:.2f})'
    # 5. lider que anda e depois para (t = 25 s): o carro tambem para, sem bater
    r = roda(T.PLANTAS['P1'], 1.3, parede=8.0, v_parede=0.6, para_em=25.0, ts=60.0)
    print(f"lider que para: v final {r['v'][-1]:.2f}, distancia final {r['folga']:.2f} m, minima {r['x_min']:.2f} m")
    assert r['x_min'] > 0.3 and r['v'][-1] < 0.02, 'lider parou e o carro nao parou'
    # 6. eco espurio a 1,5 m por 0,6 s em cruzeiro: o carro freia e volta a andar
    for nome, pl in T.PLANTAS.items():
        r = roda(pl, 1.0, parede=1e9, ts=30.0, alcance=4.0, eco=(12.0, 12.6, 1.5))
        i = np.argmax(r['t'] > 12.0)
        j = i + np.argmax(r['v'][i:] < 0.5)
        volta = r['t'][j + np.argmax(r['v'][j:] > 0.9)] - 12.6
        print(f"{nome} eco falso: v minima {r['v'][i:].min():.2f}, volta a 0,9 m/s {volta:.1f} s depois, v final {r['v'][-1]:.2f}")
        assert r['v'][-1] > 0.9, f'{nome}: nao voltou a andar depois do eco espurio'
    # 7. sem ADAS ele bate (o teste 1 so vale se a parede estava no caminho)
    r = roda(T.PLANTAS['P1'], 1.3, parede=10.0, ts=12.0, adas=False)
    assert r['x_min'] < 0.0, 'sem ADAS o carro deveria passar da parede'
    print('OK')


if __name__ == '__main__':
    teste()
