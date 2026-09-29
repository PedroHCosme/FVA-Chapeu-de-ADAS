# -*- coding: utf-8 -*-
"""Teste do main.py + analisa.py com logs SINTETICOS: roda o control_func real numa planta
conhecida (que nao e nenhum dos dois modelos antigos), grava car.csv como o carro grava e confere
que a analise (1) reconhece o ensaio, (2) recupera DB, tau, K, (3) todos cabem na pista.

    ../simulador/.venv/Scripts/python.exe analise/testa_analisa.py     (de veiculo_real/)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analisa as A

VERDADE = dict(K=1.5, DB=2.6, tau=0.27, d=0.06)   # planta sintetica (bateria cheia)


def simula_log(ensaio, pasta, pl=VERDADE, escala_G=1.0, seed=0):
    """Mesma ordem do carro: save_traj (linha) -> control_func -> set_u; planta anda com o u anterior."""
    mod = A.carrega_main(ensaio)
    A.zera(mod)
    rng = np.random.default_rng(seed)
    car, G = A.CarroFalso(), A.G_DE(pl['K']) * escala_G
    ts = mod._E['ts']
    t, th, vr, hist_t, hist_v, buf = 0.0, 0.0, 0.0, [0.0], [0.0], [0.0] * A.N_MA
    linhas, vref_ant, u_ant = [(0.01, 0.0, 0.0, 0.0)], 0.0, 0.0
    dist = vrmax = 0.0
    while t < ts:
        h = 1.0 if t == 0.0 else 0.0288 + abs(rng.normal(0, 0.003))
        t += h
        th = min(max(th + A.K_PWM * u_ant * h, 0.0), A.TH_MAX)
        vss = G * max(0.0, th - pl['DB'])
        vr = vss + (vr - vss) * np.exp(-h / pl['tau'])
        hist_t.append(t); hist_v.append(vr)
        buf = buf[1:] + [float(np.interp(t - pl['d'], hist_t, hist_v)) + rng.normal(0, 0.004)]
        car.t, car.dt, car.v = t, h, float(np.mean(buf))
        linhas.append((t, car.v, vref_ant, u_ant))
        mod.control_func(car)
        vref_ant, u_ant = car.vref, car.u
        dist += vr * h
        vrmax = max(vrmax, vr)
    Path(pasta).mkdir(parents=True, exist_ok=True)
    np.savetxt(Path(pasta) / 'car.csv', np.array(linhas), delimiter=',', header='t,v,vref,u', comments='')
    return dist, vrmax, th


def main():
    tmp = Path(tempfile.mkdtemp(prefix='analisa_'))
    corridas = [('patamares', 1.00), ('u_degrau', 0.97), ('base', 0.95), ('avanco', 0.95)]
    pastas = []
    for i, (ens, esc) in enumerate(corridas):
        pasta = tmp / f'20990101_00{i}000'
        dist, vmax, th_fim = simula_log(ens, pasta, escala_G=esc, seed=i)
        print(f"{ens:10s}: anda {dist:5.1f} m (pista {A.carrega_main('base').PISTA_M:.0f}), v real max {vmax:.2f}, pwm ao fim {th_fim:.1f} graus")
        assert dist <= A.carrega_main('base').PISTA_M - 0.5, f'{ens} passa da pista'
        assert vmax <= 1.5, f'{ens} passa de VELMAX'
        pastas.append(pasta)
    # pior cas de pista: zona morta pequena e ganho alto fazem o u_degrau andar mais
    for pl in (dict(K=2.0, DB=1.5, tau=0.23, d=0.14), dict(K=1.34, DB=3.4, tau=0.25, d=0.0)):
        for ens in ('u_degrau', 'patamares', 'base'):
            dist, vmax, th_fim = simula_log(ens, tmp / 'x', pl=pl)
            print(f"  planta K {pl['K']}, DB {pl['DB']}: {ens:9s} anda {dist:5.1f} m, v real max {vmax:.2f}")
            assert dist <= A.carrega_main('base').PISTA_M - 0.5 and vmax <= 1.5

    xs = [A.carrega(p) for p in pastas]
    for x, (ens, _) in zip(xs, corridas):
        nome, mod = A.reconhece(x)
        print(f"reconhece {ens}: {nome}")
        assert nome == ens, f'{ens} reconhecido como {nome}'
    aj = A.ajusta_conjunto(xs)
    Ks = [A.K_DE(g) for g in aj['G']]
    print(f"ajuste: DB {aj['DB']:.2f} (verdade {VERDADE['DB']})  tau {aj['tau']:.3f} ({VERDADE['tau']})  d {aj['d']:.3f} ({VERDADE['d']})"
          f"  K por corrida {np.round(Ks, 2)} (verdade {[round(VERDADE['K'] * c[1], 2) for c in corridas]})  rmse {aj['rmse']:.4f}")
    assert abs(aj['DB'] - VERDADE['DB']) < 0.4, 'DB nao recuperada'
    assert abs(aj['tau'] - VERDADE['tau']) < 0.05, 'tau nao recuperado'
    for k, (_, esc) in zip(Ks, corridas):
        assert abs(k / (VERDADE['K'] * esc) - 1) < 0.06, 'K por corrida nao recuperado'
    print('OK')


if __name__ == '__main__':
    main()
