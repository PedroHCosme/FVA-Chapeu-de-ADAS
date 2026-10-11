# -*- coding: utf-8 -*-
"""Calibra o supervisor ADAS do main.py a partir do sonar.csv de um ensaio com a caixa PARADA em patamares (em qualquer ritmo).

    ../simulador/.venv/Scripts/python.exe analise/calibra_adas.py <pasta ou sonar.csv>                    mostra os valores
    ../simulador/.venv/Scripts/python.exe analise/calibra_adas.py <pasta ou sonar.csv> --verdades 0.3,0.9,1.7   e calibra o vies
    ../simulador/.venv/Scripts/python.exe analise/calibra_adas.py --teste                                 autochecagem

Rode de veiculo_real/. Nada e escrito: cole os valores impressos em SONAR_VIES, SONAR_RUIDO e SONAR_ALCANCE do main.py.
Patamar = trecho em que a leitura fica a menos de 3 cm da mediana por >= 3 s (o deslocamento da caixa nao entra).
  ruido   = desvio das leituras nos patamares com >= 90% de leituras validas
  alcance = maior patamar com >= 90% de leituras validas (so afirmo o que foi visto)
  vies    = media dos erros (mediana do patamar menos a distancia real medida com trena); so com --verdades, na ordem do tempo.
            Patamar a mais de 0,15 m da mediana dos erros e descartado (caixa mal posta)
Nao recusa: filtra e avisa. O que o ensaio nao der fica como esta no main.py.
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sonar as S


def platos(x, tol=0.03, dur=3.0):
    """Trechos estaveis do sonar.csv, qualquer que seja o horario em que a caixa foi movida:
    [(t0, t1, mediana, desvio, % validas)] com |d - mediana do trecho| < tol por >= dur s. O deslocamento da caixa
    (varredura entre dois patamares) nao entra."""
    t, d, ok = x['t'], x['d'], x['valid'] > 0.5
    out, i, n = [], 0, len(t)
    while i < n:
        j = i
        while j + 1 < n and abs(d[j + 1] - np.median(d[i:j + 1])) < tol:
            j += 1
        if t[j] - t[i] >= dur:
            out.append((float(t[i]), float(t[j]), float(np.median(d[i:j + 1])), float(np.std(d[i:j + 1])), 100 * float(ok[i:j + 1].mean())))
        i = j + 1
    return out


def calibra(x, verdades=None):
    """({vies, ruido, alcance}, avisos). verdades = distancias reais (trena) dos patamares com eco, em ordem de tempo; sem elas
    o vies NAO e calibrado (so ruido e alcance)."""
    pl = platos(x)
    v, avisos = {}, []
    boas = [p for p in pl if p[4] >= 90]
    sem_eco = [p for p in pl if p[4] < 10]
    if not boas:
        return {}, ['nenhum patamar estavel com eco (>= 3 s dentro de 3 cm): o sensor ficou solto ou a caixa nunca parou']
    v['ruido'] = round(max(float(np.sqrt(np.mean([p[3] ** 2 for p in boas]))), 0.005), 3)
    v['alcance'] = float(np.floor(10 * max(p[2] for p in boas)) / 10)
    if verdades is not None:
        if len(verdades) != len(boas):
            avisos.append(f"{len(verdades)} distancias reais para {len(boas)} patamares com eco: vies nao calibrado")
        else:
            err = np.array([p[2] - r for p, r in zip(boas, verdades)])
            med = float(np.median(err))
            bons = np.abs(err - med) <= 0.15
            for p, r, e, b in zip(boas, verdades, err, bons):
                if not b:
                    avisos.append(f"patamar de {p[2]:.3f} m (real {r:.2f} m) descartado: erro {e:+.3f} m, fora da mediana {med:+.3f}")
            v['vies'] = round(float(np.mean(err[bons])), 3)
    else:
        avisos.append('sem --verdades (distancias reais por patamar): SONAR_VIES nao e calibrado')
    if v['alcance'] < 4.0:
        avisos.append(f"o maior patamar foi {max(p[2] for p in boas):.2f} m: o alcance de 4 m do sensor NAO foi testado")
    if sem_eco:
        avisos.append(f"{len(sem_eco)} patamar(es) sem eco (caminho livre): " + ', '.join(f't = {p[0]:.0f}-{p[1]:.0f} s' for p in sem_eco))
    return v, avisos


def acha(caminho):
    c = Path(caminho)
    achados = [c] if c.is_file() else sorted(c.rglob('sonar.csv'))
    com_caixa = [p for p in achados if not np.isnan(S.carrega(p)['alvo']).all()]   # so corridas do ensaio 'sonar'
    if not com_caixa:
        raise SystemExit(f"nenhum sonar.csv do ensaio 'sonar' (com alvo) em {caminho}")
    return com_caixa[-1]                                                           # a mais recente


def teste():
    rng = np.random.default_rng(1)
    # caixa movida em horarios soltos, sensor limpo (desvio 1 mm), caixa real a 0,3 / 0,9 / 1,7 m e vies -0,07 m
    tt = np.arange(0, 70, 0.03)
    real = np.where(tt < 6, 0.3, np.where(tt < 9, 0.3 + (tt - 6) / 3 * 0.6, np.where(tt < 25, 0.9, np.where(tt < 28, 0.9 + (tt - 25) / 3 * 0.8, 1.7))))
    dd = real - 0.07 + rng.normal(0, 0.001, len(tt))
    xp = dict(t=tt, d=dd, valid=np.ones(len(tt)), alvo=np.full(len(tt), np.nan))
    assert len(platos(xp)) == 3, platos(xp)
    vp, avp = calibra(xp, [0.3, 0.9, 1.7])
    assert abs(vp['vies'] + 0.07) < 0.005 and vp['alcance'] == 1.6 and vp['ruido'] == 0.005, (vp, avp)
    vq, _ = calibra(xp)
    assert 'vies' not in vq and vq['alcance'] == 1.6
    assert calibra(dict(t=tt, d=np.linspace(0, 3, len(tt)), valid=np.ones(len(tt)), alvo=xp['alvo']))[0] == {}   # sem patamar
    print('OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('caminho', nargs='?', help="pasta (procura sonar.csv do ensaio 'sonar'; usa o mais recente) ou o sonar.csv")
    ap.add_argument('--verdades', help='distancias reais (trena) dos patamares com eco, em ordem de tempo: 0.3,0.9,1.7')
    ap.add_argument('--teste', action='store_true')
    a = ap.parse_args()
    if a.teste:
        return teste()
    if not a.caminho:
        ap.error('falta o caminho do sonar.csv')
    p = acha(a.caminho)
    x = S.carrega(p)
    for p0 in platos(x):
        print(f"  patamar t = {p0[0]:5.1f}-{p0[1]:5.1f} s  mediana {p0[2]:.3f} m  desvio {p0[3]:.4f} m  validas {p0[4]:.0f}%")
    v, avisos = calibra(x, [float(z) for z in a.verdades.split(',')] if a.verdades else None)
    for s in avisos:
        print(f'\nAVISO: {s}')
    if not v:
        print('\nO ensaio nao deu nenhum valor: o main.py fica como esta.')
        return
    import analisa as A                                   # as constantes de projeto do supervisor vem do main.py
    mod = A.carrega_main('avanco')
    print(f'\nlog {p.parent.name}: cole no main.py\n  ' + '\n  '.join(f"{n} = {v[k]:{f}} {u}" for n, k, f, u in
                                                              (('SONAR_VIES   ', 'vies', '+.3f', 'm'), ('SONAR_RUIDO  ', 'ruido', '.3f', 'm'),
                                                               ('SONAR_ALCANCE', 'alcance', '.1f', 'm')) if k in v))
    vc = mod._v_cruzeiro(v.get('alcance', mod.SONAR_ALCANCE))
    print(f"  => teto de cruzeiro V_CRUZ = {vc:.2f} m/s (A_FREIO {mod.A_FREIO}, T_REACAO {mod.T_REACAO} s, D_PARADA {mod.D_PARADA} m)"
          f"\n  => gatilho de emergencia a 1,0 m/s: {mod._d_gatilho(1.0):.2f} m; a 1,3 m/s: {mod._d_gatilho(1.3):.2f} m")
    if vc < 1.3:
        print(f"  (o teto {vc:.2f} m/s < V_REF 1,3: com ADAS_ON as rampas de 1,3 m/s ficam em {vc:.2f})")


if __name__ == '__main__':
    main()
