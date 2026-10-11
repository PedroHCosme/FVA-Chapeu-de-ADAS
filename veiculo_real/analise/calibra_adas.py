# -*- coding: utf-8 -*-
"""Calibra o supervisor ADAS do main.py a partir do ensaio 'sonar' (sonar.csv) e escreve os valores no main.py.

    ../simulador/.venv/Scripts/python.exe analise/calibra_adas.py ..\\ensaios\\_entrada            mostra o que mudaria
    ../simulador/.venv/Scripts/python.exe analise/calibra_adas.py <pasta ou sonar.csv> --aplica    reescreve o bloco ADAS
    ... --aplica --liga                                                                              e liga ADAS_ON
    ../simulador/.venv/Scripts/python.exe analise/calibra_adas.py --teste                           autochecagem

Rode de veiculo_real/. So o bloco entre '# >>> ADAS' e '# <<< ADAS' do main.py e reescrito (SONAR_VIES, SONAR_RUIDO,
SONAR_ALCANCE; ADAS_ON so com --liga). Entradas, todas do ensaio 'sonar' (caixa a 0,3 ... 4 m e depois nada):
  vies    = media dos erros (mediana - distancia real) das caixas de 0,5 a 2 m com >= 80% de leituras validas
  ruido   = desvio das leituras nessas mesmas caixas
  alcance = maior distancia da caixa tal que ela e todas as menores tem >= 90% validas e erro (menos o vies) < 0,10 m
Nao recusa: filtra e avisa. Caixa com erro a mais de 0,15 m da mediana e descartada; o que o ensaio nao der (ex.: nenhuma
caixa valida) fica como esta no main.py; alcance baixo e gravado mesmo assim (e o que o sensor sustenta). Cada filtro vira AVISO.
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sonar as S

MAIN = Path(__file__).resolve().parent.parent / 'main.py'
LINHAS = {   # linha do bloco -> comentario (o numero e trocado, o resto da linha e refeito)
    'SONAR_VIES': 'm -- leitura menos distancia real (mediana de 0,5 a 2 m)',
    'SONAR_RUIDO': 'm -- desvio das leituras',
    'SONAR_ALCANCE': 'm -- maior distancia com >= 90% de leituras validas e erro < 0,10 m',
}


def calibra(x):
    """({vies, ruido, alcance}, avisos) a partir do dicionario do sonar.csv; (None, motivos) se nao da para calibrar."""
    tab = [r for r in S.tabela(x) if r[0] > 0]
    nada = [r for r in S.tabela(x) if r[0] == 0]
    v, avisos = {}, []
    # vies e ruido: caixas com >= 80% validas (de preferencia 0,5 a 2 m); caixa com erro a mais de 0,15 m da mediana
    # e tratada como erro do ensaio (caixa mal posta, trena) e descartada
    boas = [r for r in tab if r[2] >= 80 and not np.isnan(r[5])]
    meio = [r for r in boas if 0.5 <= r[0] <= 2.0] or boas
    if meio:
        med = float(np.median([r[5] for r in meio]))
        for r in [r for r in meio if abs(r[5] - med) > 0.15]:
            avisos.append(f"caixa de {r[0]:.1f} m descartada: erro {r[5]:+.3f} m, {abs(r[5] - med):.2f} m fora da mediana ({med:+.3f})")
        meio = [r for r in meio if abs(r[5] - med) <= 0.15]
        v['vies'] = round(float(np.mean([r[5] for r in meio])), 3)
        v['ruido'] = round(max(float(np.sqrt(np.mean([r[4] ** 2 for r in meio]))), 0.005), 3)
        if len(meio) < 3:
            avisos.append(f"vies e ruido de so {len(meio)} caixa(s): estimativa fraca")
    else:
        avisos.append("nenhuma caixa com >= 80% de leituras validas: SONAR_VIES e SONAR_RUIDO ficam como estao")
    # alcance: anda pelas caixas em ordem; a primeira que falha (poucas validas ou erro grande) fecha o alcance
    alcance = 0.0
    for a, n, pv, med, sd, err in tab:
        if pv < 90 or np.isnan(err) or abs(err - v.get('vies', 0.0)) >= 0.10:
            break
        alcance = a
    if alcance > 0:
        v['alcance'] = float(alcance)
        if alcance < 1.0:
            avisos.append(f"alcance confiavel de so {alcance:.1f} m: o teto de cruzeiro fica baixo (gravado mesmo assim: e o que o sensor sustenta)")
        if tab and alcance == max(r[0] for r in tab):
            avisos.append(f"todas as caixas ate {alcance:.1f} m foram boas: o alcance real pode ser maior, mas so afirmo o que foi medido")
    else:
        avisos.append("nem a caixa mais proxima passou (>= 90% validas e erro < 0,10 m): SONAR_ALCANCE fica como esta")
    if nada and nada[0][2] > 10:
        avisos.append(f"com nada na frente {nada[0][2]:.0f}% das leituras sairam validas (eco fantasma): o supervisor so freia por eco, confira o corredor")
    return v, avisos


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


def calibra_platos(x, verdades=None):
    """Como calibra(), mas por patamares. verdades = distancias reais (trena) dos patamares com eco, em ordem de tempo; sem elas
    o vies NAO e calibrado (so ruido e alcance). Alcance = maior patamar com >= 90% de leituras validas (so afirmo o que foi visto)."""
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


def aplica(texto, v, liga, origem):
    """Texto do main.py com o bloco ADAS reescrito (os fins de linha do arquivo sao preservados)."""
    m = re.search(r'(# >>> ADAS[^\n]*\n)(.*?)(# <<< ADAS)', texto, re.S)
    if not m:
        raise SystemExit("o main.py nao tem o bloco '# >>> ADAS ... # <<< ADAS'")
    eol = '\r\n' if '\r\n' in texto else '\n'
    bloco = m.group(2)
    novos = {n: v[k] for n, k in (('SONAR_VIES', 'vies'), ('SONAR_RUIDO', 'ruido'), ('SONAR_ALCANCE', 'alcance')) if k in v}
    for nome, val in novos.items():   # o que o ensaio nao deu fica como esta
        bloco, n = re.subn(rf'(?m)^{nome}[ \t]*=[^\r\n]*', f"{nome:<15}= {val:<6.3f} # {LINHAS[nome]} [{origem}]", bloco)
        assert n == 1, f'{nome} nao esta no bloco ADAS'
    if liga:
        bloco, n = re.subn(r'(?m)^ADAS_ON[ \t]*=[^\r\n]*', 'ADAS_ON        = True   # True = supervisor ligado', bloco)
        assert n == 1, 'ADAS_ON nao esta no bloco ADAS'
    return texto[:m.start(2)] + bloco + texto[m.start(3):]


def acha(caminho):
    c = Path(caminho)
    achados = [c] if c.is_file() else sorted(c.rglob('sonar.csv'))
    com_caixa = [p for p in achados if not np.isnan(S.carrega(p)['alvo']).all()]   # so corridas do ensaio 'sonar'
    if not com_caixa:
        raise SystemExit(f"nenhum sonar.csv do ensaio 'sonar' (com alvo) em {caminho}")
    return com_caixa[-1]                                                           # a mais recente


def teste():
    rng = np.random.default_rng(1)
    alvos, T, dt = [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 0.0], 10.0, 0.0288
    t = np.arange(0, len(alvos) * T, dt)
    alvo = np.array(alvos)[(t // T).astype(int)]
    ok = np.where(alvo == 4.0, rng.random(len(t)) < 0.5, alvo > 0)
    d = alvo + 0.03 + rng.normal(0, 0.01, len(t))
    v, av = calibra(dict(t=t, d=np.where(ok, d, 4.0), valid=ok.astype(float), alvo=alvo))
    assert abs(v['vies'] - 0.03) < 0.005 and 0.005 <= v['ruido'] < 0.02 and v['alcance'] == 3.0, (v, av)
    # sensor pior: caixa a 3 m com 60% validas -> alcance 2 m
    ok2 = np.where(alvo == 3.0, rng.random(len(t)) < 0.6, np.where(alvo == 4.0, False, alvo > 0))
    v2, _ = calibra(dict(t=t, d=np.where(ok2, d, 4.0), valid=ok2.astype(float), alvo=alvo))
    assert v2['alcance'] == 2.0, v2
    # caixa mal posta (a 1 m, erro +0,5): descartada, o vies segue ~0,03 e o alcance idem
    d3 = np.where(alvo == 1.0, d + 0.5, d)
    v3, av3 = calibra(dict(t=t, d=np.where(ok, d3, 4.0), valid=ok.astype(float), alvo=alvo))
    assert abs(v3['vies'] - 0.03) < 0.005 and any('descartada' in s for s in av3), (v3, av3)
    # so as caixas de 0,3 e 0,5 m: calibra o que da, avisa a fraqueza, nao quebra
    k = t < 20
    v4, av4 = calibra(dict(t=t[k], d=d[k], valid=np.ones(k.sum()), alvo=alvo[k]))
    assert v4['alcance'] == 0.5 and 'vies' in v4 and av4, (v4, av4)
    # nada util (nenhuma leitura valida): dicionario sem chaves, o main.py nao muda
    v5, av5 = calibra(dict(t=t, d=np.full(len(t), 4.0), valid=np.zeros(len(t)), alvo=alvo))
    assert v5 == {} and av5, (v5, av5)
    # aplica num main.py de mentira: so o bloco muda, fim de linha CRLF preservado
    falso = ("x = 1\r\n# >>> ADAS (c)\r\nADAS_ON        = False  # a\r\nSONAR_VIES     = 0.00   # b\r\n"
             "SONAR_RUIDO    = 0.02   # c\r\nSONAR_ALCANCE  = 2.00   # d\r\n# <<< ADAS\r\ny = 2\r\n")
    novo = aplica(falso, v, True, 'teste')
    assert novo.startswith('x = 1\r\n') and novo.endswith('# <<< ADAS\r\ny = 2\r\n') and '\n' not in novo.replace('\r\n', '')
    assert 'ADAS_ON        = True' in novo and 'SONAR_ALCANCE  = 3.000' in novo and f"SONAR_VIES     = {v['vies']:.3f}" in novo
    assert aplica(falso, v, False, 'teste').count('ADAS_ON        = False') == 1
    print(f"calibra: {v}")
    # por patamares: caixa movida em horarios soltos, sensor limpo (desvio 1 mm), caixa real a 0,3 / 0,9 / 1,7 m e vies -0,07 m
    tt = np.arange(0, 70, 0.03)
    real = np.where(tt < 6, 0.3, np.where(tt < 9, 0.3 + (tt - 6) / 3 * 0.6, np.where(tt < 25, 0.9, np.where(tt < 28, 0.9 + (tt - 25) / 3 * 0.8, 1.7))))
    dd = real - 0.07 + rng.normal(0, 0.001, len(tt))
    xp = dict(t=tt, d=dd, valid=np.ones(len(tt)), alvo=np.full(len(tt), np.nan))
    assert len(platos(xp)) == 3, platos(xp)
    vp, avp = calibra_platos(xp, [0.3, 0.9, 1.7])
    assert abs(vp['vies'] + 0.07) < 0.005 and vp['alcance'] == 1.6 and vp['ruido'] == 0.005, (vp, avp)
    vq, _ = calibra_platos(xp)
    assert 'vies' not in vq and vq['alcance'] == 1.6
    assert calibra_platos(dict(t=tt, d=np.linspace(0, 3, len(tt)), valid=np.ones(len(tt)), alvo=xp['alvo']))[0] == {}   # sem patamar
    print('OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('caminho', nargs='?', help="pasta (procura sonar.csv do ensaio 'sonar'; usa o mais recente) ou o sonar.csv")
    ap.add_argument('--aplica', action='store_true', help='escreve no main.py (sem isso, so mostra)')
    ap.add_argument('--liga', action='store_true', help='com --aplica: tambem ADAS_ON = True')
    ap.add_argument('--platos', action='store_true', help='calibra por patamares estaveis (caixa movida fora do ritmo de 10 s); ignora o alvo')
    ap.add_argument('--verdades', help='com --platos: distancias reais (trena) dos patamares com eco, em ordem de tempo: 0.3,0.9,1.7')
    ap.add_argument('--main', default=str(MAIN), help='main.py a reescrever')
    ap.add_argument('--teste', action='store_true')
    a = ap.parse_args()
    if a.teste:
        return teste()
    if not a.caminho:
        ap.error('falta o caminho do sonar.csv')
    p = acha(a.caminho)
    if not a.platos:
        S.relatorio(p)
    x = S.carrega(p)
    if a.platos:
        for p0 in platos(x):
            print(f"  patamar t = {p0[0]:5.1f}-{p0[1]:5.1f} s  mediana {p0[2]:.3f} m  desvio {p0[3]:.4f} m  validas {p0[4]:.0f}%")
        v, avisos = calibra_platos(x, [float(z) for z in a.verdades.split(',')] if a.verdades else None)
    else:
        v, avisos = calibra(x)
    for s in avisos:
        print(f'\nAVISO: {s}')
    if not v:
        print('\nO ensaio nao deu nenhum valor: o main.py fica como esta.')
        return
    import analisa as A                                   # as constantes de projeto do supervisor vem do main.py
    mod = A.carrega_main('avanco')
    print('\n  ' + '\n  '.join(f"{n} = {v[k]:{f}} {u}" for n, k, f, u in
                               (('SONAR_VIES   ', 'vies', '+.3f', 'm'), ('SONAR_RUIDO  ', 'ruido', '.3f', 'm'),
                                ('SONAR_ALCANCE', 'alcance', '.1f', 'm')) if k in v))
    vc = mod._v_cruzeiro(v.get('alcance', mod.SONAR_ALCANCE))
    print(f"  => teto de cruzeiro V_CRUZ = {vc:.2f} m/s (A_FREIO {mod.A_FREIO}, T_REACAO {mod.T_REACAO} s, D_PARADA {mod.D_PARADA} m)"
          f"\n  => gatilho de emergencia a 1,0 m/s: {mod._d_gatilho(1.0):.2f} m; a 1,3 m/s: {mod._d_gatilho(1.3):.2f} m")
    if vc < 1.3:
        print(f"  (o teto {vc:.2f} m/s < V_REF 1,3: com ADAS_ON as rampas de 1,3 m/s ficam em {vc:.2f})")
    if not a.aplica:
        print('\n(a seco: nada foi gravado; rode com --aplica)')
        return
    mp = Path(a.main)
    texto = mp.read_text(encoding='utf-8', newline='')
    mp.write_text(aplica(texto, v, a.liga, f'{p.parent.name}'), encoding='utf-8', newline='')
    print(f"\nescrito em {mp}" + ('  (ADAS_ON = True)' if a.liga else '  (ADAS_ON continua como estava; --liga liga)'))


if __name__ == '__main__':
    main()
