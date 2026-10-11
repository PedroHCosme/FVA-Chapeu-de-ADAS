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
PLANTA_LENTA = dict(K=1.09, DB=3.95, tau=0.715, d=0.08)   # o que os logs de 25/09 mediram


def simula_log(ensaio, pasta, pl=VERDADE, escala_G=1.0, seed=0, v_ref=None, sem_guarda=False, subida=None):
    """Mesma ordem do carro: save_traj (linha) -> control_func -> set_u; planta anda com o u anterior.
    v_ref: troca o V_REF do main.py; sem_guarda: log de antes da guarda de VELMAX (ate 02/10)."""
    mod = A.carrega_main(ensaio)
    if v_ref is not None:
        mod.V_REF = v_ref
    if sem_guarda:
        mod.V_GUARDA = None
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
        if subida and subida[1] <= dist < subida[1] + subida[2]:   # disturbio cego: -a m/s2 na rampa (dist = posicao)
            vr = max(vr - subida[0] * h, 0.0)
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


def testa_guarda():
    """Guarda de VELMAX: com v <= V_GUARDA o u e o mesmo de sem guarda; acima, u = -1."""
    mod = A.carrega_main('avanco_ff')
    vg0 = mod.V_GUARDA
    t = np.arange(1, 400) * 0.0288
    v = np.clip(1.45 * np.sin(t / 4), 0, None)           # passa de 1,35 e volta, sem chegar a 1,5 (o set_u zeraria u)
    u_com, _ = A.repassa(mod, t, v)
    mod.V_GUARDA = None
    u_sem, _ = A.repassa(mod, t, v)
    mod.V_GUARDA = vg0
    acima = np.r_[False, v[:-1] > vg0]                   # o u da linha j+1 vem do v da linha j (repassa)
    ok = ~np.isnan(u_com)
    assert np.all(u_com[ok & acima] == -1.0), 'guarda nao age acima de V_GUARDA'
    assert np.allclose(u_com[ok & ~acima], u_sem[ok & ~acima]), 'guarda mexe no u abaixo de V_GUARDA'
    print(f'guarda: u = -1 em {int((ok & acima).sum())} amostras acima de {vg0}; abaixo, identico ao sem guarda')


def testa_log_sem_guarda(tmp):
    """Log de antes da guarda: V_REF 1,45 deixa o carro acima de V_GUARDA e o u logado nao tem a guarda.
    O reconhece tem que achar 'avanco' repassando SEM guarda."""
    pasta = tmp / '20990101_semguarda'
    simula_log('avanco', pasta, v_ref=1.45, sem_guarda=True)
    x = A.carrega(pasta)
    assert x['v'].max() > A.carrega_main('avanco').V_GUARDA, 'o teste precisa passar da guarda'
    nome, mod = A.reconhece(x)
    print(f"log sem guarda (v max {x['v'].max():.2f}): {nome}, guarda do repasse = {x.get('V_GUARDA')}")
    assert nome == 'avanco' and x['V_GUARDA'] is None, f'log sem guarda reconhecido como {nome}'
    return pasta


def testa_subida(tmp):
    """Subida a V_REF 1,3 (-0,8 m/s2 entre 5 e 15 m): o reconhece separa 'subida_ff' de 'avanco_ff' pela duracao,
    e a corrida fica fora do ajuste da planta."""
    pasta = tmp / '20990101_subida'
    dist, vmax, _ = simula_log('subida_ff', pasta, subida=(0.8, 5.0, 10.0))
    pista = A.carrega_main('subida_ff')._E['pista']
    x = A.carrega(pasta)
    nome, mod = A.reconhece(x)
    x['ensaio'], x['mod'] = nome, mod
    print(f"subida_ff: anda {dist:5.1f} m (pista {pista:.0f}), v medido max {x['v'].max():.2f}: {nome}")
    assert dist <= pista - 0.5, 'subida passa da pista'
    assert nome == 'subida_ff', f'subida reconhecida como {nome}'
    assert A.fora_do_ajuste(x), 'subida entrou no ajuste da planta'
    return pasta


def testa_parada(tmp):
    """'parada': cruzeiro ate t = 8 s e depois u = -1 com vref 0. O reconhece separa de 'avanco_rt' (os u divergem apos o corte),
    a corrida fica fora do ajuste da planta, cabe na pista e o carro termina quase parado."""
    pasta = tmp / '20990101_parada'
    dist, vmax, _ = simula_log('parada', pasta)
    mod = A.carrega_main('parada')
    x = A.carrega(pasta)
    nome, m = A.reconhece(x)
    x['ensaio'], x['mod'] = nome, m
    i = np.searchsorted(x['t'], x['t'][np.argmax(x['vref'] > 0)] + mod._E['parada'])   # a referencia (e o corte) parte da 1a chamada
    print(f"parada: v no corte {x['v'][i]:.2f}, v no fim {x['v'][-1]:.2f}, anda {dist:.1f} m (pista {mod.PISTA_M:.0f}): {nome}")
    assert x['v'][i] > 1.0 and x['v'][-1] < 0.3, 'a parada nao cortou de cruzeiro ate parar'
    assert (x['u'][i + 5:] == -1.0).all(), 'depois do corte u tem que ser -1'
    assert dist <= mod.PISTA_M - 0.5, 'a parada passa da pista'
    assert nome == 'parada' and A.fora_do_ajuste(x), f'parada reconhecida como {nome}'
    return pasta


def testa_organiza(pastas):
    """organiza_logs.rotula (o catalogo) acha os mesmos ensaios. O 'avanco_rt' so casa no formato novo; a subida
    so pela cobertura; o log sem guarda so repassando sem ela."""
    import organiza_logs as O
    for ens, pasta in pastas.items():
        r = O.rotula(pasta, '20990101_000000', 'real')
        print(f"organiza {ens}: ensaio_main {r['ensaio_main']}, controlador {r['controlador']}")
        assert r['ensaio_main'] == ens, f"organiza rotulou {ens} como {r['ensaio_main']}"


def main():
    testa_guarda()
    tmp = Path(tempfile.mkdtemp(prefix='analisa_'))
    pasta_sem_guarda = testa_log_sem_guarda(tmp)
    pasta_subida = testa_subida(tmp)
    pasta_parada = testa_parada(tmp)
    pista = A.carrega_main('avanco').PISTA_M
    corridas = [('patamares', 1.00), ('u_degrau', 0.97), ('avanco', 0.95), ('avanco_ff', 0.95), ('avanco_rt', 0.95)]
    pastas = []
    for i, (ens, esc) in enumerate(corridas):
        pasta = tmp / f'20990101_00{i}000'
        dist, vmax, th_fim = simula_log(ens, pasta, escala_G=esc, seed=i)
        print(f"{ens:10s}: anda {dist:5.1f} m (pista {pista:.0f}), v real max {vmax:.2f}, pwm ao fim {th_fim:.1f} graus")
        assert dist <= pista - 0.5, f'{ens} passa da pista'
        assert vmax <= 1.5, f'{ens} passa de VELMAX'
        pastas.append(pasta)
    # pior caso de pista: zona morta pequena e ganho alto fazem o carro andar mais
    for pl in (dict(K=2.0, DB=1.5, tau=0.23, d=0.14), dict(K=1.34, DB=3.4, tau=0.25, d=0.0)):
        for ens in ('u_degrau', 'patamares', 'avanco'):
            dist, vmax, th_fim = simula_log(ens, tmp / 'x', pl=pl)
            print(f"  planta K {pl['K']}, DB {pl['DB']}: {ens:9s} anda {dist:5.1f} m, v real max {vmax:.2f}")
            assert dist <= pista - 0.5 and vmax <= 1.5

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
    # referencia decrescente, com as duas leis
    for ens in ('frenagem', 'frenagem_ff', 'frenagem_rt'):
        pasta = tmp / f'20990101_{ens}'
        dist, vmax, th_fim = simula_log(ens, pasta, pl=PLANTA_LENTA)
        print(f"{ens:11s}: anda {dist:5.1f} m, v real max {vmax:.2f}")
        assert dist <= pista - 0.5 and vmax <= 1.5
        x = A.carrega(pasta)
        nome, mod = A.reconhece(x)
        assert nome == ens, f'{ens} reconhecido como {nome}'
        x['ensaio'], x['mod'] = nome, mod
        print('\n'.join(A.bloco_degraus(x, mod._E, mod, None)))
        assert x['metr']['iae'] > 0, 'a frenagem tem que entrar no RESUMO (regra da descida usa o IAE)'
    testa_organiza({'avanco': pasta_sem_guarda, 'subida_ff': pasta_subida, 'parada': pasta_parada, 'avanco_rt': pastas[4], 'u_degrau': pastas[1],
                    'frenagem_rt': tmp / '20990101_frenagem_rt'})
    print('OK')


if __name__ == '__main__':
    main()
