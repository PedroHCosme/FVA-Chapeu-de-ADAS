# -*- coding: utf-8 -*-
"""Sintonia robusta de KP, TD, TAU_VDES e V_GUARDA do main.py
(projeto: docs/superpowers/specs/2026-10-02-sintonia-robusta-design.md).

    ../simulador/.venv/Scripts/python.exe analise/sintonia.py               varredura completa (~15 min, todos os nucleos)
    ../simulador/.venv/Scripts/python.exe analise/sintonia.py --rapido      grade grossa (confere o fluxo em ~1 min)
    ../simulador/.venv/Scripts/python.exe analise/sintonia.py --so-verifica so o ruido e a verificacao do simulador
    ../simulador/.venv/Scripts/python.exe analise/sintonia.py --teste       autochecagem (segundos)

Rode de veiculo_real/. Plantas P1/P2 = os dois ajustes de 01/10 (bateria > 88%).
Etapa 1: Ms da malha linearizada (MIGO: o maior ganho integral, 9,46*KP no acelerador, com Ms <= 1,4).
Etapa 2: o control_func do main.py numa planta nao linear: S1 (rampa), S2 (subida cega a 1,0 e 1,3 m/s; e 1,0 m/s2
para saturacao), S3 (referencia que desce); pior caso entre P1 e P2; cortes; fronteira de Pareto. VOCE escolhe o ponto.
Saidas: analise/saida/sintonia.txt e sintonia.png.
"""
import argparse, functools, itertools, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analisa as A
from acervo import caminho, linhas

TS = 0.0288                               # periodo medio do loop do carro [s]
PLANTAS = {'P1': dict(K=1.92, tau=0.615, DB=0.85, d=0.064),    # ajuste do dia, 01/10
           'P2': dict(K=2.86, tau=0.747, DB=0.85, d=0.064)}    # ajuste da corrida 211005 (K alto)
_PRE = A.carrega_main('avanco_ff')._PROJETOS['avanco_ff']
BASE = {k: _PRE[k] for k in ('KP', 'TD', 'TAU_VDES')}          # ponto de comparacao: o avanco_ff
FF = {k: _PRE[k] for k in ('K_FF', 'TAU_FF')}                  # fixo (spec, secao 2)
MS_MAX, TOL, RUIDO_MAX = 1.4, 1.02, 1.5
GRADE = dict(KP=np.round(np.arange(0.10, 0.5001, 0.025), 3), TD=np.round(np.arange(0.0, 1.2001, 0.1), 2),
             TAU_VDES=np.round(np.arange(0.2, 1.0001, 0.1), 2), V_GUARDA=(1.35, 1.375, 1.40))
GRADE_RAPIDA = dict(KP=(0.1, 0.2, 0.3, 0.4), TD=(0.0, 0.6, 1.2), TAU_VDES=(0.3, 0.5, 0.8), V_GUARDA=(1.35,))
SAIDA = A.SAIDA
X_SUBIDA, L_SUBIDA, A_SUBIDA, A_SAT = 5.0, 10.0, 0.8, 1.0      # subida: comeca a 5 m, 10 m de rampa [m/s2]
INF = float('inf')


def ms(KP, TD, K, tau, d, n_ma=A.N_MA, Ts=TS):
    """(Ms, MF em graus, MG em dB) da malha linearizada acima da zona morta.
    C = KP (1 + TD s / (1 + TD/5 s))  -- como o main.py (derivada do erro filtrada com TD/5)
    P = K/s (servos.py integra u) . 1/(tau s + 1) . atraso d . ZOH (Ts/2) . MA30 exata (car.v)"""
    w = np.logspace(-3, np.log10(np.pi / Ts), 6000)
    s = 1j * w
    C = KP * (1 + TD * s / (1 + TD / 5 * s))
    ma = (1 - np.exp(-s * n_ma * Ts)) / (n_ma * (1 - np.exp(-s * Ts)))
    L = C * K / s / (tau * s + 1) * np.exp(-s * (d + Ts / 2)) * ma
    fase = np.degrees(np.unwrap(np.angle(L)))
    ic, i180 = np.nonzero(np.abs(L) < 1)[0], np.nonzero(fase <= -180)[0]
    mf = 180 + fase[ic[0]] if len(ic) else float('nan')
    mg = -20 * np.log10(np.abs(L[i180[0]])) if len(i180) else float('inf')
    return float(np.max(np.abs(1 / (1 + L)))), float(mf), float(mg)


_MODS = {}


def _mod(ensaio):
    if ensaio not in _MODS:
        _MODS[ensaio] = A.carrega_main(ensaio)
    return _MODS[ensaio]


def simula(par, pl, cen, sig, seed=0, jitter=True):
    """Uma corrida: o control_func do main.py (ensaio cen['ensaio']) com os ganhos de 'par' (KP, TD, TAU_VDES, K_FF,
    TAU_FF, V_GUARDA) na planta 'pl'. Subida (cen['a_subida']): -a m/s2 enquanto X_SUBIDA <= x < X_SUBIDA + L_SUBIDA,
    x = posicao real (o controlador NAO sabe: disturbio cego). Retorna t, v (MA30), v_real, vref, u, pwm, x."""
    mod = _mod(cen['ensaio'])
    for k in ('KP', 'TD', 'TAU_VDES', 'K_FF', 'TAU_FF', 'V_GUARDA'):
        setattr(mod, k, par[k])
    if 'v_ref' in cen:
        mod.V_REF = cen['v_ref']
    A.zera(mod)
    planta = A.Planta(pl['K'], DB=pl['DB'], tau=pl['tau'], d=pl['d'], seed=seed, sig=sig)
    rng, car = np.random.default_rng(seed + 1000), A.CarroFalso()
    t = x = u = 0.0
    out = []
    while t < cen['ts']:
        h = TS + (abs(rng.normal(0, 0.003)) if jitter else 0.0)   # jitter do loop
        t += h
        a = -cen.get('a_subida', 0.0) if X_SUBIDA <= x < X_SUBIDA + L_SUBIDA else 0.0
        v = planta.passo(t, h, u, a)                                # a planta anda com o u anterior
        x += planta.vr * h
        car.t, car.dt, car.v = t, h, v
        mod.control_func(car)
        u = car.u                                                   # CarroFalso.set_u: u = 0 com |v| > 1,5
        out.append((t, v, planta.vr, car.vref, u, planta.pwm, x))
    return np.array(out).T


def m_s1(r):
    """Rampa 0 -> 1,3: IAE, sobressinal, pico, s travado em VELMAX, ruido de u (media |du|, ult. 3 s) -- como o analisa."""
    t, v, _, vref, u, _, _ = r
    fim = t > t[-1] - 3.0
    return dict(iae=float(np.trapezoid(np.abs(vref - v), t)), ultra=100 * (float(v.max()) / float(vref.max()) - 1),
                pico=float(v.max()), trava=float(np.sum((u == 0.0) & (v > 1.5)) * TS),
                ruido=float(np.mean(np.abs(np.diff(u[fim])))))


def m_s2(r):
    """Subida cega: IAE desde o pe da rampa, queda na subida, pico apos o topo, s acima de 1,5 apos o topo,
    erro final (ult. 2 s), trava permanente (ainda > 1,5 no fim), acelerador maximo, chegou ao topo?"""
    t, v, _, vref, u, pwm, x = r
    sub, dep = (x >= X_SUBIDA) & (x < X_SUBIDA + L_SUBIDA), x >= X_SUBIDA + L_SUBIDA
    i0 = int(np.argmax(x >= X_SUBIDA))
    alvo = float(vref[-1])
    return dict(iae=float(np.trapezoid(np.abs(vref[i0:] - v[i0:]), t[i0:])) if sub.any() else INF,
                queda=float(alvo - v[sub].min()) if sub.any() else INF,
                pico_topo=float(v[dep].max()) if dep.any() else INF,
                t_acima=float(np.sum((v > 1.5) & dep) * TS),
                erro_final=float(np.mean(v[t > t[-1] - 2.0]) - alvo),
                perm=bool(v[-1] > 1.5), th_max=float(pwm.max()), chegou=bool(dep.any()))


def m_s3(r):
    """Referencia que desce (perfil da 'frenagem': 1,0 -> 0,5 em t = 8 s a 0,9 m/s2 -> 0 em t = 13 s a 0,25 m/s2):
    IAE, quanto passa do patamar de 0,5 para baixo, s ate parar depois que vref chega a 0, pico, travado."""
    t, v, _, vref, u, _, _ = r
    t05, t00 = 8 + 0.5 / 0.9, 13 + 0.5 / 0.25                        # vref chega a 0,5 e a 0
    pl = (t >= t05) & (t < 13)
    para = np.nonzero((t > t00) & (v < 0.05))[0]
    return dict(iae=float(np.trapezoid(np.abs(vref - v), t)), abaixo=max(0.0, 0.5 - float(v[pl].min())),
                t_para=float(t[para[0]] - t00) if len(para) else INF, pico=float(v.max()),
                trava=float(np.sum((u == 0.0) & (v > 1.5)) * TS))


CENARIOS = {
    'S1':     dict(ensaio='avanco_ff', ts=13.0, v_ref=1.3, metr=m_s1),
    'S2_10':  dict(ensaio='avanco_ff', ts=30.0, v_ref=1.0, a_subida=A_SUBIDA, metr=m_s2),
    'S2_13':  dict(ensaio='avanco_ff', ts=30.0, v_ref=1.3, a_subida=A_SUBIDA, metr=m_s2),
    'S2_sat': dict(ensaio='avanco_ff', ts=30.0, v_ref=1.3, a_subida=A_SAT, metr=m_s2),
    'S3':     dict(ensaio='frenagem_ff', ts=20.0, metr=m_s3),
}
VERIF = ['20260930_211005', '20260930_212048', '20260930_212500', '20260930_213516', '20260930_214209']


def par_do_catalogo(id_):
    """Ganhos que rodaram na corrida id_ (formato antigo do main.py: kp, k_nom), sem guarda (logs de antes dela)."""
    r = linhas(id_carro=id_)[0]
    kn = float(r['k_nom'])
    return (dict(KP=float(r['kp']) / kn, TD=float(r['td_s']), TAU_VDES=float(r['tau_vdes_s']), K_FF=kn, TAU_FF=0.0,
                 V_GUARDA=None), float(r['v_ref_ms']))


def janela_real(id_, ts=13.0):
    """t, v, vref, u dos ts segundos a partir do inicio da rampa de referencia."""
    x = A.carrega(caminho(id_))
    k0 = int(np.argmax(x['vref'] > 0))
    t = x['t'] - x['t'][k0]
    w = (t >= 0) & (t <= ts)
    return t[w], x['v'][w], x['vref'][w], x['u'][w]


def calibra_sigma(id_='20260930_214209'):
    """sigma do ruido do sensor (antes da MA30) tal que o ruido de u simulado (media de |du|, ult. 3 s do S1, media
    entre P1 e P2) bata com o da corrida id_ (mesma lei que o 'avanco')."""
    t, v, vr, u = janela_real(id_)
    real = float(np.mean(np.abs(np.diff(u[t > t[-1] - 3.0]))))
    par, v_ref = par_do_catalogo(id_)
    cen = dict(CENARIOS['S1'], v_ref=v_ref)
    sim = {sig: float(np.mean([m_s1(simula(par, pl, cen, sig))['ruido'] for pl in PLANTAS.values()]))
           for sig in (0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016)}
    return min(sim, key=lambda s: abs(sim[s] - real)), real, sim


def verifica_simulador(sig):
    """Corridas de 01/10 com controlador conhecido: pico e IAE dos 13 s iniciais, real x faixa prevista por P1..P2.
    Nao e 100% cega (P1/P2 sairam desta sessao), mas o ajuste foi em malha aberta (u -> v): estas metricas de malha
    fechada nao entraram nele. Retorna (margem m [m/s], linhas do relatorio, avisos)."""
    L = [f"  {'corrida':16s} {'k_nom':>5s}  {'pico real':>9s}  {'previsto P1..P2':>15s}  {'IAE real':>8s}  {'previsto P1..P2':>15s}"]
    gaps, avisos = [], []
    for id_ in VERIF:
        par, v_ref = par_do_catalogo(id_)
        t, v, vr, u = janela_real(id_)
        pico, iae = float(v.max()), float(np.trapezoid(np.abs(vr - v), t))
        pr = [m_s1(simula(par, pl, dict(CENARIOS['S1'], v_ref=v_ref), sig)) for pl in PLANTAS.values()]
        p_lo, p_hi = min(p['pico'] for p in pr), max(p['pico'] for p in pr)
        i_lo, i_hi = min(p['iae'] for p in pr), max(p['iae'] for p in pr)
        gaps.append(max(0.0, pico - p_hi))
        L.append(f"  {id_:16s} {par['K_FF']:5.1f}  {pico:9.2f}  {p_lo:7.2f}..{p_hi:<6.2f}  {iae:8.2f}  {i_lo:7.2f}..{i_hi:<6.2f}")
        if not p_lo - 0.05 * v_ref <= pico <= p_hi + 0.05 * v_ref:
            avisos.append(f"{id_}: pico real {pico:.2f} fora de {p_lo:.2f}..{p_hi:.2f} por mais de 5 pontos de sobressinal")
        if not i_lo / 1.3 <= iae <= 1.3 * i_hi:
            avisos.append(f"{id_}: IAE real {iae:.2f} fora de {i_lo:.2f}..{i_hi:.2f} por mais de 30%")
    return max(gaps), L, avisos


PIOR = dict(iae=max, ultra=max, pico=max, trava=max, ruido=max, queda=max, pico_topo=max, t_acima=max, perm=max,
            th_max=max, chegou=min, abaixo=max, t_para=max, erro_final=lambda a, b: a if abs(a) >= abs(b) else b)


def avalia(par, sig, plantas=PLANTAS, cenarios=None):
    """{cenario: {metrica: pior caso entre as plantas}}."""
    res = {}
    for nc in cenarios or CENARIOS:
        cen = CENARIOS[nc]
        ms_ = [cen['metr'](simula(par, pl, cen, sig)) for pl in plantas.values()]
        res[nc] = {k: functools.reduce(PIOR[k], [m[k] for m in ms_]) for k in ms_[0]}
    return res


def js(r):
    """(J1, J2, J3) [m]: IAE da rampa, IAE da subida (1,0 + 1,3 m/s), IAE da descida -- pior caso."""
    return (r['S1']['iae'], r['S2_10']['iae'] + r['S2_13']['iae'], r['S3']['iae'])


def cortes(r, base, m):
    """Criterios 2 a 5 da spec (o 1, Ms, ja filtrou os pares). Lista do que falhou; vazia = passa.
    base = avanco_ff com a MESMA V_GUARDA; m = margem da verificacao do simulador [m/s]."""
    s1, s3, a, b, sat = r['S1'], r['S3'], r['S2_10'], r['S2_13'], r['S2_sat']
    f = []
    if s1['trava'] > 0 or s3['trava'] > 0 or s1['pico'] + m >= 1.5 or s3['pico'] + m >= 1.5:
        f.append('VELMAX em S1/S3')
    if not b['chegou'] or b['t_acima'] > 1.0 or abs(b['erro_final']) >= 0.02 * 1.3 or b['perm']:
        f.append('topo a 1,3')
    if not a['chegou'] or a['perm'] or sat['perm']:
        f.append('trava permanente (1,0 ou saturacao)')
    if sat['th_max'] >= A.TH_MAX - 0.1:
        f.append('acelerador no teto')
    if any(j > TOL * jb for j, jb in zip(js(r), js(base))):
        f.append('piora um objetivo')
    if s1['ruido'] > RUIDO_MAX * base['S1']['ruido']:
        f.append('ruido de u')
    return f


def pareto(itens):
    """Nao dominados em J (os tres a minimizar)."""
    # ponytail: O(n^2); trocar por ordenacao + varredura se os pontos viaveis passarem de alguns milhares
    dom = lambda b, a: all(x <= y for x, y in zip(b['J'], a['J'])) and b['J'] != a['J']
    return [a for a in itens if not any(dom(b, a) for b in itens)]


def ms_pior(kp, td):
    """(Ms, MF) no pior caso entre P1 e P2."""
    r = [ms(kp, td, pl['K'], pl['tau'], pl['d']) for pl in PLANTAS.values()]
    return max(x[0] for x in r), min(x[1] for x in r)


def etapa1(grade):
    """Pares (KP, TD) com Ms <= max(MS_MAX, Ms do avanco_ff) em P1 e P2: {(kp, td): (Ms, MF)}."""
    ms_base = ms_pior(BASE['KP'], BASE['TD'])[0]
    lim = max(MS_MAX, ms_base)
    todos = {(kp, td): ms_pior(kp, td) for kp, td in itertools.product(grade['KP'], grade['TD'])}
    return {k: v for k, v in todos.items() if v[0] <= lim}, ms_base, lim, len(todos)


def _avalia_ponto(args):
    kp, td, tv, vg, sig = args
    return args[:4], avalia(dict(KP=float(kp), TD=float(td), TAU_VDES=float(tv), V_GUARDA=vg, **FF), sig)


def etapa2(pares, grade, sig, procs):
    tarefas = [(kp, td, tv, vg, sig) for (kp, td) in pares for tv in grade['TAU_VDES'] for vg in grade['V_GUARDA']]
    res, t0 = {}, time.time()
    with Pool(procs) as pool:
        for i, (k, r) in enumerate(pool.imap_unordered(_avalia_ponto, tarefas, chunksize=4), 1):
            res[k] = r
            if i % max(1, len(tarefas) // 20) == 0:
                print(f"  {i}/{len(tarefas)} pontos ({time.time() - t0:.0f} s)", flush=True)
    return res


def figura(itens, sel, base_r, sig):
    """Pontos viaveis (J1 x J2, cor = J3) e respostas no tempo da fronteira contra o avanco_ff."""
    fig, ax = A.plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    if itens:
        J = np.array([a['J'] for a in itens])
        sc = ax[0, 0].scatter(J[:, 0], J[:, 1], c=J[:, 2], s=8, cmap='viridis')
        fig.colorbar(sc, ax=ax[0, 0], label='J3: IAE descida [m]')
    Jb = js(base_r)
    ax[0, 0].plot(Jb[0], Jb[1], 'k*', ms=14, label='avanco_ff')
    for i, a in enumerate(sel):
        ax[0, 0].annotate(f'#{i + 1}', a['J'][:2], fontsize=9, color='r')
    ax[0, 0].set(xlabel='J1: IAE rampa [m]', ylabel='J2: IAE subida 1,0 + 1,3 [m]', title='pontos que passam nos cortes (pior caso P1/P2)')
    ax[0, 0].legend()
    curvas = [('avanco_ff', dict(BASE, V_GUARDA=1.35, **FF), 'k')] + [
        (f'#{i + 1}', dict(KP=a['par'][0], TD=a['par'][1], TAU_VDES=a['par'][2], V_GUARDA=a['par'][3], **FF), None) for i, a in enumerate(sel)]
    for axx, (nc, pl, tit) in zip((ax[0, 1], ax[1, 0], ax[1, 1]), (('S1', 'P2', 'S1 rampa (P2, K alto)'),
                                                                   ('S2_13', 'P2', 'S2 subida a 1,3 m/s (P2)'),
                                                                   ('S3', 'P1', 'S3 referencia que desce (P1)'))):
        for nome, par, cor in curvas:
            r = simula(par, PLANTAS[pl], CENARIOS[nc], sig)
            axx.plot(r[0], r[1], color=cor, lw=2 if cor else 1, label=nome)
        axx.plot(r[0], r[3], 'k--', lw=0.8, label='vref')
        axx.axhline(1.5, color='r', ls=':', lw=0.8)
        axx.set(title=tit, xlabel='t [s]', ylabel='v medido (MA30) [m/s]')
        axx.grid(alpha=.3)
        axx.legend(fontsize=7)
    return fig


def relatorio(P, pares, res, m, sig):
    """Cortes, fronteira de Pareto (ate 8 pontos) e a linha pronta de cada ponto para o main.py."""
    vgs = sorted({k[3] for k in res})
    base = {vg: res[(BASE['KP'], BASE['TD'], BASE['TAU_VDES'], vg)] for vg in vgs}
    for vg, rb in base.items():
        f = cortes(rb, rb, m)
        J = js(rb)
        P(f"  avanco_ff com V_GUARDA {vg}: J1 {J[0]:.2f}  J2 {J[1]:.2f}  J3 {J[2]:.2f}" + (f"   (NAO passa: {', '.join(f)})" if f else ''))
    itens = [dict(par=k, J=js(r), r=r, Ms=pares[k[:2]][0], MF=pares[k[:2]][1]) for k, r in res.items() if not cortes(r, base[k[3]], m)]
    front = pareto(itens)
    P(f"  {len(itens)} de {len(res)} pontos passam em todos os cortes; {len(front)} na fronteira de Pareto")
    sel = sorted(front, key=lambda a: a['J'][1])
    if len(sel) > 8:
        sel = [sel[i] for i in np.unique(np.round(np.linspace(0, len(sel) - 1, 8)).astype(int))]
    pl_db = {k: dict(v, DB=2.5) for k, v in PLANTAS.items()}           # sensibilidade a zona morta (fora do Pareto)
    P("\n=== fronteira de Pareto (pior caso P1/P2; a 1a linha e o avanco_ff com V_GUARDA 1,35) ===")
    P(f"  {'':9s} {'KP':>5s} {'TD':>4s} {'TAUV':>4s} {'VGUAR':>5s} | {'J1':>6s} {'J2':>6s} {'J3':>6s} | {'queda':>5s} {'topo':>5s} {'>1,5':>4s} | "
      f"{'os%':>5s} {'osDB':>5s} | {'Ms':>4s} {'MF':>4s} | ruido u")
    b0 = (BASE['KP'], BASE['TD'], BASE['TAU_VDES'], vgs[0])
    tab = [('avanco_ff', b0, base[vgs[0]], pares[b0[:2]])] + [(f'#{i + 1}', a['par'], a['r'], (a['Ms'], a['MF'])) for i, a in enumerate(sel)]
    for nome, (kp, td, tv, vg), r, (Ms, MF) in tab:
        db = avalia(dict(KP=kp, TD=td, TAU_VDES=tv, V_GUARDA=vg, **FF), sig, pl_db, ['S1'])['S1']
        J, s2 = js(r), r['S2_13']
        P(f"  {nome:9s} {kp:5.3f} {td:4.1f} {tv:4.1f} {vg:5.3f} | {J[0]:6.2f} {J[1]:6.2f} {J[2]:6.2f} | {s2['queda']:5.2f} {s2['pico_topo']:5.2f} "
          f"{s2['t_acima']:4.1f} | {r['S1']['ultra']:5.1f} {db['ultra']:5.1f} | {Ms:4.2f} {MF:4.0f} | {r['S1']['ruido']:.4f}")
    P("  J1 IAE rampa, J2 IAE subida (1,0 + 1,3), J3 IAE descida [m] | a 1,3 m/s na subida: queda e pico apos o topo [m/s], s acima de 1,5")
    P("  | sobressinal da rampa % (DB 0,85 / DB 2,5: sensibilidade, fora do Pareto) | Ms, MF [graus] pior caso | ruido de u (media |du|)")
    if not sel:
        P("  FRONTEIRA VAZIA: nenhum ponto melhora sem piorar algo -> o avanco_ff ja e nao dominado nesta grade (spec, secao 8).")
    P("\n=== linhas prontas para o main.py (escolha UMA) ===")
    for nome, (kp, td, tv, vg), _, _ in tab[1:]:
        P(f"  {nome}: 'avanco_rt': dict(KP={kp:g}, TD={td:g}, TAU_VDES={tv:g}, K_FF={FF['K_FF']:g}, TAU_FF={FF['TAU_FF']:g}),   V_GUARDA = {vg:g}")
    return itens, sel, base[vgs[0]]


def teste():
    # a guarda (u igual ao sem guarda abaixo de V_GUARDA) e testada no testa_analisa.py (testa_guarda)
    m_, mf, _ = ms(1.0, 0.0, 1.0, 0.0, 0.0, n_ma=1, Ts=1e-6)       # L = 1/s: |S| < 1 sempre, MF 90
    assert abs(m_ - 1.0) < 1e-3 and abs(mf - 90.0) < 0.5, f'Ms de 1/s: {m_:.4f}, MF {mf:.1f} (esperado 1 e 90)'
    print(f'Ms de L = 1/s: {m_:.4f} (esperado 1), MF {mf:.1f} graus (90)')
    for nome, pl in PLANTAS.items():
        m_, mf, mg = ms(BASE['KP'], BASE['TD'], pl['K'], pl['tau'], pl['d'])
        print(f"avanco_ff em {nome}: Ms {m_:.2f}, MF {mf:.0f} graus, MG {mg:.1f} dB")
    par = dict(BASE, V_GUARDA=None, **FF)
    for nome, pl in PLANTAS.items():                                    # mesmo control_func, mesma planta, sem ruido
        r = simula(par, pl, CENARIOS['S1'], sig=0.0, jitter=False)
        meu = 100 * max(0.0, r[2].max() / 1.3 - 1)                      # v real, como o analisa.malha
        mod = A.carrega_main('avanco_ff')
        mod.V_GUARDA = None
        ref, _ = A.malha(mod, pl['K'], pl['DB'], pl['tau'], pl['d'], ts=13.0)
        print(f"S1 avanco_ff em {nome}: sobressinal {meu:.2f}% (analisa.malha {ref:.2f}%)")
        assert abs(meu - ref) < 1.0, 'o simulador da sintonia nao bate com o analisa.malha'
    r = simula(par, PLANTAS['P2'], CENARIOS['S2_13'], sig=0.0, jitter=False)
    m2 = m_s2(r)
    print(f"S2 a 1,3 sem guarda em P2: queda {m2['queda']:.2f}, pico apos o topo {m2['pico_topo']:.2f}, trava permanente {m2['perm']}")
    assert m2['chegou'] and m2['perm'], 'sem guarda o topo a 1,3 tem que travar (medido em 02/10: 1,82 m/s)'
    r = simula(dict(par, V_GUARDA=1.35), PLANTAS['P2'], CENARIOS['S2_13'], sig=0.0, jitter=False)
    assert not m_s2(r)['perm'], 'com guarda 1,35 o topo a 1,3 nao pode travar'
    print('OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--teste', action='store_true', help='autochecagem rapida')
    ap.add_argument('--so-verifica', action='store_true', help='so calibra o ruido e verifica o simulador')
    ap.add_argument('--rapido', action='store_true', help='grade grossa: confere o fluxo em ~1 min')
    ap.add_argument('--procs', type=int, default=max(1, (os.cpu_count() or 2) - 1), help='processos em paralelo')
    a = ap.parse_args()
    if a.teste:
        return teste()
    L = []

    def P(s=''):
        print(s, flush=True)
        L.append(s)
    sig, real, sim = calibra_sigma()
    P("=== ruido do sensor (calibrado) ===")
    P(f"  ruido de u real (20260930_214209, ult. 3 s): {real:.4f}; simulado por sigma: " + ', '.join(f'{s}: {v:.4f}' for s, v in sim.items()))
    P(f"  sigma escolhido: {sig}")
    m, Lv, avisos = verifica_simulador(sig)
    P("\n=== verificacao do simulador (13 s iniciais das corridas de 01/10) ===")
    for s in Lv:
        P(s)
    P(f"  margem m (pico real acima do maior previsto) = {m:.3f} m/s   <- somada aos picos nas checagens de VELMAX")
    for s in avisos:
        P(f"  AVISO: {s}  -> a ordem dos candidatos pode nao se repetir no carro")
    if a.so_verifica:
        return
    grade = GRADE_RAPIDA if a.rapido else GRADE
    pares, ms_base, lim, n = etapa1(grade)
    P(f"\n=== etapa 1 (frequencia) ===\n  {len(pares)} de {n} pares (KP, TD) com Ms <= {lim:.2f} em P1 e P2 (avanco_ff: Ms {ms_base:.2f})")
    for nome, pl in PLANTAS.items():
        P(f"  avanco_ff em {nome}: Ms {{:.2f}}, MF {{:.0f}} graus, MG {{:.1f}} dB".format(*ms(BASE['KP'], BASE['TD'], pl['K'], pl['tau'], pl['d'])))
    t0 = time.time()
    P("\n=== etapa 2 (tempo) ===")
    res = etapa2(pares, grade, sig, a.procs)
    P(f"  {len(res)} pontos simulados em {time.time() - t0:.0f} s")
    itens, sel, base_r = relatorio(P, pares, res, m, sig)
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / 'sintonia.txt').write_text('\n'.join(L), encoding='utf-8')
    figura(itens, sel, base_r, sig).savefig(SAIDA / 'sintonia.png', dpi=110)
    print(f"\nsaidas: {SAIDA / 'sintonia.txt'} e {SAIDA / 'sintonia.png'}")


if __name__ == '__main__':
    main()
