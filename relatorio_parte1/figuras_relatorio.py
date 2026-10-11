# -*- coding: utf-8 -*-
"""Gera TODAS as figuras e tabelas do relatorio (Parte 1, ADAS longitudinal) a partir dos logs em logs/.

Uso (nesta pasta):   python figuras_relatorio.py
Saida: imgs/*.pdf, tab_aeb.tex, numeros.tex (macros \\num... usadas no texto) e numeros.txt.
Os logs sao copias de simulador/logs/<id>/car.csv (CoppeliaSim) e veiculo_real/adas/roxo/<id>/{car,sonar}.csv (carro real).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(AQUI, 'logs', 'sim')
REAL = os.path.join(AQUI, 'logs', 'real')
IMG = os.path.join(AQUI, 'imgs')

plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': 0.3, 'axes.titlesize': 8,
                     'legend.fontsize': 7, 'lines.linewidth': 1.2, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
AZUL, VERM, VERDE, CINZA = '#1f5fa8', '#c0392b', '#2e8b57', '#777777'

num = {}   # nome -> string, vira numeros.tex


def csv(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))


def simcar(k):
    return csv(os.path.join(SIM, k, 'car.csv'))


def realcar(k):
    return csv(os.path.join(REAL, k, 'car.csv'))


def realsonar(k):
    return csv(os.path.join(REAL, k, 'sonar.csv'))


def vir(x, nd=2):
    return f'{x:.{nd}f}'.replace('.', ',')


def macro(nome, valor):
    num[nome] = valor


# ---------------------------------------------------------------- modelo identificado (IDENTIFICACAO.md)
K_, C_, A0_ = 1.00, 0.00425, 0.096


def simula(t, u, v0):
    v = np.empty_like(t)
    v[0] = v0
    for i in range(len(t) - 1):
        vd = K_ * u[i] - C_ * v[i] * abs(v[i]) - A0_ * np.tanh(v[i] / 0.05)
        v[i + 1] = v[i] + vd * (t[i + 1] - t[i])
    return v


# ================================================================ 1) on-off (simulador)
def fig_onoff():
    d = simcar('20260915_200353')
    t, v, u = d['t'], d['v'], d['u']
    i = np.where(np.diff(u) != 0)[0] + 1
    lig = [k for k in i if u[k] == 1 and t[k] > 4]
    des = [k for k in i if u[k] == 0 and t[k] > 4]
    v_on, v_off = np.mean(v[lig]), np.mean(v[des])
    T = np.mean(np.diff(t[lig]))
    sel = t > 4.0
    macro('OnOffVon', vir(v_on, 3)); macro('OnOffVoff', vir(v_off, 3)); macro('OnOffT', vir(T, 2))
    macro('OnOffVmin', vir(v[sel].min(), 2)); macro('OnOffVmax', vir(v[sel].max(), 2))
    macro('OnOffTrocas', str(len(i)))
    acel = (v[i[1]] - v[i[0]]) / (t[i[1]] - t[i[0]])            # subida inicial, u = 1
    macro('OnOffAcel', vir(acel, 2))
    macro('OnOffAtraso', vir((v_off - 2.5) / acel, 2))          # ultrapasse / aceleracao = atraso efetivo
    fig, ax = plt.subplots(2, 1, figsize=(3.3, 2.6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax[0].plot(t, v, color=AZUL)
    ax[0].axhline(2.5, color=CINZA, ls='--', lw=0.8)
    ax[0].set_ylabel('$v$ [m/s]')
    ax[1].step(t, u, where='post', color=VERM)
    ax[1].set_ylabel('$u$'); ax[1].set_xlabel('$t$ [s]'); ax[1].set_ylim(-0.1, 1.1)
    # ampliacao dentro do painel de v, na regiao vazia abaixo do patamar (t > 5 s, v < 1,5 m/s)
    ax2 = ax[0].inset_axes([0.50, 0.10, 0.47, 0.42])
    z = (t > 6.0) & (t < 8.5)
    ax2.plot(t[z], v[z], color=AZUL, lw=1.0); ax2.axhline(2.5, color=CINZA, ls='--', lw=0.8)
    ax2.tick_params(labelsize=5.5, pad=1); ax2.grid(False)
    ax2.set_title('detalhe do ciclo-limite', fontsize=5.5, pad=1)
    fig.savefig(os.path.join(IMG, 'onoff.pdf')); plt.close(fig)


# ================================================================ 2) identificacao: previsao em malha aberta
def fig_ident():
    logs = ['20260915_200353', '20260915_200647', '20260915_200750']
    rot = ['on-off 2,5 m/s', '$u$=1 depois 0', '$u$=1 depois 0 (outro)']
    fig, ax = plt.subplots(1, 3, figsize=(6.7, 1.9), sharey=False)
    rmss, pcs = [], []
    for a, k, r in zip(ax, logs, rot):
        d = simcar(k)
        m = d['t'] <= 8.0
        t, v, u = d['t'][m], d['v'][m], d['u'][m]
        vm = simula(t, u, v[0])
        e = vm - v
        rmss.append(np.sqrt(np.mean(e ** 2)))
        pcs.append(100 * np.abs(e).max() / v.max())
        a.plot(t, v, color=AZUL, label='medido')
        a.plot(t, vm, color=VERM, ls='--', label='modelo')
        a.set_title(f'{r}: rms = {vir(rmss[-1], 3)} m/s', fontsize=7)
        a.set_xlabel('$t$ [s]')
    ax[0].set_ylabel('$v$ [m/s]'); ax[0].legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'ident.pdf')); plt.close(fig)
    macro('IdentRmsMax', vir(max(rmss), 3)); macro('IdentRmsMin', vir(min(rmss), 3))
    macro('IdentErrMaxPc', vir(max(pcs), 1))


# ================================================================ 3) controle PI+FF no simulador (plano + rampa)
def fig_sim_controle():
    d = simcar('20260923_173603')
    t, v, u = d['t'], d['v'], d['u']
    vdes = np.minimum(1.0, 0.9 * t)
    pre = (t > 1) & (t < 6)
    sobre = 100 * (v[pre].max() - 1.0)
    # rampa: primeiro instante, depois de 5 s, em que v cai abaixo de 0,97
    k0 = np.argmax((t > 5) & (v < 0.97))
    kmin = k0 + np.argmin(v[k0:k0 + 60])
    k98 = kmin + np.argmax(v[kmin:] >= 0.98)
    macro('SimOvershoot', vir(max(sobre, 0.0), 2)); macro('SimDipVmin', vir(v[kmin], 2))
    macro('SimTRec', vir(t[k98] - t[k0], 2)); macro('SimTdip', vir(t[k0], 1))
    # chacoalho de u antes/depois do filtro TAU_U (mesma metrica da documentacao: std(diff(u)), t >= 8 s)
    ant, dep = simcar('20260922_234940'), simcar('20260923_090619')
    ch = lambda dd: np.std(np.diff(dd['u'][dd['t'] >= 8.0]))
    macro('SimChAntes', vir(ch(ant), 4)); macro('SimChDepois', vir(ch(dep), 4))
    macro('SimChRed', vir(100 * (1 - ch(dep) / ch(ant)), 0))
    fig, ax = plt.subplots(1, 3, figsize=(6.7, 2.0))
    ax[0].plot(t, v, color=AZUL, label='$v$'); ax[0].plot(t, vdes, color=CINZA, ls='--', label='$v_{des}$ (rampa)')
    ax[0].axvspan(t[k0], t[k98], color='orange', alpha=0.2, label='subida da rampa')
    ax[0].set_xlabel('$t$ [s]'); ax[0].set_ylabel('$v$ [m/s]'); ax[0].legend(loc='lower right')
    ax[1].plot(t, u, color=VERM); ax[1].set_xlabel('$t$ [s]'); ax[1].set_ylabel('$u$'); ax[1].set_title('comando (rampa em laranja)')
    ax[1].axvspan(t[k0], t[k98], color='orange', alpha=0.2)
    for dd, c, r in ((ant, CINZA, 'sem filtro em $u$'), (dep, VERM, '$\\tau_u$ = 0,1 s')):
        m = (dd['t'] >= 8) & (dd['t'] <= 12.5)
        ax[2].plot(dd['t'][m], dd['u'][m], color=c, label=r, lw=0.9)
    ax[2].set_xlabel('$t$ [s]'); ax[2].set_title('chacoalho de $u$ após a rampa'); ax[2].legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'sim_controle.pdf')); plt.close(fig)


# ================================================================ 4) carro real: cruzeiro + corte (planta e parada)
# Controladores do carro real: C1 = preset 'avanco_ff' do main.py, C2 = preset 'avanco_rt'. A lei de cada log vem do repasse do
# main.py sobre o log (analise/analisa.py, reconhece(); nas corridas com ADAS, so a janela antes de o supervisor agir).
# O log 231136 e uma corrida com ADAS (faixas coloridas).
CRUZEIRO = [('20260930_223649', '$C_2$'), ('20260930_231136', '$C_1$ + ADAS')]
LEI_C1 = {'231012', '231108', '231136'}      # as demais corridas com AEB rodaram C2
ENS = {}                                    # log -> 'E1'... (ordem do log), preenchido em tab_aeb()


def sombreia(a, k):
    s = realsonar(k)
    for cod, cor in ((1, '#ffe08a'), (2, '#f5a3a3')):
        m = (s['adas'] == cod).astype(int)
        for i0, i1 in zip(np.where(np.diff(np.r_[0, m, 0]) == 1)[0], np.where(np.diff(np.r_[0, m, 0]) == -1)[0]):
            a.axvspan(s['t'][i0], s['t'][min(i1, len(s['t']) - 1)], color=cor, alpha=0.6, lw=0)


def fig_real_cruzeiro():
    n = len(CRUZEIRO)
    fig, ax = plt.subplots(2, n, figsize=(4.8, 2.6), sharex='col', gridspec_kw={'height_ratios': [2, 1]}, squeeze=False)
    for j, (k, lei) in enumerate(CRUZEIRO):
        d = realcar(k)
        t, v, u, vr = d['t'], d['v'], d['u'], d['vref']
        if os.path.exists(os.path.join(REAL, k, 'sonar.csv')):
            sombreia(ax[0, j], k); sombreia(ax[1, j], k)
        ax[0, j].plot(t, vr, color=CINZA, ls='--', label='$v_{ref}$'); ax[0, j].plot(t, v, color=AZUL, label='$v$')
        ax[0, j].set_title(f'{lei} (log {k[9:]})'); ax[1, j].plot(t, u, color=VERM)
        ax[1, j].set_xlabel('$t$ [s]')
        ref = vr.max()
        k95 = np.argmax(v >= 0.95 * ref)
        c = 'AB'[j]   # nome de macro so aceita letras
        macro(f'Cz{c}Tempo', vir(t[k95], 1))
        macro(f'Cz{c}Sobre', vir(max(0.0, 100 * (v.max() - ref) / ref), 1))
        macro(f'Cz{c}Vmax', vir(v.max(), 2))
    ax[0, 0].set_ylabel('$v$ [m/s]'); ax[1, 0].set_ylabel('$u$'); ax[0, 0].legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'real_cruzeiro.pdf')); plt.close(fig)


def fig_real_parada():
    k = '20260930_224401'
    d = realcar(k)
    t, v, u, x, vr = d['t'], d['v'], d['u'], d['x'], d['vref']
    kc = np.where((vr[1:] < 0.01) & (vr[:-1] >= 0.01))[0][0] + 1
    tc = t[kc]
    kp = kc + np.argmax(v[kc:] < 0.05)
    macro('ParV', vir(np.interp(tc, t, v), 2)); macro('ParDist', vir(x[kp] - x[kc], 2))
    macro('ParTempo', vir(t[kp] - tc, 1))
    fig, ax = plt.subplots(2, 1, figsize=(3.3, 2.6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax[0].plot(t, vr, color=CINZA, ls='--', label='$v_{ref}$'); ax[0].plot(t, v, color=AZUL, label='$v$')
    ax[0].axvline(tc, color=VERM, lw=0.8); ax[0].set_ylabel('$v$ [m/s]'); ax[0].legend(loc='upper right')
    ax[1].plot(t, u, color=VERM); ax[1].set_ylabel('$u$'); ax[1].set_xlabel('$t$ [s]')
    fig.savefig(os.path.join(IMG, 'real_parada.pdf')); plt.close(fig)


# ================================================================ 5) AEB: eventos, tabela, figuras
T_REACAO, A_FREIO, D_PARADA = 0.7, 0.5, 0.1      # os valores de veiculo_real/main.py


def d_gat(v):
    return v * T_REACAO + v * v / (2.0 * A_FREIO) + D_PARADA


def eventos(k):
    c, s = realcar(k), realsonar(k)
    ad, ts = s['adas'], s['t']
    val = s['valid'] > 0
    ini = np.where((ad[1:] == 2) & (ad[:-1] != 2))[0] + 1
    out = []
    for i in ini:
        j = i
        while j < len(ad) and ad[j] == 2:
            j += 1
        t0, t1 = ts[i], ts[min(j, len(ts) - 1)]
        v0 = float(np.interp(t0, c['t'], c['v']))
        m = (c['t'] > t0) & (c['v'] < 0.05)
        tp = float(c['t'][m][0]) if m.any() else np.nan
        fim = tp if np.isfinite(tp) else min(t1, t0 + 4.0)
        w2 = val & (ts >= t0) & (ts <= fim)
        dmin = float(s['d'][w2].min()) if w2.any() else np.nan
        out.append(dict(run=k, v0=v0, d0=float(s['d'][i]), dg=d_gat(v0), dt=tp - t0, dmin=dmin))
    return out


def tab_aeb():
    runs = sorted(r for r in os.listdir(REAL)
                  if os.path.exists(os.path.join(REAL, r, 'sonar.csv')) and (realsonar(r)['adas'] == 2).any())
    todos = {r: eventos(r) for r in runs}
    ev = [todos[r][0] for r in runs]                       # 1o corte de cada corrida (o corte a partir do cruzeiro)
    f = lambda x, nd=2: '--' if not np.isfinite(x) else vir(x, nd)
    ENS.update({r: f'E{i + 1}' for i, r in enumerate(runs)})
    L = [f"{ENS[e['run']]} & {e['run'][9:]} & {'$C_1$' if e['run'][9:] in LEI_C1 else '$C_2$'} & {f(e['v0'])} & {f(e['d0'])} & {f(e['dg'])} & {f(e['dt'], 1)} & {f(e['dmin'] if np.isfinite(e['dt']) else np.nan)} & {len(todos[e['run']])}"
         for e in ev]
    # sem '\\' na ultima linha: o texto fecha a tabela (evita \bottomrule logo apos o fim do \input)
    open(os.path.join(AQUI, 'tab_aeb.tex'), 'w', encoding='utf-8').write(' \\\\\n'.join(L) + '\n')
    dt = np.array([e['dt'] for e in ev]); dm = np.array([e['dmin'] for e in ev]); v0 = np.array([e['v0'] for e in ev])
    ok = np.isfinite(dt)
    macro('AebCorridas', str(len(runs))); macro('AebAcion', str(sum(len(v) for v in todos.values())))
    macro('AebParou', str(int(ok.sum())))
    macro('AebDtMin', vir(dt[ok].min(), 1)); macro('AebDtMax', vir(dt[ok].max(), 1)); macro('AebDtMed', vir(np.median(dt[ok]), 1))
    macro('AebDmMin', vir(dm[ok].min(), 2)); macro('AebDmMax', vir(dm[ok].max(), 2)); macro('AebDmMed', vir(np.median(dm[ok]), 2))
    macro('AebVmin', vir(v0.min(), 2)); macro('AebVmax', vir(v0.max(), 2))


def fig_aeb():
    runs = ['20260930_230029', '20260930_231242']
    fig, ax = plt.subplots(2, 2, figsize=(6.7, 3.3), sharex='col')
    for col, k in enumerate(runs):
        c, s = realcar(k), realsonar(k)
        val = s['valid'] > 0
        for a in ax[:, col]:
            sombreia(a, k)
        ax[0, col].plot(c['t'], c['vref'], color=CINZA, ls='--', label='$v_{ref}$')
        ax[0, col].plot(c['t'], c['v'], color=AZUL, label='$v$')
        ax[0, col].set_title(f'{ENS[k]} (log {k[9:]})')
        ax[1, col].plot(s['t'][val], s['d'][val], '.', ms=2, color=VERDE, label='sonar $d$')
        ax[1, col].plot(s['t'], d_gat(np.clip(np.interp(s['t'], c['t'], c['v']), 0, None)), color=VERM, lw=0.9,
                        label='$d_{gat}(v)$')
        ax[1, col].set_ylim(0, 4.2); ax[1, col].set_xlabel('$t$ [s]')
    ax[0, 0].set_ylabel('$v$ [m/s]'); ax[1, 0].set_ylabel('$d$ [m]')
    ax[0, 0].legend(loc='upper right'); ax[1, 0].legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'aeb.pdf')); plt.close(fig)


def fig_aeb_demo():
    k = '20260930_231739'
    c, s = realcar(k), realsonar(k)
    val = s['valid'] > 0
    ad = s['adas']
    fig, ax = plt.subplots(2, 1, figsize=(6.7, 2.7), sharex=True)
    for a in ax:
        sombreia(a, k)
    ax[0].plot(c['t'], c['vref'], color=CINZA, ls='--', label='$v_{ref}$'); ax[0].plot(c['t'], c['v'], color=AZUL, label='$v$')
    ax[0].set_ylabel('$v$ [m/s]'); ax[0].legend(loc='upper right', ncol=2)
    ax[1].plot(s['t'][val], s['d'][val], '.', ms=2, color=VERDE)
    ax[1].plot(s['t'], d_gat(np.clip(np.interp(s['t'], c['t'], c['v']), 0, None)), color=VERM, lw=0.9)
    ax[1].set_ylim(0, 4.2); ax[1].set_ylabel('$d$ [m]'); ax[1].set_xlabel('$t$ [s]')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'aeb_demo.pdf')); plt.close(fig)
    macro('DemoDmin', vir(s['d'][val].min(), 2))
    macro('DemoEv', str(int(np.sum((ad[1:] == 2) & (ad[:-1] != 2)))))


if __name__ == '__main__':
    os.makedirs(IMG, exist_ok=True)
    fig_onoff(); fig_ident(); fig_sim_controle(); fig_real_cruzeiro(); fig_real_parada()
    tab_aeb(); fig_aeb(); fig_aeb_demo()
    with open(os.path.join(AQUI, 'numeros.tex'), 'w', encoding='utf-8') as f:
        for k, v in num.items():
            f.write(f'\\newcommand{{\\num{k}}}{{{v}}}\n')
    with open(os.path.join(AQUI, 'numeros.txt'), 'w', encoding='utf-8') as f:
        for k, v in num.items():
            f.write(f'{k} = {v}\n')
    print(open(os.path.join(AQUI, 'numeros.txt'), encoding='utf-8').read())
