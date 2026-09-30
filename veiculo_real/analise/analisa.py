# -*- coding: utf-8 -*-
"""Analisa os logs (car.csv) dos ensaios do carro real, sem nenhum passo manual.

    python analise/analisa.py tools/base tools/patamares ...   pastas de ensaio da GUI (tools/<ensaio>/<carro>/<log>/car.csv)
    python analise/analisa.py PASTA_DO_LOG ...                  ou a pasta de um log, ou o proprio car.csv
    python analise/analisa.py ... --todos                       nao descarta corridas de bateria diferente

Voce sempre passa os caminhos: nada e procurado sozinho. A GUI baixa toda a pasta logs/ do carro a
cada coleta, entao o mesmo log aparece em varias pastas: vale a primeira que voce passou (passe em
ordem cronologica). Passe TODOS os logs da mesma bateria juntos (o ajuste da planta usa o
conjunto). Rode a partir de veiculo_real/ com o venv do projeto (simulador/.venv). Para cada log:
  1. descobre QUE ensaio foi, repassando o control_func do main.py sobre o v logado (o u
     tem que bater); o car.csv nao traz o nome do ensaio;
  2. reconstroi o pwm (integral de u, como o servos.py) e ajusta a planta
     u -> pwm -> zona morta DB e ganho G -> lag tau -> atraso d -> MA30;
  3. compara com os dois modelos antigos (previsao cega) e com o ajustado agora;
  4. no fim, projeta KP/TD/TAU_VDES para a planta ajustada e imprime a linha para
     colar em _PROJETOS['auto'] do main.py.
Saidas: <pasta do log>/analise.png e analise.txt.
"""
import argparse
import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True
REAL = Path(__file__).resolve().parents[1]

# ---- o que o codigo do carro diz (servos.py, car.py) -----------------------------------
K_PWM = np.rad2deg(0.4 * 0.08 * 5.16)             # graus/s de acelerador por unidade de u
TH_MAX = (1.5 + 8.9370) / 0.092294 - 95.0         # limite do acelerador [graus] (atingido em subida)
N_MA = 30                                         # car.py: v_filt = MovingAverage(n=30)
G_DE = lambda K: K / K_PWM                        # K [m/s^2 por u] -> G [m/s por grau]
K_DE = lambda G: G * K_PWM

# modelos de 24/09 (matlab-control/modelos_longitudinais.m); previsao CEGA nos logs novos
MODELOS = {
    'Minimos Quadrados no Erro de Saida': dict(G=G_DE(1.6), DB=3.42, tau=0.251, d=0.0),
    'SystemID (OE)': dict(G=G_DE(1.34), DB=1.5, tau=0.228, d=0.144),
}
DB_REF, D_REF = 2.5, 0.10   # so para comparar o K entre corridas (nao sao resultado)


# ---- main.py como modulo (mesma tecnica do capitulo/figuras.py) ------------------------
def carrega_main(ensaio='base'):
    """Carrega veiculo_real/main.py com ENSAIO trocado; bibliotecas de hardware viram mocks."""
    src = (REAL / 'main.py').read_text(encoding='utf-8')
    src = re.sub(r"^ENSAIO = '[^']*'", f"ENSAIO = '{ensaio}'", src, count=1, flags=re.M)
    sys.path.insert(0, str(REAL))
    try:
        while True:
            for m in [m for m in sys.modules if m == 'fva_car' or m.startswith('fva_car.')]:
                del sys.modules[m]
            mod = importlib.util.module_from_spec(importlib.util.spec_from_loader('main_real', loader=None))
            try:
                exec(compile(src, str(REAL / 'main.py'), 'exec'), mod.__dict__)
                return mod
            except ModuleNotFoundError as e:
                if e.name.split('.')[0] == 'fva_car':
                    raise
                sys.modules[e.name] = MagicMock()
    finally:
        sys.path.pop(0)


class CarroFalso:
    v_filt = type('F', (), {'n': N_MA})()
    atuador = type('A', (), {'trim_throttle': 0.0})()   # o 'freio_esc' escreve o trim aqui
    t = dt = v = vref = u = 0.0

    def set_u(self, u):                       # car.py set_u: +-1 e protecao |v| > VELMAX
        self.u = 0.0 if abs(self.v) > 1.5 else float(np.clip(u, -1.0, 1.0))


def zera(mod):
    mod._estado.clear()
    mod._vdes_filt.reset(0.0)


def repassa(mod, t, v):
    """Iteracao j usa a linha j (t, v, dt) e o u dela aparece na linha j+1 do log."""
    zera(mod)
    car, u, vr = CarroFalso(), np.full(len(t), np.nan), np.full(len(t), np.nan)
    for j in range(1, len(t) - 1):
        car.t, car.dt, car.v = t[j], t[j] - t[j - 1], v[j]
        mod.control_func(car)
        u[j + 1], vr[j + 1] = car.u, car.vref
    return u, vr


_MODS = {}   # um main.py carregado por ensaio


def reconhece(x):
    """(nome do ensaio, modulo) cujo controlador reproduz o u logado; ('desconhecido', None) se nenhum."""
    melhor = ('desconhecido', None, 0.0, None)
    if not _MODS:
        _MODS.update({n: carrega_main(n) for n in carrega_main('base').ROTEIRO})
    for nome, mod in _MODS.items():
        v_ref0 = mod.V_REF
        # V_REF e uma constante que voce edita no main.py entre corridas: tenta tambem o alvo que o log mostra
        for v_ref in ({v_ref0, round(float(x['vref'].max()), 2)} if 'aberto' not in mod._E else {v_ref0}):
            mod.V_REF = v_ref
            u, _ = repassa(mod, x['t'], x['v'])
            ok = (x['t'] <= mod._E['ts'] + 0.05) & ~np.isnan(u)
            ok[:3] = False
            acerto = float(np.mean(np.abs(u[ok] - x['u'][ok]) < 2e-3)) if ok.any() else 0.0
            if acerto > melhor[2]:
                melhor = (nome, mod, acerto, v_ref)
        mod.V_REF = v_ref0
    if melhor[2] > 0.97:
        x['V_REF'] = melhor[3]      # quem usar o modulo (malha) tem que por mod.V_REF = x['V_REF']
        return melhor[0], melhor[1]
    return 'desconhecido', None


# ---- dados e planta --------------------------------------------------------------------
def pwm(t, u):
    """Integra u como o servos.py. A linha k guarda o u aplicado entre t[k-1] e t[k]."""
    th = np.zeros_like(t)
    for k in range(1, len(t)):
        th[k] = min(max(th[k - 1] + K_PWM * u[k] * (t[k] - t[k - 1]), 0.0), TH_MAX)
    return th


def carrega(pasta):
    a = np.genfromtxt(Path(pasta) / 'car.csv', delimiter=',', names=True)
    x = {c: np.asarray(a[c], float) for c in ('t', 'v', 'vref', 'u')}
    x['th'] = pwm(x['t'], x['u'])
    Ts = float(np.median(np.diff(x['t'][2:])))                 # grade uniforme (jitter do loop)
    tg = x['t'][1] + Ts * np.arange(int((x['t'][-1] - x['t'][1]) / Ts) + 1)
    x.update(Ts=Ts, tg=tg, vg=np.interp(tg, x['t'], x['v']), thg=np.interp(tg, x['t'], x['th']),
             ug=np.interp(tg, x['t'], x['u']), nome=Path(pasta).name)
    return x


def preve(thg, Ts, G, DB, tau, d):
    """Planta em malha aberta na grade uniforme: v que o log (MA30) mostraria."""
    vss = np.r_[0.0, G * np.maximum(0.0, thg[:-1] - DB)]
    a = np.exp(-Ts / tau)
    v = lfilter([1 - a], [1, -a], vss)
    if d > 0:
        tt = np.arange(len(v)) * Ts
        v = np.interp(tt - d, tt, v, left=0.0)
    c = np.cumsum(np.r_[np.zeros(N_MA), v])
    return (c[N_MA:] - c[:-N_MA]) / N_MA


rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
fit_pct = lambda y, yh: 100.0 * (1.0 - np.linalg.norm(y - yh) / np.linalg.norm(y - np.mean(y)))


def ajusta_um(x, DB=DB_REF, d=D_REF):
    """(G, tau) de UMA corrida com DB e d fixos: serve so para comparar K entre corridas."""
    r = lambda p: preve(x['thg'], x['Ts'], p[0], DB, p[1], d) - x['vg']
    return least_squares(r, [0.15, 0.3], bounds=([0.02, 0.05], [0.5, 2.0])).x


def ajusta_conjunto(xs, DB=None):
    """Um (DB, tau, d) para todas as corridas (sao do carro/ESC) e um G por corrida (depende
    da bateria). DB fixo se dado. Retorna dict(DB, tau, d, G=[...], rmse)."""
    m = len(xs)
    def desempacota(p):
        db = DB if DB is not None else p[0]
        return db, p[1], p[2], p[3:]
    def res(p):
        db, tau, d, G = desempacota(p)
        return np.concatenate([preve(x['thg'], x['Ts'], G[i], db, tau, d) - x['vg'] for i, x in enumerate(xs)])
    p0 = [2.5, 0.3, 0.05] + [0.15] * m
    lo = [0.0, 0.05, 0.0] + [0.02] * m
    hi = [8.0, 1.5, 0.4] + [0.5] * m
    p = least_squares(res, p0, bounds=(lo, hi)).x
    db, tau, d, G = desempacota(p)
    return dict(DB=float(db), tau=float(tau), d=float(d), G=list(map(float, G)), rmse=rmse(res(p)))


# ---- metricas por ensaio: numeros para VOCE julgar (nenhum veredito) -------------------
def _t_de(tt, cond):
    i = np.nonzero(cond)[0]
    return float(tt[i[0]]) if len(i) else float('nan')


def bloco_rampa(x, E, mod, ajx):
    """Malha fechada com rampa e patamar unico: metricas + sugestoes (lista de linhas)."""
    t, v, vr, u, th = x['t'], x['v'], x['vref'], x['u'], x['th']
    k0 = int(np.argmax(vr > 0))
    alvo = float(vr.max())
    ok = (t >= t[k0]) & (t <= E['ts'] + 0.05)
    tt, vv, uu, rr, tht = t[ok] - t[k0], v[ok], u[ok], vr[ok], th[ok]
    fim = tt > tt[-1] - 3.0

    def acomoda(f):   # ultimo instante fora da faixa; nan = nao acomodou ate o fim
        fo = np.nonzero(np.abs(vv - alvo) > f * alvo)[0]
        return 0.0 if not len(fo) else (float(tt[fo[-1]]) if fo[-1] < len(tt) - 1 else float('nan'))

    ip = int(np.argmax(vv))
    e = vv[ip:] - alvo
    e = e[np.abs(e) > 0.01 * alvo]
    m = dict(partida=_t_de(tt, vv > 0.05), subida=_t_de(tt, vv >= 0.9 * alvo) - _t_de(tt, vv >= 0.1 * alvo),
             pico=float(vv[ip]), ultra=100 * (float(vv[ip]) / alvo - 1), t_pico=float(tt[ip]), a5=acomoda(0.05), a2=acomoda(0.02),
             cruz=int(np.sum(np.sign(e[1:]) != np.sign(e[:-1]))) if len(e) > 1 else 0,
             ondula=float(np.ptp(vv[fim])), erro_final=float(np.mean(vv[tt > tt[-1] - 2.0]) - alvo),
             iae=float(np.trapezoid(np.abs(rr - vv), tt)), umax=float(uu.max()), umin=float(uu.min()),
             sat=100 * float(np.mean(np.abs(uu) >= 0.99)), ruido=float(np.mean(np.abs(np.diff(uu[fim])))), pwm_fim=float(np.mean(tht[fim])))
    nan_s = lambda a: 'nao acomodou' if np.isnan(a) else f'{a:.1f} s'
    L = [f"  {x['nome']}  {x['ensaio']}  alvo {alvo:.2f} m/s" + (f"   (KP {mod.KP_MOD}, TD {mod.TD_DERIV}, TAU_VDES {mod.TAU_VDES}, K_NOM {mod.K_NOM})" if mod else ''),
         f"    partida (v > 0,05)             {m['partida']:.1f} s          <- zona morta do ESC, nao e atraso da malha",
         f"    subida 10% -> 90%              {m['subida']:.1f} s",
         f"    pico                           {m['pico']:.2f} m/s  (sobressinal {m['ultra']:+.1f}%) em t = {m['t_pico']:.1f} s",
         f"    acomodacao +-5% / +-2%         {nan_s(m['a5'])} / {nan_s(m['a2'])}   (contada desde o inicio da rampa)",
         f"    cruzamentos do alvo apos o pico {m['cruz']}      oscilacao pico a pico (ult. 3 s)  {m['ondula']:.3f} m/s",
         f"    erro final (ult. 2 s)          {m['erro_final']:+.3f} m/s   pwm no fim {m['pwm_fim']:.1f} graus",
         f"    IAE contra vref                {m['iae']:.2f} m",
         f"    u: max {m['umax']:+.2f}  min {m['umin']:+.2f}  saturado (|u| >= 0,99) {m['sat']:.1f}% do tempo   ruido de u (media |du|, ult. 3 s) {m['ruido']:.4f}"]
    sug = []
    K = K_DE(ajx['G']) if ajx else None
    if ajx and mod:
        os_sim, st_sim = malha(mod, K, ajx['DB'], ajx['tau'], ajx['d'], ts=E['ts'])
        L.append(f"    modelo ajustado previa (K {K:.2f} desta corrida): sobressinal {os_sim:.1f}%, acomoda +-5% em {st_sim:.1f} s   <- diferenca grande = modelo nao explica o carro")
        if mod and K > 1.15 * mod.K_NOM:
            sug.append(f"K medido ({K:.2f}) e {100 * (K / mod.K_NOM - 1):.0f}% maior que K_NOM ({mod.K_NOM}): a malha fica mais agressiva que a projetada. "
                       f"Suba K_NOM para ~{K:.2f} (FF fica exato e KP/K_NOM cai na mesma proporcao).")
        if mod and K < 0.85 * mod.K_NOM:
            sug.append(f"K medido ({K:.2f}) e {100 * (1 - K / mod.K_NOM):.0f}% menor que K_NOM ({mod.K_NOM}): resposta mais lenta que a projetada. "
                       f"Se a bateria esta fraca, troque/recarregue; se {K:.2f} e o K normal do carro, o valor seria K_NOM = {K:.2f}, "
                       "mas compare antes as duas opcoes simuladas no bloco da planta ajustada (K menor que K_NOM e o lado seguro).")
    if m['ultra'] > 5:
        sug.append("sobressinal alto: reduza KP (menos ganho de malha) ou ative o derivativo (ENSAIO 'avanco': TD da margem de fase); TAU_VDES maior suaviza a referencia.")
    if m['ultra'] < 0.5 and (np.isnan(m['a5']) or m['a5'] > 4.0):
        sug.append("sem sobressinal e lento: ha folga para subir KP (ou 'avanco'); confira K medido acima.")
    if m['partida'] > 1.5:
        sug.append("partida > 1,5 s: zona morta grande (ou bateria fraca). Rode 'u_degrau' para medir DB; compensar a zona morta piorou o sobressinal na simulacao.")
    if m['sat'] > 3:
        sug.append("u saturou: reduza TAXA_RAMPA_V ou KP (a saturacao esconde o ganho real e degrada a margem de fase).")
    if m['cruz'] >= 2 or m['ondula'] > 0.05:
        sug.append("oscila em regime: reduza KP; se TD > 0, reduza TD (amplifica ruido).")
    if m['ruido'] > 0.03:
        sug.append("u muito ruidoso: se TD > 0, reduza TD ou aumente o filtro (TD/5).")
    if abs(m['erro_final']) > 0.03:
        sug.append("erro final != 0 sem integral: K caindo com a bateria ou perturbacao; o integrador de pwm deveria zerar o erro (o carro chegou ao fim antes de acomodar?).")
    if sug:
        L.append("    sugestoes:")
        L += [f"      - {s}" for s in sug]
    return L


def bloco_degraus(x, E, mod, ajx):
    """Patamares: por degrau, e a curva estatica v x pwm lida direto do fim de cada degrau."""
    t, v, th, u, perfil = x['t'], x['v'], x['th'], x['u'], E['perfil']
    t0 = float(t[np.argmax(x['vref'] > 0)])
    dv = np.gradient(np.convolve(v, np.ones(15) / 15, 'same'), t)   # dv/dt com ~0,4 s de suavizacao
    L = [f"  {x['nome']}  {x['ensaio']}   (KP {mod.KP_MOD}, TD {mod.TD_DERIV}, TAU_VDES {mod.TAU_VDES}, K_NOM {mod.K_NOM})",
         "    degrau        v final   erro    ultrapassa  entra em +-0,05  pwm no fim (0,7 s)  u sat.  |dv/dt| max"]
    fins, ant, pts = [p[0] for p in perfil[1:]] + [E['ts']], 0.0, []
    for (ti, alvo, *_), tf in zip(perfil, fins):   # 3o item opcional do perfil = taxa da referencia
        j = (t >= t0 + ti) & (t < t0 + tf)
        if j.sum() > 5:
            tj, vj = t[j] - t[j][0], v[j]
            sobe = alvo > ant
            ultra = float(vj.max() - alvo) if sobe else float(alvo - vj.min())
            fora = np.nonzero(np.abs(vj - alvo) > 0.05)[0]
            tacom = float(tj[fora[-1]]) if len(fora) else 0.0
            vf, pf = float(np.mean(vj[tj > tj[-1] - 0.7])), float(np.mean(th[j][tj > tj[-1] - 0.7]))
            sat = 100 * float(np.mean(np.abs(u[j]) >= 0.99))
            taxa = float(dv[j].max() if sobe else -dv[j].min())
            L.append(f"    {'sobe ' if sobe else 'desce'} -> {alvo:.1f}   {vf:6.2f}  {vf - alvo:+.2f}   {max(0.0, ultra):6.2f}      {tacom:5.1f} s        {pf:6.2f}      {sat:4.1f}%   {taxa:5.2f} m/s2")
            if tacom < (tj[-1] - 0.7) and alvo > 0.05:
                pts.append((pf, vf))          # degrau acomodado: ponto da curva estatica
        ant = alvo
    if len(pts) >= 3 and np.ptp([p[1] for p in pts]) > 0.3:
        G, b = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
        L.append(f"    reta v = G (pwm - DB) pelos {len(pts)} degraus acomodados: G = {G:.3f} m/s por grau (K = {K_DE(G):.2f}), DB = {-b / G:.2f} graus"
                 "   <- leitura direta, sem modelo dinamico")
    else:
        L.append("    (menos de 3 degraus acomodados cobrindo > 0,3 m/s: sem reta v x pwm)")
    return L


def bloco_aberto(x, E):
    """u_degrau: por nivel de u, quando o carro sai do lugar e com que pwm."""
    t, v, u, th = x['t'], x['v'], x['u'], x['th']
    L = [f"  {x['nome']}  {x['ensaio']}  (malha aberta)",
         "    u      subida do u  partida (v>0,05)  atraso    pwm na partida  pico v   pico pwm"]
    pos, k, pts = u > 0.01, 1, []
    while k < len(t):
        if pos[k] and not pos[k - 1]:
            j = k
            while j < len(t) and pos[j]:
                j += 1
            seg = slice(k, j)
            nivel = float(np.median(u[seg]))
            i = np.nonzero(v[k:] > 0.05)[0]
            tm = float(t[k + i[0]]) if len(i) else float('nan')
            pm = float(np.interp(tm, t, th)) if not np.isnan(tm) else float('nan')
            atraso = tm - float(t[k])
            v_pico = float(v[k:min(len(t), j + 60)].max())
            L.append(f"    {nivel:.2f}   {t[k]:6.2f} s     {tm:6.2f} s      {atraso:5.2f} s   {pm:8.2f} graus  {v_pico:6.2f}  {float(th[seg].max()):7.2f}")
            if not np.isnan(atraso):
                pts.append((1.0 / nivel, atraso))
            k = j
        k += 1
    if len(pts) >= 2:
        a, b = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
        L.append(f"    atraso = a/u + b:  a = {a:.2f} s, b = {b:.2f} s  ->  DB ~ a x {K_PWM:.2f} = {a * K_PWM:.2f} graus")
        L.append("    (aproximado: o limiar v>0,05 e a MA30 somam ~0,3 s em b e um pouco em a; compare com a DB do ajuste dinamico)")
    return L


def bloco_freio(x, E):
    """freio_esc: um pulso abaixo do neutro por nivel. O log nao tem o trim; o vref leva a marca
    (-0,01 drenando; -max(nivel/10, 0,02) no pulso). Numeros para comparar cada nivel com o 0 (so atrito)."""
    t, v, vr = x['t'], x['v'], x['vref']
    L = [f"  {x['nome']}  {x['ensaio']}  (EXPLORATORIO; o pulso e lido da marca no vref)",
         "    nivel   inicio    dur    v no inicio   v 0,45 s apos o fim   queda media   v minimo   parou em   distancia ate parar"]
    marca = vr < -0.015
    k, base = 1, None
    while k < len(t):
        if marca[k] and not marca[k - 1]:
            j = k
            while j < len(t) and marca[j]:
                j += 1
            nivel = 0.0 if vr[k] > -0.03 else round(-10 * float(np.median(vr[k:j])), 1)
            ta, tb = float(t[k]), float(t[j - 1])
            v0 = float(np.interp(ta, t, v))
            v1 = float(np.interp(tb + 0.45, t, v))     # a MA30 atrasa ~0,43 s: o efeito do pulso aparece depois
            queda = (v0 - v1) / (tb + 0.45 - ta)
            resto = slice(k, len(t))
            pos = np.nonzero(v[resto] < 0.05)[0]
            kp = k + int(pos[0]) if len(pos) else None
            parou = f"{t[kp] - ta:5.1f} s" if kp is not None else "    -  "
            dist = f"{float(np.trapezoid(v[k:kp], t[k:kp])):5.2f} m" if kp is not None else "    -  "
            vmin = float(v[k:min(len(t), j + 100)].min())
            L.append(f"    {nivel:4.0f}   {ta:6.1f} s  {tb - ta:4.2f} s   {v0:6.2f} m/s      {v1:6.2f} m/s        {queda:5.2f} m/s2   {vmin:+6.2f}   {parou}   {dist}")
            if nivel == 0.0:
                base = queda
            elif base:
                L[-1] += f"   ({queda / base:.1f}x o atrito)"
            k = j
        k += 1
    L.append("    v minimo < 0 = o carro deu re (o pulso passou da parada). queda media >> a do nivel 0 = o ESC freia; igual = so atrito.")
    return L


# ---- margem de fase e projeto ----------------------------------------------------------
def margem_fase(KP, TD, K_nom, K, tau, d, Ts=0.0288, extra=0.03):
    """PM [graus] da malha linear: C(s)*K*MA30*atraso/(s(tau s+1)), C = KP/K_nom*(1+TD s/(TD/5 s+1))."""
    w = np.logspace(-2, 1.3, 4000)
    s = 1j * w
    C = KP / K_nom * (1 + (TD * s / (TD / 5 * s + 1) if TD > 0 else 0))
    MA = sum(np.exp(-s * k * Ts) for k in range(N_MA)) / N_MA
    L = C * K * MA * np.exp(-s * (d + extra)) / (s * (tau * s + 1))
    ph = np.unwrap(np.angle(L))
    i = np.nonzero(np.abs(L) < 1)[0]
    return float(180 + np.degrees(ph[i[0]])) if len(i) else np.nan


def malha(mod, K, DB, tau, d, ts=12.0, Ts=0.0288):
    """Malha fechada com o control_func do main.py e a planta ajustada."""
    zera(mod)
    car, n, G = CarroFalso(), round(ts / Ts), G_DE(K)
    a, th, vr, uk = np.exp(-Ts / tau), 0.0, 0.0, 0.0
    atraso, buf = np.zeros(round(d / Ts) + 1), np.zeros(N_MA)
    t, v = np.arange(n) * Ts, np.zeros(n)
    for k in range(1, n):
        vss = G * max(0.0, th - DB)
        th = min(max(th + K_PWM * uk * Ts, 0.0), TH_MAX)
        vr = vss + (vr - vss) * a
        atraso = np.r_[vr, atraso[:-1]]
        buf = np.r_[atraso[-1], buf[:-1]]
        car.t, car.dt, car.v = t[k], Ts, float(buf.mean())
        mod.control_func(car)
        uk = car.u
        v[k] = vr
    alvo = mod.V_REF
    fora = np.nonzero(np.abs(v - alvo) > 0.05 * alvo)[0]
    return 100 * max(0.0, v.max() / alvo - 1), float(t[fora[-1]]) if len(fora) else 0.0


def projeta(pl):
    """Varre KP, TD, TAU_VDES na simulacao com a planta ajustada. Compara com o 'base'."""
    mod = carrega_main('base')
    K_nom = float(np.clip(round(pl['K'], 2), 0.8, 2.5))
    plantas = {'nominal': (pl['K'], pl['tau']), 'alto': (1.25 * pl['K'], 1.5 * pl['tau']), 'baixo': (0.8 * pl['K'], pl['tau'])}

    def avalia(KP, TD, TV, Kn):
        mod.KP_MOD, mod.TD_DERIV, mod.TAU_VDES, mod.K_NOM = KP, TD, TV, Kn
        r = {n: malha(mod, K, pl['DB'], tau, pl['d']) for n, (K, tau) in plantas.items()}
        pm = margem_fase(KP, TD, Kn, plantas['alto'][0], plantas['alto'][1], pl['d'])
        return r, pm

    base = avalia(0.6, 0.0, 0.3, 1.6)
    melhor = None
    for KP in np.arange(0.3, 1.01, 0.1):
        for TD in (0.0, 0.4, 0.6, 0.8):
            for TV in (0.3, 0.5):
                r, pm = avalia(round(KP, 2), TD, TV, K_nom)
                # nao pode ser pior que o 'base' nos casos alto (sobressinal) e baixo (acomodacao)
                viavel = (r['nominal'][0] <= 3 and r['baixo'][0] <= 3 and pm >= 55
                          and r['alto'][0] <= base[0]['alto'][0] + 0.5 and r['baixo'][1] <= base[0]['baixo'][1] + 0.5)
                custo = r['nominal'][1] + 0.5 * r['alto'][1]
                if viavel and (melhor is None or custo < melhor[0]):
                    melhor = (custo, dict(KP=round(float(KP), 2), TD=TD, TAU_VDES=TV, K_NOM=K_nom), r, pm)
    return base, melhor


# ---- logs: sempre os caminhos que voce passa (nada de busca automatica) ----------------
def pastas_dos_logs(caminhos):
    """Cada caminho pode ser: um car.csv; a pasta de um log; ou uma pasta de ensaio como a da GUI
    (tools\\base -> roxo\\AAAAMMDD_HHMMSS\\car.csv), onde os car.csv sao procurados em qualquer
    profundidade. A GUI baixa TODA a pasta logs/ do carro a cada coleta, entao o mesmo log aparece
    em varias pastas de ensaio: fica so a primeira ocorrencia (mesmo nome e mesmo tamanho)."""
    achados, vistos = [], set()
    for c in map(Path, caminhos):
        if str(c).lower().endswith('.csv'):
            logs = [c.parent]
        elif (c / 'car.csv').is_file():
            logs = [c]
        else:
            logs = sorted(f.parent for f in c.rglob('car.csv')) if c.is_dir() else []
        if not logs:
            sys.exit(f'nao achei nenhum car.csv em: {c}\n(passe a pasta do ensaio, ex.: tools\\base, a pasta de um log ou o proprio car.csv)')
        for p in logs:
            chave = (p.name, (p / 'car.csv').stat().st_size)
            if chave not in vistos:
                vistos.add(chave)
                achados.append(p)
    return achados


def ajuste_cego(grupo, i):
    """Previsao da corrida i por um modelo que NAO a viu: DB, tau e d vem do ajuste das outras
    corridas; so o G (bateria) e refeito nesta, com 1 parametro."""
    a = ajusta_conjunto([g for k, g in enumerate(grupo) if k != i])
    x = grupo[i]
    r = lambda p: preve(x['thg'], x['Ts'], p[0], a['DB'], a['tau'], a['d']) - x['vg']
    G = least_squares(r, [0.15], bounds=([0.02], [0.5])).x[0]
    return preve(x['thg'], x['Ts'], G, a['DB'], a['tau'], a['d'])


# ---- relatorio -------------------------------------------------------------------------
def figura(x, ident, mods, ajustado, texto):
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout='constrained')
    t, tg = x['t'], x['tg']
    axs[0].plot(t, x['v'], 'k', lw=2, label='v medido (MA30)')
    if (x['vref'] > 0).any():
        axs[0].plot(t, x['vref'], 'k--', lw=1, label='vref')
        if x['ensaio'] != 'patamares':   # faixa de +-5% em torno do alvo final
            alvo = float(x['vref'].max())
            axs[0].axhspan(0.95 * alvo, 1.05 * alvo, color='0.9', zorder=0, label='+-5% do alvo')
    for (nome, m), cor in zip(mods.items(), ('#1f77b4', '#2ca02c')):
        yh = preve(x['thg'], x['Ts'], m['G'], m['DB'], m['tau'], m['d'])
        axs[0].plot(tg, yh, color=cor, lw=1.3, label=f"{nome} (cego): rmse {rmse(yh - x['vg']):.3f}, fit {fit_pct(x['vg'], yh):.0f}%")
    if ajustado is not None:
        yh = preve(x['thg'], x['Ts'], ajustado['G'], ajustado['DB'], ajustado['tau'], ajustado['d'])
        axs[0].plot(tg, yh, color='#d62728', lw=1.3, label=f"ajustado (conjunto): rmse {rmse(yh - x['vg']):.3f}, fit {fit_pct(x['vg'], yh):.0f}%")
    axs[0].set_ylabel('v [m/s]'); axs[0].legend(fontsize=7, loc='upper left'); axs[0].grid(alpha=.3)
    axs[0].set_title(f"{x['nome']}: {ident}", fontsize=10)
    axs[1].plot(t, x['u'], 'C0'); axs[1].set_ylabel('u'); axs[1].grid(alpha=.3)
    axs[2].plot(t, x['th'], 'C1'); axs[2].set_ylabel('pwm reconstruido [graus]'); axs[2].set_xlabel('t [s]'); axs[2].grid(alpha=.3)
    if ajustado is not None:
        axs[2].axhline(ajustado['DB'], color='r', ls=':', lw=1, label=f"DB ajustada {ajustado['DB']:.2f}")
        axs[2].legend(fontsize=7)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pastas', nargs='+', help='pasta(s) do log (ou car.csv) que voce acabou de coletar')
    ap.add_argument('--todos', action='store_true', help='nao descarta corridas de bateria diferente')
    args = ap.parse_args()

    pastas = pastas_dos_logs(args.pastas)

    xs, ident = [], {}
    for p in pastas:
        x = carrega(p)
        nome, mod = reconhece(x)
        x['ensaio'], x['mod'] = nome, mod
        xs.append(x)
        # nome de uma pasta acima (tools\base\roxo\<log>) e so dica: quem manda e o u do log
        dica = next((a.name for a in p.parents[:4] if a.name in _MODS), None)
        aviso = ''
        if dica and dica != nome:
            aviso = (f"   <-- ATENCAO: a pasta diz '{dica}', mas o u do log " +
                     ("nao bate com nenhum ensaio do main.py atual" if nome == 'desconhecido' else f"corresponde a '{nome}'"))
        print(f"{x['nome']}: {nome}{aviso}")

    # so entram no ajuste corridas que MEXEM o carro; K por corrida denuncia a bateria
    # Malha aberta (u_degrau, ou log com vref = 0 e u > 0) entra mesmo devagar: o que importa e quando o carro
    # comeca a andar, nao a velocidade que atinge (com bateria fraca ela fica ~0,2 m/s).
    for x in xs:
        x['aberto'] = (x['mod'] is not None and 'aberto' in x['mod']._E) or (x['mod'] is None and not (x['vref'] > 0).any() and x['u'].max() > 0.01)
    # 'freio_esc' tem pulso de freio que o modelo da planta nao tem: fica fora do ajuste
    for x in xs:
        x['freio'] = x['mod'] is not None and 'freio' in x['mod']._E
    uteis = [x for x in xs if x['v'].max() > (0.05 if x['aberto'] else 0.3)]
    for x in uteis:
        g, tau = ajusta_um(x)
        x['K_corrida'], x['tau_corrida'] = K_DE(g), tau
    # so as corridas de malha fechada denunciam a bateria pelo K (a aberta lenta tem K mal determinado)
    fechadas = [x for x in uteis if not x['aberto'] and not x['freio']]
    if fechadas and not args.todos:
        Kmed = float(np.median([x['K_corrida'] for x in fechadas]))
        for x in uteis:
            x['bateria_ok'] = x['aberto'] or (0.8 * Kmed <= x['K_corrida'] <= 1.25 * Kmed)
    else:
        for x in uteis:
            x['bateria_ok'] = True
    grupo = [x for x in uteis if x['bateria_ok'] and not x['freio']]

    linhas = []
    P = linhas.append
    P(f"=== {len(xs)} log(s), {len(uteis)} com movimento, {len(grupo)} no ajuste ===")
    for x in uteis:
        P(f"  {x['nome']}  {x['ensaio']:10s}  K(DB={DB_REF},d={D_REF}) = {x['K_corrida']:.2f}  tau = {x['tau_corrida']:.2f}"
          + ('   (pulso de freio: fora do ajuste da planta)' if x['freio'] else
             '' if x['bateria_ok'] else '   <-- FORA do grupo (bateria/ganho diferente): nao entra no ajuste'))

    aj = None
    if grupo:
        aj = ajusta_conjunto(grupo)
        aj['K'] = float(np.median([K_DE(g) for g in aj['G']]))
        P(f"\n=== planta ajustada (corridas do grupo, DB/tau/d comuns, G por corrida) ===")
        P(f"  DB = {aj['DB']:.2f} graus   tau = {aj['tau']:.3f} s   d = {aj['d']:.3f} s   K = {aj['K']:.2f} m/s^2 por u   rmse = {aj['rmse']:.4f} m/s")
        P("  DB  = zona morta do ESC: graus de acelerador (acima do neutro) que o carro precisa de antes de comecar a andar")
        P("  G   = ganho: m/s de velocidade final por grau acima da DB  (v final = G x (pwm - DB));  K = G x 9,46 = m/s^2 por unidade de u")
        P("  Uso: o K vira o K_NOM do projeto 'auto' (abaixo); DB, tau e d entram na simulacao que escolhe KP/TD. O main.py em si so usa K_NOM, KP, TD, TAU_VDES.")
        # velocidades em que o carro ficou >= 1 s (patamares): so elas separam G de DB
        niveis = np.concatenate([np.round(x['vg'][x['vg'] > 0.15] / 0.1) * 0.1 for x in grupo])
        val, cont = np.unique(niveis, return_counts=True)
        fixos = val[cont >= 1.0 / np.median([x['Ts'] for x in grupo])]
        amplitude = float(fixos.max() - fixos.min()) if len(fixos) else 0.0
        P("  rmse com DB fixo (o quanto o ajuste 'sente' a zona morta):")
        P('    ' + '  '.join(f"DB {db:.1f}: {ajusta_conjunto(grupo, DB=db)['rmse']:.4f}" for db in (0.5, 1.5, 2.5, 3.5, 4.5)))
        if amplitude < 0.4:
            P(f"  ATENCAO: o carro ficou >= 1 s em patamares que cobrem so {amplitude:.1f} m/s. Nesta faixa so a COMBINACAO G x (pwm - DB) e conhecida:"
              " varias duplas (G, DB) explicam igualmente os dados, entao DB e G isolados nao sao confiaveis (o K, sim, perto dessa velocidade)."
              " Rode 'patamares' e 'u_degrau' e passe esses logs junto.")
        # o K medido bate com o K_NOM do main.py? (K_NOM e o unico numero de planta que o controlador usa)
        K_atual, Ks = _MODS['base'].K_NOM, [K_DE(g) for g in aj['G']]
        dif = 100 * (aj['K'] / K_atual - 1)
        P(f"\n  K_NOM no main.py = {K_atual}   K medido (mediana das corridas) = {aj['K']:.2f}   "
          f"(por corrida: {min(Ks):.2f} a {max(Ks):.2f})   diferenca {dif:+.0f}%")
        if abs(dif) <= 15:
            P("  -> dentro de +-15%: pode manter o K_NOM (o projeto cobre K de 1,3 a 2,0).")
        else:
            P(f"  -> FORA de +-15% (hoje 'K_NOM = {K_atual}', linha perto do topo do main.py)"
              + ("   (K ainda incerto: falta patamares/u_degrau)" if amplitude < 0.4 else ""))
            # controlador final = preset 'avanco'. K_NOM = K medido NAO e o melhor: com a planta lenta (tau ~0,7 s) o
            # cruzamento KP*K/K_NOM = KP fica alto demais; o bom e KP*K/K_NOM ~ 0,55, isto e K_NOM ~ 1,4 x K
            mk = carrega_main('avanco')
            P(f"     Simulado na planta ajustada (preset 'avanco', KP {mk.KP_MOD}, TD {mk.TD_DERIV}): sobressinal / acomodacao +-5%")
            cands = sorted({K_atual, *(round(f * aj['K'], 2) for f in (1.0, 1.25, 1.5, 1.75, 2.0))})
            res = {}
            for kn in cands:
                mk.K_NOM = kn
                res[kn] = {n: malha(mk, aj['K'] * f_k, aj['DB'], aj['tau'] * f_t, aj['d']) for n, (f_k, f_t) in
                           {'K medido': (1, 1), 'K x1,25 e tau x1,5': (1.25, 1.5)}.items()}
                P(f"       K_NOM = {kn:<5} (KP*K/K_NOM = {mk.KP_MOD * aj['K'] / kn:.2f}): "
                  + '   '.join(f"{n}: {o:4.1f}% / {s:3.1f} s" for n, (o, s) in res[kn].items()))
            alto = 'K x1,25 e tau x1,5'
            if res[K_atual]['K medido'][0] <= 3.0:
                o, s = res[K_atual]['K medido']
                P(f"     Sugestao: MANTER 'K_NOM = {K_atual}' (no K medido: {o:.1f}% / {s:.1f} s; K_NOM acima do K real e o lado seguro). "
                  f"No caso alto da {res[K_atual][alto][0]:.0f}%: se quiser mais folga, suba K_NOM (malha mais lenta), veja as linhas acima.")
            else:
                bons = [kn for kn in cands if res[kn]['K medido'][0] <= 2.0]
                if bons:
                    melhor_kn = min(bons, key=lambda kn: (res[kn][alto][0] > 8.0, res[kn]['K medido'][1]))
                    P(f"     Sugestao: 'K_NOM = {melhor_kn}' (<= 2% de sobressinal no K medido, o mais rapido; prefere <= 8% no caso alto).")
                else:
                    P("     Nenhuma opcao testada fica <= 2% no K medido: a planta mudou muito; reveja KP/TD com o 'auto'.")
            P("     K_NOM acima do K real e o lado SEGURO (malha mais lenta, sem sobressinal). Se o K varia muito entre corridas, e a bateria: troque/recarregue.")
        # previsao cega dos modelos antigos + ajustado, por corrida
        P("\n=== previsao por corrida: rmse [m/s] (fit %) ===")
        P("  Os dois modelos antigos preveem SEM ver esta corrida. 'ajustado agora' JA VIU todas (nota otimista, so mede o ajuste).")
        P("  'ajustado sem esta' e a prova honesta: DB, tau e d das outras corridas; so o G e refeito aqui.")
        cab = ['Minimos Quadrados (24/09)', 'SystemID (24/09)', 'ajustado agora', 'ajustado sem esta']
        P('  ' + f"{'corrida':17s}" + ''.join(f'{n:>27s}' for n in cab))
        for i, x in enumerate(grupo):
            cols = [preve(x['thg'], x['Ts'], m['G'], m['DB'], m['tau'], m['d']) for m in MODELOS.values()]
            cols.append(preve(x['thg'], x['Ts'], aj['G'][i], aj['DB'], aj['tau'], aj['d']))
            cols.append(ajuste_cego(grupo, i) if len(grupo) >= 2 else None)
            P('  ' + f"{x['nome']:17s}" + ''.join(f"{'-':>27s}" if c is None else f"{rmse(c - x['vg']):>20.3f} ({fit_pct(x['vg'], c):3.0f}%)" for c in cols))

    # metricas por ensaio (sem veredito: a leitura e sua)
    P("\n=== metricas por ensaio ===")
    for x in uteis:
        mod = x['mod']
        if mod:
            mod.V_REF = x['V_REF']
        E = mod._E if mod else dict(ts=float(x['t'][-1]), perfil=None)
        i = next((k for k, g in enumerate(grupo) if g is x), None)
        ajx = dict(aj, G=aj['G'][i]) if (aj is not None and i is not None) else None
        if x['aberto']:
            linhas_x = bloco_aberto(x, E)
        elif x['freio']:
            linhas_x = bloco_freio(x, E)
        elif not (x['vref'] > 0).any():
            continue
        elif E['perfil'] is not None:
            linhas_x = bloco_degraus(x, E, mod, ajx)
        else:
            linhas_x = bloco_rampa(x, E, mod, ajx)
        linhas += linhas_x

    # projeto
    if aj is not None:
        P("\n=== projeto do controlador na planta ajustada (simulacao com o control_func do main.py) ===")
        base, melhor = projeta(aj)
        fmt = lambda r: '  '.join(f"{n}: {o:4.1f}%/{s:3.1f} s" for n, (o, s) in r.items())
        P(f"  base (KP 0,6 TD 0 TV 0,3 K_NOM 1,6): {fmt(base[0])}   PM(alto) {base[1]:.0f} graus")
        if melhor is None:
            P("  nenhuma combinacao atende (<=3% nominal e baixo, PM >= 55 e nao pior que o base nos casos alto/baixo): mantenha 'base'.")
        else:
            _, c, r, pm = melhor
            P(f"  auto {c}: {fmt(r)}   PM(alto) {pm:.0f} graus")
            P(f"  (sobressinal/acomodacao; alto = K x1,25 e tau x1,5; baixo = K x0,8)")
            P(f"  base -> auto:  acomodacao nominal {base[0]['nominal'][1]:.1f} -> {r['nominal'][1]:.1f} s,  sobressinal alto {base[0]['alto'][0]:.1f}% -> {r['alto'][0]:.1f}%,"
              f"  acomodacao baixo {base[0]['baixo'][1]:.1f} -> {r['baixo'][1]:.1f} s,  PM(alto) {base[1]:.0f} -> {pm:.0f} graus   (a decisao e sua)")
            P("  para testar no carro, cole em main.py, _PROJETOS, e use ENSAIO = 'auto':")
            P(f"      'auto'  : dict(KP={c['KP']}, TD={c['TD']}, TAU_VDES={c['TAU_VDES']}, K_NOM={c['K_NOM']}),")

    feitos = {x['ensaio'] for x in xs}
    ordem = ['base', 'patamares', 'u_degrau', 'avanco', 'v_alta']
    faltam = [e for e in ordem if e not in feitos]
    P(f"\n=== roteiro: feitos {sorted(feitos - {'desconhecido'})}; proximo: {faltam[0] if faltam else 'nenhum (roteiro completo)'} ===")

    texto = '\n'.join(linhas)
    print('\n' + texto)
    falhas = []
    for x, pasta in zip(xs, pastas):
        i = next((k for k, g in enumerate(grupo) if g is x), None)
        ajx = dict(aj, G=aj['G'][i]) if (aj is not None and i is not None) else None
        fig = figura(x, x['ensaio'], MODELOS, ajx, texto)
        try:   # arquivo aberto em outro programa (visualizador de imagem) nao pode derrubar o relatorio
            fig.savefig(pasta / 'analise.png', dpi=110)
            (pasta / 'analise.txt').write_text(texto, encoding='utf-8')
        except OSError as e:
            falhas.append(f"{pasta / 'analise.png'} ({e.strerror or e})")
        plt.close(fig)
    print(f"\nfiguras: <pasta do log>/analise.png ({len(xs) - len(falhas)} de {len(xs)})")
    for f in falhas:
        print(f"  NAO consegui gravar {f}: feche o arquivo se estiver aberto e rode de novo")


if __name__ == '__main__':
    main()
