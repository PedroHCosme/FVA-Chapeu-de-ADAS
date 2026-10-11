# -*- coding: utf-8 -*-
"""Analisa os logs (car.csv) dos ensaios do carro real, sem nenhum passo manual.

    python analise/analisa.py ../ensaios/real/roxo/2026-10-01   uma pasta da arvore ensaios/ (um arquivo por log)
    python analise/analisa.py ARQUIVO.csv ...                    ou arquivos dessa arvore (acervo.py: id -> caminho)
    python analise/analisa.py PASTA_DO_LOG ...                   ou a coleta crua da GUI: <log>/car.csv (tambem aceita)
    python analise/analisa.py ... --todos                        nao descarta corridas de bateria diferente

Voce sempre passa os caminhos: nada e procurado sozinho. Se o mesmo log aparecer em mais de um caminho,
vale o primeiro (mesmo id e mesmo tamanho). Passe TODOS os logs da mesma bateria juntos (o ajuste da
planta usa o conjunto). Rode a partir de veiculo_real/ com o venv do projeto (simulador/.venv). Para cada log:
  1. descobre QUE ensaio foi, repassando o control_func do main.py sobre o v logado (o u
     tem que bater); o car.csv nao traz o nome do ensaio;
  2. reconstroi o pwm (integral de u, como o servos.py) e ajusta a planta
     u -> pwm -> zona morta DB e ganho G -> lag tau -> atraso d -> MA30;
  3. confere o K medido contra o controlador: ganho de malha KP*K e, no 'avanco_ff', se K <= K_FF;
  4. metricas por corrida e, no fim, o RESUMO: uma linha por ensaio (media das corridas) para o A/B;
Saidas: analise/saida/<id>.png e <id>.txt para logs da arvore ensaios/ (os dados nao recebem arquivo gerado);
<pasta do log>/analise.png e analise.txt para a coleta crua.
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
SAIDA = REAL / 'analise' / 'saida'                # analise.png/txt dos logs da arvore ensaios/ (gitignored)

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
HISTORICO = REAL / 'analise' / 'ensaios_historicos.py'


def carrega_main(ensaio=None):
    """Carrega veiculo_real/main.py (lei avanco_rt + ADAS); bibliotecas de hardware viram mocks. Com `ensaio`, poe por cima o
    ensaio historico de analise/ensaios_historicos.py (referencia, preset de ganhos, malha aberta, sonar): o main.py de antes de 11/10."""
    src = (REAL / 'main.py').read_text(encoding='utf-8')
    sys.path.insert(0, str(REAL))
    try:
        while True:
            for m in [m for m in sys.modules if m == 'fva_car' or m.startswith('fva_car.')]:
                del sys.modules[m]
            mod = importlib.util.module_from_spec(importlib.util.spec_from_loader('main_real', loader=None))
            try:
                exec(compile(src, str(REAL / 'main.py'), 'exec'), mod.__dict__)
                if ensaio:
                    mod.ENSAIO = ensaio
                    exec(compile(HISTORICO.read_text(encoding='utf-8'), str(HISTORICO), 'exec'), mod.__dict__)
                mod.ADAS_ON = False   # a analise repassa so a lei de controle (o supervisor mexeria em V_REF/vdes dos logs antigos)
                return mod
            except ModuleNotFoundError as e:
                if e.name.split('.')[0] == 'fva_car':
                    raise
                sys.modules[e.name] = MagicMock()
    finally:
        sys.path.pop(0)


class CarroFalso:
    v_filt = type('F', (), {'n': N_MA})()
    atuador = type('A', (), {'trim_throttle': 0.0})()
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


def guardas(mod, x):
    """Variantes da guarda de VELMAX a repassar: a do main.py e, se o log passou dela, tambem sem guarda
    (logs de ate 02/10 rodaram sem). Abaixo da guarda as duas dao o mesmo u: uma so basta."""
    vg = getattr(mod, 'V_GUARDA', None)
    return [vg, None] if vg is not None and float(np.max(x['v'])) > vg else [vg]


def acerto(mod, x):
    """(fracao das amostras em que o u repassado bate com o logado, amostras comparadas): o 2o numero desempata a favor
    de quem cobre mais do log (ex.: 'avanco', 13 s, bate com os 13 s iniciais de uma 'frenagem')."""
    u, _ = repassa(mod, x['t'], x['v'])
    ok = (x['t'] <= mod._E['ts'] + 0.05) & ~np.isnan(u)
    ok[:3] = False
    return (float(np.mean(np.abs(u[ok] - x['u'][ok]) < 2e-3)) if ok.any() else 0.0), int(ok.sum())


def reconhece(x):
    """(nome do ensaio, modulo) cujo controlador reproduz o u logado; ('desconhecido', None) se nenhum."""
    melhor = ('desconhecido', None, (0.0, 0), None, None)
    if not _MODS:
        _MODS.update({n: carrega_main(n) for n in carrega_main('avanco').ROTEIRO})
    for nome, mod in _MODS.items():
        v_ref0, vg0 = mod.V_REF, getattr(mod, 'V_GUARDA', None)
        for vg in guardas(mod, x):
            mod.V_GUARDA = vg
            # V_REF e uma constante que voce edita no main.py entre corridas: tenta tambem o alvo que o log mostra
            for v_ref in ({v_ref0, round(float(x['vref'].max()), 2)} if 'aberto' not in mod._E else {v_ref0}):
                mod.V_REF = v_ref
                ac, n = acerto(mod, x)
                chave = (round(ac, 3), n)   # empate: vence quem cobre mais do log
                if chave > melhor[2]:
                    melhor = (nome, mod, chave, v_ref, vg)
        mod.V_REF, mod.V_GUARDA = v_ref0, vg0
    if melhor[2][0] > 0.97:
        x['V_REF'], x['V_GUARDA'] = melhor[3], melhor[4]   # quem usar o modulo (malha) tem que repor os dois
        return melhor[0], melhor[1]
    return 'desconhecido', None


SIM = dict(K=1.0, A0=0.096, C=0.00425, EPS=0.05, KP=4.0, KI=2.67, TT=0.5, TV=0.3, TU=0.10)   # simulador/main.py (rodou no carro em 24/09)


def lei_simulador(t, v, vref, I=True, ff_modelo=True, ff_ref=True, filtro_u=True):
    """Lei do simulador/main.py (PI + inversao do modelo + FF da referencia + filtro em u) repassada sobre um log; os flags
    desligam um termo. Iteracao j usa v[j] e grava u na linha j+1. Na parada do ultrassom (vref volta a 0 depois de a rampa
    comecar) roda set_vel(0) no lugar: o controlador fica congelado."""
    K, A0, C, EPS, KP, KI, TT, TV, TU = (SIM[k] for k in ('K', 'A0', 'C', 'EPS', 'KP', 'KI', 'TT', 'TV', 'TU'))
    vf = uf = Ii = up = ubp = 0.0
    out = np.full(len(t), np.nan)
    comecou = False
    for j in range(1, len(t) - 1):
        dt, vdes = t[j] - t[j - 1], vref[j + 1]
        comecou |= vdes > 0
        if comecou and vdes == 0:
            continue
        vp = vf
        if dt > 0:
            vf += np.clip(dt / TV, 0, 1) * (vdes - vf)
        ff = (vdes - vp) / TV if ff_ref else 0.0
        e = vf - v[j]
        if I:
            Ii += dt * (KI * e + (up - ubp) / TT)
        atrito = (A0 * np.tanh(v[j] / EPS) + C * v[j] * abs(v[j])) if ff_modelo else 0.0
        ub = (KP * e + Ii + ff + atrito) / K
        uc = float(np.clip(ub, 0.0, 1.0))
        if filtro_u:
            uf = uf + np.clip(dt / TU, 0, 1) * (uc - uf) if dt > 0 else uf
        else:
            uf = uc
        up, ubp = uf, ub
        out[j + 1] = 0.0 if abs(v[j]) > 1.5 else uf
    return out


# ---- dados e planta --------------------------------------------------------------------
def fora_do_ajuste(x):
    """Corrida de subida (disturbio) ou de parada (u = -1 no fim): o modelo da planta nao as descreve (o K sairia viesado)."""
    return x['mod'] is not None and bool(x['mod']._E.get('subida') or x['mod']._E.get('parada'))


def pwm(t, u):
    """Integra u como o servos.py. A linha k guarda o u aplicado entre t[k-1] e t[k]."""
    th = np.zeros_like(t)
    for k in range(1, len(t)):
        th[k] = min(max(th[k - 1] + K_PWM * u[k] * (t[k] - t[k - 1]), 0.0), TH_MAX)
    return th


RE_LOG = re.compile(r'^\d{8}T\d{6}__')   # nome de arquivo da arvore ensaios/: <id_carro>__<tipo>...


def arq_do_log(p):
    """O csv de um log: o proprio arquivo (arvore ensaios/) ou <pasta>/car.csv (coleta crua)."""
    p = Path(p)
    return p if p.is_file() else p / 'car.csv'


def nome_do_log(p):
    """id do carro (AAAAMMDD_HHMMSS) para a arvore ensaios/; nome da pasta para a coleta crua."""
    p = Path(p)
    return p.stem.split('__')[0].replace('T', '_') if RE_LOG.match(p.name) else p.name


def carrega(pasta):
    a = np.genfromtxt(arq_do_log(pasta), delimiter=',', names=True)
    x = {c: np.asarray(a[c], float) for c in ('t', 'v', 'vref', 'u')}
    x['th'] = pwm(x['t'], x['u'])
    Ts = float(np.median(np.diff(x['t'][2:])))                 # grade uniforme (jitter do loop)
    tg = x['t'][1] + Ts * np.arange(int((x['t'][-1] - x['t'][1]) / Ts) + 1)
    x.update(Ts=Ts, tg=tg, vg=np.interp(tg, x['t'], x['v']), thg=np.interp(tg, x['t'], x['th']),
             ug=np.interp(tg, x['t'], x['u']), nome=nome_do_log(pasta))
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


def par_str(mod):
    """Ganhos do preset como o main.py os usa (TAU_FF so no 'avanco_ff')."""
    s = f"KP {mod.KP}, TD {mod.TD}, TAU_VDES {mod.TAU_VDES}, K_FF {mod.K_FF}"
    return s + (f", TAU_FF {mod.TAU_FF}" if mod.TAU_FF else '')


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
             sat=100 * float(np.mean(np.abs(uu) >= 0.99)), ruido=float(np.mean(np.abs(np.diff(uu[fim])))), pwm_fim=float(np.mean(tht[fim])),
             trava=float(np.sum((uu == 0.0) & (vv > 1.5)) * x['Ts']), alvo=alvo, os_sim=float('nan'))
    x['metr'] = m                             # o RESUMO junta as corridas de cada ensaio
    nan_s = lambda a: 'nao acomodou' if np.isnan(a) else f'{a:.1f} s'
    L = [f"  {x['nome']}  {x['ensaio']}  alvo {alvo:.2f} m/s" + (f"   ({par_str(mod)})" if mod else ''),
         f"    partida (v > 0,05)             {m['partida']:.1f} s          <- zona morta do ESC, nao e atraso da malha",
         f"    subida 10% -> 90%              {m['subida']:.1f} s",
         f"    pico                           {m['pico']:.2f} m/s  (sobressinal {m['ultra']:+.1f}%) em t = {m['t_pico']:.1f} s",
         f"    acomodacao +-5% / +-2%         {nan_s(m['a5'])} / {nan_s(m['a2'])}   (contada desde o inicio da rampa)",
         f"    cruzamentos do alvo apos o pico {m['cruz']}      oscilacao pico a pico (ult. 3 s)  {m['ondula']:.3f} m/s",
         f"    erro final (ult. 2 s)          {m['erro_final']:+.3f} m/s   pwm no fim {m['pwm_fim']:.1f} graus",
         f"    travado em VELMAX              {m['trava']:.1f} s   (u = 0 com v > 1,5: o set_u zera u e o acelerador fica parado)",
         f"    IAE contra vref                {m['iae']:.2f} m",
         f"    u: max {m['umax']:+.2f}  min {m['umin']:+.2f}  saturado (|u| >= 0,99) {m['sat']:.1f}% do tempo   ruido de u (media |du|, ult. 3 s) {m['ruido']:.4f}"]
    if ajx and mod:   # K x KP / K_FF: uma vez so, no bloco da planta (vale para a sessao toda)
        K = K_DE(ajx['G'])
        os_sim, st_sim = malha(mod, K, ajx['DB'], ajx['tau'], ajx['d'], ts=E['ts'])
        m['os_sim'] = os_sim
        L.append(f"    modelo ajustado previa (K {K:.2f} desta corrida): sobressinal {os_sim:.1f}%, acomoda +-5% em {st_sim:.1f} s   <- diferenca grande = modelo nao explica o carro")
    return L


def bloco_degraus(x, E, mod, ajx):
    """Patamares: por degrau, e a curva estatica v x pwm lida direto do fim de cada degrau."""
    t, v, th, u, perfil = x['t'], x['v'], x['th'], x['u'], E['perfil']
    t0 = float(t[np.argmax(x['vref'] > 0)])
    dv = np.gradient(np.convolve(v, np.ones(15) / 15, 'same'), t)   # dv/dt com ~0,4 s de suavizacao
    L = [f"  {x['nome']}  {x['ensaio']}   ({par_str(mod)})",
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
    w = (t >= t0) & (t <= t0 + E['ts'] + 0.05)
    x['metr'] = dict(alvo=float(x['vref'].max()), ultra=float('nan'), a5=float('nan'), os_sim=float('nan'),   # RESUMO: so IAE
                     iae=float(np.trapezoid(np.abs(x['vref'][w] - v[w]), t[w])),
                     trava=float(np.sum((u[w] == 0.0) & (v[w] > 1.5)) * x['Ts']))
    L.append(f"    IAE contra vref {x['metr']['iae']:.2f} m   travado em VELMAX {x['metr']['trava']:.1f} s")
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


# ---- malha fechada simulada ------------------------------------------------------------
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


class Planta:
    """Planta nao linear para simular a malha: u -> pwm (integra, clip) -> v_ss = G max(0, pwm - DB) -> lag tau -> atraso d
    -> MA30 (+ ruido). Na descida a desaceleracao nao passa de a_coast (o carro so 'coasta'; 0,45 = maximo observado em 212026).
    passo(t, h, u, dist) devolve o v que o log mostraria; dist [m/s2] e um disturbio aditivo (rampa)."""
    def __init__(self, K, DB=3.5, tau=0.72, d=0.08, a_coast=0.45, seed=0, sig=0.004):
        self.G, self.DB, self.tau, self.d, self.a_coast = G_DE(K), DB, tau, d, a_coast
        self.pwm = self.vr = 0.0
        self.t_hist, self.v_hist = [0.0], [0.0]
        self.buf = [0.0] * N_MA
        self.rng, self.sig = np.random.default_rng(seed), sig

    def passo(self, t, h, u, dist=0.0):
        self.pwm = min(max(self.pwm + K_PWM * u * h, 0.0), TH_MAX)
        vss = self.G * max(0.0, self.pwm - self.DB)
        a = max((vss - self.vr) / self.tau, -self.a_coast) + dist
        self.vr = max(self.vr + a * h, 0.0)
        self.t_hist.append(t); self.v_hist.append(self.vr)
        self.buf = self.buf[1:] + [float(np.interp(t - self.d, self.t_hist, self.v_hist)) + self.rng.normal(0, self.sig)]
        return float(np.mean(self.buf))


# ---- logs: sempre os caminhos que voce passa (nada de busca automatica) ----------------
def pastas_dos_logs(caminhos):
    """Cada caminho pode ser: um arquivo da arvore ensaios/ (AAAAMMDDTHHMMSS__tipo...csv); uma pasta dessa
    arvore (ex.: ensaios/real/roxo/2026-10-01); um car.csv ou a pasta de um log da coleta crua da GUI; ou uma
    pasta com coletas cruas (car.csv em qualquer profundidade). Se o mesmo log aparecer mais de uma vez
    fica so a primeira ocorrencia (mesmo id e mesmo tamanho). Retorna uma lista de 'logs': arquivo ou pasta."""
    achados, vistos = [], set()
    for c in map(Path, caminhos):
        if c.is_file() and RE_LOG.match(c.name):
            logs = [c]
        elif str(c).lower().endswith('.csv'):
            logs = [c.parent]
        elif (c / 'car.csv').is_file():
            logs = [c]
        elif c.is_dir():
            logs = sorted(f.parent for f in c.rglob('car.csv')) + sorted(f for f in c.rglob('*.csv') if RE_LOG.match(f.name))
        else:
            logs = []
        if not logs:
            sys.exit(f'nao achei nenhum log em: {c}\n(passe uma pasta de ensaios/real/roxo/<data>, um arquivo dessa arvore, a pasta de um log ou um car.csv)')
        for p in logs:
            chave = (nome_do_log(p), arq_do_log(p).stat().st_size)
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
def figura(x, ident, mods, ajustado):
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
    uteis = [x for x in xs if x['v'].max() > (0.05 if x['aberto'] else 0.3)]
    for x in uteis:
        g, tau = ajusta_um(x)
        x['K_corrida'], x['tau_corrida'] = K_DE(g), tau
    # so as corridas de malha fechada denunciam a bateria pelo K (a aberta lenta tem K mal determinado)
    fechadas = [x for x in uteis if not x['aberto'] and not fora_do_ajuste(x)]
    if fechadas and not args.todos:
        Kmed = float(np.median([x['K_corrida'] for x in fechadas]))
        for x in uteis:
            x['bateria_ok'] = x['aberto'] or (0.8 * Kmed <= x['K_corrida'] <= 1.25 * Kmed)
    else:
        for x in uteis:
            x['bateria_ok'] = True
    grupo = [x for x in uteis if x['bateria_ok'] and not fora_do_ajuste(x)]

    linhas = []
    P = linhas.append
    P(f"=== {len(xs)} log(s), {len(uteis)} com movimento, {len(grupo)} no ajuste ===")
    P(f"  (K relativo = DB {DB_REF} e d {D_REF} fixos: so compara corridas para achar bateria diferente; o K de verdade e o da planta ajustada)")
    for x in uteis:
        P(f"  {x['nome']}  {x['ensaio']:11s}  K relativo {x['K_corrida']:.2f}  tau {x['tau_corrida']:.2f}"
          + ('   (subida: fora do ajuste da planta)' if fora_do_ajuste(x) else
             '' if x['bateria_ok'] else '   <-- FORA do grupo (bateria/ganho diferente): nao entra no ajuste'))

    aj = None
    if grupo:
        aj = ajusta_conjunto(grupo)
        aj['K'] = float(np.median([K_DE(g) for g in aj['G']]))
        P(f"\n=== planta ajustada (corridas do grupo, DB/tau/d comuns, G por corrida) ===")
        P(f"  DB = {aj['DB']:.2f} graus   tau = {aj['tau']:.3f} s   d = {aj['d']:.3f} s   K = {aj['K']:.2f} m/s^2 por u   rmse = {aj['rmse']:.4f} m/s")
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
        # o K medido contra o que os controladores assumem (o K e o unico numero de planta que o main.py usa)
        Ks = [K_DE(g) for g in aj['G']]
        aj['incerto'] = amplitude < 0.4
        P(f"\n  K medido (mediana das corridas) = {aj['K']:.2f}   (por corrida: {min(Ks):.2f} a {max(Ks):.2f})"
          + ("   <- K INCERTO: rode 'patamares' e 'u_degrau' (aviso acima)" if aj['incerto'] else ''))
        for nome in ('avanco', 'avanco_ff', 'avanco_rt'):
            mk = _MODS.get(nome)
            if mk is None:
                continue
            P(f"  '{nome}': malha KP*K = {mk.KP}*{aj['K']:.2f} = {mk.KP * aj['K']:.2f} (projeto 0,55-0,6)"
              + (f";  K {'<=' if aj['K'] <= mk.K_FF else '>'} K_FF {mk.K_FF}" if mk.TAU_FF else ''))
            if mk.TAU_FF and aj['K'] > mk.K_FF:   # o feedforward empurraria mais do que o carro precisa
                aj.setdefault('ff', f"ATENCAO: K medido {aj['K']:.2f} acima do K_FF do '{nome}' ({mk.K_FF}): mais sobressinal que o simulado")
        # previsao por corrida
        P("\n=== previsao por corrida: rmse [m/s] (fit %) ===")
        P("  'ajustado agora' JA VIU todas as corridas (so mede o ajuste). 'ajustado sem esta' e a prova honesta: DB, tau e d das outras corridas; so o G e refeito aqui.")
        cab = ['ajustado agora', 'ajustado sem esta']
        P('  ' + f"{'corrida':17s}" + ''.join(f'{n:>22s}' for n in cab))
        for i, x in enumerate(grupo):
            cols = [preve(x['thg'], x['Ts'], aj['G'][i], aj['DB'], aj['tau'], aj['d']), ajuste_cego(grupo, i) if len(grupo) >= 2 else None]
            P('  ' + f"{x['nome']:17s}" + ''.join(f"{'-':>22s}" if c is None else f"{rmse(c - x['vg']):>15.3f} ({fit_pct(x['vg'], c):3.0f}%)" for c in cols))

    # metricas por ensaio (sem veredito: a leitura e sua)
    P("\n=== metricas por ensaio ===")
    for x in uteis:
        mod = x['mod']
        if mod:
            mod.V_REF, mod.V_GUARDA = x['V_REF'], x['V_GUARDA']
        E = mod._E if mod else dict(ts=float(x['t'][-1]), perfil=None)
        i = next((k for k, g in enumerate(grupo) if g is x), None)
        ajx = dict(aj, G=aj['G'][i]) if (aj is not None and i is not None) else None
        if x['aberto']:
            linhas_x = bloco_aberto(x, E)
        elif not (x['vref'] > 0).any():
            continue
        elif E['perfil'] is not None:
            linhas_x = bloco_degraus(x, E, mod, ajx)
        else:
            linhas_x = bloco_rampa(x, E, mod, ajx)
        linhas += linhas_x

    # RESUMO: o que decide o A/B, uma linha por ensaio de rampa (media das corridas, min a max entre parenteses)
    P("\n=== RESUMO ===")
    cont = {}
    for x in xs:
        cont[x['ensaio']] = cont.get(x['ensaio'], 0) + 1
    P("  logs: " + ', '.join(f"{n} x{c}" for n, c in cont.items()))
    if aj is not None:
        P(f"  planta: K {aj['K']:.2f}, tau {aj['tau']:.2f} s, DB {aj['DB']:.2f} graus"
          + ("   (K INCERTO: faltam 'patamares'/'u_degrau')" if aj['incerto'] else ''))
        if aj.get('ff'):
            P(f"  {aj['ff']}")
    lotes = {}
    for x in uteis:
        if 'metr' in x:
            lotes.setdefault((x['ensaio'], round(x['metr']['alvo'], 2)), []).append(x['metr'])
    if lotes:
        P(f"  {'ensaio @ alvo':20s} {'n':>2s}   {'sobressinal %':19s} {'acomoda +-5%':13s} {'IAE [m]':8s} {'travado VELMAX':15s} {'modelo previa %':15s}")
        for (ens, alvo), ms in lotes.items():
            ov, a5, iae = [m['ultra'] for m in ms], [m['a5'] for m in ms], [m['iae'] for m in ms]
            n_nan = int(np.sum(np.isnan(a5)))
            acom = 'nao acomodou' if n_nan == len(a5) else f"{np.nanmean(a5):.1f} s" + (f" ({n_nan} n/a)" if n_nan else '')
            sim = [m['os_sim'] for m in ms if not np.isnan(m['os_sim'])]
            sob = '-' if np.isnan(ov).all() else f"{np.mean(ov):5.1f} ({min(ov):.1f} a {max(ov):.1f})"
            if sob == '-':                    # frenagem: so IAE e travado
                acom = '-'
            P(f"  {ens + ' @ ' + f'{alvo:.2f}':20s} {len(ms):2d}   {sob:21s} {acom:13s} {np.mean(iae):<8.2f} "
              f"{sum(m['trava'] for m in ms):5.1f} s{'':9s} {(f'{np.mean(sim):5.1f}' if sim else '-'):15s}")
        P("  (sobressinal e acomodacao contra o alvo; IAE = integral de |vref - v|, menor e melhor; travado = s somados com u = 0 e v > 1,5;")
        P("   modelo previa = sobressinal simulado na planta ajustada: medido bem acima dele = o simulador subestima, como em 25/09)")
        P("  (frenagem e patamares: so IAE e travado (o sobressinal de cada degrau esta no relatorio da corrida); subida: sobressinal = pico apos o topo)")

    texto = '\n'.join(linhas)
    print('\n' + texto)
    if '\n=== RESUMO ===' in linhas:   # so o RESUMO, num arquivo so (sobrescrito a cada analise)
        SAIDA.mkdir(parents=True, exist_ok=True)
        cab = f"analisa.py sobre {len(xs)} log(s): {', '.join(x['nome'] for x in xs)}"
        (SAIDA / 'RESUMO.txt').write_text(cab + '\n' + '\n'.join(linhas[linhas.index('\n=== RESUMO ==='):]), encoding='utf-8')
        print(f"\nresumo: {SAIDA / 'RESUMO.txt'}")
    falhas = []
    for x, pasta in zip(xs, pastas):
        i = next((k for k, g in enumerate(grupo) if g is x), None)
        ajx = dict(aj, G=aj['G'][i]) if (aj is not None and i is not None) else None
        fig = figura(x, x['ensaio'], MODELOS, ajx)
        if Path(pasta).is_file():   # arvore ensaios/: os dados nao recebem arquivo gerado
            png, txt = SAIDA / f"{x['nome']}.png", SAIDA / f"{x['nome']}.txt"
        else:
            png, txt = pasta / 'analise.png', pasta / 'analise.txt'
        try:   # arquivo aberto em outro programa (visualizador de imagem) nao pode derrubar o relatorio
            png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png, dpi=110)
            txt.write_text(texto, encoding='utf-8')
        except OSError as e:
            falhas.append(f"{png} ({e.strerror or e})")
        plt.close(fig)
    print(f"\nfiguras: {SAIDA if any(Path(p).is_file() for p in pastas) else '<pasta do log>'}/*.png ({len(xs) - len(falhas)} de {len(xs)})")
    for f in falhas:
        print(f"  NAO consegui gravar {f}: feche o arquivo se estiver aberto e rode de novo")


if __name__ == '__main__':
    main()
