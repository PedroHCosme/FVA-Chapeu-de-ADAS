# -*- coding: utf-8 -*-
"""Porta de entrada dos logs do carro: classifica, nomeia e guarda cada log NOVO na arvore ensaios/.

A GUI baixa a pasta logs/ INTEIRA do carro a cada coleta (e o carro nunca a esvazia), entao o que chega vem
cheio de logs que ja temos. Fluxo:
  1. na GUI, aba de coleta, destino = ensaios/_entrada          (ela cria _entrada/<carro>/<id>/car.csv)
  2. ../simulador/.venv/Scripts/python.exe analise/organiza_logs.py            # a seco: mostra o que faria
  3. ../simulador/.venv/Scripts/python.exe analise/organiza_logs.py --aplica   # copia, confere o md5, atualiza o
     ensaios/catalogo.csv e esvazia a entrada (so o que ja esta guardado; o que ficou retido continua la)
Rode de veiculo_real/. Pode passar outras pastas no lugar da entrada (ex.: um pen drive de um colega).
  --reclassifica [--aplica]   refaz controlador e parametros de TODOS os logs ja guardados (apos mudar a classificacao)

O que ele faz com cada car.csv (decide pelo md5 e pelo conteudo, nunca pelo nome da pasta):
  novo               classifica, escolhe o nome e a pasta de data e guarda
  ja no acervo       mesmo md5 do catalogo: so sai da entrada
  truncada           prefixo de um log que ja temos (baixada com o arquivo ainda sendo gravado): sai da entrada
  RETIDO incompleto  cortado no meio e sem versao completa: baixe de novo
  RETIDO conflito    mesmo id, conteudo diferente (nao e prefixo): decisao sua
  RETIDO data        data real que nao e terca/quinta (ou e posterior ao download): confirme com --data

Nome do arquivo:  <id_carro>__<tipo>[__<parametros>][__nao-moveu].csv      em  real/roxo/<data real>/
So entra no nome o que se SABE: a aparencia do sinal.
  tipo u-*   : u e a entrada (malha aberta)      u-degrau (u constante por nivel), u-nulo (u = 0 o tempo todo)
  tipo vref-*: a referencia e a entrada (malha fechada), pela forma do vref logado:
               rampa (sobe em >= 0.3 s) | degrau (< 0.3 s) | escada (varios niveis, o 2o e maior que o 1o)
               descida (varios niveis, o 2o e menor que o 1o) | pulsos (vref volta a 0 e sobe de novo)
  parametros : u0.5-0.8 | v1.3 | v1.0-0.6-1.0-0.6-0.2-0.0     (niveis de u | alvo(s) de vref, lidos do sinal)
  nao-moveu  : v_max < 0.05 m/s.

catalogo.csv: o que nao esta no sinal fica FORA do nome e vai para as colunas. Parametros do controlador (kp ... niveis_freio_graus):
so os que MUDAM o u e que o log confirma: o repasse da lei do main.py (com estes valores) reproduz o u logado em >= 97%
das amostras (acerto_replay). Vazio = nao se aplica a esse tipo de ensaio OU nao identificado. NAO registramos o que nao deixa rastro
no log (PISTA_M, ATRASO_PARTIDA, ts, camera, us_buzzer): copiar o valor atual do main.py seria registrar o que nao sabemos que rodou.
  kp [rad/s]  td_s [s]  tau_vdes_s [s]  k_nom [m/s^2 por u]  v_ref_ms [m/s]  taxa_rampa_v_ms2 [m/s^2]  perfil_vref (t:alvo[@taxa];...)
  niveis_u (u-degrau, lidos do sinal)  v_corte_ms  pwm_corte_graus  (corte do u-degrau)
  v_freio_ms  t_sobe_s  t_pulso_s  niveis_freio_graus  (freio_esc)      parametros_extra: JSON, lei do simulador (logs de 24/09)

Data real = dia em que o arquivo chegou ao PC pela 1a vez (o relogio do PC esta certo; o do carro nao, e o atraso
muda de sessao para sessao). Aula so as tercas e quintas. Um log rodado numa aula e baixado so depois (o carro nao
apaga nada) chega com a data errada: confirme com  --data ID=AAAA-MM-DD  (ou --data AAAA-MM-DD para todos).
"""
import argparse, contextlib, csv, hashlib, io, json, os, shutil, sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent
REAL = AQUI.parent
RAIZ = REAL.parent
DEST = RAIZ / 'ensaios'
os.chdir(REAL)                      # analisa.py usa caminhos relativos a veiculo_real/
sys.path.insert(0, str(AQUI))
with contextlib.redirect_stdout(io.StringIO()):
    import analisa as A

V_MIN_MOVEU = 0.05    # m/s: v_max abaixo disso = o carro nao saiu do lugar
V_MAX = 1.5           # car.py: |v| > VELMAX forca u = 0
DIAS_DE_AULA = (1, 3)  # weekday(): terca, quinta

PROJ = {'avanco': (.8, .6, .5), 'base': (.6, 0., .3)}
KN = (1.0, 1.09, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.75, 5.0)
# lei do simulador/main.py que rodou no carro em 24/09 (copia de capitulo/figuras.py:143)
SIM = dict(K=1.0, A0=0.096, C=0.00425, EPS=0.05, KP=4.0, KI=2.67, TT=0.5, TV=0.3, TU=0.10)

PARAM_COLS = ['kp', 'td_s', 'tau_vdes_s', 'v_ref_ms', 'taxa_rampa_v_ms2', 'perfil_vref', 'niveis_u', 'v_corte_ms',
              'pwm_corte_graus', 'v_freio_ms', 't_sobe_s', 't_pulso_s', 'niveis_freio_graus', 'parametros_extra']
COLUNAS = (['arquivo', 'id_carro', 'data_carro', 'data_real', 'data_real_origem', 'baixado_em', 'dominio', 'veiculo', 'tipo', 'malha',
            'params', 'controlador', 'ensaio_main', 'projeto', 'k_nom'] + PARAM_COLS +
           ['vref_max', 'dur_s', 'v_max', 'moveu', 'origem_rotulo', 'acerto_replay', 'md5', 'bytes', 'n_copias', 'caminhos_originais', 'nota'])
RELABEL = ['controlador', 'ensaio_main', 'projeto', 'k_nom', 'origem_rotulo', 'acerto_replay'] + PARAM_COLS   # o que o --reclassifica refaz


def fmt(x):
    s = f'{x:.2f}'.rstrip('0')
    return s + '0' if s.endswith('.') else s


def g(x):
    return f'{x:g}' if isinstance(x, (int, float, np.floating)) else x


def md5(b):
    return hashlib.md5(b).hexdigest()


def completo(b):
    """O logger grava linha a linha: um arquivo cortado no meio nao termina em \\n ou termina com linha curta."""
    if not b.endswith(b'\n'):
        return False
    linhas = b.rstrip(b'\n').split(b'\n')
    return len(linhas) > 2 and linhas[-1].count(b',') == linhas[0].count(b',')


def platos(y, t, tol, min_s):
    out, i = [], 0
    while i < len(y):
        j = i
        while j + 1 < len(y) and abs(y[j + 1] - y[i]) <= tol:
            j += 1
        if t[j] - t[i] >= min_s:
            out.append(round(float(np.median(y[i:j + 1])), 2))
        i = j + 1
    return out


def forma(x):
    """So a aparencia do sinal. Retorna dict com 'tipo' (vref-* decidido aqui pela forma do vref)."""
    t, u, vr, v = x['t'] - x['t'][0], x['u'], x['vref'], x['v']
    f = dict(n=len(t), dur=float(t[-1]), vmax=float(v.max()), vrmax=float(vr.max()), moveu=bool(v.max() >= V_MIN_MOVEU))
    if vr.max() == 0:
        if np.abs(u).max() < 1e-2:      # ruido numerico (~2e-4) conta como u = 0; o menor degrau real e 0.3
            f['tipo'] = 'u-nulo'
        else:
            f['tipo'] = 'u-degrau'
            f['niveis'] = sorted(set(p for p in platos(u, t, 1e-3, 1.0) if p > 0.05))
        return f
    pl = platos(vr, t, 0.01, 2.0)
    f['pl'] = pl
    nz_idx = [i for i, p in enumerate(pl) if p > 0.05]
    nz = [pl[i] for i in nz_idx]
    zero_no_meio = any(abs(pl[i]) <= 0.05 for i in range(nz_idx[0], nz_idx[-1])) if nz_idx else False
    i0, i95 = np.argmax(vr > 0.01), np.argmax(vr >= 0.95 * vr.max())
    sobe = float(t[i95] - t[i0])
    if zero_no_meio:
        f['tipo'] = 'vref-pulsos'
    elif nz and max(nz) - min(nz) >= 0.1:
        f['tipo'] = 'vref-descida' if nz[1] < nz[0] else 'vref-escada'
    else:
        f['tipo'] = 'vref-degrau' if sobe < 0.3 else 'vref-rampa'
    return f


_mods = {}
def _mod(e):
    if e not in _mods:
        _mods[e] = A.carrega_main(e)
    return _mods[e]


def perfil_str(perfil):
    """perfil de vref do ROTEIRO como 't:alvo[@taxa];...'; vazio = rampa ate V_REF."""
    return ';'.join(f'{g(p[0])}:{g(p[1])}' + (f'@{g(p[2])}' if len(p) > 2 else '') for p in perfil) if perfil else ''


def parametros(m):
    """Parametros customizaveis do main.py que MUDAM o u deste ensaio (valores do modulo no instante do acerto)."""
    E = m._E
    if 'aberto' in E:
        return dict(v_corte_ms=g(m.V_CORTE), pwm_corte_graus=g(m.PWM_CORTE))
    p = dict(td_s=g(m.TD), tau_vdes_s=g(m.TAU_VDES),   # kp vai no formato antigo (ver repasse)
             taxa_rampa_v_ms2=g(m.TAXA_RAMPA_V), perfil_vref=perfil_str(E.get('perfil')))
    if m.TAU_FF:                                         # feedforward do 'avanco_ff' (K_FF, TAU_FF)
        p['parametros_extra'] = json.dumps(dict(k_ff=m.K_FF, tau_ff_s=m.TAU_FF))
    if not E.get('perfil'):                              # V_REF so age na rampa simples
        p['v_ref_ms'] = g(m.V_REF)
    return p


# modulo(s) do main.py que podem ter gerado cada forma de vref (so escolhe o que repassar)
CANDIDATOS = {'vref-descida': ['frenagem', 'frenagem_ff', 'frenagem_rt'], 'vref-escada': ['patamares'], 'vref-pulsos': []}   # pulsos: freio_esc, saiu do main.py
RAMPA = ['avanco', 'avanco_ff', 'avanco_rt', 'subida_ff', 'subida_rt', 'parada']   # vref-rampa (e o que nao estiver acima)


def acerto(m, x):
    """(fracao das amostras em que o u bate, amostras comparadas): o 2o numero desempata a favor de quem cobre mais do log."""
    u, _ = A.repassa(m, x['t'], x['v'])
    ok = ~np.isnan(u)
    ok[:3] = False
    ok &= x['t'] <= m._E['ts'] + 0.05
    return (float(np.mean(np.abs(u[ok] - x['u'][ok]) < 2e-3)) if ok.any() else 0.0), int(ok.sum())


def repasse(x, f):
    """Melhor (acerto, info): o control_func do main.py reproduz o u logado (|dif| < 2e-3)? Tenta o formato antigo
    (kp, k_nom: logs ate 02/10) e o preset como esta no main.py (unico jeito de casar o 'avanco_rt'), com e sem a guarda
    de VELMAX (logs de antes dela). Empate: vence quem cobre mais do log (a 'subida' de 17 s contra o 'avanco' de 13 s)."""
    melhor, chave = (0.0, None), (0.0, 0)
    for e in CANDIDATOS.get(f['tipo'], RAMPA):
        m = _mod(e)
        pre, vg0 = m._PROJETOS[m.PROJETO], m.V_GUARDA
        ff = pre['TAU_FF'] > 0                           # 'avanco_ff'/'avanco_rt': feedforward proprio
        # (rotulo do projeto, k_nom, kp registrado, KP, TD, TAU_VDES, K_FF)
        tentativas = [(pn, kn, par[0], par[0] / kn, par[1], par[2], pre['K_FF'] if ff else kn) for pn, par in PROJ.items() for kn in KN]
        tentativas.append((m.PROJETO, None, pre['KP'], pre['KP'], pre['TD'], pre['TAU_VDES'], pre['K_FF']))
        for vg in A.guardas(m, x):
            m.V_GUARDA = vg
            for pn, kn, kp_reg, kp, td, tau, kff in tentativas:
                m.KP, m.TD, m.TAU_VDES, m.K_FF = kp, td, tau, kff
                m.V_REF = round(f['vrmax'], 2)
                ac, n = acerto(m, x)
                if (round(ac, 3), n) > chave:
                    chave = (round(ac, 3), n)
                    proj = pn + ('_ff' if ff and kn is not None else '')
                    p = parametros(m)
                    if kn is None:   # formato novo: a coluna kp e do formato antigo (kp = KP*k_nom); o KP vai no JSON
                        p['parametros_extra'] = json.dumps(dict(json.loads(p.get('parametros_extra') or '{}'), kp_u_por_ms=kp))
                    else:
                        p['kp'] = g(kp_reg)
                    melhor = (ac, dict(ensaio=e, proj=proj, kn=kn, params=p))
                if ac >= 0.999 and m._E['ts'] + 0.05 >= x['t'][-1]:   # bateu e cobre o log inteiro: nada ganha disso
                    m.V_GUARDA = vg0
                    return melhor
        m.V_GUARDA = vg0
    return melhor


def repasse_aberto(x, f):
    """u-degrau: repassa o _u_aberto do main.py com os niveis programados e com os niveis lidos do sinal."""
    m = _mod('u_degrau')
    programados = m._E['aberto']
    melhor = (0.0, None)
    for niveis in (list(programados), list(f['niveis'])):
        m._E['aberto'] = niveis
        try:
            ac, _ = acerto(m, x)
        finally:
            m._E['aberto'] = programados
        if ac > melhor[0]:
            melhor = (ac, dict(params=parametros(m)))
    return melhor


def lei_simulador(t, v, vref, ff_ref=True):
    """Copia de capitulo/figuras.py:143 (aquele arquivo executa tudo ao ser importado): controlador do simulador/main.py."""
    K, A0, C, EPS = SIM['K'], SIM['A0'], SIM['C'], SIM['EPS']
    KP, KI, TT, TV, TU = SIM['KP'], SIM['KI'], SIM['TT'], SIM['TV'], SIM['TU']
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
        Ii += dt * (KI * e + (up - ubp) / TT)
        ub = (KP * e + Ii + ff + A0 * np.tanh(v[j] / EPS) + C * v[j] * abs(v[j])) / K
        uc = float(np.clip(ub, 0.0, 1.0))
        uf = uf + np.clip(dt / TU, 0, 1) * (uc - uf) if dt > 0 else uf
        up, ubp = uf, ub
        out[j + 1] = 0.0 if abs(v[j]) > V_MAX else uf
    return out


def forense_sim(x):
    for nome, kw in (('sim-orig', {}), ('sim-orig-sem-ff', dict(ff_ref=False))):
        u = lei_simulador(x['t'], x['v'], x['vref'], **kw)
        ok = ~np.isnan(u)
        if ok.any() and float(np.sqrt(np.mean((u - x['u'])[ok] ** 2))) < 1e-3:
            return nome, kw
    return None, None


def rotula(arq, id_, dom):
    """Tudo que o log permite saber: tipo/params do nome (pela forma do sinal) e controlador + parametros (por repasse)."""
    x = A.carrega(arq)
    f = forma(x)
    r = dict(f, controlador='', ensaio_main='', projeto='', k_nom='', origem_rotulo='', acerto='', nota='', **{c: '' for c in PARAM_COLS})
    if f['tipo'].startswith('u-'):
        r['params'] = ''
        if f['tipo'] == 'u-degrau':
            r['params'] = 'u' + '-'.join(fmt(p) for p in f['niveis'])
            r['niveis_u'] = '-'.join(fmt(p) for p in f['niveis'])
            r['origem_rotulo'] = 'forma'
            if dom == 'real':
                ac, info = repasse_aberto(x, f)
                r['acerto'] = round(ac, 3)
                if ac >= 0.97:
                    r.update(ensaio_main='u_degrau', origem_rotulo='repasse main.py', **info['params'])
                else:
                    r['nota'] = f'repasse do u_degrau atual = {ac:.0%} (ensaio de uma versao anterior do main.py?): v_corte/pwm_corte nao identificados'
        return r
    if dom == 'sintetico':
        r['origem_rotulo'] = 'forma'
    else:
        sim, kw = forense_sim(x) if id_.startswith('20260924') else (None, None)      # 24/09 rodou o controlador do simulador
        if sim:
            extra = dict(lei='simulador/main.py', **SIM, u_min=0.0, u_max=1.0, ff_ref=kw.get('ff_ref', True))
            r.update(controlador=sim, origem_rotulo='repasse simulador/main.py', acerto=1.0, parametros_extra=json.dumps(extra))
        else:
            ac, info = repasse(x, f)
            # formato novo (preset como esta no main.py) nao tem k_nom: o controlador e so o nome do projeto
            ctl = info and (info['proj'] if info['kn'] is None else f"{info['proj']}-kn{fmt(info['kn'])}")
            if ac >= 0.97:
                r.update(controlador=ctl, ensaio_main=info['ensaio'], projeto=info['proj'],
                         k_nom='' if info['kn'] is None else g(info['kn']), origem_rotulo='repasse main.py',
                         acerto=round(ac, 3), **info['params'])
            else:
                r.update(controlador='desconhecido', origem_rotulo='nenhum', acerto=round(ac, 3))
                if info:   # formato antigo da nota quando ha k_nom (o --aplica so acrescenta notas NOVAS)
                    r['nota'] = (f"melhor repasse {info['ensaio']}/{info['proj']}/kn{fmt(info['kn'])} = {ac:.0%}" if info['kn'] is not None
                                 else f"melhor repasse {info['ensaio']}/{ctl} = {ac:.0%}")
    escada = r['tipo'] in ('vref-escada', 'vref-descida')
    v = '-'.join(fmt(p) for p in f['pl']) if escada else fmt(f['vrmax'])
    r['params'] = 'v' + v            # o controlador NAO entra no nome: so no catalogo
    return r


def carrega_catalogo(dest):
    p = dest / 'catalogo.csv'
    if not p.exists():
        return []
    with open(p, encoding='utf-8', newline='') as fh:
        return [{c: r.get(c, '') for c in COLUNAS} for r in csv.DictReader(fh)]


def escreve_catalogo(dest, linhas):
    with open(dest / 'catalogo.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, COLUNAS)
        w.writeheader()
        w.writerows(sorted(linhas, key=lambda r: r['arquivo']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fontes', nargs='*', help='pasta(s) com os car.csv baixados (padrao: <destino>/_entrada)')
    ap.add_argument('--aplica', action='store_true', help='grava (copia, atualiza o catalogo, esvazia a entrada); padrao: a seco')
    ap.add_argument('--data', action='append', default=[], metavar='[ID=]AAAA-MM-DD',
                    help='data real confirmada por voce, para um id (20260930_214209=2026-10-01) ou para todos os logs novos')
    ap.add_argument('--reclassifica', action='store_true', help='refaz controlador e parametros de todos os logs ja guardados')
    ap.add_argument('--destino', default=str(DEST), help=argparse.SUPPRESS)       # so para teste
    args = ap.parse_args()
    dest = Path(args.destino)
    entrada = dest / '_entrada'
    acervo = carrega_catalogo(dest)

    if args.reclassifica:
        mudou = 0
        for r in acervo:
            novo = rotula(dest / r['arquivo'], r['id_carro'], r['dominio'])
            if r['ensaio_main'] and not novo['ensaio_main']:   # ensaio que saiu do main.py: fica o rotulo de quando rodou
                print(f"  {r['id_carro']}  {r['controlador'] or '-':16s} mantido ('{r['ensaio_main']}' nao esta mais no main.py)")
                continue
            assert (novo['tipo'], novo['params']) == (r['tipo'], r['params']), f"{r['arquivo']}: o nome mudaria ({novo['tipo']}, {novo['params']})"
            antes = {c: r[c] for c in RELABEL}
            r.update({c: novo[c] for c in RELABEL if c != 'acerto_replay'}, acerto_replay=novo['acerto'])
            if novo['nota'] and novo['nota'] not in r['nota']:
                r['nota'] = (r['nota'] + '; ' if r['nota'] else '') + novo['nota']
            mudou += {c: str(v) for c, v in antes.items()} != {c: str(r[c]) for c in RELABEL}   # o csv guarda texto
            print(f"  {r['id_carro']}  {r['controlador'] or '-':16s} acerto {r['acerto_replay'] or '-'}  kp={r['kp'] or '-'} td={r['td_s'] or '-'} tau_vdes={r['tau_vdes_s'] or '-'} k_nom={r['k_nom'] or '-'}")
        print(f'{len(acervo)} logs reclassificados, {mudou} com colunas alteradas')
        if args.aplica:
            escreve_catalogo(dest, acervo)
            print('catalogo.csv regravado')
        else:
            print('(a seco: nada foi gravado; rode com --aplica)')
        return

    fontes = [Path(f) for f in args.fontes] or [entrada]
    data_de = {a.split('=')[0]: a.split('=')[1] for a in args.data if '=' in a}
    data_todos = next((a for a in args.data if '=' not in a), None)
    md5_acervo = {r['md5']: r for r in acervo}
    id_acervo = {r['id_carro']: r for r in acervo}
    if acervo and not acervo[0]['kp'] and any(r['dominio'] == 'real' and r['controlador'].endswith(('kn1.6', 'kn4.0', 'kn3.5')) and not r['kp'] for r in acervo):
        print('AVISO: ha logs no catalogo sem parametros do controlador; rode --reclassifica --aplica\n')

    arqs = sorted(p for f in fontes for p in f.rglob('car.csv'))
    dados = {p: p.read_bytes() for p in arqs}
    por_md5 = {}
    for p, b in dados.items():
        por_md5.setdefault(md5(b), []).append(p)
    por_id = {}
    for h, ps in por_md5.items():
        por_id.setdefault(ps[0].parent.name, []).append(h)

    # 1. cada conteudo vira uma decisao: (status, paths, detalhe). 'novo' ainda passa pela checagem de data adiante.
    decisao = {}
    for id_, hs in por_id.items():
        hs = sorted(hs, key=lambda h: -len(dados[por_md5[h][0]]))
        for h in hs:
            ps, b = por_md5[h], dados[por_md5[h][0]]
            if h in md5_acervo:
                tree = dest / md5_acervo[h]['arquivo']
                decisao[h] = ('ja-no-acervo', ps, '') if tree.exists() and md5(tree.read_bytes()) == h else \
                    ('RETIDO acervo', ps, f'o catalogo cita {md5_acervo[h]["arquivo"]} mas o arquivo nao confere')
                continue
            ref = None                                          # o log completo que ja temos (acervo) ou o maior desta entrada
            if id_ in id_acervo:
                ref = (dest / id_acervo[id_]['arquivo']).read_bytes()
            elif h != hs[0] and completo(dados[por_md5[hs[0]][0]]):
                ref = dados[por_md5[hs[0]][0]]
            if ref is not None and ref != b:
                decisao[h] = ('truncada', ps, 'prefixo de um log completo') if ref.startswith(b) else \
                    ('RETIDO conflito', ps, 'mesmo id, conteudo diferente (nao e prefixo)')
            elif not completo(b):
                decisao[h] = ('RETIDO incompleto', ps, 'cortado no meio e sem versao completa: baixe de novo')
            else:
                decisao[h] = ('novo', ps, '')

    # 2. rotulo + nome + data dos novos
    novos = []
    for h, (st, ps, det) in sorted(decisao.items(), key=lambda kv: kv[1][1][0].parent.name):
        if st != 'novo':
            continue
        rep = ps[0]
        id_ = rep.parent.name
        ncol = dados[rep].split(b'\n', 1)[0].count(b',') + 1
        if ncol not in (4, 14):
            decisao[h] = ('RETIDO formato', ps, f'{ncol} colunas (o carro grava 14, scripts gravam 4)')
            continue
        dom = 'sintetico' if ncol == 4 else 'real'
        veiculo = rep.parent.parent.name
        if dom == 'real' and veiculo == entrada.name:
            decisao[h] = ('RETIDO pasta', ps, 'coloque o log dentro de uma pasta com o nome do carro (a GUI faz isso: _entrada/roxo/<id>/car.csv)')
            continue
        baixado = datetime.fromtimestamp(min(p.stat().st_mtime for p in ps))
        if dom == 'real':
            if id_ in data_de:
                data, origem = data_de[id_], 'confirmada pelo usuario (--data)'
            elif data_todos:
                data, origem = data_todos, 'confirmada pelo usuario (--data)'
            else:
                data, origem = baixado.strftime('%Y-%m-%d'), '1o download no PC (relogio do PC)'
            if data > baixado.strftime('%Y-%m-%d'):
                decisao[h] = ('RETIDO data', ps, f'{data} e posterior ao download ({baixado:%Y-%m-%d}): um ensaio nao roda depois de baixado')
                continue
            if datetime.strptime(data, '%Y-%m-%d').weekday() not in DIAS_DE_AULA:
                decisao[h] = ('RETIDO data', ps, f'{data} nao e terca/quinta; se foi aula em outro dia ou baixado depois, confirme com --data {id_}=AAAA-MM-DD')
                continue
            pasta = f'real/roxo/{data}'
        else:
            data, origem, pasta = '', 'sintetico', 'sintetico'
        r = rotula(rep, id_, dom)
        t_carro = datetime.strptime(id_, '%Y%m%d_%H%M%S')
        nome = '__'.join([t_carro.strftime('%Y%m%dT%H%M%S'), r['tipo']] + ([r['params']] if r['params'] else [])
                         + ([] if r['moveu'] else ['nao-moveu'])) + '.csv'
        linha = dict(
            arquivo=f'{pasta}/{nome}', id_carro=id_, data_carro=t_carro.isoformat(), data_real=data, data_real_origem=origem,
            baixado_em=baixado.strftime('%Y-%m-%d %H:%M'), dominio=dom, veiculo=veiculo,
            tipo=r['tipo'], malha='aberta' if r['tipo'].startswith('u-') else 'fechada', params=r['params'],
            controlador=r['controlador'], ensaio_main=r['ensaio_main'], projeto=r['projeto'], k_nom=r['k_nom'],
            vref_max=round(r['vrmax'], 2), dur_s=round(r['dur'], 1), v_max=round(r['vmax'], 2), moveu=r['moveu'],
            origem_rotulo=r['origem_rotulo'], acerto_replay=r['acerto'], md5=h, bytes=len(dados[rep]), n_copias=len(ps),
            caminhos_originais=';'.join(p.relative_to(RAIZ).as_posix() if RAIZ in p.parents else p.as_posix() for p in ps),
            nota=r['nota'], **{c: r[c] for c in PARAM_COLS})
        assert set(linha) == set(COLUNAS), set(linha) ^ set(COLUNAS)
        novos.append(linha)
    # carimbo do carro quase colado no log anterior (<20 min) mas com outra data real: a data pelo download pode estar errada
    todos = sorted([r for r in acervo if r['dominio'] == 'real'] + [r for r in novos if r['dominio'] == 'real'], key=lambda r: r['id_carro'])
    for a, b in zip(todos, todos[1:]):
        gap = (datetime.fromisoformat(b['data_carro']) - datetime.fromisoformat(a['data_carro'])).total_seconds() / 60
        if b in novos and b['data_real'] != a['data_real'] and 0 <= gap < 20:
            b['nota'] = (b['nota'] + '; ' if b['nota'] else '') + (
                f"CONFIRMAR DATA: carimbo do carro {gap:.0f} min depois de {a['id_carro']} ({a['data_real']}), mas data real {b['data_real']}")
    nomes = {r['arquivo'] for r in acervo}
    for r in novos:
        assert r['arquivo'] not in nomes, r['arquivo']
        nomes.add(r['arquivo'])

    # 3. relatorio
    print(f'{len(arqs)} car.csv em {", ".join(map(str, fontes))} -> {len(decisao)} conteudos distintos')
    for st, n in sorted(Counter(v[0] for v in decisao.values()).items()):
        print(f'  {st:18s} {n}')
    for r in novos:
        print(f"  NOVO  {r['id_carro']} -> {r['arquivo']}  [{r['controlador'] or '-'}]" + (f"  <{r['nota']}>" if 'CONFIRMAR' in r['nota'] else ''))
    for h, (st, ps, det) in decisao.items():
        if st.startswith('RETIDO'):
            print(f"  {st}: {ps[0].parent.name}: {det}")

    # 4. aplica
    if not args.aplica:
        print('\n(a seco: nada foi copiado nem apagado; rode com --aplica)')
        return
    for r in novos:
        d = dest / r['arquivo']
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(por_md5[r['md5']][0], d)
        assert md5(d.read_bytes()) == r['md5'], r['arquivo']
    if novos:
        escreve_catalogo(dest, acervo + novos)
    novos_md5 = {r['md5'] for r in novos}
    apagados = 0
    for h, (st, ps, det) in decisao.items():
        if st in ('ja-no-acervo', 'truncada') or h in novos_md5:
            for p in ps:
                if entrada in p.parents:                     # so a entrada e esvaziada, e so o que ja esta guardado
                    p.unlink()
                    apagados += 1
    if entrada.exists():                                     # pastas de corrida que sobraram vazias (corrida que nao gravou)
        for d in sorted((d for d in entrada.rglob('*') if d.is_dir()), key=lambda d: -len(d.parts)):
            if not any(d.iterdir()):
                d.rmdir()
    print(f'\ncopiado e conferido (md5): {len(novos)} log(s) novo(s); {apagados} arquivo(s) tirado(s) da entrada; '
          f'{sum(1 for v in decisao.values() if v[0].startswith("RETIDO"))} retido(s) continuam la')


if __name__ == '__main__':
    main()
