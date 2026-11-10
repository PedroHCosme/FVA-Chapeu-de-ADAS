# -*- coding: utf-8 -*-
########################################
# Disciplina: Topicos em Engenharia de Controle e Automacao IV (ENG075):
# Fundamentos de Veiculos Autonomos - 2026/2
# Professores: Armando Alves Neto e Leonardo A. Mozelli
# Cursos: Engenharia de Controle e Automacao
# DELT - Escola de Engenharia
# Universidade Federal de Minas Gerais
########################################
# CARRO REAL: segue a fita amarela com o Stanley (camera), velocidade pelo PD + feedforward
# e supervisor ADAS (ultrassom). Cada corrida grava car.csv, sonar.csv e lateral.csv.
#
# O carro so esterca sozinho com a chave de DIRECAO do radio em AUTO (em RC o car.set_steer obedece ao radio).
# A tracao deste arquivo (car.set_u) e sempre automatica, qualquer que seja a chave.
from fva_car import Car
from fva_car.filter import AlphaFilter, MovingAverage
from collections import deque
import numpy as np
import threading
import time
import os

########################################
# CONDICOES DO ENSAIO
########################################
V_REF          = 1.0    # m/s -- velocidade de cruzeiro na reta
V_CURVA        = 1.0    # m/s -- alvo com a direcao no batente (entre 0 e 20 graus cai linear de V_REF ate aqui)
TAXA_RAMPA_V   = 0.9    # m/s^2 -- inclinacao maxima da referencia, subindo e descendo
TEMPO_CORRIDA  = 40.0   # s -- duracao da corrida
ADAS_ON        = False  # False = sem supervisor (SEGUREM o carro)
VIDEO_VISAO    = True   # transmite a visao para a aba Camera da GUI, porta 8080
VIDEO_HZ       = 10.0   # Hz -- quadros do video; a compressao so roda com a GUI conectada
TELEMETRIA_HZ  = 10.0   # Hz -- linhas DATA para a GUI
GRAVA_QUADROS  = True   # guarda quadros crus da camera para testar a visao offline (analise/compara_filtros.py)
GRAVA_HZ       = 2.0    # Hz -- quadros guardados por segundo; vao para logs/<corrida>/quadros/ depois que o carro para

########################################
# LIMITES DE SEGURANCA
########################################
V_GUARDA       = 1.6    # m/s -- acima disto u = -1 (None desliga). So protege abaixo do VELMAX 1,5 do car.py: acima dele o car.py zera u e o acelerador congela
ACELERADOR_MAX = 20.0   # graus -- teto do acelerador. So limita abaixo do teto do servos.py (~18 graus). Graus -> m/s muda com bateria e piso
D_SEM_FITA     = 0.6    # m -- andou isto sem ver fita: freia ate ela voltar
T_SEM_FITA     = 1.0    # s -- este tempo sem medida da fita: freia tambem (nao depende do encoder)
E_MAX_FITA     = 0.6    # m -- medida com |e| maior que isto nao e a fita: descarta

########################################
# BUSCA DA FITA: fita perdida -> 1) arco para a frente, para o lado onde ela sumiu -> 2) re pelo caminho ja percorrido -> 3) desiste e para
# As duas chaves False: so freia e espera a fita aparecer, como antes.
########################################
BUSCA_ARCO     = False  # etapa 1
BUSCA_RE       = False  # etapa 2 (sem sensor atras: SEGUREM o carro nos primeiros testes)
V_BUSCA        = 0.3    # m/s -- velocidade do arco
D_ARCO         = 1.0    # m -- maior distancia do arco
T_ARCO         = 5.0    # s -- maior tempo do arco (vale mesmo com o encoder lendo zero)
V_RE           = 0.25   # m/s -- velocidade da re
K_RE           = 0.075  # u por m/s de erro -- 1/3 de KP: o servos.py integra a re 3 vezes mais rapido
ACELERADOR_RE_MAX = 12.0   # graus -- teto do acelerador na re
D_RE_MAX       = 2.0    # m -- maior distancia da re (inclui desfazer o arco)
T_RE_MAX       = 16.0   # s -- maior tempo da re
T_PARAR        = 4.0    # s -- espera o carro parar antes de trocar de marcha
V_PARADO       = 0.05   # m/s -- abaixo disto conta como parado
T_RETOMADA     = 0.3    # s -- depois de voltar para a frente, tempo para a visao ter quadro novo
V_RETOMADA     = 0.5    # m/s -- teto de velocidade depois de reencontrar a fita, ate andar D_PROGRESSO nela
MAX_BUSCAS     = 3      # buscas seguidas sem andar D_PROGRESSO na fita: desiste (ex.: fita acabou)
D_PROGRESSO    = 2.0    # m -- andou isto seguindo a fita: zera a contagem de buscas
MIN_PIXELS_BUSCA = 150  # pixels usados para aceitar a fita durante a busca (mais exigente: nao confundir com porta ou caixa)
TRECHO_BUSCA   = 0.25   # m -- trecho minimo de fita para aceitar durante a busca

########################################
# DIRECAO: Stanley
#   delta = psi_e + arctan(K_STANLEY * e / (K_SOFT + v))
#   e     [m]   distancia lateral da fita ao centro do eixo dianteiro (> 0 = fita a esquerda do carro)
#   psi_e [rad] angulo da fita em relacao ao eixo do carro (> 0 = fita aponta para a esquerda)
########################################
K_STANLEY      = 2.33   # 1/s -- taxa de convergencia do erro lateral
K_SOFT         = 0.2    # m/s -- evita K_STANLEY*e/v explodir com v pequeno
SINAL_DIRECAO  = 1.0    # +1 se a fita a esquerda vira as rodas para a ESQUERDA; senao -1
TRIM_DIRECAO   = 1.0    # graus -- somado a todo comando de direcao (servos.set_trim)
L_CARRO        = 0.36   # m -- entre-eixos; preve a fita entre quadros e nos vaos
BATENTE        = np.deg2rad(20.0)   # rad -- esterco maximo do carro

########################################
# CAMERA -> CHAO: pinhole sem rolagem. x para a frente, y para a esquerda, origem no centro do eixo dianteiro
########################################
ALTURA_CAMERA  = 0.255  # m -- altura do centro da lente ao chao
X_CAMERA       = 0.175  # m -- lente a frente (+) ou atras (-) do eixo dianteiro
FOV_HORIZONTAL = 60.0   # graus -- abertura horizontal da webcam
LINHA_HORIZONTE = 100   # pixel -- linha da imagem onde o chao some ao longe: da a inclinacao da camera
RESOLUCAO      = (640, 480)   # (largura, altura) do camera.py

########################################
# VISAO: fita amarela em HSV na escala do OpenCV (H 0-179, S e V 0-255); o camera.py entrega BGR
########################################
H_MIN          = 15
H_MAX          = 42
S_MIN          = 55     # fita velha/desbotada: 40; pegando madeira clara: 75-90
V_MIN          = 60
FILTRO_RELATIVO = False # True = aguenta reflexo: compara a amarelidade com o chao da mesma linha, em vez de exigir saturacao
AMARELO_REL_MIN = 16    # min(R,G) - B acima da mediana da linha (com reflexo a fita fica em 13-28, o chao perto de 0)
H_MAX_REL      = 45     # H maximo do filtro relativo
FAIXA_VISAO   = (0.30, 0.95)   # fracao da altura da imagem que a visao usa (acima: corredor e portas)
X_MAX_VISAO    = 2.0    # m -- chao mais longe que isto nao entra (la cada pixel vale varios cm)
JANELA_FITA    = 0.12   # m -- so contam pixels amarelos a menos disto da fita prevista
JANELA_FINA    = 0.03   # m -- segunda passada do ajuste, e tolerancia do RANSAC
TRECHO_CURVA   = 0.5    # m -- fita vista num trecho maior que isto: ajusta arco; menor: mantem a curvatura prevista
C_MAX          = 0.6    # 1/m -- teto da meia curvatura medida (raio de 0,8 m)
MIN_PIXELS     = 40     # pixels amarelos minimos para medir (pedaco de fita picotada ja basta)
MAX_PIXELS     = 3000   # acima disto usa uma amostra regular dos pixels

########################################
# VELOCIDADE: PD + feedforward (analise/sintonia.py, ponto #5 da fronteira de Pareto)
#   u = KP*erro + KP*TD*d(erro)/dt + (dvdes/dt + TAU_FF*d2vdes/dt2) / K_FF      erro = media_movel(vdes) - v
# O servos.py integra u, entao no acelerador esta lei e um PI + feedforward (analise/prova_integrador.py).
########################################
KP             = 0.225  # u por m/s de erro
TD             = 0.6    # s -- tempo derivativo (0 desliga); derivada filtrada com TD/5
TAU_VDES       = 0.2    # s -- filtro de 1a ordem da referencia
K_FF           = 2.9    # m/s^2 por u -- o MAIOR K medido, para o feedforward nunca empurrar demais
TAU_FF         = 0.61   # s -- atraso da planta que o feedforward compensa (0 desliga)

########################################
# ADAS (ultrassom). Calibracao do sensor: analise/calibra_adas.py
#   emergencia: d < d_gatilho(v) -> u = -1, travado ate d > D_LIBERA
#   camada suave: limita a referencia pela distancia de frenagem e pelo lider que anda
#   Ultrassom sem eco = caminho livre.
########################################
SONAR_VIES     = 0.00   # m -- leitura menos distancia real
SONAR_RUIDO    = 0.005  # m -- desvio das leituras
SONAR_ALCANCE  = 2.900  # m -- maior distancia com leituras confiaveis
A_FREIO        = 0.5    # m/s^2 -- desaceleracao so por atrito (analise/parada.py, 12 paradas)
T_REACAO       = 0.7    # s -- do corte (u = -1) ate a velocidade cair 5% (analise/parada.py)
D_PARADA       = 0.1    # m -- folga entre o para-choque e o obstaculo, parado
A_PLAN         = 0.30   # m/s^2 -- a camada suave planeja frear com isto (< A_FREIO: sobra margem)
D_LIBERA       = 1.5    # m -- parado por obstaculo, so volta a andar com distancia maior que esta
TH_GAP         = 1.0    # s -- seguindo lider: distancia desejada = D_PARADA + TH_GAP * v
K_ACC          = 0.4    # 1/s -- vdes <= v_lider + K_ACC * (d - distancia desejada)
JANELA_AP      = 0.6    # s -- janela da reta d(t) que da a velocidade de aproximacao
JANELA_MEDIANA = 0.35   # s -- mediana das leituras (3 leituras a 10 Hz): rejeita eco espurio
T_SEGURA_ECO   = 1.0    # s -- sumiu o eco com obstaculo perto: segura a ultima leitura por este tempo
D_SEGURA_ECO   = 1.0    # m -- "perto" da linha acima
V_LIDER_MIN    = 0.2    # m/s -- acima disto o obstaculo conta como lider que anda

_estado = {}                          # memoria compartilhada entre as chamadas e as threads
_vdes_filt = AlphaFilter(alpha=1.0)   # alpha recalculado a cada chamada com car.dt

########################################
# CONTROLE DE VELOCIDADE
########################################
def _filtra_referencia(car, vdes_bruto):
	# devolve a referencia filtrada e a derivada dela
	if car.dt > 0:
		_vdes_filt.alpha = np.clip(car.dt / TAU_VDES, 0.0, 1.0)
	vdes_anterior = _vdes_filt.value
	vdes = _vdes_filt.filter(vdes_bruto)
	dvdes_dt = (vdes_bruto - vdes_anterior) / TAU_VDES
	return vdes, dvdes_dt

def _erro_velocidade(car, vdes):
	# referencia atrasada pela MESMA media movel do car.v: o P nao reage ao atraso do sensor
	vdes_ma = _estado.setdefault('vdes_ma', MovingAverage(n=car.v_filt.n))
	return vdes_ma.filter(vdes) - car.v

def _derivada_erro(car, erro):
	# derivada filtrada com TD/5: devolve a fase que a media movel tira
	if TD <= 0.0 or car.dt <= 0:
		return 0.0
	erro_ant = _estado.get('erro_ant', erro)
	derro_dt = _estado.get('derro', 0.0)
	b = car.dt / (TD / 5.0 + car.dt)
	derro_dt += b * ((erro - erro_ant) / car.dt - derro_dt)
	_estado['derro'], _estado['erro_ant'] = derro_dt, erro
	return derro_dt

def _derivada2_referencia(car, dvdes_dt):
	if TAU_FF <= 0.0 or car.dt <= 0:
		return 0.0
	dvdes_ant = _estado.get('dvdes_ant', dvdes_dt)
	_estado['dvdes_ant'] = dvdes_dt
	return (dvdes_dt - dvdes_ant) / car.dt

def controlador_longitudinal(car, vdes_bruto):
	vdes, dvdes_dt = _filtra_referencia(car, vdes_bruto)
	erro = _erro_velocidade(car, vdes)
	derro_dt = _derivada_erro(car, erro)
	d2vdes_dt2 = _derivada2_referencia(car, dvdes_dt)

	termo_p = KP * erro
	termo_d = KP * TD * derro_dt
	termo_ff = (dvdes_dt + TAU_FF * d2vdes_dt2) / K_FF
	u = termo_p + termo_d + termo_ff

	if V_GUARDA is not None and car.v > V_GUARDA:
		u = -1.0
	return float(np.clip(u, -1.0, 1.0))   # u < 0 baixa o acelerador (o carro so desacelera por atrito)

def _zera_lei():
	# freando: filtros e derivadas voltam a zero, a retomada nao parte de estado velho
	_vdes_filt.reset(0.0)
	if 'vdes_ma' in _estado:
		_estado['vdes_ma'].reset(0.0)
	for chave in ('derro', 'erro_ant', 'dvdes_ant'):
		_estado.pop(chave, None)

def _alvo_velocidade(car):
	# mais devagar quanto mais a direcao esterca (nunca mais rapido, mesmo com V_CURVA acima de V_REF)
	queda_max = max(V_REF - V_CURVA, 0.0)
	fracao_esterco = min(abs(getattr(car, 'st', 0.0)) / BATENTE, 1.0)
	v_alvo = V_REF - queda_max * fracao_esterco
	if _estado.get('buscas', 0) > 0:   # acabou de perder a fita: devagar ate andar D_PROGRESSO nela
		v_alvo = min(v_alvo, V_RETOMADA)
	return v_alvo

def _rampa_referencia(car, v_alvo):
	# a referencia anda ate o alvo com no maximo TAXA_RAMPA_V: o feedforward deriva duas vezes, degrau vira pico
	passo = TAXA_RAMPA_V * (car.t - _estado.get('t_vdes', car.t))
	vdes = _estado.get('vdes', 0.0)
	vdes += float(np.clip(v_alvo - vdes, -passo, passo))
	_estado['vdes'], _estado['t_vdes'] = vdes, car.t
	return vdes

def _fita_perdida(car):
	longe_sem_fita = _estado.get('sem_fita', 0.0) > D_SEM_FITA
	tempo_sem_fita = _estado.get('visto') and car.t - _estado['linha'][0] > T_SEM_FITA
	return longe_sem_fita or tempo_sem_fita

def control_func(car):
	_atualiza_busca(car)
	modo = _estado.get('busca')
	if modo not in (None, 'arco'):
		_controle_manobra(car, modo)
		return
	_registra_trilha(car)
	v_alvo = V_BUSCA if modo == 'arco' else _alvo_velocidade(car)
	vdes = _rampa_referencia(car, v_alvo)
	frear = False
	if ADAS_ON:
		vdes, frear = _adas(car, vdes)
	if modo is None and _fita_perdida(car):
		vdes, frear = 0.0, True
	if frear:
		_zera_lei()
	car.vref = vdes                           # so para o log e a GUI
	u = controlador_longitudinal(car, vdes)   # roda sempre: os filtros nao ficam velhos
	car.set_u(-1.0 if frear else u)

########################################
# ADAS
########################################
def _d_gatilho(v):
	# distancia que o carro precisa para parar a partir de v
	d_reacao = v * T_REACAO
	d_frenagem = v * v / (2.0 * A_FREIO)
	return d_reacao + d_frenagem + D_PARADA

def _v_cruzeiro(alcance):
	# maior velocidade cujo gatilho cabe no alcance do sensor
	raiz = np.sqrt(T_REACAO ** 2 + 2.0 * (alcance - D_PARADA) / A_FREIO)
	return A_FREIO * (-T_REACAO + raiz)

V_CRUZ = _v_cruzeiro(SONAR_ALCANCE)

def _distancia_sonar(car, s):
	# distancia atual [m]; inf = caminho livre
	d_lido, valido = _estado.get('sonar', (4.0, False))
	if valido:
		d = d_lido - SONAR_VIES
		s['ult'] = (car.t, d)
		return d
	t_ultima, d_ultima = s['ult']
	if car.t - t_ultima < T_SEGURA_ECO and d_ultima < D_SEGURA_ECO:
		return d_ultima
	return np.inf

def _mediana_distancia(car, s, d):
	# atualiza as janelas de leituras e devolve a distancia filtrada, descontada a margem de ruido
	novo_hist = [(car.t, d)] if np.isfinite(d) else []
	s['hist'] = [(t, x) for t, x in s['hist'] if car.t - t < JANELA_AP] + novo_hist
	s['buf'] = [(t, x) for t, x in s['buf'] if car.t - t < JANELA_MEDIANA] + [(car.t, d)]
	return float(np.median([x for _, x in s['buf']])) - 2.0 * SONAR_RUIDO

def _velocidade_aproximacao(s, v):
	# -inclinacao da reta d(t): (v_ap, v_lider). Sem leituras suficientes: (0, None)
	h = s['hist']
	if len(h) < 4 or h[-1][0] - h[0][0] < 0.5 * JANELA_AP:
		return 0.0, None
	tt, dd = np.array(h).T
	v_ap = -float(np.polyfit(tt - tt[-1], dd, 1)[0])
	return v_ap, max(v - v_ap, 0.0)

def _atualiza_emergencia(s, d, v, v_gatilho):
	if not s['frear'] and v > 0.05 and d < _d_gatilho(v_gatilho):
		s['frear'] = True
	elif s['frear'] and d > D_LIBERA and d > _d_gatilho(v) + 0.3:
		s['frear'] = False

def _alvo_suave(vdes, d, v, v_lider):
	if not np.isfinite(d):
		return min(vdes, V_CRUZ)
	v_frenagem = float(np.sqrt(2.0 * A_PLAN * max(d - D_PARADA, 0.0)))
	alvo = min(vdes, V_CRUZ, v_frenagem)
	if v_lider is not None and v_lider > V_LIDER_MIN:
		d_desejada = D_PARADA + TH_GAP * v
		alvo = min(alvo, v_lider + K_ACC * (d - d_desejada))
	return alvo

def _adas(car, vdes):
	# devolve (vdes, frear); o sensor chega em _estado['sonar'] = (d, valid) a cada ciclo
	s = _estado.setdefault('adas', dict(buf=[], hist=[], frear=False, v_out=0.0, ult=(-1e9, 0.0)))
	d = _mediana_distancia(car, s, _distancia_sonar(car, s))
	v = max(car.v, 0.0)
	v_ap, v_lider = _velocidade_aproximacao(s, v)
	_atualiza_emergencia(s, d, v, max(v, v_ap))   # o gatilho vale para o mais rapido dos dois
	if s['frear']:
		s['v_out'] = 0.0
		_estado['adas_cod'] = 2
		return 0.0, True
	alvo = _alvo_suave(vdes, d, v, v_lider)
	s['v_out'] = min(alvo, s['v_out'] + TAXA_RAMPA_V * max(car.dt, 0.0))   # sobe devagar, desce na hora
	_estado['adas_cod'] = 1 if s['v_out'] < vdes - 1e-3 else 0
	return s['v_out'], False

########################################
# VISAO: filtro da cor
########################################
def _hsv(img):
	# matiz, saturacao e valor de uma imagem BGR, na escala do OpenCV
	arr = img.astype(np.float32)
	b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
	valor = np.maximum(np.maximum(r, g), b)
	delta = valor - np.minimum(np.minimum(r, g), b)
	d = np.maximum(delta, 1e-5)
	matiz_r = (30.0 * (g - b) / d) % 180.0
	matiz_g = 30.0 * ((b - r) / d + 2.0)
	matiz_b = 30.0 * ((r - g) / d + 4.0)
	matiz = np.where(valor == r, matiz_r, np.where(valor == g, matiz_g, matiz_b))
	saturacao = 255.0 * delta / np.maximum(valor, 1e-5)
	return matiz, saturacao, valor, delta

def _sem_pixels_isolados(m):
	# exige vizinho amarelo na horizontal e na vertical
	limpa = m.copy()
	limpa[:, 1:-1] &= m[:, :-2] | m[:, 2:]
	limpa[1:-1, :] &= m[:-2, :] | m[2:, :]
	return limpa

def _amarelo_acima_do_chao(img):
	# o reflexo soma branco aos 3 canais: a saturacao despenca, mas min(R,G) - B continua acima do chao
	arr = img.astype(np.float32)
	b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
	amarelidade = np.minimum(r, g) - b
	return amarelidade - np.median(amarelidade, axis=1, keepdims=True)

def mascara_amarela(img):
	matiz, saturacao, valor, delta = _hsv(img)
	if FILTRO_RELATIVO:
		amarelo = (_amarelo_acima_do_chao(img) >= AMARELO_REL_MIN) & (matiz >= H_MIN) & (matiz <= H_MAX_REL) & (valor >= V_MIN)
	else:
		amarelo = (delta > 0) & (matiz >= H_MIN) & (matiz <= H_MAX) & (saturacao >= S_MIN) & (valor >= V_MIN)
	return _sem_pixels_isolados(amarelo)

########################################
# VISAO: camera -> chao
########################################
def homografia(pixels, chao):
	# homografia por 4 pontos (DLT com H[2,2] = 1)
	A, b = [], []
	for (u, v), (x, y) in zip(pixels, chao):
		A += [[u, v, 1, 0, 0, 0, -u * x, -v * x], [0, 0, 0, u, v, 1, -u * y, -v * y]]
		b += [x, y]
	h = np.linalg.solve(np.array(A, float), np.array(b, float))
	return np.append(h, 1.0).reshape(3, 3)

def homografia_pinhole():
	# projeta 4 pontos do chao pela camera (altura, inclinacao pelo horizonte, foco pelo FOV) e calibra com eles
	largura, altura = RESOLUCAO
	foco = (largura / 2.0) / np.tan(np.deg2rad(FOV_HORIZONTAL) / 2.0)
	inclinacao = np.arctan((altura / 2.0 - LINHA_HORIZONTE) / foco)
	eixo_optico = np.array([np.cos(inclinacao), 0.0, -np.sin(inclinacao)])
	direita_img = np.array([0.0, -1.0, 0.0])
	baixo_img = np.array([-np.sin(inclinacao), 0.0, -np.cos(inclinacao)])
	chao = [(1.0, 0.3), (1.0, -0.3), (2.0, 0.6), (2.0, -0.6)]
	pixels = []
	for x, y in chao:
		P = np.array([x - X_CAMERA, y, -ALTURA_CAMERA])
		profundidade = P @ eixo_optico
		pixels.append((largura / 2.0 + foco * (P @ direita_img) / profundidade,
		               altura / 2.0 + foco * (P @ baixo_img) / profundidade))
	return homografia(pixels, chao)

H_CHAO = homografia_pinhole()

def no_chao(u, v):
	# pixel (coluna, linha) -> (x, y) no chao [m]; aceita numeros ou vetores
	x, y, z = H_CHAO @ np.vstack([np.ravel(u), np.ravel(v), np.ones(np.size(u))])
	if np.ndim(u):
		return x / z, y / z
	return float(x[0] / z[0]), float(y[0] / z[0])

########################################
# VISAO: ajuste da fita. A fita e um arco de circulo (reta = curvatura 0), escrito como
#   y = s + p*x + k*(x^2 + y^2)      circ = (s, p, k)
# E linear em (s, p, k): minimos quadrados direto, sem singularidade na reta (k = 0).
########################################
def _ransac(x, y, tentativas=60):
	# reta com mais pixels perto: acha a fita do zero mesmo com outras coisas amarelas no chao
	rng = np.random.default_rng(0)
	melhor, n_melhor = None, 0
	for i, j in rng.integers(len(x), size=(tentativas, 2)):
		if abs(x[j] - x[i]) < 0.05:
			continue
		inclinacao = (y[j] - y[i]) / (x[j] - x[i])
		n = np.count_nonzero(np.abs(y - y[i] - inclinacao * (x - x[i])) < JANELA_FINA)
		if n > n_melhor:
			melhor, n_melhor = (y[i] - inclinacao * x[i], inclinacao, 0.0), n
	return melhor

def _circulo(e, psi, c):
	# (s, p, k) do circulo que passa em (0, e) com inclinacao tan(psi) e y''/2 = c
	t = np.tan(psi)
	k = c / (1.0 + t * t + 2.0 * c * e)
	s = e - k * e * e
	p = t * (1.0 - 2.0 * k * e)
	return s, p, k

def _no_eixo(s, p, k):
	# (e, psi_e, c) do circulo em x = 0: raiz de k*y^2 - y + s = 0 perto de s; derivadas implicitas
	e = 2.0 * s / (1.0 + np.sqrt(max(1.0 - 4.0 * k * s, 0.0)))
	denominador = 1.0 - 2.0 * k * e
	t = p / denominador
	c = k * (1.0 + t * t) / denominador
	return e, np.arctan(t), c

def _distancia(circ, x, y):
	# distancia aproximada [m] de cada ponto ao circulo: |F| / |grad F|
	s, p, k = circ
	F = s + p * x + k * (x * x + y * y) - y
	grad_F = np.hypot(p + 2.0 * k * x, 2.0 * k * y - 1.0)
	return np.abs(F) / grad_F

def _ajusta_curvatura_fixa(xs, ys, k):
	# trecho curto nao determina a curvatura: fica k, ajusta so posicao e inclinacao
	p, s = np.polyfit(xs, ys - k * (xs * xs + ys * ys), 1)
	return s, p, k

def _trecho_determina_arco(xs):
	# trecho longo e espalhado (nao 2 montinhos)
	return np.ptp(xs) > TRECHO_CURVA and len(np.unique(np.round(xs, 1))) >= 4

def _ajusta_arco(xs, ys):
	# (s, p, k) livres, ou None se os pixels nao os determinam
	A = np.column_stack([np.ones_like(xs), xs, xs * xs + ys * ys])
	arco, _, posto, _ = np.linalg.lstsq(A, ys, rcond=None)
	return tuple(arco) if posto == 3 else None

def _refina_circulo(circ, x, y):
	# duas passadas: pixels perto da curva atual -> novo ajuste. Devolve (circ, perto) ou None
	for janela in (JANELA_FITA, JANELA_FINA):
		perto = _distancia(circ, x, y) < janela
		if np.count_nonzero(perto) < MIN_PIXELS:
			return None
		xs, ys = x[perto], y[perto]
		circ = _ajusta_curvatura_fixa(xs, ys, circ[2])
		if _trecho_determina_arco(xs):
			circ = _ajusta_arco(xs, ys) or circ
	return circ, perto

def _pixels_amarelos(img):
	# (colunas, linhas) dos pixels amarelos na FAIXA_VISAO, no maximo ~MAX_PIXELS
	altura = img.shape[0]
	linha_ini = int(altura * FAIXA_VISAO[0])
	linha_fim = int(altura * FAIXA_VISAO[1])
	vv, uu = np.nonzero(mascara_amarela(img[linha_ini:linha_fim]))
	if len(uu) > MAX_PIXELS:
		passo = len(uu) // MAX_PIXELS + 1
		vv, uu = vv[::passo], uu[::passo]
	return uu, vv + linha_ini

def mede_fita(img, prevista):
	# (e, psi_e, c) da fita no eixo dianteiro, ou None. Usa todo pixel amarelo do chao: na fita picotada basta algum pedaco.
	# prevista = (e, psi_e, c) esperado agora (so contam pixels perto dele) ou None (acha do zero).
	uu, vv = _pixels_amarelos(img)
	if len(uu) < MIN_PIXELS:
		return None
	x, y = no_chao(uu, vv)
	no_alcance = (x > 0.0) & (x < X_MAX_VISAO)
	x, y, uu, vv = x[no_alcance], y[no_alcance], uu[no_alcance], vv[no_alcance]
	_estado['desenho'] = (uu, vv, None, None)
	if len(x) < MIN_PIXELS:
		return None
	circ = _ransac(x, y) if prevista is None else _circulo(*prevista)
	if circ is None:
		return None
	if _estado.get('busca') is not None and prevista is None:
		circ = (circ[0], circ[1], _estado['fita'][2])   # reencontrando na curva: parte da curvatura conhecida, a reta do RANSAC erra psi_e
	ajuste = _refina_circulo(circ, x, y)
	if ajuste is None:
		return None
	circ, perto = ajuste
	_estado['desenho'] = (uu, vv, perto, circ)
	e, psi, c = _no_eixo(*circ)
	return float(e), float(psi), float(np.clip(c, -C_MAX, C_MAX))

def _medida_confiavel():
	# durante a busca so aceita trecho longo com muitos pixels
	uu, vv, perto, _ = _estado['desenho']
	if np.count_nonzero(perto) < MIN_PIXELS_BUSCA:
		return False
	x, _ = no_chao(uu[perto], vv[perto])
	return np.ptp(x) >= TRECHO_BUSCA

def processa_quadro(img, t_quadro):
	# mede a fita perto da prevista (ou do zero, se faz tempo que nao ve) e guarda em _estado['linha']
	buscando = _estado.get('busca') is not None
	recente = not buscando and _estado.get('visto') and _estado.get('sem_fita', 0.0) < D_SEM_FITA
	medida = mede_fita(img, _estado.get('fita') if recente else None)
	if medida is not None and abs(medida[0]) > E_MAX_FITA:   # o RANSAC achou outra coisa amarela
		medida = None
	if medida is not None and buscando and not _medida_confiavel():
		medida = None
	if medida is not None:
		_estado['linha'], _estado['visto'] = (t_quadro, *medida), True
		_estado['n_linha'] = _estado.get('n_linha', 0) + 1
	return medida

########################################
# DIRECAO
########################################
def stanley(e, psi, v):
	return psi + np.arctan2(K_STANLEY * e, K_SOFT + max(v, 0.0))

def _anda_fita(e, psi, c, ds, delta):
	# a fita vista depois de o carro andar ds com esterco delta (> 0 = esquerda)
	tan_psi = np.tan(psi)
	tan_delta = np.tan(delta)
	deslocamento = ds * (tan_psi - tan_delta) + c * ds * ds   # a fita em x = ds vira o novo x = 0; a roda anda de lado
	e_novo = e + deslocamento
	giro_fita = np.arctan(tan_psi + 2.0 * c * ds)           # a fita vira (curva)
	giro_carro = ds * tan_delta / L_CARRO                   # o carro gira
	return e_novo, giro_fita - giro_carro

def fita_prevista(car):
	# parte da ultima medida e anda com o carro desde aquele quadro: cobre o atraso da camera e os vaos da fita.
	# _estado['sem_fita'] = metros andados sem medida nova
	t_quadro, e, psi, c = _estado['linha']
	n_linha = _estado.get('n_linha', 0)
	if _estado.get('n_usada') != n_linha:   # medida nova (pelo contador: o 1o quadro pode ter o mesmo t)
		_estado['n_usada'], _estado['t_fita'], _estado['sem_fita'], _estado['fita'] = n_linha, t_quadro, 0.0, (e, psi, c)
	e, psi, c = _estado['fita']
	ds = max(car.v, 0.0) * (car.t - _estado['t_fita'])
	e, psi = _anda_fita(e, psi, c, ds, SINAL_DIRECAO * car.st)
	_estado['t_fita'], _estado['fita'], _estado['sem_fita'] = car.t, (e, psi, c), _estado['sem_fita'] + ds
	return e, psi

def direcao(car):
	# esterco [rad] para o car.set_steer
	e, psi = fita_prevista(car)
	modo = _estado.get('busca')
	if modo is None:
		return SINAL_DIRECAO * stanley(e, psi, car.v)
	if modo == 'arco':
		return SINAL_DIRECAO * _estado['lado_busca'] * BATENTE
	if modo == 're':
		return _esterco_re(car)
	return car.st   # parado, trocando de marcha: rodas onde estao

########################################
# BUSCA DA FITA: maquina de estados
#   None -> arco -> re_parando -> re -> re_saindo -> retomando -> None (fita achada) ou desistiu
########################################
CODIGO_BUSCA = {None: 0, 'arco': 1, 're_parando': 2, 're': 3, 're_saindo': 4, 'retomando': 5, 'desistiu': 6}   # coluna busca do lateral.csv

def _muda_busca(car, modo):
	_estado['busca'], _estado['t_busca'], _estado['d_busca'] = modo, None, 0.0   # t_busca: marcado no proximo ciclo (trocar de marcha trava o laco)
	print(f"BUSCA: {modo or 'fita achada'} (t = {car.t:.1f} s)", flush=True)

def _tempo_na_etapa(car):
	if _estado['t_busca'] is None:
		_estado['t_busca'] = car.t
	return car.t - _estado['t_busca']

def _reinicia_lei():
	# depois de manobrar a velocidade recomeca do zero, sem salto na referencia
	_zera_lei()
	_estado['vdes'] = 0.0
	_estado.pop('t_vdes', None)

def _registra_trilha(car):
	# caminho recente para a frente, (ds, esterco): a re refaz o mesmo caminho ao contrario
	if not BUSCA_RE:
		return
	ds = max(car.v, 0.0) * car.dt
	if ds <= 0.0:
		return
	trilha = _estado.setdefault('trilha', deque())
	trilha.append((ds, getattr(car, 'st', 0.0)))
	_estado['trilha_m'] = _estado.get('trilha_m', 0.0) + ds
	while _estado['trilha_m'] > D_RE_MAX:
		_estado['trilha_m'] -= trilha.popleft()[0]

def _esterco_re(car):
	# anda para tras na trilha: o mesmo esterco, de re, refaz o mesmo arco
	trilha = _estado.get('trilha')
	restante = abs(car.v) * car.dt
	esterco = car.st
	while trilha and restante > 0.0:
		ds, esterco = trilha[-1]
		if ds <= restante:
			restante -= ds
			trilha.pop()
			_estado['trilha_m'] -= ds
		else:
			trilha[-1] = (ds - restante, esterco)
			_estado['trilha_m'] -= restante
			restante = 0.0
	return esterco

def _desiste(car):
	car.bz.beep(1.0)
	_muda_busca(car, 'desistiu')

def _etapa_re(car):
	if BUSCA_RE and _estado.get('trilha'):
		_reinicia_lei()
		_muda_busca(car, 're_parando')
	else:
		_desiste(car)

def _comeca_busca(car):
	_estado['buscas'] = _estado.get('buscas', 0) + 1
	if _estado['buscas'] > MAX_BUSCAS:
		_desiste(car)
	elif BUSCA_ARCO:
		e, psi, _ = _estado['fita']
		_estado['lado_busca'] = 1.0 if stanley(e, psi, V_BUSCA) >= 0.0 else -1.0
		_muda_busca(car, 'arco')
	else:
		_etapa_re(car)

def _conta_progresso(car):
	_estado['d_seguindo'] = _estado.get('d_seguindo', 0.0) + max(car.v, 0.0) * car.dt
	if _estado['d_seguindo'] >= D_PROGRESSO:
		_estado['buscas'] = 0

def _atualiza_busca(car):
	modo = _estado.get('busca')
	perdida = _fita_perdida(car)
	if modo is None:
		if not (BUSCA_ARCO or BUSCA_RE):
			return
		if perdida and _estado.get('visto'):
			_estado['d_seguindo'] = 0.0
			_comeca_busca(car)
		elif not perdida:
			_conta_progresso(car)
		return
	_estado['d_busca'] += abs(car.v) * car.dt
	tempo = _tempo_na_etapa(car)
	parado = abs(car.v) < V_PARADO or tempo >= T_PARAR
	if modo == 'arco':
		if not perdida:
			_muda_busca(car, None)
		elif _estado['d_busca'] >= D_ARCO or tempo >= T_ARCO:
			_etapa_re(car)
	elif modo == 're_parando' and parado:
		car.set_reverse()
		_reinicia_lei()
		_muda_busca(car, 're')
	elif modo == 're':
		if not perdida or not _estado.get('trilha') or _estado['d_busca'] >= D_RE_MAX or tempo >= T_RE_MAX:
			_muda_busca(car, 're_saindo')
	elif modo == 're_saindo' and parado:
		car.set_forward()
		_reinicia_lei()
		_muda_busca(car, 'retomando')
	elif modo == 'retomando' and tempo >= T_RETOMADA:
		if perdida:
			_desiste(car)
		else:
			_muda_busca(car, None)
	elif modo == 'desistiu' and not perdida:
		_muda_busca(car, None)

def _u_re(car):
	# o servos.py integra u: P na velocidade vira PI no acelerador. Com o teto, nunca acelera acima de ACELERADOR_RE_MAX
	u = K_RE * (V_RE - abs(car.v))
	if abs(car.atuador.th_pwm) >= np.deg2rad(ACELERADOR_RE_MAX):
		u = min(u, 0.0)
	return u

def _controle_manobra(car, modo):
	# re: anda para tras devagar; demais etapas: parado
	if modo == 're':
		car.vref = -V_RE
		car.set_u(_u_re(car))
	else:
		car.vref = 0.0
		car.set_u(-1.0)

########################################
# VIDEO: o quadro com o que a visao fez, para a aba Camera da GUI
#   vermelho = amarelo ignorado | verde = pixels usados | magenta = fita ajustada | branco = eixo do carro | ciano = FAIXA_VISAO
########################################
COR_IGNORADO, COR_USADO = (0, 0, 255), (0, 255, 0)
COR_FITA, COR_EIXO, COR_FAIXA = (255, 0, 255), (255, 255, 255), (255, 255, 0)

def _desenha_pontos(vis, x, y, cor):
	# pontos do chao [m] -> quadrados de 5 pixels (o video sai em meia resolucao)
	a, b, c = np.linalg.inv(H_CHAO) @ np.vstack([x, y, np.ones(len(x))])
	for u, v in zip((a / c).astype(int), (b / c).astype(int)):
		if 2 <= u < vis.shape[1] - 2 and 2 <= v < vis.shape[0] - 2:
			vis[v - 2:v + 3, u - 2:u + 3] = cor

def _desenha_fita(vis, circ, xs):
	s, p, k = circ
	a = s + p * xs + k * xs * xs
	discriminante = 1.0 - 4.0 * k * a
	ok = discriminante >= 0.0
	y = 2.0 * a[ok] / (1.0 + np.sqrt(discriminante[ok]))
	_desenha_pontos(vis, xs[ok], y, COR_FITA)

def _texto_video(car):
	e, psi, _ = _estado.get('fita', (0.0, 0.0, 0.0))
	idade = car.t - _estado['linha'][0]
	return f"e {100 * e:+.1f} cm  psi {np.rad2deg(psi):+.1f}  st {np.rad2deg(car.st):+.1f}  v {car.v:.2f}  idade {idade:.2f} s"

def quadro_visao(img, car):
	import cv2   # o camera.py ja depende dele; os testes no PC nao
	vis = img.copy()
	altura = vis.shape[0]
	vis[[int(altura * FAIXA_VISAO[0]), int(altura * FAIXA_VISAO[1]) - 1]] = COR_FAIXA
	xs = np.arange(0.3, X_MAX_VISAO, 0.05)
	_desenha_pontos(vis, xs, np.zeros(len(xs)), COR_EIXO)
	uu, vv, perto, circ = _estado.get('desenho', (np.array([], int), np.array([], int), None, None))
	vis[vv, uu] = COR_IGNORADO
	if circ is not None:
		vis[vv[perto], uu[perto]] = COR_USADO
		_desenha_fita(vis, circ, xs)
	texto = _texto_video(car)
	cv2.putText(vis, texto, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
	cv2.putText(vis, texto, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
	return vis

def inicia_video(car):
	# sem video o carro anda igual: camera.py antigo no carro ou porta ocupada nao param a corrida
	if not VIDEO_VISAO:
		return
	try:
		car.cam.start_stream(fps=VIDEO_HZ)
		_estado['video'] = True
	except (AttributeError, OSError) as erro:
		print(f"VIDEO desligado: {erro!r}", flush=True)

########################################
# THREAD DE VISAO
########################################
def _guarda_quadro(img, t_quadro):
	# so guarda a referencia: a escrita em disco fica para depois da corrida, para nao atrasar a visao
	if GRAVA_QUADROS and t_quadro - _estado.get('t_gravado', -1e9) >= 1.0 / GRAVA_HZ:
		_estado['t_gravado'] = t_quadro
		_estado.setdefault('quadros', []).append((t_quadro, img))

def vision_func(car, stop_event):
	video = _estado.get('video', False)
	while not stop_event.is_set():
		t_quadro = car.t
		img = car.get_image(gray=False)
		if img is None:
			time.sleep(0.01)
			continue
		processa_quadro(img, t_quadro)
		_guarda_quadro(img, t_quadro)
		if video and car.t - _estado.get('t_video', -1.0) >= 1.0 / VIDEO_HZ:
			_estado['t_video'] = car.t
			try:
				car.cam.publish_frame(quadro_visao(img, car))
			except Exception as erro:   # o video nunca derruba a visao: sem esta thread a fita congela
				print(f"VIDEO desligado: {erro!r}", flush=True)
				video = False

########################################
# LOGS E TELEMETRIA
########################################
def _salva_sonar(car, linhas):
	# t [s], d [m], valid (0 = sem eco por 0,3 s), alvo [m] (sempre nan), adas (0 livre, 1 limitando a referencia, 2 emergencia)
	np.savetxt(os.path.join(car.logfile, 'sonar.csv'), np.array(linhas, float).reshape(-1, 5), delimiter=',',
	           header='t,d,valid,alvo,adas', comments='', fmt='%.4f')

def _salva_lateral(car, linhas):
	# t [s], t_quadro [s] (t - t_quadro = idade da medida), e_med e psi_e_med (a medida desse quadro),
	# e e psi_e (a fita prevista agora, que o Stanley usa) [m, rad], st [rad], sem_fita [m], busca (CODIGO_BUSCA)
	np.savetxt(os.path.join(car.logfile, 'lateral.csv'), np.array(linhas, float).reshape(-1, 9), delimiter=',',
	           header='t,t_quadro,e_med,psi_e_med,e,psi_e,st,sem_fita,busca', comments='', fmt='%.4f')

def _salva_quadros(car):
	# PNG sem perda, em BGR. O nome tem o t_quadro, o mesmo do lateral.csv
	quadros = _estado.get('quadros', [])
	if not quadros:
		return
	import cv2   # o camera.py ja depende dele; os testes no PC nao
	pasta = os.path.join(car.logfile, 'quadros')
	os.makedirs(pasta, exist_ok=True)
	for t_quadro, img in quadros:
		cv2.imwrite(os.path.join(pasta, f'quadro_{t_quadro:07.3f}.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
	print(f"{len(quadros)} quadros salvos em {pasta}", flush=True)

def _registra_ciclo(car, dist, valid, sonar_log, lateral_log):
	sonar_log.append((car.t, dist, int(valid), np.nan, _estado.get('adas_cod', 0)))
	t_quadro, e_med, psi_med, _ = _estado['linha']
	e, psi, _ = _estado['fita']
	lateral_log.append((car.t, t_quadro, e_med, psi_med, e, psi, car.st, _estado['sem_fita'], CODIGO_BUSCA[_estado.get('busca')]))

def _envia_telemetria(car):
	# linha DATA para a GUI, subamostrada
	if car.t - _estado.get('t_telemetria', -1e9) < 1.0 / TELEMETRIA_HZ:
		return
	_estado['t_telemetria'] = car.t
	print(
		f"DATA,"
		f"{car.t:.2f},"
		f"{car.p[0]:.2f},"
		f"{car.p[1]:.2f},"
		f"{car.v:.2f},"
		f"{car.vref:.2f},"
		f"{car.a:.2f},"
		f"{car.u:.2f},"
		f"{car.w:.2f},"
		f"{car.th:.2f}",
		flush=True
	)

########################################
# CORRIDA
########################################
def _imprime_configuracao():
	print(f"V_REF {V_REF} m/s, {TEMPO_CORRIDA:.0f} s, PD + FF (KP {KP}, TD {TD}, TAU_VDES {TAU_VDES}, K_FF {K_FF}, TAU_FF {TAU_FF}), "
	      f"Stanley (K {K_STANLEY}, K_SOFT {K_SOFT}, sinal {SINAL_DIRECAO:+.0f}, trim {TRIM_DIRECAO} graus), "
	      f"acelerador max {ACELERADOR_MAX} graus, ADAS {'ligado' if ADAS_ON else 'DESLIGADO'}, "
	      f"filtro {'relativo' if FILTRO_RELATIVO else 'saturacao'}, grava quadros {'sim' if GRAVA_QUADROS else 'nao'}, "
	      f"busca arco {'sim' if BUSCA_ARCO else 'nao'}, re {'sim' if BUSCA_RE else 'nao'}", flush=True)

def _cria_carro():
	car = Car({
		'ts'               : TEMPO_CORRIDA,
		'save'             : True,
		'logfile'          : 'logs/',
		'camera'           : True,
		'us_buzzer'        : True,
		'initial_position' : [0, 0, np.deg2rad(0)],
	})
	car.atuador.set_trim(steer=np.deg2rad(TRIM_DIRECAO))
	car.atuador.max_throttle = min(car.atuador.max_throttle, np.deg2rad(ACELERADOR_MAX))
	return car

def _laco_de_controle(car, sonar_log, lateral_log):
	while car.t < TEMPO_CORRIDA:
		if not car.step():
			break
		car.set_steer(direcao(car))
		dist, valid = car.get_distance()
		_estado['sonar'] = (dist, valid)
		control_func(car)
		_registra_ciclo(car, dist, valid, sonar_log, lateral_log)
		_envia_telemetria(car)

def _salva_logs(car, sonar_log, lateral_log):
	if not getattr(car, 'traj', None):
		return
	car.save()
	_salva_sonar(car, sonar_log)
	if lateral_log:
		_salva_lateral(car, lateral_log)

def corrida(car):
	sonar_log, lateral_log = [], []
	stop_event = threading.Event()
	thread_visao = None
	try:
		car.start_mission()
		_estado['linha'] = (car.t, 0.0, 0.0, 0.0)   # ainda sem fita
		inicia_video(car)
		thread_visao = threading.Thread(target=vision_func, args=(car, stop_event), daemon=True)
		thread_visao.start()
		_laco_de_controle(car, sonar_log, lateral_log)   # o controle fica na thread principal
	finally:
		stop_event.set()
		if thread_visao is not None:
			thread_visao.join(timeout=1.0)
		_salva_logs(car, sonar_log, lateral_log)   # salva mesmo com erro: o log e o que explica o erro
		car.close()
		_salva_quadros(car)   # demora alguns segundos: so depois que o carro parou

########################################
# main
########################################
if __name__ == "__main__":
	_imprime_configuracao()
	corrida(_cria_carro())
	print('Terminou...')
