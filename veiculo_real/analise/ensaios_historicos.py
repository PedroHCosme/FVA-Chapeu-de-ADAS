# -*- coding: utf-8 -*-
"""Os ensaios que o main.py do carro tinha ate 11/10/2026 (ENSAIO, ROTEIRO, presets), congelados so para a analise
repassar os logs antigos (analisa.reconhece, organiza_logs, sintonia, parada).

Nao roda sozinho: analisa.carrega_main(ensaio) executa este arquivo DENTRO do namespace do main.py (que ja tem ENSAIO,
V_REF, TAXA_RAMPA_V, o ADAS e controlador_longitudinal) e troca a referencia, os ganhos e o control_func pelos do ensaio.
A lei e o supervisor continuam sendo os do main.py: uma fonte so. Os valores abaixo sao os de quando cada log rodou, nao
os atualize. Historico dos controladores: analise/HISTORICO_CONTROLADORES.md.
"""
PISTA_M        = 500    # m -- pista disponivel; o ensaio se recusava a rodar se passasse de PISTA_M - 1
ATRASO_PARTIDA = 1.0    # s -- tempo ate o carro sair do lugar (zona morta), so para estimar a distancia
V_CORTE        = 0.7    # m/s -- 'u_degrau' para de acelerar aqui...
PWM_CORTE      = 8.0    # graus -- ...ou quando o acelerador estimado chega aqui
T_SONAR        = 10.0   # s -- 'sonar': tempo da caixa em cada distancia

# perfil de referencia: lista de (t [s], alvo [m/s]) ou (t, alvo, taxa [m/s^2]); None = rampa ate V_REF
ROTEIRO = {
	'avanco'     : dict(projeto='avanco',    ts=13.0, perfil=None),   # lei escolhida (P + D + feedforward), rampa ate V_REF, ~15 m
	'avanco_ff'  : dict(projeto='avanco_ff', ts=13.0, perfil=None),   # a mesma malha com o feedforward completo (K_FF, TAU_FF)
	'avanco_rt'  : dict(projeto='avanco_rt', ts=35.0, perfil=None),   # a mesma rampa com a lei re-sintonizada (analise/sintonia.py)
	'frenagem'   : dict(projeto='avanco',    ts=20.0, perfil=[(0, 1.0), (8, 0.5), (13, 0.0, 0.25)]),   # referencia DECRESCENTE, ~11 m: 1,0 > 0,5 rapido > 0 devagar
	'frenagem_ff': dict(projeto='avanco_ff', ts=20.0, perfil=[(0, 1.0), (8, 0.5), (13, 0.0, 0.25)]),
	'frenagem_rt': dict(projeto='avanco_rt', ts=20.0, perfil=[(0, 1.0), (8, 0.5), (13, 0.0, 0.25)]),

	# subida: parte ~5 m antes da rampa, V_REF que voce editar (1,0 ou 1,3). 17 s (a 1,3 m/s ~20 m: 5 de plano + rampa + topo; a 1,0 ~15 m).
	# Fora do ajuste da planta (o disturbio enviesaria o K); a duracao diferente da do 'avanco' e o que separa os dois na analise.
	'subida_ff'  : dict(projeto='avanco_ff', ts=17.0, perfil=None, subida=True, pista=24.0),
	'subida_rt'  : dict(projeto='avanco_rt', ts=17.0, perfil=None, subida=True, pista=24.0),

	# parada: cruzeiro em V_REF ate t = 8 s (~8 m a 1,3 m/s) e CORTA (u = -1, vref 0: a mesma acao da emergencia do ADAS);
	# mede a distancia e o tempo de parada. 7 s depois para (ts). Segurem o carro. Lei 'avanco_rt'.
	'parada'     : dict(projeto='avanco_rt', ts=15.0, perfil=None, parada=8.0),

	'sonar'      : dict(ts=72.0, sonar=[0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 0.0]),   # carro PARADO, alguem segura uma caixa nestas distancias [m] (0 = sem caixa)
	'u_degrau'   : dict(ts=30.0, aberto=[0.5, 0.8, 1.0, 1.0]),                 # malha ABERTA: niveis de u a partir do repouso (zona morta e atraso)
	'patamares'  : dict(projeto='avanco',    ts=16.0,                          # degraus de referencia 0,6 > 0,9 > 1,1 > 0,8 > 0,5 m/s (separa ganho G de zona morta)
	                    perfil=[(0, 0.6), (3.5, 0.9), (6.5, 1.1), (9.5, 0.8), (12, 0.5), (13.5, 0.0)]),
}
_E = ROTEIRO[ENSAIO]

# presets de ganhos (KP, TD, TAU_VDES, K_FF, TAU_FF: significado no main.py). O 'avanco_rt' e o que o main.py usa hoje.
_PROJETOS = {
	'avanco'   : dict(KP=0.2, TD=0.6, TAU_VDES=0.5, K_FF=4.0, TAU_FF=0.0),
	'avanco_ff': dict(KP=0.2, TD=0.6, TAU_VDES=0.5, K_FF=2.9, TAU_FF=0.61),
	'avanco_rt': dict(KP=0.225, TD=0.6, TAU_VDES=0.2, K_FF=2.9, TAU_FF=0.61),   # analise/sintonia.py, ponto #5, V_GUARDA 1,35 (02/10)
}
PROJETO = _E.get('projeto', 'avanco')
KP, TD, TAU_VDES, K_FF, TAU_FF = (_PROJETOS[PROJETO][k] for k in ('KP', 'TD', 'TAU_VDES', 'K_FF', 'TAU_FF'))


def _pontos(perfil, taxa):
	# vertices (t, v) do perfil: cada alvo e alcancado com a taxa dada, ate o proximo comecar
	pts = [(0.0, 0.0)]
	for t_i, alvo, *r in perfil:
		v_i = float(np.interp(t_i, *zip(*pts)))
		pts = [p for p in pts if p[0] < t_i] + [(t_i, v_i), (t_i + abs(alvo - v_i) / (r[0] if r else taxa), alvo)]
	return pts

_PONTOS = _pontos(_E['perfil'], TAXA_RAMPA_V) if _E.get('perfil') else None

def _vdes(tau):
	# referencia tau segundos depois da primeira chamada do controle
	if 'parada' in _E and tau >= _E['parada']:
		return 0.0
	if _PONTOS is None:
		return min(V_REF, TAXA_RAMPA_V * tau)
	return float(np.interp(tau, *zip(*_PONTOS)))

def _u_degrau(car, s):
	# cada nivel de u a partir do repouso ate v >= V_CORTE (ou acelerador estimado >= PWM_CORTE); depois
	# u = -1 ate o acelerador zerar (u = 0 SEGURARIA a velocidade: o servos.py integra u) e espera parar
	k_pwm = np.rad2deg(0.4 * 0.08 * 5.16)   # graus/s de acelerador por unidade de u
	dt = car.dt if 0.0 < car.dt < 0.5 else 0.0
	s.setdefault('fase', 'espera'); s.setdefault('i', 0); s.setdefault('pwm', 0.0)
	s['tf'] = s.get('tf', 0.0) + dt
	def vai(f):
		s['fase'], s['tf'] = f, 0.0
	niveis = _E['aberto']
	if s['fase'] == 'espera':
		if s['tf'] > 1.0 and abs(car.v) < 0.05 or s['tf'] > 4.0:
			vai('sobe' if s['i'] < len(niveis) else 'fim')
		return 0.0
	if s['fase'] == 'sobe':
		u = min(niveis[s['i']], 1.0)
		s['pwm'] = min(s['pwm'] + k_pwm * u * dt, 25.0)
		if car.v >= V_CORTE or s['pwm'] >= PWM_CORTE or s['tf'] > 6.0:
			vai('desce')
		return u
	if s['fase'] == 'desce':
		s['pwm'] = max(s['pwm'] - k_pwm * dt, 0.0)
		if s['pwm'] <= 0.0 and s['tf'] > 0.3:
			s['i'] += 1
			vai('espera')
		return -1.0
	return 0.0

def _sonar(car, s):
	# carro parado (u = -1 mantem o acelerador em 0); a cada T_SONAR s a caixa muda de distancia
	t0 = s.setdefault('t0', car.t)
	k = min(int((car.t - t0) // T_SONAR), len(_E['sonar']) - 1)
	s['alvo'] = _E['sonar'][k]
	car.vref = 0.0
	car.set_u(-1.0)

def control_func(car):
	if 'aberto' in _E:
		car.vref = 0.0
		car.set_u(_u_degrau(car, _estado))
		return
	if 'sonar' in _E:
		_sonar(car, _estado)
		return
	t0 = _estado.setdefault('t0', car.t)   # a referencia comeca na primeira chamada
	vdes = _vdes(car.t - t0)
	frear = 'parada' in _E and car.t - t0 >= _E['parada']   # 'parada': depois do corte, u = -1 direto (nao a lei)
	if ADAS_ON and not frear:
		vdes, frear = _adas(car, vdes)
		if frear:
			_zera_lei()
	car.vref = vdes                        # so para o log e a GUI
	u = controlador_longitudinal(car, vdes)   # roda sempre: os filtros nao ficam velhos
	car.set_u(-1.0 if frear else u)
