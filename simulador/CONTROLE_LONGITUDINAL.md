# CONTROLE_LONGITUDINAL.md — do setpoint ao controlador completo

> Companheiro do [IDENTIFICACAO.md](IDENTIFICACAO.md). Aquele documento chega
> até o modelo `v̇ = K·u − c·v·|v| − a₀·sign(v) − g·senθ`. Este parte dali e
> monta o controlador que acelera **e** freia, incluindo rampa, sem medir `θ`.
> Cada decisão abaixo tem uma frase de "por que não a alternativa mais óbvia"
> — é a parte que mais vale ler antes da aula.

---

## 1. O erro de partida: pegar uma fórmula pronta

A primeira tentativa foi:

```
u(t) = ½ρv²C_dA  +  Kp·e + Ki∫e·dτ + Kd·de/dt
```

Isso é a estrutura *genérica* de cruise-control de livro-texto. Ela está
errada aqui por cinco motivos, cada um instrutivo:

1. **Falta o `a₀`** — o termo dominante do modelo (atrito seco, 0,096 m/s²).
   Sem ele o comando erra sistematicamente em baixa velocidade.
2. **O arrasto usa `ρ, C_d, A`** (aerodinâmica de manual) em vez do `c` que
   foi **medido** por regressão nos seus próprios dados. Você tem dado; usa o
   dado, não a fórmula genérica.
3. **Falta `g·senθ`.**
4. **`Ki` sem justificativa** — "PID tem Ki" não é motivo. Integrador serve
   pra rejeitar **viés constante**, não qualquer erro.
5. **`Kd` sem justificativa** — derivada amplifica ruído, e o sistema tem
   ruído medido justamente nos piores momentos (troca de comando).

**Lição:** a lei de controle se deriva do seu modelo identificado e da
evidência nos seus dados, não se importa de uma referência genérica e depois
se tenta encaixar as variáveis do seu problema nela.

---

## 2. `K+` vs `K-`: como não confundir ruído com efeito real

Rodando `melhora.py` (que separa `K` para `u>0` e `u<0`), saiu `K+=0,993,
K-=0,986` numa execução e `K+=1,001, K-=1,005` noutra — **o sinal da
diferença inverteu** entre as duas execuções, porque o número de ensaios em
`logs/` mudou entre elas (o simulador é determinístico: rodar de novo só
ajuda se for uma condição nova, cf. IDENTIFICACAO.md §11.2).

**Regra:** uma diferença que muda de sinal quando você adiciona mais dados
**é ruído**, não um efeito físico. `K+` e `K-` não justificam dois
feedforwards separados. O `K` do modelo validado (IDENTIFICACAO.md, com
intervalo de confiança e validação cega) é um só: `K≈1,00`.

---

## 3. Onde a assimetria acelerar/frear É real

A tabela de resíduo (`v̇` medido menos `v̇` previsto pelo modelo), separada
por proximidade de uma troca de comando:

```
                          n     media do residuo      rms
troca -> acelerando      212        +0,0210          0,1028
troca -> freando         152        +0,0007          0,1627
```

Duas assinaturas diferentes, e **nenhuma delas é "o K muda"**:

- **Entrando em aceleração:** viés real e repetível de **+0,021 m/s²** (o
  carro acelera mais do que o modelo prevê, por ~0,3 s). Isso é uma
  correção que o **feedforward** resolve — viés é exatamente o que
  feedforward sabe compensar.
- **Entrando em frenagem:** sem viés (média ≈0), mas variância alta
  (imprevisível). Isso feedforward **não resolve** — não tem uma constante
  pra somar num erro que não tem padrão. Só um **feedback com ganho/banda
  suficiente** absorve isso.

**Por que não LQG para a frenagem?** LQG = filtro de Kalman (estima estado
não medido) + LQR (ganho ótimo). Aqui `v` é medido direto — não há estado
escondido pra estimar — e não se sabe se o ruído é Gaussiano ou estrutural
(folga mecânica, stick-slip). LQG otimiza contra um modelo de ruído que
pode estar errado; um `P` bem dimensionado, mais barato, já ataca o problema
certo (banda/ganho).

---

## 4. Por que feedforward sozinho não serve

Feedforward é malha aberta: aplica o `u` que o modelo prediz, sem olhar o
erro real. Se o modelo tiver qualquer desvio — e tem, é isso que a tabela
acima mostra — o erro não vai a zero sozinho. Quem fecha erro de regime é
**feedback**. A lei final por isso é feedforward (para o que é sistemático e
conhecido) **mais** feedback (para o que sobra).

## 5. Por que integrador na rampa, mas não na frenagem

Mesma pergunta, duas respostas diferentes — é o ponto mais importante deste
documento:

- **Frenagem:** erro de média ≈0, alta variância → **não** integra. Um `I`
  ali só acumula ruído (e ruído integrado é um passeio aleatório, pode
  divergir mesmo com média zero).
- **Rampa:** `g·senθ` é um viés **constante** enquanto durar a subida — e o
  carro não tem sensor de `θ` no simulador (`car.py` só tem `get_yaw()`,
  documentado como não-confiável; o IMU do IDENTIFICACAO.md §"Rota A" é do
  **carro real**, não existe aqui). Como a inclinação pode variar entre
  cenas, tratar como **distúrbio desconhecido** e deixar o integrador
  aprender (Rota B do IDENTIFICACAO.md) generaliza melhor do que hard-codar
  um `θ` medido uma vez.

**Consequência de projeto:** o integrador não pode ficar ligado o tempo
todo (senão absorve o ruído da frenagem) nem pode zerar a cada troca de
comando (senão perde a estimativa de `g·senθ` bem na hora que mais precisa
dela, e dá overshoot). Ele **congela** (mantém o valor) durante a janela de
transiente e só integra fora dela.

---

## 6. A lei de controle completa

Variáveis: `e = vdes − v`; `dt = 0,05 s`; `K=1,00`, `c=0,00425`,
`a₀=0,096` (IDENTIFICACAO.md); janela de transiente = 0,3 s (6 amostras,
igual à métrica que a mediu).

```
a_cmd    = Kv·e + I

viés     = 0,021   se (dentro da janela de 0,3s E a troca foi para acelerar)
           0       caso contrário

u_bruto  = (a_cmd + a0·tanh(v/ε) + c·v·|v| − viés) / K

u        = clip(u_bruto, −1, 1)

# integrador: congela dentro da janela de transiente, senão:
I[k+1]   = I[k] + dt·( Ki·e + (u − u_bruto)/Tt )
```

**Por que `− viés` e não `+ viés`:** o resíduo positivo significa que o
carro *ganha* aceleração de graça nessa janela. Pra atingir o mesmo `a_cmd`,
o controlador precisa pedir **menos** `u` do que o modelo puro pediria —
por isso subtrai antes de dividir por `K`.

**Por que `(u − u_bruto)/Tt` e não o contrário:** teste de sinal —
`u_bruto=1,5`, `u=1,0` (saturou, pedindo mais do que o atuador dá).
`u − u_bruto = −0,5`: puxa o integrador **pra baixo**, exatamente o
comportamento certo (para de "empurrar" um comando que o atuador já não
consegue cumprir). Isso é anti-windup por *back-calculation*.

## 7. Os três ganhos — chute inicial justificado, não ajuste fino

| Ganho | Valor | Por quê |
|---|---|---|
| `Kv` | 2,0 rad/s | banda de malha fechada ≈ `Kv` (depois que o feedforward cancela a não-linearidade conhecida, resta `v̇≈−Kv·v`). O atraso de 1 amostra (`Td=0,05s`) limita a banda útil a ~5 rad/s (IDENTIFICACAO.md); `Kv=2` deixa margem de ~2,5×. |
| `Ki` | 0,4 · s⁻² | regra prática: integral ~5× mais lento que o proporcional. `Ti = 5/Kv = 5s` → `Ki = Kv/Ti = 0,4`. |
| `Tt` | 0,5 s | heurística de Åström para back-calculation, `Tt=√(Ti·Td)`; aqui `Td` é o atraso de transporte de 1 amostra (0,05s): `√(5·0,05)=0,5`. |

Estes números **não foram tunados empiricamente** — são o ponto de partida
que a teoria justifica. Depois de rodar, meça overshoot, erro em regime na
rampa, e tempo de acomodação, e ajuste.

## 8. Onde está no código

> **Atualizado após §11:** este documento usa `Kv` no texto (nome que veio
> primeiro). No código atual é `KP_MOD` — renomeado pra `Kp` porque é o nome
> que a turma já conhece de PID (`Kp`/`Ki`/`Kd`/`Ti`/`Td`). `Kv=Kp` em todo
> lugar deste arquivo. `Tt` (anti-windup, back-calculation) **não** é `Td`
> (tempo derivativo) — não existe `Kd`/`Td` aqui, de propósito (§ abaixo).

Implementado em [main.py](main.py): `controlador_longitudinal()` +
`control_func()` (os ensaios de identificação — frenagem/rampa/curva/parada —
saíram do arquivo depois que o controlador ficou pronto; estão no histórico
do git). Os ganhos (`KP_MOD, KI_MOD, TT_MOD, TAXA_RAMPA_V`) estão nos
globais do topo do arquivo.

## 9. O que ainda não foi validado (não afirme que funciona sem rodar)

- Se `Kv=2, Ki=0,4, Tt=0,5` de fato estabilizam sem overshoot — só a
  simulação mostra.
- Se a janela de 0,3s é suficiente pra rampa (o `g·senθ≈0,63 m/s²` pode
  levar mais que isso pra o integrador convergir — `Ki=0,4` demora
  `0,63/(0,4·e)` segundos pra acumular, dependendo do `e` médio nesse
  trecho).
- Se o gate de "entrando em aceleração" (baseado em troca de sinal de `u`)
  dispara do jeito certo em malha fechada — ele foi desenhado a partir de
  ensaios em **malha aberta** (comando degrau); em malha fechada o `u`
  varia suavemente na maior parte do tempo, e só muda de sinal
  abruptamente perto de inverter o sentido do movimento. Rode e confira
  nos logs se `entrando_accel` está disparando quando devia.

Roda o `MODO='completo'` na cena da rampa e na cena plana, compara com o
`ensaio_parada` antigo, e traz os logs — essa é a validação que falta.

---

## 10. Pós-aula: o carro real não tem freio (atualização grande)

A aula trouxe um fato que muda tudo acima: **o carro real não tem freio
ativo.** Conferido em [servos.py](../veiculo_real/fva_car/servos.py) — em
marcha `FORWARD` o throttle só vai de `0` a `max` (linha 181); reverter torque
exige parar e trocar de marcha, com 5s de espera. Ou seja: **em movimento, a
única forma de desacelerar é soltar o acelerador e deixar o arrasto (`c·v²`,
`a₀`) fazerem o trabalho.** Não existe `u<0` em uso normal.

Isso invalida a premissa por trás de boa parte das seções 1-9: `K-`, a tabela
"troca → freando", o `U_FREIO=-0,7` do `ensaio_frenagem` — tudo isso mede um
mecanismo (torque reverso em movimento) que só existe no simulador. O colega
que trouxe `u ∈ [0,1]` (sem negativo) para a aula estava certo nessa
restrição, mesmo que o resto do controlador dele fosse simples demais
(setpoint fixo, sem inversão de modelo, sem ligar no perfil de aproximação).

### O que mudou no controlador

- **`u = clip(u_bruto, 0, 1)`** — nunca negativo.
- **Caiu a correção de viés de "troca → acelerando"** e o gate por tempo. Eles
  mediam a transição *freio ativo → acelerar*, que não existe mais nesse
  regime. Se um log novo (só com `u≥0`) mostrar um transiente parecido na
  transição *coast → acelerar*, remede e reintroduz — não reaproveitei o
  número antigo porque ele vem de um regime diferente.
- **`A_FREIO` deixou de ser uma frenagem ativa assumida (`0,5`) e virou uma
  fração conservadora do arrasto passivo.** E aqui teve uma surpresa: testei
  em malha fechada contra o modelo identificado (não só a conta de coast
  livre) e descobri que uma margem "generosa" (`A_FREIO` próximo ou acima de
  `a₀`) dá **mais** overshoot, não menos — o atraso da malha de controle come
  a margem perto do alvo. `A_FREIO = 0,3·a₀` zerou o overshoot em pista plana
  e em rampa de subida (erro final <1mm, testado contra o modelo). Isso
  importa porque **sem marcha à ré, ultrapassar o alvo é erro permanente** —
  não tem como voltar. Por isso o critério aqui não é "quase certo", é "nunca
  otimista".
- **Filtro passa-baixa na referência (`TAU_VDES`)**, dentro de
  `controlador_longitudinal`, em cima do perfil já rampado — atende ao pedido
  do professor de referência filtrada, não só rampada.

### O limite físico que isso expõe (não resolvido — decisão em aberto)

Testei também **rampa de descida** (mesma inclinação, sentido contrário): o
carro **nunca para**. Ele acelera até a velocidade terminal onde
`c·v² + a₀ = g·senθ` (≈11,2 m/s pra `θ=3,7°` — bem acima de `V_MAX`) e segue
assim, porque em nenhuma velocidade alcançável o arrasto passivo sozinho
vence a gravidade. **Isso não é bug de sintonia — é físico.** Um carro sem
freio não consegue segurar uma descida além do que o arrasto dissipa,
qualquer que seja o controlador.

**Decisão em aberto pra levar ao professor:** o escopo do exercício inclui
descida, ou só subida/plano? Se incluir descida, a única saída dentro do que
o hardware permite é a marcha à ré como "freio de emergência" (parar, trocar
de marcha, aplicar torque reverso) — o que significa detectar a situação
(velocidade crescendo com `u=0`, ou `θ` abaixo de um limiar) e tratar como um
modo de operação separado, não como o controlador contínuo de cruzeiro.

**Decidido:** descida está fora de escopo. Inicialmente isso foi implementado
com um guard de posição (`X_TOPO=4,0`, empírico, de 2 logs reais) que soltava
o acelerador antes do topo da rampa. Removido depois: é redundante com o
próprio `u=clip(u_bruto,0,1)` do controlador — numa descida `v` cresce acima
de `vdes`, o erro fica bem negativo, `u_bruto` vai a negativo e satura em 0
sozinho (mesmo teste de `θ=-3,7°` no `check_cruzeiro.py`: `u[fim]=0.000`,
sem intervenção externa). A diferença é que sem o guard o integrador/filtro
continuam rodando com anti-windup em vez de congelar no valor de quando
cruzou `X_TOPO` — estritamente melhor. E, mais importante: `X_TOPO` era um
número amarrado à posição x=4m *dessa cena específica* (mesmo pecado do
`X_ALVO` que já tinha sido removido antes) — não tem por que existir numa
malha que só deveria depender de `v`, não de `p`.

---

## 11. Ajuste fino com dado real (Kv, Ki, taxa da rampa, feedforward de dvdes/dt)

Depois de trocar `X_ALVO`/perfil de posição por cruzeiro puro (`vdes` sobe em
rampa até `V_REF` e fica lá por tempo indeterminado — sem alvo de posição,
que não fazia sentido pro objetivo de manter velocidade), veio uma rodada de
ajuste em cima de log real.

**`Ki`: `Ti=5s → Ti=1s` (`KI_MOD=0,4→2,0`).** No log `20260922_220158`, a
subida da rampa em `t=7s` levou **4,25s** pra `v` voltar a 95% do alvo — o
integrador lento (escolhido originalmente pra não perseguir o ruído da
transição frear-ativo→acelerar, regime que não existe mais nesse controlador)
também demorava pra "aprender" a gravidade nova. Varri `Ti` de 5 a 0,7s em
malha fechada (sem sinal de instabilidade nem no mais agressivo testado);
fiquei em `Ti=1s` — corta a recuperação pra ~2,5s sem piorar o overshoot em
pista plana.

**`Kv`: `2,0 → 3,0`.** O overshoot de partida (log com `V_REF=1`) caiu de
8,7% pra 5,8% subindo `Kv`. Testei com o atraso de 1 amostra incluído no
modelo de verificação (antes esse teste não tinha atraso nenhum — otimista
demais pra decisão de ganho) e não vi oscilação até `Kv=5`, mas não fui até
lá: o teste ainda não inclui o filtro do sensor de `v`
(`AlphaFilter(alpha=0,6)` em `car.py`) nem o round-trip real do ZMQ, então
`Kv=3` fica de meio-termo, não no teto testado.

**Feedforward de `dvdes/dt` — o que resolveu overshoot e velocidade de
acomodação ao mesmo tempo.** Só subir `Kv`/`Ki` ou a taxa da rampa tem
trade-off (mais rápido = mais overshoot). A saída foi avisar o controlador
que a referência **está subindo** antes do erro aparecer: como `vdes` já
passa por um filtro passa-baixa (`τ·dy/dt = x−y`), a derivada desse filtro
sai de graça, sem diferenciar nenhum sinal ruidoso:

```
dvdes/dt ≈ (vdes_bruto − vdes_filtrado) / TAU_VDES
a_cmd    = Kv·e + I + dvdes/dt
```

Isso zerou o overshoot no modelo (era 5,8%) mantendo a mesma velocidade de
subida. **Confirmado com dado real:** overshoot medido no log
`20260922_230516` foi **0,80%** (pedido do usuário: máximo 3%).

**`TAXA_RAMPA_V`: `0,3 → 0,9`.** Com o feedforward de `dvdes/dt` ativo, uma
rampa mais rápida deixou de custar overshoot — varri `0,6` a `3,0 m/s²` e
achei o ponto ótimo em `0,9-0,95` (`t98≈1,8s`, contra `3,6s` antes). Acima
disso o atuador satura e fica saturado por mais tempo, e o tempo de
acomodação **piora** de novo em vez de melhorar (não é "quanto mais rápido a
rampa, melhor").

**Ponto final de `Kp`/`Ki`: `4,0` / `2,67` (`Ti=1,5s`).** Depois de subir pra
`Kp=6` e ver o comando `u` chacoalhar mais (`std` da variação entre amostras
subiu 70%), escrevi a função de transferência de malha fechada (planta
compensada = integrador puro `1/s`, PI, atraso de 1 amostra via Padé de 1ª
ordem — [lugar_das_raizes.py](lugar_das_raizes.py)) e fiz o lugar das raízes.
**Correção importante:** isso mostrou que o "teto de ~5 rad/s" citado antes
neste documento (seção 7, tabela de ganhos) não é o limite de instabilidade
real — o lugar das raízes fica com raízes reais (sem par complexo, sistema
superamortecido) até `Kp≈39 rad/s`, quase 8× mais alto. Ou seja, **o chatter
em `Kp=6` não era proximidade de instabilidade** — é amplificação de ruído de
medição proporcional ao ganho (efeito normal de `P` alto, nada a ver com
margem). A discrepância entre o `~5 rad/s` original e o `~39 rad/s` do lugar
das raízes não foi resolvida — pode ser que a regra original fosse
conservadora de propósito, ou que a aproximação de Padé de 1ª ordem seja
otimista demais pro atraso real. Ficou em `Kp=4` como meio-termo prático
(mais rápido que `3`, sem o chatter visível do `6`), não por ter achado o
ponto ótimo analítico.

**`TAU_VDES`: `0,4 → 0,3s`.** É esse filtro que responde ao pedido do
professor de filtrar a rampa de entrada — confirmado testando três variantes
(filtro+feedforward / nada / só feedforward analítico sem filtro): só o
feedforward, mesmo exato, deixa 3,78% de overshoot (o "cotovelo" onde a
rampa vira reta continua abrupto); o filtro é o que resolve isso de fato.
Varrendo `TAU_VDES`, `0,3s` é o menor valor que ainda zera o overshoot
(`t98` cai de 2,00s pra 1,50s); abaixo disso já vaza overshoot.

**Armadilha encontrada limpando o código depois:** `A_FREIO` (usado só pelo
`ensaio_parada`) tinha sido baixado pra `0,3·a₀` — valor certo pro regime
*sem* freio ativo do `ensaio_completo`, mas o `ensaio_parada` **retém** freio
ativo (`car.set_u` não trava `u` em `[0,1]` lá, só o `controlador_longitudinal`
faz isso). Reusar a mesma variável entre os dois ensaios silenciosamente
quebrou o `ensaio_parada` (perfil de aproximação artificialmente lento demais).
Corrigido: `A_FREIO` voltou a ser exclusivo do `ensaio_parada`, com seu valor
original.

---

## 12. Distância de parada por coastdown (implicação pra desvio de obstáculo)

Sem freio ativo (§10), a única forma de desacelerar em pista plana é soltar o
acelerador e deixar `v̇ = -c·v|v| - a₀·sign(v)` (atrito seco + arrasto
quadrático) frear o carro sozinho. Pergunta que interessa pro resto do stack
(planejamento, desvio de obstáculo): **quantos metros o carro anda até parar,
partindo de `V_REF`?**

Pela regra da cadeia, `dv/dt = v·(dv/dx)`, então em pista plana e `v>0`:

```
v·dv/dx = -c·v² - a₀
```

Separável direto em `x` (não precisa passar por `t`):

```
v·dv/(c·v²+a₀) = -dx
```

Integra os dois lados, de `v=V0` (início do coast) até `v=0` (parado) do lado
esquerdo, e de `x=0` até `x=x_parada` do lado direito. Substituição
`u=c·v²+a₀` (⟹ `du=2c·v·dv`, ou seja `v·dv=du/(2c)`) no lado esquerdo:

```
∫ v·dv/(c·v²+a₀)  =  (1/2c)·∫ du/u  =  (1/2c)·ln(u)  =  (1/2c)·ln(c·v²+a₀)
```

Aplicando os limites (`v=V0 → v=0`) do lado esquerdo e (`0 → x_parada`) do
lado direito:

```
(1/2c)·[ln(a₀) − ln(c·V0²+a₀)]  =  −x_parada
```

Troca o sinal dos dois lados e junta os logs (`ln(a₀) − ln(c·V0²+a₀) =
−ln((c·V0²+a₀)/a₀)`):

```
x_parada = (1/2c)·ln((c·V0²+a₀)/a₀) = (1 / (2·c))·ln(1 + c·V0²/a₀)
```

**Cuidado com a notação:** `1/2c` é ambígua — o denominador é `2·c` inteiro,
não `c/2` (erro que já aconteceu tentando calcular isso à mão, ver conversa).
Sempre parenteize: `1 / (2·c)`.

Com `c=0,00425`, `a₀=0,096` e `V0=V_REF=1 m/s`:

```
x_parada = (1 / (2·0,00425))·ln(1 + 0,00425·1²/0,096) = 117,647·ln(1,0443) ≈ 5,10 m
```

**5 metros pra parar vindo de só 1 m/s, sem nenhum obstáculo detectado ainda
— é bastante para o porte do carrinho.** Isso confirma que o desvio de
obstáculo que vier depois não pode assumir frenagem ativa: a distância de
detecção mínima precisa ser calibrada em cima dessa fórmula (este §), crescendo
com `V_REF` (mas em `ln`, não linear — dobrar `V_REF` não dobra `x_parada`,
cresce mais devagar que isso, mas ainda cresce).

Fórmula geral, reaproveitável pra qualquer `V0`, não só `V_REF=1`:

```
x_parada(V0) = (1 / (2·c))·ln(1 + c·V0²/a₀)
```

**Tempo de parada (`t_parada`).** Integral diferente da de `x` — não dá pra
reaproveitar a mesma substituição, porque agora não tem a regra da cadeia
trocando `dt` por `dx`. Parte direto da EDO original:

```
dv/dt = -c·v² - a₀   ⟹   dv/(c·v²+a₀) = -dt
```

A primitiva de `1/(c·v²+a₀)` é a forma padrão `1/(a²+x²)` → `arctan`:

```
∫ dv/(c·v²+a₀) = (1/√(a₀c))·arctan(v·√(c/a₀))
```

Aplicando os limites (`v=V0 → v=0`, `t=0 → t=t_parada`), com `arctan(0)=0`:

```
t_parada(V0) = (1/√(a₀c)) · arctan(V0·√(c/a₀))
```

Com `V0=1 m/s`: `t_parada = (1/√(0,096·0,00425))·arctan(1·√(0,00425/0,096)) ≈ 10,27 s`.

**Resumo dos dois, saindo de `V_REF=1 m/s`, só de coast:** `x_parada≈5,10 m`,
`t_parada≈10,27 s`.

---

## 13. Chacoalho em serra durante a recuperação da rampa: filtro em `u`, não em `v`, não `Kd`

Log real na cena de rampa: `v` sobe limpo até `1 m/s` (t≈2s), fica em regime
até bater no início da subida em t≈6s (dip até ~0,83), recupera — mas `u`
mostra uma serra de alta frequência entre t≈8-12,5s, com `v` em regime
alcançado (não é anti-windup, `u` está longe de saturar em 0 ou 1).

**Primeira ideia descartada: `Kd` (ação derivativa).** Errada por construção
— já está documentado em [main.py:53-59](main.py) que `Kd` fica de fora
justamente porque derivada amplifica ruído de alta frequência (multiplicar
por `jω` no domínio da frequência cresce com `ω`). Um sinal em serra É ruído
de alta frequência; diferenciar isso pioraria, não resolveria. A ferramenta
certa pra atacar alta frequência sem mexer no ganho de baixa frequência é um
filtro **passa-baixa**, não passa-alta/derivativo.

**Onde filtrar: `u`, não `v`.** As duas opções atrasam a malha, mas coisas
diferentes:
- Filtrar `v` (medição) atrasa o que o controlador **enxerga como erro** —
  `Kp·erro`, `I` e o anti-windup passam a reagir a uma velocidade desatualizada.
  Como existe um distúrbio real (a rampa) que precisa de reação rápida, isso
  piora a recuperação.
- Filtrar `u` (comando), depois do `clip(u_bruto,0,1)` e antes do atuador,
  suaviza só o que chega no motor — o erro continua calculado em cima do `v`
  real, sem atraso extra.

**Anti-windup precisa usar o `u` pós-filtro.** O back-calculation
(`(u_prev - u_bruto_prev)/Tt`, [main.py](main.py) §6) compara o que foi
**realmente aplicado** no atuador com o que o cálculo bruto pedia. Se o
filtro entra depois do `clip` mas o anti-windup ainda usasse o `u` pré-filtro,
ele ficaria cego pro efeito do filtro — passaria a comparar com um valor que
nunca foi de fato mandado pro motor. `st['u_prev']` guarda o `u` pós-filtro
(implementado em [main.py](main.py), `controlador_longitudinal()`).

**Validação (ruído sintético em `v`, não é o ruído real do ZMQ — é só um
proxy pra confirmar a direção do efeito antes de testar em log real):**
cenário plano até t=6s, rampa 3,7° depois, ruído gaussiano `σ=0,006 m/s`
somado à leitura de `v` (a física usa o `v` limpo; o controlador só vê o
ruidoso). Métrica de chacoalho: `std(diff(u))` depois do distúrbio. Métrica
de resposta: tempo até `v` voltar a 98% de `V_REF` depois do dip.

| `TAU_U` | chacoalho (`std(diff(u))`) | tempo de recuperação |
|---|---|---|
| 0 (sem filtro) | 0,0419 | 2,57 s |
| 0,05 s | 0,0419 | 2,57 s |
| 0,10 s | 0,0220 (-47%) | 2,71 s (+0,14s) |
| 0,15 s | 0,0181 (-57%) | 2,73 s (+0,16s) |
| 0,20 s | 0,0164 (-61%) | 2,73 s |
| 0,30 s | 0,0149 (-64%) | 2,71 s |

O tempo de recuperação praticamente não piora em nenhum valor (confirma a
escolha de filtrar `u`, não `v`) — os ganhos de chacoalho é que têm
retorno decrescente. Mas tem um custo que essa tabela não mostra: o filtro em
`u` também mexe no overshoot de partida em pista plana (cenário sem ruído,
`check_cruzeiro.py`), que estava calibrado em 0% via `TAU_VDES` (§11) assumindo
`u` sem atraso extra:

| `TAU_U` | overshoot de partida (pista plana, sem ruído) |
|---|---|
| 0 / 0,02 / 0,05 s | 0,000% |
| 0,08 s | 0,192% |
| 0,10 s | 0,546% |
| 0,12 s | 0,869% |
| 0,15 s | 1,313% |

**Escolhido: `TAU_U=0,10s`.** Trade-off, não o valor que mais reduz ruído:
`0,15` reduz mais chacoalho (-57% vs -47%), mas quase triplica o overshoot de
partida (1,31% vs 0,55%) pelo ganho marginal de chacoalho ser pequeno depois
de `0,10` (retorno decrescente claro na tabela acima). `0,10` fica bem dentro
do orçamento de 3% de overshoot com folga de sobra.

**Validado com log real** (`TAU_U=0,10`), comparando o log de antes da mudança
(`logs/20260922_234940`) com um novo log depois (`logs/20260923_090619`),
mesma métrica (`std(diff(u))` em `t≥8s`, depois do transiente da rampa):

| | chacoalho (`std(diff(u))`, t≥8s) | tempo de recuperação (98%) |
|---|---|---|
| ANTES (sem `TAU_U`) | 0,02613 | 2,35 s |
| DEPOIS (`TAU_U=0,10`) | 0,00578 (**-78%**) | 2,45 s (+0,10s) |

Reduziu mais do que o modelo sintético previa (-47%) — o ruído real
provavelmente tem mais energia em alta frequência do que o `σ=0,006`
gaussiano chutado pra reproduzir a serra. A direção do efeito bateu (chacoalho
cai, recuperação não piora de forma relevante); a magnitude só o log real
confirmou, o sintético só serviu pra decidir o `TAU_U` antes de gastar uma
rodada de simulação. Com -78% de redução já alcançado sem custo relevante na
recuperação, subir `TAU_U` além disso não compensa: o overshoot de partida
cresce rápido (§ tabela acima, 5× de `0,10` pra `0,25`) por um ganho marginal
de chacoalho pequeno — retorno decrescente já visível mesmo no modelo
sintético. Decidido: fica em `TAU_U=0,10`.
