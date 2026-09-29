function M = modelos_longitudinais()
%MODELOS_LONGITUDINAIS  Os dois modelos longitudinais do carro roxo com a bateria carregada
%(logs de 24/09 em veiculo_real/dados2_sem_feedforward/roxo).
%
%   M = modelos_longitudinais();
%     M.capitulo, M.ident   os dois modelos: parâmetros e a planta linear P
%     M.H                   a média de 30 amostras do car.py (sensor), discreta em M.Ts
%     M.simula(m, t, u)     o v medido que o modelo m prevê para o u de um log
%                           (colunas t e u do car.csv, com a amostragem irregular do carro)
%   modelos_longitudinais   sem saída: confere os dois contra os 3 logs e plota o desempenho
%                           dos controladores projetados com cada modelo (compara_controladores)
%
% ESTRUTURA (a mesma nos dois; só os números mudam)
%   u -> integrador do servos.py (K_PWM graus/s por unidade de u, limitado a [0, TH_MAX]) = pwm
%     -> zona morta e ganho: v_regime = G*max(0, pwm - DB)
%     -> primeira ordem tau -> atraso puro d -> v real
%     -> média de 30 amostras do car.py -> v medido
%   O integrador e a média vêm do código; DB, G, tau e d vêm dos dados.
%   P é a parte linear de u até o v real, sem a zona morta, discretizada como em simula:
%   o lag responde ao pwm da amostra anterior e d vira round(d/Ts) amostras.
%   Para projetar: controlSystemDesigner(m.P, C, M.H), com a média no sensor. A zona morta
%   fica fora do projeto linear: com planta integradora, a malha a compensa como offset.
%
% OS DOIS MODELOS
%   capitulo  A planta nominal do projeto (veiculo_real/capitulo): o ajuste do 003330, com
%             K arredondado para K_nom = 1,6 (a bateria boa deu K = 1,49 / 1,60 / 1,61).
%             Mínimos quadrados no erro de saída, no Python (figuras.py).
%             Previsão cega deixa-um-de-fora: rmse 0,063 / 0,023 / 0,044 m/s.
%   ident     System Identification Toolbox, estimado em 002634 + 002751. Receita: rode
%             prepara_ident.m; entrada = média de 30 de max(0, pwm - 1,5); Polynomial
%             Models, OE [1 1 6]. DB e ordens escolhidos por validação cruzada entre os dois
%             ensaios de estimação, só entre modelos cujo degrau nunca fica negativo.
%             Previsão cega em 003330: fit 91,5% (rmse 0,042 m/s). O modelo do capítulo,
%             reajustado nos mesmos dois ensaios, dá 91,4%: empate.
%
% ONDE CONCORDAM E ONDE NÃO
%   Todos os ensaios bons terminaram em ~1,85 m/s: um único ponto da reta v = G*(pwm - DB).
%   Os modelos passam por esse ponto com retas diferentes (DB e G se compensam) e dividem
%   de jeitos diferentes o atraso entre lag e atraso puro. pwm de regime [graus]:
%         v [m/s]   capitulo   ident
%          1,85       14,4      14,6    <- onde estão os dados
%          1,0         9,3       8,6
%          0,5         6,4       5,0
%   Com o controlador do main.py (KP 0,6, K_NOM 1,6) os dois dão ~67 graus de margem e ~1%
%   de sobressinal: para o controle longitudinal, tanto faz qual usar.
%   O que decide: ensaios em outras velocidades (vref 1,0 e 0,5 com o controlador novo).
%   Valem só com a bateria carregada; a bateria fraca é outra planta (K 0,9-1,1, tau 0,6-0,76 s).

Ts = 0.0288;                              % período mediano do laço nesses logs
K_PWM = rad2deg(0.4 * 0.08 * 5.16);       % servos.py/car.py: graus/s de pwm por unidade de u
TH_MAX = (1.5 + 8.9370) / 0.092294 - 95;  % teto do pwm acima do neutro (velmax 1,5)

M.Ts = Ts;
M.H = tf(ones(1, 30)/30, [1 zeros(1, 29)], Ts);
M.capitulo = modelo(1.6/K_PWM, 3.42, 0.251, 0, K_PWM, TH_MAX, Ts);
M.ident = modelo(0.14131, 1.5, 0.22797, 5*Ts, K_PWM, TH_MAX, Ts);
M.simula = @simula;
if nargout == 0
    confere(M)
    compara_controladores(M)
    clear M   % só a conferência e o gráfico, sem despejar a struct na tela
end
end

function m = modelo(G, DB, tau, d, K_PWM, TH_MAX, Ts)
a = exp(-Ts/tau);
m = struct('G', G, 'DB', DB, 'tau', tau, 'd', d, 'K', G*K_PWM, 'K_PWM', K_PWM, 'TH_MAX', TH_MAX);
m.P = tf(K_PWM*Ts, [1 -1], Ts) * tf([0 (1 - a)*G], [1 -a], Ts, 'InputDelay', round(d/Ts));
end

function v = simula(m, t, u)
% A mesma conta de planta() em capitulo/figuras.py. A linha k do log guarda o u aplicado
% entre t(k-1) e t(k).
th = zeros(size(t));
vr = zeros(size(t));
for k = 2:numel(t)
    h = t(k) - t(k-1);
    th(k) = min(max(th(k-1) + m.K_PWM*u(k)*h, 0), m.TH_MAX);
    vss = m.G*max(0, th(k-1) - m.DB);
    vr(k) = vss + (vr(k-1) - vss)*exp(-h/m.tau);
end
v = filter(ones(1, 30)/30, 1, interp1(t, vr, t - m.d, 'linear', 0));
end

function confere(M)
logs = fullfile(fileparts(mfilename('fullpath')), '..', 'veiculo_real', 'dados2_sem_feedforward', 'roxo');
for n = ["20260924_002634" "20260924_002751" "20260924_003330"]
    T = readtable(fullfile(logs, n, 'car.csv'));
    e = rms([M.simula(M.capitulo, T.t, T.u), M.simula(M.ident, T.t, T.u)] - T.v);
    fprintf('%s  rmse: capitulo %.3f  ident %.3f m/s\n', n, e);
end
% o capítulo foi ajustado no 003330 e o ident o prevê às cegas
assert(e(1) < 0.02 && e(2) < 0.05, 'os modelos não reproduzem mais o 003330');
end

function compara_controladores(M)
% O procedimento de projeto do capítulo (KP 0,6 e K_NOM = o K do modelo) aplicado a cada
% modelo dá dois controladores. Cada um roda nas duas plantas e no pior caso do capítulo.
KP = 0.6;
virgula = @(x, casas) strrep(sprintf('%.*f', casas, x), '.', ',');
C = struct('nome', {'capítulo', 'ident'}, 'K_NOM', {1.6, M.ident.K}, ...
           'cor', {[0.165 0.471 0.839], [0.922 0.408 0.204]});
pior = modelo(2.0/M.capitulo.K_PWM, 3.42, 0.40, 0, M.capitulo.K_PWM, M.capitulo.TH_MAX, M.Ts);
plantas = {M.capitulo, M.ident, pior};
titulos = {'planta do capítulo (K 1,6)', ['planta do ident (K ' virgula(M.ident.K, 2) ')'], ...
           'pior caso do capítulo (K 2,0, τ 0,4 s)'};
figure(Name = 'Controladores do capítulo e do ident', Position = [100 100 1150 560]);
tl = tiledlayout(2, 3, TileSpacing = 'compact');
title(tl, {['Controlador do main.py (KP 0,6) projetado com cada modelo: K\_NOM = 1,6 (capítulo) ou ' ...
            virgula(M.ident.K, 2) ' (ident)'], ...
           'rampa de 0,9 m/s² até 1 m/s; legenda: sobressinal · acomodação na faixa de 5%'});
fprintf('\n%-10s %-7s %-40s %11s %9s\n', 'projeto', 'K_NOM', 'planta', 'sobressinal', 'acomoda');
for j = 1:3
    av = nexttile(j);      hold(av, 'on'); grid(av, 'on');
    au = nexttile(j + 3);  hold(au, 'on'); grid(au, 'on');
    for c = 1:2
        [t, v, u] = malha(plantas{j}, KP, C(c).K_NOM, 1.0, 12, M.Ts);
        s = 100*(max(v) - 1);
        acomoda = t(find(abs(v - 1) > 0.05, 1, 'last'));   % última saída da faixa de 5%
        plot(av, t, v, Color = C(c).cor, LineWidth = 1.5, ...
             DisplayName = sprintf('%s: %s%% · %s s', C(c).nome, virgula(s, 1), virgula(acomoda, 1)));
        plot(au, t, u, Color = C(c).cor, LineWidth = 1.5);
        fprintf('%-10s %-7.2f %-40s %10.1f%% %7.1f s\n', C(c).nome, C(c).K_NOM, titulos{j}, s, acomoda);
    end
    plot(av, t, min(1, 0.9*t), '--', Color = [0.54 0.53 0.51], DisplayName = 'referência');
    title(av, titulos{j});  ylabel(av, 'v real [m/s]');  ylim(av, [0 1.35]);
    legend(av, Location = 'southeast');
    ylabel(au, 'u');  xlabel(au, 't [s]');  ylim(au, [-1.05 1.05]);
end
for c = 1:2
    [~, pm] = margin(KP/C(c).K_NOM*pior.P*M.H);
    fprintf('projeto %s: margem de fase no pior caso = %.1f graus (critério do capítulo: 55)\n', C(c).nome, pm);
end
end

function [t, v, u] = malha(m, KP, K_NOM, vref, ts, Ts)
% O controlador do main.py (controlador_longitudinal + control_func) na planta não linear
% do modelo m, em passos fixos de Ts. Devolve o v real e o u aplicado.
n = round(ts/Ts);
t = (0:n-1)'*Ts;
v = zeros(n, 1);  u = zeros(n, 1);
bufv = zeros(30, 1);  bufr = zeros(30, 1);  atraso = zeros(round(m.d/Ts) + 1, 1);
a = exp(-Ts/m.tau);
th = 0;  vr = 0;  vf = 0;  uk = 0;
for k = 2:n
    % planta: um passo com o u anterior, na mesma discretização de simula e de P
    vss = m.G*max(0, th - m.DB);
    th = min(max(th + m.K_PWM*uk*Ts, 0), m.TH_MAX);
    vr = vss + (vr - vss)*a;
    atraso = [vr; atraso(1:end-1)];
    bufv = [atraso(end); bufv(1:end-1)];
    vm = mean(bufv);                              % o v que o car.py entrega
    % controlador do main.py
    vdes = min(vref, 0.9*(t(k) - t(2)));          % rampa TAXA_RAMPA_V
    vp = vf;
    vf = vf + min(Ts/0.3, 1)*(vdes - vf);         % filtro TAU_VDES
    bufr = [vf; bufr(1:end-1)];                   % a mesma média de 30 na referência
    uk = min(max((KP*(mean(bufr) - vm) + (vdes - vp)/0.3)/K_NOM, -1), 1);
    if abs(vm) > 1.5, uk = 0; end                 % car.py: acima de VELMAX, u = 0
    v(k) = vr;  u(k) = uk;
end
end
