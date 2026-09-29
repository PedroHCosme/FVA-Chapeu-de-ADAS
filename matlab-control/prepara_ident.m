%% prepara_ident.m - logs do carro real prontos para o System Identification app
%
% Rode (F5 no editor) e depois digite  ident  no Command Window.
% Cria no workspace, todos com saída v [m/s]:
%
%   est_u,   val_u     entrada u   (comando do controlador, -1 a 1)
%   est_pwm, val_pwm   entrada pwm (acelerador acima do neutro [graus] = integral de u)
%   desc_u,  desc_pwm  os 4 ensaios com a bateria descarregando: outra planta, não misture
%
% est_* junta 002634 e 002751 (2 experimentos, bateria carregada) e val_* é o
% 003330, que fica fora da estimação. O modelo de referência no fim do script
% usa o mesmo recorte, então o fit dele se compara com o da janela Model output.

LOGS = fullfile(fileparts(mfilename('fullpath')), '..', 'veiculo_real', 'dados2_sem_feedforward', 'roxo');

% do código do carro (servos.py, car.py), as mesmas de veiculo_real/capitulo/figuras.py
K_PWM  = rad2deg(0.4 * 0.08 * 5.16);      % [graus/s por unidade de u] GAIN_TORQUE_FORWARD*RW*MASS
TH_MAX = (1.5 + 8.9370) / 0.092294 - 95;  % [graus] teto do acelerador acima do neutro (velmax 1,5)
N_MA   = 30;                              % car.py: v sai de uma média móvel de 30 amostras

EST  = ["20260924_002634" "20260924_002751"];
VAL  =  "20260924_003330";
DESC = ["20260924_003552" "20260924_003904" "20260924_004516" "20260924_004851"];
% fora: 000844 (código do professor, bateria fraca), 002158/002236/002318 (carro
% na mão), 002506 (parado), 003019/003157/003238 (o ultrassom não deixou andar)

nomes = [EST VAL DESC];
logs = cell(size(nomes));
dts = [];
for i = 1:numel(nomes)
    logs{i} = readtable(fullfile(LOGS, nomes(i), 'car.csv'));
    dts = [dts; diff(logs{i}.t(2:end))]; %#ok<AGROW> o 1o intervalo é o ~1 s que o start_mission dorme
end
% o ident só estima com amostragem uniforme e o laço do carro tem jitter: tudo vai
% para uma grade com o período mediano do laço (~28,8 ms, não os 20 ms nominais)
Ts = round(median(dts), 4);

dados = cell(numel(nomes), 2);
for i = 1:numel(nomes)
    [dados{i,1}, dados{i,2}] = para_iddata(logs{i}, nomes(i), Ts, K_PWM, TH_MAX);
end
grupo = @(lista, c) merge(dados{ismember(nomes, lista), c});
est_u  = grupo(EST, 1);   est_pwm  = grupo(EST, 2);
val_u  = grupo(VAL, 1);   val_pwm  = grupo(VAL, 2);
desc_u = grupo(DESC, 1);  desc_pwm = grupo(DESC, 2);

%% Modelo de referência: o do capítulo, ajustado só em est e simulado às cegas em val
% u -> integrador (pwm) -> zona morta DB e ganho G -> 1a ordem tau -> atraso d -> média de 30
% parâmetros = ajusta(["20260924_002634", "20260924_002751"]) de capitulo/figuras.py
ref = struct('G', 0.14920, 'DB', 2.1926, 'tau', 0.22796, 'd', 0.09725);  % [m/s/grau] [graus] [s] [s]
t = val_pwm.SamplingInstants;
y = val_pwm.y;
a = exp(-Ts/ref.tau);
v = filter([0 1-a], [1 -a], ref.G*max(0, val_pwm.u - ref.DB));  % 1a ordem, entrada segura no passo
v = interp1(t, v, t - ref.d, 'linear', 0);                        % atraso puro
ref.v = filter(ones(1, N_MA)/N_MA, 1, v);                         % média móvel do sensor
ref.fit = 100*(1 - norm(y - ref.v)/norm(y - mean(y)));            % a mesma conta do fit do ident

figure(Name = 'Modelo de referência em val');
plot(t, y, 'k', t, ref.v, '--', LineWidth = 1.2);
grid on; xlabel('t [s]'); ylabel('v [m/s]');
legend('medido (003330)', sprintf('referência: fit %.1f%%', ref.fit), Location = 'southeast');

fprintf('Ts = %.4f s. No workspace: est_u, val_u, est_pwm, val_pwm, desc_u, desc_pwm\n', Ts);
fprintf('Referência em val (003330): fit = %.1f%%, rmse = %.3f m/s\n', ref.fit, rms(y - ref.v));
clear i logs dts dados grupo t y a v

function [du, dp] = para_iddata(T, nome, Ts, K_PWM, TH_MAX)
% Um log -> dois iddata na grade uniforme (entrada u e entrada pwm, saída v).
t = T.t;
% acelerador como a thread do servos.py: integra u e limita a [0, TH_MAX].
% A linha k do log guarda o u aplicado entre t(k-1) e t(k).
th = zeros(size(t));
for k = 2:numel(t)
    th(k) = min(max(th(k-1) + K_PWM*T.u(k)*(t(k) - t(k-1)), 0), TH_MAX);
end
% a grade começa em t(2): entre t(1) e t(2) o start_mission dorme ~1 s sem medir
% nada, e interpolar ali inventaria amostras (em t(2) o carro já está parado, u = 0)
tu = (t(2):Ts:t(end))';
v = interp1(t, T.v, tu);          % v já é suave (média móvel): linear basta
u = interp1(t, T.u, tu, 'next');  % em cada instante vale o u da linha seguinte
p = interp1(t, th, tu);           % pwm é rampa entre amostras: linear é exato
com = {'Tstart', tu(1), 'OutputName', 'v', 'OutputUnit', 'm/s', 'ExperimentName', char(nome)};
du = iddata(v, u, Ts, 'InputName', 'u', com{:});
dp = iddata(v, p, Ts, 'InputName', 'pwm', 'InputUnit', 'graus', 'InterSample', 'foh', com{:});
end
