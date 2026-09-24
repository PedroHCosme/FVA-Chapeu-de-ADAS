"""Compara o modelo do colega (ajuste só do coastdown de 1 ensaio) com o
nosso modelo identificado (IDENTIFICACAO.md), validando ambos por simulação
em malha aberta contra o log real dele.
"""
import numpy as np
import matplotlib.pyplot as plt

LOG = r"C:\Users\Usuario\Downloads\car.csv"
MASS = 6.3

# --- nosso modelo (simulador/IDENTIFICACAO.md) ---
K_NOSSO, C_NOSSO, A0_NOSSO = 1.00, 0.00425, 0.096


def carrega(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d["t"], d["v"], d["a"], d["u"]


def ajusta_colega1(t, v, a, u):
    """calculo.py: -m*a = F0 + F2*v^2 (regressão linear em v^2)."""
    mask = (u == 0) & (v > 0.2) & (a < 0)
    x = v[mask] ** 2
    y = -MASS * a[mask]
    F2, F0 = np.polyfit(x, y, 1)
    return F0, F2  # N, N/(m/s)^2


def ajusta_colega2(t, v, a, u):
    """calculo2.py: desaceleração média / g -> coeficiente único."""
    mask = (u == 0) & (v > 0.2) & (a < 0)
    desac_media = -a[mask].mean()
    f = desac_media / 9.81
    return desac_media  # m/s^2, constante


def simula(v0, t0, t1, dt, deriv):
    n = int(round((t1 - t0) / dt))
    ts = t0 + np.arange(n + 1) * dt
    vs = np.empty(n + 1)
    vs[0] = v0
    for i in range(n):
        vs[i + 1] = vs[i] + dt * deriv(vs[i])
    return ts, vs


def metrica(v_real, v_pred):
    err = v_pred - v_real
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((v_real - v_real.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    erro_final = abs(v_pred[-1] - v_real[-1])
    return rmse, mae, r2, erro_final


def main():
    t, v, a, u = carrega(LOG)

    F0, F2 = ajusta_colega1(t, v, a, u)
    desac_media = ajusta_colega2(t, v, a, u)
    print("=== Modelo colega (calculo.py, ajuste v^2) ===")
    print(f"F0 = {F0:.4f} N   F2 = {F2:.4f} N/(m/s)^2")
    print(f"-> a0_colega = {F0/MASS:.4f} m/s^2   c_colega = {F2/MASS:.5f} 1/m")
    print("\n=== Modelo colega2 (calculo2.py, média) ===")
    print(f"desaceleracao_media = {desac_media:.4f} m/s^2 (constante, sem termo v^2)")
    print("\n=== Nosso modelo (IDENTIFICACAO.md) ===")
    print(f"K={K_NOSSO}  c={C_NOSSO}  a0={A0_NOSSO}")

    # janela de coastdown real (u==0 após o degrau) para validação
    i_start = np.argmax((u == 0) & (t > 1.0))  # primeira amostra u=0 após o degrau de subida
    t0, v0 = t[i_start], v[i_start]
    t1 = t[-1]
    dt = 0.05

    a0_colega, c_colega = F0 / MASS, F2 / MASS
    deriv_colega1 = lambda vv: -a0_colega - c_colega * vv ** 2
    deriv_colega2 = lambda vv: -desac_media
    deriv_nosso = lambda vv: -A0_NOSSO * np.sign(vv) - C_NOSSO * vv * abs(vv)

    ts, v_colega1 = simula(v0, t0, t1, dt, deriv_colega1)
    _, v_colega2 = simula(v0, t0, t1, dt, deriv_colega2)
    _, v_nosso = simula(v0, t0, t1, dt, deriv_nosso)

    v_real_interp = np.interp(ts, t, v)

    print("\n=== Métricas de validação (previsão cega do coastdown, malha aberta) ===")
    print(f"{'modelo':<18}{'RMSE (m/s)':<14}{'MAE (m/s)':<14}{'R2':<10}{'erro final (m/s)'}")
    for nome, vp in [("colega (v^2)", v_colega1), ("colega2 (média)", v_colega2), ("nosso", v_nosso)]:
        rmse, mae, r2, ef = metrica(v_real_interp, vp)
        print(f"{nome:<18}{rmse:<14.4f}{mae:<14.4f}{r2:<10.4f}{ef:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(t[(t >= t0)], v[(t >= t0)], "k.", ms=3, alpha=0.5, label="dado real")
    plt.plot(ts, v_colega1, label="modelo colega (v²)")
    plt.plot(ts, v_colega2, label="modelo colega2 (média)")
    plt.plot(ts, v_nosso, label="nosso modelo")
    plt.xlabel("t (s)")
    plt.ylabel("v (m/s)")
    plt.title("Coastdown do colega: dado real vs modelos (malha aberta)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figuras/comparacao_colega.png", dpi=150)
    print("\nFigura salva em figuras/comparacao_colega.png")


if __name__ == "__main__":
    main()
