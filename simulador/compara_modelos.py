# -*- coding: utf-8 -*-
"""Compara Modelo 1 (nosso, IDENTIFICACAO.md) x Modelo 2 (colega, coastdown unico).

Cross-validacao: cada modelo e testado em coastdowns que ele NAO viu.
- Modelo 2 foi ajustado no log do colega -> testa-lo nos NOSSOS logs e previsao cega.
- Modelo 1 foi ajustado nos nossos logs -> testa-lo no log DELE e previsao cega.
Metricas sem R2 (R2 nao discrimina em decaimento monotonico: baseline "media"
e ruim, entao qualquer modelo que capte a queda ja pontua ~0.98).
"""
import numpy as np
import glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASS = 6.3
DT = 0.05
LOG_COLEGA = r"C:\Users\Usuario\Downloads\car.csv"

# --- Modelo 1: nosso (IDENTIFICACAO.md) ---
K1, C1, A0_1 = 1.00, 0.00425, 0.096
deriv1 = lambda v: -A0_1 * np.tanh(v / 0.05) - C1 * v * abs(v)

# --- Modelo 2: colega (ajustado no coastdown do log dele) ---
def ajusta_modelo2(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    u, v, a = d["u"], d["v"], d["a"]
    mask = (u == 0) & (v > 0.2) & (a < 0)
    F2, F0 = np.polyfit(v[mask] ** 2, -MASS * a[mask], 1)   # calculo.py
    a0, c = F0 / MASS, F2 / MASS
    desac_media = -a[mask].mean()                            # calculo2.py
    return a0, c, desac_media


def coastdown_blocks(path, vmin=0.5, nmin=20):
    """extrai blocos contiguos de u==0 em movimento (previsao cega de coastdown)."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    t, u, v = d["t"], d["u"], d["v"]
    blocks = []
    i = 0
    n = len(u)
    while i < n:
        if u[i] == 0 and v[i] > vmin:
            j = i
            while j < n and u[j] == 0:
                j += 1
            if j - i >= nmin:
                blocks.append((t[i:j], v[i:j]))
            i = j
        else:
            i += 1
    return blocks


def simula(t, v0, deriv):
    v = np.empty_like(t)
    v[0] = v0
    for i in range(len(t) - 1):
        v[i + 1] = v[i] + deriv(v[i]) * (t[i + 1] - t[i])
        if v[i + 1] < 0:
            v[i + 1] = 0.0
    return v


def metricas(vr, vp):
    e = vp - vr
    rmse = np.sqrt(np.mean(e ** 2))
    mae = np.mean(np.abs(e))
    emax = np.abs(e).max()
    ef = abs(vp[-1] - vr[-1])          # erro no ponto final (parada)
    return rmse, mae, emax, ef


def main():
    a0_2, c_2, desac2 = ajusta_modelo2(LOG_COLEGA)
    deriv2 = lambda v: -a0_2 - c_2 * v * abs(v)          # Modelo 2 (v^2)
    deriv2b = lambda v: -desac2                          # Modelo 2 variante media

    print("Modelo 1 (nosso):  K=%.2f  c=%.5f  a0=%.3f" % (K1, C1, A0_1))
    print("Modelo 2 (colega): a0=%.4f  c=%.5f   [+ variante media=%.4f m/s2]" % (a0_2, c_2, desac2))

    # ---- coleta todos os coastdowns: log do colega + nossos logs ----
    # 20260901: log mais antigo, coastdown desacelera ~0.74 m/s2 a v=3 (5x o teto
    # da pista plana) e o v ate inverte -> outra cena/condicao. Nenhum dos dois
    # modelos cobre isso; nao e discriminador justo (regra "nao misture cenas").
    IGNORA = ["20260901"]
    fontes = [("colega", LOG_COLEGA)]
    for csv in sorted(glob.glob(os.path.join("logs", "*", "car.csv"))):
        if any(s in csv for s in IGNORA):
            continue
        fontes.append((os.path.basename(os.path.dirname(csv)), csv))

    linhas = []          # (nome, vmax, m1..., m2...)
    plot_blocks = []     # guarda alguns pra figura
    for nome, path in fontes:
        for k, (t, vr) in enumerate(coastdown_blocks(path)):
            vmax = vr.max()
            v1 = simula(t, vr[0], deriv1)
            v2 = simula(t, vr[0], deriv2)
            v3 = simula(t, vr[0], deriv2b)
            linhas.append((f"{nome}#{k}", vmax, metricas(vr, v1), metricas(vr, v2)))
            plot_blocks.append((f"{nome}#{k}", t, vr, v1, v2, v3, vmax))

    # ---- tabela ----
    print("\n" + "=" * 96)
    print("VALIDACAO CRUZADA POR COASTDOWN  (RMSE / MAE / erro max / erro no ponto de parada, em m/s)")
    print("=" * 96)
    h = f"{'coastdown':<16}{'vmax':>6}  |{'M1 rmse':>9}{'M1 mae':>8}{'M1 emax':>9}{'M1 efim':>9}  |{'M2 rmse':>9}{'M2 mae':>8}{'M2 emax':>9}{'M2 efim':>9}"
    print(h)
    print("-" * len(h))
    agg1 = np.zeros(4); agg2 = np.zeros(4); N = 0
    for nome, vmax, m1, m2 in linhas:
        print(f"{nome:<16}{vmax:>6.2f}  |{m1[0]:>9.4f}{m1[1]:>8.4f}{m1[2]:>9.4f}{m1[3]:>9.4f}  "
              f"|{m2[0]:>9.4f}{m2[1]:>8.4f}{m2[2]:>9.4f}{m2[3]:>9.4f}")
        agg1 += np.array(m1); agg2 += np.array(m2); N += 1
    print("-" * len(h))
    a1, a2 = agg1 / N, agg2 / N
    print(f"{'MEDIA':<16}{'':>6}  |{a1[0]:>9.4f}{a1[1]:>8.4f}{a1[2]:>9.4f}{a1[3]:>9.4f}  "
          f"|{a2[0]:>9.4f}{a2[1]:>8.4f}{a2[2]:>9.4f}{a2[3]:>9.4f}")

    # separacao por faixa de velocidade (onde o c subdimensionado do M2 machuca)
    print("\nErro RMS medio por faixa de velocidade inicial:")
    for lo, hi in [(0, 2), (2, 10)]:
        sel = [(m1, m2) for _, vmax, m1, m2 in linhas if lo <= vmax < hi]
        if not sel:
            continue
        r1 = np.mean([m1[0] for m1, _ in sel]); r2 = np.mean([m2[0] for _, m2 in sel])
        print(f"  vmax in [{lo},{hi}) m/s  ({len(sel)} coastdowns):  M1={r1:.4f}  M2={r2:.4f} m/s")

    # ---- figura: os coastdowns de maior velocidade ----
    plot_blocks.sort(key=lambda b: -b[6])
    sel = plot_blocks[:6]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.ravel()
    for ax, (nome, t, vr, v1, v2, v3, vmax) in zip(axs, sel):
        ax.plot(t, vr, "k.", ms=4, label="dado real")
        ax.plot(t, v1, "-", color="#ff7f0e", lw=2, label="Modelo 1")
        ax.plot(t, v2, "-", color="#2ca02c", lw=2, label="Modelo 2 (v²)")
        ax.plot(t, v3, "--", color="#4fc3f7", lw=2, label="Modelo 3 (média)")
        ax.set_title(f"{nome}   vmax={vmax:.2f} m/s", fontsize=10)
        ax.set_xlabel("t (s)"); ax.set_ylabel("v (m/s)"); ax.grid(alpha=.3)
    axs[0].legend(fontsize=9)
    fig.suptitle("Validacao cruzada: coastdowns de maior velocidade — Modelo 1 x Modelo 2 x Modelo 3", fontsize=13)
    fig.tight_layout()
    fig.savefig("figuras/comparacao_modelos.png", dpi=140)
    print("\n-> figuras/comparacao_modelos.png")


if __name__ == "__main__":
    main()
