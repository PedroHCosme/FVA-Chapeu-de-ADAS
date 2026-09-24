import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
t, v, a, u = d["t"], d["v"], d["a"], d["u"]
x, y = d["x"], d["y"]

fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
ax[0].step(t, u, where="post"); ax[0].set_ylabel("u [m/s^2] (cmd)"); ax[0].grid(alpha=.3)
ax[1].plot(t, v); ax[1].set_ylabel("v [m/s]"); ax[1].grid(alpha=.3)
ax[2].plot(t, a); ax[2].set_ylabel("a [m/s^2] (num.)"); ax[2].grid(alpha=.3)
ax[3].plot(t, np.hypot(x - x[0], y - y[0])); ax[3].set_ylabel("dist. percorrida [m]")
ax[3].set_xlabel("t [s]"); ax[3].grid(alpha=.3)
for A in ax:
    A.axvline(2.0, color="r", ls="--", lw=.8)
fig.suptitle("Resposta longitudinal: degrau u=1 por 2 s, depois u=0")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)

# --- numeros ---
dt = np.diff(t)
print(f"dt: mean={dt.mean():.4f} min={dt.min():.4f} max={dt.max():.4f} n={len(t)}")
on = (t > 0.05) & (t <= 2.0)
print(f"v em t=2s: {np.interp(2.0, t, v):.3f} m/s   v max: {v.max():.3f}")
# ajuste 1a ordem na subida: v(t)=K(1-exp(-t/tau))
vf = v[on][-1]
# tau = tempo para 63.2% de vf
print(f"63.2%% de v(2s) = {0.632*vf:.3f} -> t = {np.interp(0.632*vf, v[on], t[on]):.3f} s")
# ganho: dv/dt inicial
sl = np.polyfit(t[(t > .1) & (t < .5)], v[(t > .1) & (t < .5)], 1)[0]
print(f"inclinacao inicial dv/dt ~ {sl:.3f} m/s^2 (u comandado = 1.0)")
# coast down
off = t > 2.05
va = v[off]
print(f"apos u=0: v(2.05)={va[0]:.3f} -> v(final)={va[-1]:.3f}, parou em t~"
      f"{t[off][np.argmax(np.abs(va) < 0.02)] if np.any(np.abs(va) < 0.02) else float('nan'):.2f} s")
des = np.polyfit(t[off][:20], va[:20], 1)[0]
print(f"desaceleracao media logo apos corte: {des:.3f} m/s^2  (mu*g = {0.05*9.81:.3f})")
print(f"v final: {v[-1]:.4f}   dist total: {np.hypot(x-x[0], y-y[0])[-1]:.3f} m")
