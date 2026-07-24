import numpy as np
import matplotlib.pyplot as plt
from vho_sim import simulate

SCHEMES = ["I-VHO", "D-VHO(0.5s)", "D-VHO(1s)", "LA-VHO"]
COLORS = {"I-VHO": "#2f6b3a", "D-VHO(0.5s)": "#4aa3e0",
          "D-VHO(1s)": "#b5482f", "LA-VHO": "#d63bb0"}
MARK = {"I-VHO": "s", "D-VHO(0.5s)": "D", "D-VHO(1s)": "o", "LA-VHO": "x"}

def run_scheme(name, Nu, room, **kw):
    if name == "I-VHO":
        return simulate("I-VHO", Nu, room, **kw)
    if name == "D-VHO(0.5s)":
        return simulate("D-VHO", Nu, room, dwell=0.5, **kw)
    if name == "D-VHO(1s)":
        return simulate("D-VHO", Nu, room, dwell=1.0, **kw)
    if name == "LA-VHO":
        return simulate("LA-VHO", Nu, room, threshold=6, **kw)
    raise ValueError(name)


# ======================================================================
# FIGURA 5 (reproduzida): NVHO vs numero de usuarios, para as duas salas
# ======================================================================
Nu_range = list(range(1, 11))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, room, title in zip(axes, ["small", "big"], ["4 APs VLC (sala 5.6x5.6m)", "9 APs VLC (sala 8x8m)"]):
    for sc in SCHEMES:
        vals = [run_scheme(sc, Nu, room, seed=100 + Nu)["NVHO"] for Nu in Nu_range]
        ax.plot(Nu_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)
    ax.set_xlabel("Número de usuários (Nu)")
    ax.set_ylabel("N_VHO (handovers médios)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("Fig. 5 (reproduzida) — Número médio de handovers vs Nu")
fig.tight_layout()
fig.savefig("output/fig5_nvho.png", dpi=150)
plt.close(fig)

# ======================================================================
# FIGURA 6: QoE vs velocidade (Nu fixo) e QoE vs Nu (velocidade aleatoria)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

vel_range = np.arange(0.2, 1.01, 0.1)
ax = axes[0]
for sc in SCHEMES:
    vals = [run_scheme(sc, 5, "big", velocity_range=(v, v), seed=7)["QoE"] for v in vel_range]
    ax.plot(vel_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)
ax.set_xlabel("Velocidade do usuário (m/s)")
ax.set_ylabel("QoE")
ax.set_title("QoE vs velocidade (Nu=5)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
for sc in SCHEMES:
    vals = [run_scheme(sc, Nu, "big", seed=100 + Nu)["QoE"] for Nu in Nu_range]
    ax.plot(Nu_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)
ax.set_xlabel("Número de usuários (Nu)")
ax.set_ylabel("QoE")
ax.set_title("QoE vs Nu (9 APs)")
ax.grid(alpha=0.3)

fig.suptitle("Fig. 6 (reproduzida) — Comparação de QoE")
fig.tight_layout()
fig.savefig("output/fig6_qoe.png", dpi=150)
plt.close(fig)

# ======================================================================
# FIGURA 7: Perda de pacotes vs velocidade e vs Nu
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
for sc in SCHEMES:
    vals = [run_scheme(sc, 5, "big", velocity_range=(v, v), seed=7)["packet_loss"] for v in vel_range]
    ax.plot(vel_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)
ax.set_xlabel("Velocidade do usuário (m/s)")
ax.set_ylabel("Perda de pacotes (%)")
ax.set_title("Perda de pacotes vs velocidade (Nu=5)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
for sc in SCHEMES:
    vals = [run_scheme(sc, Nu, "big", seed=100 + Nu)["packet_loss"] for Nu in Nu_range]
    ax.plot(Nu_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)
ax.set_xlabel("Número de usuários (Nu)")
ax.set_ylabel("Perda de pacotes (%)")
ax.set_title("Perda de pacotes vs Nu (9 APs)")
ax.grid(alpha=0.3)

fig.suptitle("Fig. 7 (reproduzida) — Comparação de perda de pacotes")
fig.tight_layout()
fig.savefig("output/fig7_packetloss.png", dpi=150)
plt.close(fig)

# ======================================================================
# Tabela-resumo (Nu=2 e Nu=10) para conferência qualitativa com o artigo
# ======================================================================
print(f"{'Esquema':14s} {'Nu':>3s} {'Sala':>6s} {'NVHO':>8s} {'QoE':>6s} {'PktLoss%':>9s}")
for room in ["small", "big"]:
    for Nu in [2, 10]:
        for sc in SCHEMES:
            r = run_scheme(sc, Nu, room, seed=100 + Nu)
            print(f"{sc:14s} {Nu:3d} {room:>6s} {r['NVHO']:8.2f} {r['QoE']:6.2f} {r['packet_loss']:9.2f}")

print("\nOK - figuras salvas em output/fig5_nvho.png, fig6_qoe.png, fig7_packetloss.png")
