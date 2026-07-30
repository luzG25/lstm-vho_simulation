import numpy as np
import matplotlib.pyplot as plt
from vho_sim import simulate
from lstm_policy import LSTMHandoverPolicy

# load the model
lstm_policy = LSTMHandoverPolicy(
    model_path="output/lstm_vho_model.pt",
    scaler_path="output/lstm_vho_scaler.pkl",
    meta_path="output/lstm_vho_meta.json",
)

SCHEMES = ["I-VHO", "D-VHO(0.5s)", "D-VHO(1s)", "LA-VHO", "LSTM-VHO"]
COLORS = {"I-VHO": "#2f6b3a", "D-VHO(0.5s)": "#4aa3e0",
          "D-VHO(1s)": "#b5482f", "LA-VHO": "#d63bb0", "LSTM-VHO": "#e0a020"}
MARK = {"I-VHO": "s", "D-VHO(0.5s)": "D", "D-VHO(1s)": "o", "LA-VHO": "x", "LSTM-VHO": "^"}

def run_scheme(name, Nu, room, **kw):
    if name == "I-VHO":
        return simulate("I-VHO", Nu, room, **kw)
    if name == "D-VHO(0.5s)":
        return simulate("D-VHO", Nu, room, dwell=0.5, **kw)
    if name == "D-VHO(1s)":
        return simulate("D-VHO", Nu, room, dwell=1.0, **kw)
    if name == "LA-VHO":
        return simulate("LA-VHO", Nu, room, threshold=6, **kw)
    if name == "LSTM-VHO":
        return simulate("LSTM-VHO", Nu, room, lstm_policy=lstm_policy, **kw)
    raise ValueError(name)


# ======================================================================
# FIGURE 5: NVHO vs number of users (9 APs)
# ======================================================================
Nu_range = list(range(1, 11))
plt.figure(figsize=(6, 4.5))
for sc in SCHEMES:
    vals = [run_scheme(sc, Nu, "big", seed=100 + Nu)["NVHO"] for Nu in Nu_range]
    plt.plot(Nu_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)
plt.xlabel("Number of Users (Nu)")
plt.ylabel("Average Number of Handovers")
plt.title("NVHO vs Number of Users (9 VLC APs)")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("output/fig5_nvho_9aps.png", dpi=150)
plt.close()

# ======================================================================
# FIGURE 6: QoE vs velocity (Nu fixed) and QoE vs Nu (random velocity)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

vel_range = np.arange(0.2, 1.01, 0.1)
ax = axes[0]
for sc in SCHEMES:
    vals = [run_scheme(sc, 5, "big", velocity_range=(v, v), seed=7)["QoE"] for v in vel_range]
    ax.plot(vel_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)
ax.set_xlabel("Velocity of the user (m/s)")
ax.set_ylabel("QoE")
ax.set_title("QoE vs velocity (Nu=5)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
for sc in SCHEMES:
    vals = [run_scheme(sc, Nu, "big", seed=100 + Nu)["QoE"] for Nu in Nu_range]
    ax.plot(Nu_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)
ax.set_xlabel("Number of users (Nu)")
ax.set_ylabel("QoE")
ax.set_title("QoE vs Nu (9 APs)")
ax.grid(alpha=0.3)

fig.suptitle("Fig. 6 —  QoE Comparison")
fig.tight_layout()
fig.savefig("output/fig6_qoe.png", dpi=150)
plt.close(fig)


#======================================================================
# FIGURE: QoE vs Number of Users (9 APs)
# ======================================================================
plt.figure(figsize=(6, 4.5))

for sc in SCHEMES:
    vals = [run_scheme(sc, Nu, "big", seed=100 + Nu)["QoE"] for Nu in Nu_range]
    plt.plot(Nu_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)

plt.xlabel("Number of Users (Nu)")
plt.ylabel("QoE")
plt.title("QoE vs Number of Users (9 VLC APs)")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("output/fig6_qoe_9aps.png", dpi=150)
plt.close()

# ======================================================================
# FIGURE 7: Packet Loss vs velocity  vs Nu
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
for sc in SCHEMES:
    vals = [run_scheme(sc, 5, "big", velocity_range=(v, v), seed=7)["packet_loss"] for v in vel_range]
    ax.plot(vel_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)
ax.set_xlabel("Velocity of the user (m/s)")
ax.set_ylabel("Packet Loss(%)")
ax.set_title("Packet Loss vs velocity (Nu=5)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
for sc in SCHEMES:
    vals = [run_scheme(sc, Nu, "big", seed=100 + Nu)["packet_loss"] for Nu in Nu_range]
    ax.plot(Nu_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)
ax.set_xlabel("Number of users (Nu)")
ax.set_ylabel("Packet Loss(%)")
ax.set_title("Packet Loss vs Nu (9 APs)")
ax.grid(alpha=0.3)

fig.suptitle("Fig. 7  — Packet loss comparation")
fig.tight_layout()
fig.savefig("output/fig7_packetloss.png", dpi=150)
plt.close(fig)

# ======================================================================
# FIGURE: Packet Loss vs Number of Users (9 APs)
# ======================================================================
plt.figure(figsize=(6, 4.5))

for sc in SCHEMES:
    vals = [run_scheme(sc, Nu, "big", seed=100 + Nu)["packet_loss"] for Nu in Nu_range]
    plt.plot(Nu_range, vals, marker=MARK[sc], color=COLORS[sc], label=sc)

plt.xlabel("Number of Users (Nu)")
plt.ylabel("Packet Loss (%)")
plt.title("Packet Loss vs Number of Users (9 VLC APs)")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("output/fig7_packetloss_9aps.png", dpi=150)
plt.close()

# ======================================================================
# Resume table (Nu=2 e Nu=10) 
# ======================================================================
print(f"{'Esquema':14s} {'Nu':>3s} {'Sala':>6s} {'NVHO':>8s} {'QoE':>6s} {'PktLoss%':>9s}")
for room in ["small", "big"]:
    for Nu in [2, 10]:
        for sc in SCHEMES:
            r = run_scheme(sc, Nu, room, seed=100 + Nu)
            print(f"{sc:14s} {Nu:3d} {room:>6s} {r['NVHO']:8.2f} {r['QoE']:6.2f} {r['packet_loss']:9.2f}")

print("\nOK - figures saved in /fig5_nvho.png, fig6_qoe.png, fig7_packetloss.png")