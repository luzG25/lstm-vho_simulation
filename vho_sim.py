import numpy as np

# ----------------------------------------------------------------------
# Parametres
# ----------------------------------------------------------------------
R_VLC = 1.5          # cover radius of each AP VLC (m)
V_MIN, V_MAX = 0.2, 1.0   # velocity of user (m/s)
DT = 0.25             # time step of simulation (s)
MEAN_TURN_INTERVAL = 4.0  # average time between direction changes (s) (fluid-flow)

DELAY_COST = 0.28     # cost (in unit of QoE) for handover 
QOE_VLC_OK = 5.0       # QoE when the device is conected e transmiting in VLC
QOE_RF = 3.2            # QoE when connect in WLAN/RF
QOE_OUTAGE = 0.5        # QoE when "locked"  trying to use VLC block (loss)


def room_layout(room):
    """Returns (list of VLC AP positions, room side)."""
    if room == "small":       # 5.6 x 5.6 x 3 m, 4 APs
        L = 5.6
        c = [1.4, 4.2]
        aps = [(x, y) for x in c for y in c]
    elif room == "big":       # 8 x 8 x 3 m, 9 APs
        L = 8.0
        c = [1.333, 4.0, 6.667]
        aps = [(x, y) for x in c for y in c]
    else:
        raise ValueError(room)
    return np.array(aps), L


def blocking_probability(Nu):
    """
    Probability of optical channel blockage (shadowing) as a function of
    the number of active users Nu.
    """
    p1 = 0.035  # Instant blocking probability due to one user passing by
    return 1.0 - (1.0 - p1) ** Nu


def simulate(scheme, Nu, room, n_iter=250, sim_time=400.0,
             dwell=None, threshold=None, velocity_range=(V_MIN, V_MAX),
             seed=None, log_decisions=False, verbose=False, lstm_policy=None):
    """
    Simulates `n_iter` users (independent iterations) moving randomly
    around the room for `sim_time` seconds, making vertical handover decisions
    according to `scheme` in {'I-VHO', 'D-VHO', 'LA-VHO', 'LSTM-VHO'}.

    For scheme='LSTM-VHO', it is mandatory to provide `lstm_policy`, an instance
    of lstm_policy.LSTMHandoverPolicy loading the model trained in
    train_lstm_handover.py. Until a user's history fills the window required
    by the model (`lstm_policy.window_size` steps), the scheme behaves
    like I-VHO (using link_available directly).

    If log_decisions=True, the return value includes the key "log": a list of
    dictionaries with one entry per (user, time step), containing the
    position, link state, and handover decision made at that moment—useful
    for compiling a dataset of handover decisions.

    Returns a dict with averages of: NVHO (handovers/iteration), QoE, packet_loss (%)
    """
    print(f"\n>>>Starting Simulation: scheme={scheme} | Nu={Nu} | room={room} "
          f"| n_iter={n_iter} | sim_time={sim_time}s")

    if verbose:
        print(f"[simulate] scheme={scheme} Nu={Nu} room={room} n_iter={n_iter} "
              f"sim_time={sim_time} dwell={dwell} threshold={threshold} "
              f"log_decisions={log_decisions}")

    if scheme == "LSTM-VHO" and lstm_policy is None:
        raise ValueError("scheme='LSTM-VHO' requer o argumento lstm_policy="
                          "<instância de LSTMHandoverPolicy> (ver lstm_policy.py)")


    rng = np.random.default_rng(seed)
    aps, L = room_layout(room)
    n_steps = int(sim_time / DT)

    print(f"    room '{room}' loaded: {len(aps)} APs VLC, side={L}m, "
          f"{n_steps} steps of simulation (DT={DT}s)")

    # --- vector  (one user for line) ---
    pos = rng.uniform(0, L, size=(n_iter, 2))
    ang = rng.uniform(0, 2 * np.pi, size=n_iter)
    vel = rng.uniform(*velocity_range, size=n_iter)
    next_turn = rng.exponential(MEAN_TURN_INTERVAL, size=n_iter)

    pb = blocking_probability(Nu)

    mode_vlc = np.zeros(n_iter, dtype=bool)   # True = connected by VLC
    blocked_timer = np.zeros(n_iter)          # for D-VHO (dwell)
    la_pending_timer = np.zeros(n_iter)       
    handovers = np.zeros(n_iter)
    time_vlc_ok = np.zeros(n_iter)
    time_rf = np.zeros(n_iter)
    time_outage = np.zeros(n_iter)
    t_elapsed = np.zeros(n_iter)

    force_rf = (Nu >= threshold) if (scheme == "LA-VHO" and threshold is not None) else False

    log = [] if log_decisions else None

    # slider buffer for LSTM-VHO: (n_iter, window_size, n_features)
    # ["x","y","vel","link_available","geo_cov","shadow_blocked","mode_before"]
    if scheme == "LSTM-VHO":
        lstm_window = lstm_policy.window_size
        lstm_nfeat = lstm_policy.n_features
        feat_buffer = np.zeros((n_iter, lstm_window, lstm_nfeat), dtype=np.float32)
        print(f"    LSTM-VHO: buffer deslizante inicializado "
              f"(window_size={lstm_window}, n_features={lstm_nfeat}) — "
              f"aquecimento tipo I-VHO durante os primeiros {lstm_window} passos")

    for step in range(n_steps):
        t_elapsed += DT
        # -- direction changes (fluid-flow model) --
        turn_now = t_elapsed >= next_turn
        if turn_now.any():
            ang[turn_now] = rng.uniform(0, 2 * np.pi, size=turn_now.sum())
            next_turn[turn_now] = t_elapsed[turn_now] + rng.exponential(
                MEAN_TURN_INTERVAL, size=turn_now.sum())

        # -- movement and reflexing on the walls --
        pos[:, 0] += vel * np.cos(ang) * DT
        pos[:, 1] += vel * np.sin(ang) * DT
        for k in (0, 1):
            below = pos[:, k] < 0
            above = pos[:, k] > L
            pos[below, k] = -pos[below, k]
            pos[above, k] = 2 * L - pos[above, k]
            ang[below | above] = rng.uniform(0, 2 * np.pi, size=(below | above).sum())

        # -- geometric cover VLC (between the AP cover) --
        d = np.linalg.norm(pos[:, None, :] - aps[None, :, :], axis=2)
        geo_cov = (d <= R_VLC).any(axis=1)

        # -- shadowing --
        shadow_blocked = rng.random(n_iter) < pb
        link_available = geo_cov & (~shadow_blocked)

        # ------------------- handover decision -------------------
        if scheme == "I-VHO":
            new_mode = link_available

        elif scheme == "D-VHO":
            blocked_timer = np.where(link_available, 0.0, blocked_timer + DT)
            switch_to_rf = mode_vlc & (~link_available) & (blocked_timer >= dwell)
            new_mode = np.where(switch_to_rf, False,
                                 np.where(link_available, True, mode_vlc))

        elif scheme == "LA-VHO":
            if force_rf:
                new_mode = np.zeros(n_iter, dtype=bool)
            else:
                # second stage: selection via cost-minimization function
                # handovers -> short time hysteresis (stability)
                # before switching in any direction, representing the
                # minimization of handover cost via gradient descent.
                la_dwell = 1.5
                stable_state = (link_available == mode_vlc)
                la_pending_timer = np.where(stable_state, 0.0, la_pending_timer + DT)
                do_switch = (~stable_state) & (la_pending_timer >= la_dwell)
                new_mode = np.where(do_switch, link_available, mode_vlc)

        elif scheme == "LA-VHO2":
            # threshold passa a representar o custo do handover
            ho_cost = threshold / 10.0

            # custo esperado em VLC
            cost_vlc = np.where(
                link_available,
                0.0,          # ligação boa
                1.0           # ligação bloqueada
            )

            # custo esperado em RF
            cost_rf = np.full(n_iter, 0.3)

            # penalizar mudanças desnecessárias
            cost_vlc += (~mode_vlc) * ho_cost
            cost_rf  += (mode_vlc) * ho_cost

            prefer_vlc = cost_vlc < cost_rf

            # histerese
            stable_state = (prefer_vlc == mode_vlc)
            la_pending_timer = np.where(
                stable_state,
                0.0,
                la_pending_timer + DT
            )

            do_switch = (~stable_state) & (la_pending_timer >= dwell)

            new_mode = np.where(
                do_switch,
                prefer_vlc,
                mode_vlc
            )

        elif scheme == "LSTM-VHO":
            # construct the feature for the current step in the SAME order used during training
            current_features = np.stack([
                pos[:, 0], pos[:, 1], vel,
                link_available.astype(np.float32),
                geo_cov.astype(np.float32),
                shadow_blocked.astype(np.float32),
                mode_vlc.astype(np.float32),   # mode_before
            ], axis=1)

            feat_buffer = np.roll(feat_buffer, -1, axis=1)
            feat_buffer[:, -1, :] = current_features

            if step >= lstm_window - 1:
                if step == lstm_window - 1:
                    print(f" LSTM-VHO: warm-up completed at step {step + 1} "
                          f"— neural network begins making decisions from this point on.")
                new_mode = lstm_policy.decide(feat_buffer)
            else:
                # warm-up: window still incomplete -> behaves like I-VHO
                new_mode = link_available

        else:
            raise ValueError(scheme)

        switched = new_mode != mode_vlc
        handovers += switched

        # -- metrics of QoE / packet loss --
        vlc_ok = new_mode & link_available
        vlc_outage = new_mode & (~link_available)   # Trying VLC but blocked -> loss
        rf_on = ~new_mode

        time_vlc_ok += vlc_ok * DT
        time_outage += vlc_outage * DT
        time_rf += rf_on * DT

        if log_decisions:
            for u in range(n_iter):
                log.append({
                    "scheme": scheme,
                    "Nu": Nu,
                    "room": room,
                    "step": step,
                    "t": float(t_elapsed[u]),
                    "user": u,
                    "x": float(pos[u, 0]),
                    "y": float(pos[u, 1]),
                    "vel": float(vel[u]),
                    "geo_cov": bool(geo_cov[u]),
                    "shadow_blocked": bool(shadow_blocked[u]),
                    "link_available": bool(link_available[u]),
                    "mode_before": bool(mode_vlc[u]),
                    "action_switch": bool(switched[u]),
                    "mode_after": bool(new_mode[u]),
                })

        mode_vlc = new_mode

        if (step + 1) % max(1, n_steps // 5) == 0:
            print(f"    [step {step + 1}/{n_steps}] "
                  f"(t={t_elapsed[0]:.1f}s de {sim_time}s) — "
                  f"Average handovers so far: {handovers.mean():.2f}")

    total_time = n_steps * DT
    avg_nvho = handovers.mean()
    packet_loss = 100.0 * (time_outage / total_time)
    qoe = (QOE_VLC_OK * time_vlc_ok + QOE_RF * time_rf + QOE_OUTAGE * time_outage) / total_time
    qoe = qoe - DELAY_COST * handovers / (total_time / 10.0)  # custo de sinalização
    qoe = np.clip(qoe, 0, 5)

    print(f">>> Simulation Concluded: NVHO={avg_nvho:.2f} | QoE={qoe.mean():.2f} | "
          f"packet_loss={packet_loss.mean():.2f}%"
          + (f" | log com {len(log)} linhas" if log_decisions else ""))

    result = {
        "NVHO": avg_nvho,
        "QoE": qoe.mean(),
        "packet_loss": packet_loss.mean(),
    }
    if log_decisions:
        result["log"] = log

    return result