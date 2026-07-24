"""
Reprodução da simulação de handover vertical location-aware (LA-VHO) em rede
híbrida VLC/WLAN, comparada com I-VHO e D-VHO (t=0.5s, t=1s), baseada em:

A. Zeshan and T. Baykas, "Location Aware Vertical Handover in a VLC/WLAN
Hybrid Network," IEEE Access, vol. 9, pp. 129810-129819, 2021.

OBS. IMPORTANTE (transparência metodológica):
O artigo não publica código nem todos os parâmetros finos do modelo
(pesos w_u/w_s exatos, custo de delay d_c, forma exata da probabilidade de
bloqueio Pb(Nu) fora da eq. 15 simbólica, etc.). Esta reimplementação segue
fielmente a ARQUITETURA descrita (Seção IV, Algoritmo 1, Tabela 1) e usa
modelos simplificados, mas fisicamente coerentes, para as partes não
especificadas numericamente. As tendências qualitativas reproduzidas batem
com o artigo (I-VHO pior em handovers e melhor em perda de pacotes, D-VHO
sensível ao dwell time, LA-VHO domina em handovers e QoE, sobretudo quando
Nu ultrapassa o limiar), mas os valores percentuais exatos NÃO são
reproduzidos (é matematicamente impossível sem os parâmetros ocultos).
"""

import numpy as np

# ----------------------------------------------------------------------
# Parâmetros do sistema (Tabela 1 do artigo)
# ----------------------------------------------------------------------
R_VLC = 1.5          # raio de cobertura de cada AP VLC (m)
V_MIN, V_MAX = 0.2, 1.0   # velocidade do usuário (m/s)
DT = 0.25             # passo de tempo da simulação (s)
MEAN_TURN_INTERVAL = 4.0  # tempo médio entre mudanças de direção (s) (fluid-flow)

DELAY_COST = 0.28     # custo (em unidades de QoE) por handover realizado
QOE_VLC_OK = 5.0       # QoE quando conectado e transmitindo bem via VLC
QOE_RF = 3.2            # QoE quando conectado via WLAN/RF
QOE_OUTAGE = 0.5        # QoE quando "preso" tentando usar VLC bloqueado (perda)


def room_layout(room):
    """Retorna (lista de posições de APs VLC, lado da sala)."""
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
    Probabilidade de bloqueio do canal óptico (sombreamento) em função do
    número de usuários ativos Nu, eq. (15) do artigo (forma qualitativa:
    cresce monotonicamente e satura). Calibrada para produzir o mesmo tipo
    de degradação de BER/bloqueio relatado na Fig. 3 do artigo (crescente
    e saturante com Nu).
    """
    p1 = 0.035  # prob. de bloqueio instantâneo devido a 1 usuário passando
    return 1.0 - (1.0 - p1) ** Nu


def simulate(scheme, Nu, room, n_iter=250, sim_time=400.0,
             dwell=None, threshold=None, velocity_range=(V_MIN, V_MAX),
             seed=None):
    """
    Simula `n_iter` usuários (iterações independentes) andando aleatoriamente
    pela sala durante `sim_time` segundos, decidindo handovers verticais
    segundo `scheme` in {'I-VHO','D-VHO','LA-VHO'}.

    Retorna dict com médias de: NVHO (handovers/iteração), QoE, packet_loss (%)
    """
    rng = np.random.default_rng(seed)
    aps, L = room_layout(room)
    n_steps = int(sim_time / DT)

    # --- estado vetorizado (um "usuário" por linha) ---
    pos = rng.uniform(0, L, size=(n_iter, 2))
    ang = rng.uniform(0, 2 * np.pi, size=n_iter)
    vel = rng.uniform(*velocity_range, size=n_iter)
    next_turn = rng.exponential(MEAN_TURN_INTERVAL, size=n_iter)

    pb = blocking_probability(Nu)

    mode_vlc = np.zeros(n_iter, dtype=bool)   # True = conectado via VLC
    blocked_timer = np.zeros(n_iter)          # p/ D-VHO (dwell)
    la_pending_timer = np.zeros(n_iter)       # p/ histerese do LA-VHO
    handovers = np.zeros(n_iter)
    time_vlc_ok = np.zeros(n_iter)
    time_rf = np.zeros(n_iter)
    time_outage = np.zeros(n_iter)
    t_elapsed = np.zeros(n_iter)

    force_rf = (Nu >= threshold) if (scheme == "LA-VHO" and threshold is not None) else False

    for step in range(n_steps):
        t_elapsed += DT
        # -- mudanças de direção (fluid-flow model, Seção V) --
        turn_now = t_elapsed >= next_turn
        if turn_now.any():
            ang[turn_now] = rng.uniform(0, 2 * np.pi, size=turn_now.sum())
            next_turn[turn_now] = t_elapsed[turn_now] + rng.exponential(
                MEAN_TURN_INTERVAL, size=turn_now.sum())

        # -- movimento e reflexão nas paredes --
        pos[:, 0] += vel * np.cos(ang) * DT
        pos[:, 1] += vel * np.sin(ang) * DT
        for k in (0, 1):
            below = pos[:, k] < 0
            above = pos[:, k] > L
            pos[below, k] = -pos[below, k]
            pos[above, k] = 2 * L - pos[above, k]
            ang[below | above] = rng.uniform(0, 2 * np.pi, size=(below | above).sum())

        # -- cobertura geométrica VLC (dentro do raio de algum AP) --
        d = np.linalg.norm(pos[:, None, :] - aps[None, :, :], axis=2)
        geo_cov = (d <= R_VLC).any(axis=1)

        # -- bloqueio por sombreamento (shadowing), eq. (10)-(15) --
        shadow_blocked = rng.random(n_iter) < pb
        link_available = geo_cov & (~shadow_blocked)

        # ------------------- decisão de handover -------------------
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
                # segunda etapa: seleção por função de custo minimizando
                # handovers -> pequena histerese temporal (estabilidade)
                # antes de comutar em qualquer direção, representando a
                # minimização do custo de handover via gradiente descendente.
                la_dwell = 1.5
                stable_state = (link_available == mode_vlc)
                la_pending_timer = np.where(stable_state, 0.0, la_pending_timer + DT)
                do_switch = (~stable_state) & (la_pending_timer >= la_dwell)
                new_mode = np.where(do_switch, link_available, mode_vlc)
        else:
            raise ValueError(scheme)

        switched = new_mode != mode_vlc
        handovers += switched

        # -- métricas de QoE / perda de pacotes --
        vlc_ok = new_mode & link_available
        vlc_outage = new_mode & (~link_available)   # tentando VLC mas bloqueado -> perda
        rf_on = ~new_mode

        time_vlc_ok += vlc_ok * DT
        time_outage += vlc_outage * DT
        time_rf += rf_on * DT

        mode_vlc = new_mode

    total_time = n_steps * DT
    avg_nvho = handovers.mean()
    packet_loss = 100.0 * (time_outage / total_time)
    qoe = (QOE_VLC_OK * time_vlc_ok + QOE_RF * time_rf + QOE_OUTAGE * time_outage) / total_time
    qoe = qoe - DELAY_COST * handovers / (total_time / 10.0)  # custo de sinalização
    qoe = np.clip(qoe, 0, 5)

    return {
        "NVHO": avg_nvho,
        "QoE": qoe.mean(),
        "packet_loss": packet_loss.mean(),
    }
