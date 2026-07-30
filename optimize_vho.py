import numpy as np
import pandas as pd
from vho_sim import simulate
from hybrid_algorithm import HybridMO

# parâmetros a otimizar: [dwell (D-VHO), threshold (LA-VHO), Nu_ref]
DIM = 2
# [handover_cost, dwell]
LB = np.array([0.1, 0.2])
UB = np.array([2.0, 3.0])

def objective(params):

    ho_cost, dwell = params

    r = simulate(
        "LA-VHO2",
        Nu=6,
        room="big",
        threshold=ho_cost,
        dwell=dwell,
        seed=42
    )

    return np.array([
        -r["QoE"],
        r["packet_loss"],
        r["NVHO"]
    ])

hybrid = HybridMO(objective, dim=DIM, lb=LB, ub=UB, exchange_interval=2)
pareto_fitness, pareto_params = hybrid.optimize()
best_idx = np.argmin(pareto_fitness[:, 1])
best_params = pareto_params[best_idx]

# gera o dataset de decisões usando os parâmetros otimizados
result = simulate("LA-VHO2", Nu=6, room="big",
                   threshold=int(round(best_params[1])),
                   dwell=best_params[0],
                   n_iter=50, log_decisions=True, seed=123)

df = pd.DataFrame(result["log"])
df.to_csv("output/handover_decisions_dataset.csv", index=False)