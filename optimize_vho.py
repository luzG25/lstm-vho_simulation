import numpy as np
import pandas as pd
from vho_sim import simulate
from hybrid_algorithm import HybridMO

# parâmetros a otimizar: [dwell (D-VHO), threshold (LA-VHO), Nu_ref]
DIM = 2
LB = np.array([0.2, 2])
UB = np.array([2.0, 10])

def objective(params):
    dwell, threshold = params
    threshold = int(round(threshold))
    r = simulate("LA-VHO", Nu=6, room="big", threshold=threshold, seed=42)
    # objetivos: minimizar NVHO e minimizar perda de pacotes
    return np.array([r["NVHO"], r["packet_loss"]])

hybrid = HybridMO(objective, dim=DIM, lb=LB, ub=UB, exchange_interval=2)
pareto_fitness, pareto_params = hybrid.optimize()
best_idx = np.argmin(pareto_fitness[:, 1])
best_params = pareto_params[best_idx]

# gera o dataset de decisões usando os parâmetros otimizados
result = simulate("LA-VHO", Nu=6, room="big",
                   threshold=int(round(best_params[1])),
                   n_iter=50, log_decisions=True, seed=123)

df = pd.DataFrame(result["log"])
df.to_csv("output/handover_decisions_dataset.csv", index=False)