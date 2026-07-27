# ============================================================
# COMPARAÇÃO MULTIOBJETIVO
#
# MOWGA vs NSGAIII vs MOPSO vs MOWCA vs HYBRID
#
# HYBRID COM CONTROLE DE ITERAÇÕES
#
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import time
from hybrid_algorithm import HybridMO

# ============================================================
# CONFIGURAÇÕES
# ============================================================

np.random.seed(42)

# ============================================================
# BENCHMARKS
# ============================================================

def zdt1(x):

    f1 = x[0]

    g = 1 + 9*np.sum(x[1:])/(len(x)-1)

    f2 = g * (1 - np.sqrt(f1/g))

    return np.array([f1, f2])

def zdt2(x):

    f1 = x[0]

    g = 1 + 9*np.sum(x[1:])/(len(x)-1)

    f2 = g * (1 - (f1/g)**2)

    return np.array([f1, f2])

def zdt3(x):

    f1 = x[0]

    g = 1 + 9*np.sum(x[1:])/(len(x)-1)

    f2 = g * (
        1
        - np.sqrt(f1/g)
        - (f1/g)*np.sin(10*np.pi*f1)
    )

    return np.array([f1, f2])

def sphere_mo(x):

    f1 = np.sum(x**2)

    f2 = np.sum((x-2)**2)

    return np.array([f1, f2])

def rastrigin_mo(x):

    f1 = (
        10*len(x)
        + np.sum(
            x**2 - 10*np.cos(2*np.pi*x)
        )
    )

    f2 = np.sum((x-1)**2)

    return np.array([f1, f2])

def ackley_mo(x):

    f1 = (
        -20*np.exp(
            -0.2*np.sqrt(np.mean(x**2))
        )
        - np.exp(
            np.mean(np.cos(2*np.pi*x))
        )
        + 20
        + np.e
    )

    f2 = np.sum((x+1)**2)

    return np.array([f1, f2])

def griewank_mo(x):

    f1 = (
        np.sum(x**2)/4000
        - np.prod(
            np.cos(
                x / np.sqrt(
                    np.arange(1, len(x)+1)
                )
            )
        )
        + 1
    )

    f2 = np.sum((x-3)**2)

    return np.array([f1, f2])

def schwefel_mo(x):

    f1 = (
        418.9829*len(x)
        - np.sum(
            x*np.sin(np.sqrt(np.abs(x)))
        )
    )

    f2 = np.sum((x+2)**2)

    return np.array([f1, f2])

def rosenbrock_mo(x):

    f1 = np.sum(
        100*(x[1:] - x[:-1]**2)**2
        + (1 - x[:-1])**2
    )

    f2 = np.sum((x-1.5)**2)

    return np.array([f1, f2])

def dtlz_like(x):

    mid = len(x)//2

    f1 = np.sum(x[:mid]**2)

    f2 = np.sum(x[mid:]**2)

    return np.array([f1, f2])

benchmarks = {

    "ZDT1": zdt1,
    "ZDT2": zdt2,
    "ZDT3": zdt3,
    "Sphere": sphere_mo,
    "Rastrigin": rastrigin_mo,
    "Ackley": ackley_mo,
    "Griewank": griewank_mo,
    "Schwefel": schwefel_mo,
    "Rosenbrock": rosenbrock_mo,
    "DTLZ-Like": dtlz_like
}

# ============================================================
# PARETO
# ============================================================

# ============================================================
# EXECUÇÃO
# ============================================================

results = {}
times = {}

# ============================================================
# EXECUTA BENCHMARKS
# ============================================================

for bench_name, func in benchmarks.items():
    print(f"\nBenchmark: {bench_name}")
    results[bench_name] = {}
    times[bench_name] = {}

    # ========================================================
    # HYBRID
    # ========================================================

    print("  -> HYBRID")

    hybrid = HybridMO(

        func,

        mowga_iters=1,
        nsgaiii_iters=2,
        mopso_iters=3,
        mowca_iters=1,

        exchange_interval=10
    )

    start = time.perf_counter()

    hybrid_pf = hybrid.optimize()

    end = time.perf_counter()

    elapsed = end - start

    times[bench_name]["HYBRID"] = elapsed

    results[bench_name]["HYBRID"] = hybrid_pf

# ============================================================
# PLOTS PARETO
# ============================================================

fig, axes = plt.subplots(
    5,
    2,
    figsize=(16, 24)
)

axes = axes.flatten()

markers = {

    "MOWGA": "o",
    "NSGAIII": "s",
    "MOPSO": "^",
    "MOWCA": "x",
    "HYBRID": "*"
}

for idx, (
    bench_name,
    bench_results
) in enumerate(results.items()):

    ax = axes[idx]

    for alg_name, pf in bench_results.items():

        ax.scatter(

            pf[:, 0],
            pf[:, 1],

            s=30,

            marker=markers[alg_name],

            label=alg_name,

            alpha=0.7
        )

    ax.set_title(
        f"Pareto Front - {bench_name}"
    )

    ax.set_xlabel("f1")

    ax.set_ylabel("f2")

    ax.grid(True)

    ax.legend()

plt.tight_layout()

plt.show()