# hybrid_algorithm.py

import numpy as np
import matplotlib.pyplot as plt


POP_SIZE = 60
DIM = 10
ITERATIONS = 3

LB = -5
UB = 5


def dominates(a, b):

    return (
        np.all(a <= b)
        and np.any(a < b)
    )


def pareto_front(fitness):

    front = []

    for i in range(len(fitness)):

        dominated = False

        for j in range(len(fitness)):

            if i != j:

                if dominates(
                    fitness[j],
                    fitness[i]
                ):
                    dominated = True
                    break

        if not dominated:
            front.append(i)

    return front


# ============================================================
# BASE
# ============================================================

class BaseMO:

    def __init__(self, func, dim=DIM, lb=LB, ub=UB, pop_size=POP_SIZE, n_obj=2):

        self.func = func

        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size

        self.population = np.random.uniform(
            lb,
            ub,
            (pop_size, dim)
        )

        self.velocity = np.zeros(
            (pop_size, dim)
        )

        self.fitness = np.zeros(
            (pop_size, n_obj)
        )

    def evaluate(self):

        for i in range(self.pop_size):

            self.fitness[i] = self.func(
                self.population[i]
            )

# ============================================================
# MOWGA
# ============================================================

class MOWGA(BaseMO):

    def step(self):

        self.evaluate()

        pf = pareto_front(self.fitness)

        print(f"    [MOWGA]   pareto_front={len(pf)}/{self.pop_size}")

        elites = self.population[pf]

        new_pop = []

        while len(new_pop) < self.pop_size:

            p1 = elites[
                np.random.randint(len(elites))
            ]

            p2 = elites[
                np.random.randint(len(elites))
            ]

            alpha = np.random.rand()

            child = (
                alpha*p1
                + (1-alpha)*p2
            )

            child += np.random.normal(
                0,
                0.1,
                self.dim
            )

            child = np.clip(
                child,
                self.lb,
                self.ub
            )

            new_pop.append(child)

        self.population = np.array(new_pop)

# ============================================================
# NSGAIII SIMPLIFICADO
# ============================================================

class NSGAIII(BaseMO):

    def step(self):

        self.evaluate()

        pf = pareto_front(self.fitness)

        print(f"    [NSGAIII] pareto_front={len(pf)}/{self.pop_size}")

        leaders = self.population[pf]

        for i in range(self.pop_size):

            leader = leaders[
                np.random.randint(len(leaders))
            ]

            rand = np.random.uniform(
                -1,
                1,
                self.dim
            )

            self.population[i] += (

                0.3
                * rand
                * (
                    leader
                    - self.population[i]
                )
            )

        self.population = np.clip(
            self.population,
            self.lb,
            self.ub
        )

# ============================================================
# MOPSO
# ============================================================

class MOPSO(BaseMO):

    def step(self):

        self.evaluate()

        pf = pareto_front(self.fitness)

        print(f"    [MOPSO]   pareto_front={len(pf)}/{self.pop_size}")

        leaders = self.population[pf]

        w = 0.7
        c1 = 1.5
        c2 = 1.5

        for i in range(self.pop_size):

            leader = leaders[
                np.random.randint(len(leaders))
            ]

            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)

            self.velocity[i] = (

                w*self.velocity[i]

                + c1*r1*(
                    leader
                    - self.population[i]
                )

                + c2*r2*(
                    leader
                    - self.population[i]
                )
            )

            self.population[i] += (
                self.velocity[i]
            )

        self.population = np.clip(
            self.population,
            self.lb,
            self.ub
        )

# ============================================================
# MOWCA
# ============================================================

class MOWCA(BaseMO):

    def step(self):

        self.evaluate()

        pf = pareto_front(self.fitness)

        print(f"    [MOWCA]   pareto_front={len(pf)}/{self.pop_size}")

        seas = self.population[pf]

        for i in range(self.pop_size):

            sea = seas[
                np.random.randint(len(seas))
            ]

            flow = (
                np.random.rand()
                * (
                    sea
                    - self.population[i]
                )
            )

            self.population[i] += flow

        self.population = np.clip(
            self.population,
            self.lb,
            self.ub
        )

# ============================================================
# HYBRID
# ============================================================

class HybridMO:

    def __init__(
        self,
        func,

        dim=DIM,
        lb=LB,
        ub=UB,
        pop_size=POP_SIZE,

        mowga_iters=1,
        nsgaiii_iters=2,
        mopso_iters=3,
        mowca_iters=1,

        exchange_interval=10
    ):

        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size

        self.mowga = MOWGA(func, dim, lb, ub, pop_size)

        self.nsgaiii = NSGAIII(func, dim, lb, ub, pop_size)

        self.mopso = MOPSO(func, dim, lb, ub, pop_size)

        self.mowca = MOWCA(func, dim, lb, ub, pop_size)

        self.mowga_iters = mowga_iters
        self.nsgaiii_iters = nsgaiii_iters
        self.mopso_iters = mopso_iters
        self.mowca_iters = mowca_iters

        self.exchange_interval = exchange_interval

    # ========================================================

    def exchange(self):

        print("  >> EXCHANGE (troca de elites entre algoritmos)")

        algorithms = [

            self.mowga,
            self.nsgaiii,
            self.mopso,
            self.mowca
        ]

        names = ["MOWGA", "NSGAIII", "MOPSO", "MOWCA"]

        elites = []

        for name, alg in zip(names, algorithms):

            alg.evaluate()

            pf = pareto_front(
                alg.fitness
            )

            elite = alg.population[pf]

            print(f"     - {name}: {len(elite)} elites disponiveis")

            elites.append(elite)

        for i, alg in enumerate(algorithms):

            external = []

            for j, e in enumerate(elites):

                if i != j:
                    external.extend(e)

            external = np.array(external)

            if len(external) == 0:
                continue

            replace_n = min(
                5,
                len(external)
            )

            idx = np.random.choice(
                self.pop_size,
                replace_n,
                replace=False
            )

            chosen = external[
                np.random.choice(
                    len(external),
                    replace_n
                )
            ]

            alg.population[idx] = chosen

            print(f"     - {names[i]}: {replace_n} individuos substituidos por elites externas")

    # ========================================================

    def step(self, t, T):

        for _ in range(self.mowga_iters):
            self.mowga.step()

        for _ in range(self.nsgaiii_iters):
            self.nsgaiii.step()

        for _ in range(self.mopso_iters):
            self.mopso.step()

        for _ in range(self.mowca_iters):
            self.mowca.step()

        if (
            (t + 1)
            % self.exchange_interval
            == 0
        ):
            self.exchange()

    # ========================================================

    def optimize(self):

        print(f"\n=== INICIANDO OTIMIZACAO HYBRID (dim={self.dim}, pop_size={self.pop_size}, iters={ITERATIONS}) ===")

        for t in range(ITERATIONS):

            print(f"\n--- Iteracao {t + 1}/{ITERATIONS} ---")

            self.step(
                t,
                ITERATIONS
            )

            if (t + 1) % 10 == 0 or t == ITERATIONS - 1:
                self.mowga.evaluate()
                self.nsgaiii.evaluate()
                self.mopso.evaluate()
                self.mowca.evaluate()
                combined = np.vstack([
                    self.mowga.fitness,
                    self.nsgaiii.fitness,
                    self.mopso.fitness,
                    self.mowca.fitness,
                ])
                print(f"  [resumo] fitness medio combinado ate agora: {combined.mean(axis=0)}")

        print("\n=== OTIMIZACAO FINALIZADA, calculando fronteira de Pareto final ===")

        all_fit = []
        all_pop = []

        algorithms = [

            self.mowga,
            self.nsgaiii,
            self.mopso,
            self.mowca
        ]

        for alg in algorithms:

            alg.evaluate()

            all_fit.extend(
                alg.fitness
            )

            all_pop.extend(
                alg.population
            )

        all_fit = np.array(all_fit)
        all_pop = np.array(all_pop)

        pf = pareto_front(all_fit)

        print(f"Fronteira de Pareto final: {len(pf)} solucoes de {len(all_fit)} avaliadas")

        return all_fit[pf], all_pop[pf]