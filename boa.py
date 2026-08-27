"""
Butterfly Optimization Algorithm (BOA), after Arora & Singh (2019),
"Butterfly optimization algorithm: a novel approach for global optimization",
Soft Computing 23:715-734.

Pure numpy, population-based metaheuristic minimizer. Plays the same role
here that Genetic Algorithm / Artificial Bee Colony played in the reference
papers' curve-fitting pipelines: optimize a handful of free curve parameters
to minimize a sum-of-squared-error fitness function.

Each butterfly i at position x_i emits a "fragrance":
    f_i = c * I_i^a
where I_i is the stimulus intensity (here: 1 / (1 + fitness), so better
solutions smell stronger), c is sensor modality, a is the power exponent.
With probability p, a butterfly moves toward the current global best
(global search); otherwise it moves relative to two random butterflies
(local search).
"""
import numpy as np


def butterfly_optimize(fitness_fn, bounds, n_pop=30, n_iter=200, p=0.8,
                        c0=0.01, c1=0.25, a=0.3, seed=None, verbose=False):
    """
    fitness_fn: callable, takes array shape (n_pop, n_dim) -> array (n_pop,)
                of fitness values (lower = better).
    bounds: array shape (n_dim, 2) of (lo, hi) per dimension.
    c grows linearly from c0 to c1 over the run (standard BOA enhancement -
    increases exploitation strength as the search progresses).
    Returns: (best_x, best_fitness, history)
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=np.float64)
    n_dim = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    span[span == 0] = 1.0

    x = lo + rng.random((n_pop, n_dim)) * (hi - lo)
    fitness = fitness_fn(x)
    best_idx = np.argmin(fitness)
    g_best = x[best_idx].copy()
    g_best_fit = fitness[best_idx]
    history = [g_best_fit]

    for it in range(n_iter):
        c = c0 + (c1 - c0) * it / max(1, n_iter - 1)
        # normalize fitness to [0,1] (1 = best) so intensity/fragrance is
        # well-scaled regardless of the fitness function's magnitude
        fmin, fmax = fitness.min(), fitness.max()
        norm = (fmax - fitness) / (fmax - fmin + 1e-12)
        fragrance = c * (norm + 1e-6) ** a

        r = rng.random(n_pop)
        new_x = x.copy()

        global_mask = r < p
        if global_mask.any():
            r2 = rng.random((global_mask.sum(), n_dim)) ** 2
            new_x[global_mask] = (x[global_mask]
                                   + (r2 * g_best[None, :] - x[global_mask])
                                   * fragrance[global_mask, None])

        local_mask = ~global_mask
        if local_mask.any():
            n_local = local_mask.sum()
            j = rng.integers(0, n_pop, n_local)
            k = rng.integers(0, n_pop, n_local)
            r2 = rng.random((n_local, n_dim)) ** 2
            new_x[local_mask] = (x[local_mask]
                                  + (r2 * x[j] - x[k])
                                  * fragrance[local_mask, None])

        new_x = np.clip(new_x, lo, hi)
        new_fitness = fitness_fn(new_x)

        improved = new_fitness < fitness
        x[improved] = new_x[improved]
        fitness[improved] = new_fitness[improved]

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < g_best_fit:
            g_best_fit = fitness[best_idx]
            g_best = x[best_idx].copy()
        history.append(g_best_fit)

        if verbose and it % 20 == 0:
            print(f"    BOA iter {it}: best fitness = {g_best_fit:.6g}")

    return g_best, g_best_fit, history


if __name__ == '__main__':
    # Self-test 1: minimize sum of squares (sphere function), known optimum 0
    def sphere(x):
        return np.sum(x ** 2, axis=1)

    bounds = np.array([[-10, 10]] * 5)
    best_x, best_f, hist = butterfly_optimize(sphere, bounds, n_pop=30, n_iter=150, seed=0)
    print("sphere: best_x =", best_x, "best_f =", best_f)
    assert best_f < 1e-3, f"BOA failed to converge on sphere function: {best_f}"

    # Self-test 2: minimize a 2D curve-fitting-style problem - find the
    # midpoint of two known points (should converge near their average)
    target = np.array([3.0, -2.0])

    def dist_fitness(x):
        return np.sum((x - target[None, :]) ** 2, axis=1)

    bounds2 = np.array([[-20, 20], [-20, 20]])
    best_x2, best_f2, _ = butterfly_optimize(dist_fitness, bounds2, n_pop=20, n_iter=150, seed=1)
    print("target test: best_x =", best_x2, "expected", target)
    assert np.allclose(best_x2, target, atol=0.15)

    print("ALL BOA SELF-TESTS PASSED")
