import numpy as np
from pymoo.core.problem import Problem

import logging

def f1(x) -> float:
    return float(np.sum(np.abs(x)) / len(x))


def f2(x, p: float = None, weight=None, classes=None) -> float:
    if classes is None:
        classes = np.zeros_like(x)
    if weight is None:
        weight = 0.5 * np.ones_like(x)
    else:
        weight = normalized_weight(weight, classes)
    if p is None:
        p = -np.inf
    pairs = np.zeros_like(x)
    for i, a in enumerate(x):
        dists = [abs(a-b) for j, b in enumerate(x) if i!=j and classes[i] == classes[j]]
        if len(dists):
            pairs[i] = min(dists)
        else:
            pairs[i] = np.nan
    #pairs = [min([abs(a - b) for j, b in enumerate(x) if i != j and classes[i] == classes[j]]) for i, a in enumerate(x)]
    pairs = pairs * weight
    pairs = pairs[np.isfinite(pairs)]
    if p < 0.0 and 0.0 in pairs: #this is to avoid division by zero error. Numpy handles it correctly, but this edge-case should probably be handled somehow...
        z = 0.0
    else:
        z = np.linalg.norm(pairs, p)
            
    return float(1.0 - z)


def normalized_weight(weight, classes=None):
    if len(weight) <= 1:
        return weight
    if classes is None:
        norm = sum([1/w for w in sorted(weight)[:-1]])
        return weight / norm
    # recursively build PS for each class:
    assert len(weight) == len(classes)
    x = np.zeros_like(weight)
    for c in set(classes):
        idx = np.where(c == classes)
        wc = normalized_weight(weight[idx])
        x[idx] = wc
    return x


def fitness(x:np.array, p : float = -np.inf, weight : np.array = None, classes: np.array = None):
    if weight is None:
        weight = .5 * np.ones_like(x)
    x = np.clip(x, 0.0, 1.0)
    return f1(x), f2(x, p=p, weight=weight, classes=classes)


def optimal_set(dim=10, samples=50, weight=None, classes=None):
    if weight  is None:
        weight = np.ones(dim)
    if classes is not None:
        # recursively build PS for each class:
        x = np.zeros((samples, dim))
        for c in set(classes):
            idx = np.where(c == classes)
            local_ps = optimal_set(dim=len(weight[idx]), weight=weight[idx], samples=samples).T
            x.T[idx] = local_ps
        return x
    full = np.array([0.0])
    if dim > 1:
        var = 0.0
        sol = np.zeros(dim)
        for i, w in sorted(enumerate(weight), key=lambda x: x[1], reverse=True)[1:]:
            var +=  1 / w
            sol[i] = var
        full = sol * (1.0 / max(sol))
    return np.array([full * scale for scale in np.linspace(0.0, 1.0, num=samples, endpoint=True)])

def sampled_front(dim=10, samples=50, p=-np.inf, weight=None, classes=None):
    s = optimal_set(dim=dim, samples=samples, weight=weight, classes=classes)
    return [fitness(x, p=p, weight=weight, classes=classes) for x in s]

def get_classes(ctype, classes, N):
    if classes is not None:
        if ctype is not None:
            logging.warn("Setting both ctype and classes, ctype will be ignored")
        return np.array(classes)
    if ctype is None or ctype == "none":
        return np.zeros(N, dtype=int)
    if ctype == "half":
        first = np.zeros(N // 2, dtype=int)
        second = np.ones(N - N // 2, dtype=int)
        return np.concatenate([first, second])
    if ctype == "G3":
        return np.array([i // 3 for i in range(N)], dtype=int)
    if ctype == "G4":
        return np.array([i // 4 for i in range(N)], dtype=int)
    raise NotImplementedError("ctype not known")


def get_weights(wtype, weights, N):
    if weights is not None:
        if wtype is not None:
            logging.warn(f"setting wtype {wtype} and weights {weights}, wtype is ignored")
        return np.array(weights)
    if wtype is None or wtype == "equal":
        return np.ones(N)
    if wtype == "shallow":
        return np.linspace(1.0, 0.9, N)
    if wtype == "shallow":
        return np.linspace(1.0, 0.9, N)
    if wtype == "steep":
        return np.linspace(1.0, 0.1, N)
    raise NotImplementedError("wtype not known")


class MACO(Problem): # MACO = multi-agent coordination problem
    def __init__(self, n_var=10, p=-np.inf, weights=None, classes=None, ctype=None, wtype=None):
        classes = get_classes(ctype, classes, n_var)
        weights = get_weights(wtype, weights, n_var)
        
        assert len(weights) == n_var
        assert len(classes) == n_var
        self.classes = classes
        self.weights = weights
        self.p = p
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=np.zeros((n_var,)), xu=np.ones((n_var,)), elementwise=True)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.column_stack(fitness(x, p=self.p, weight=self.weights, classes=self.classes)).astype(float)

    def _calc_pareto_front(self, n_pareto_points=100):
        s = sampled_front(dim=self.n_var, samples=n_pareto_points, p=self.p, weight=self.weights, classes=self.classes)
        return np.array(s)

    def _calc_pareto_set(self, n_pareto_points=100):
        return np.array(optimal_set(dim=self.n_var, samples=n_pareto_points, weight=self.weights, classes=self.classes))
    