import numpy as np


def fsum(*x):
    return np.sum([*x], axis=0)


def fprod(*x):
    return np.prod([*x], axis=0)


def linear(left: float, right: float):
    return lambda x: np.clip((x - left) / (right - left), 0, 1)


def triangular(left: float, peak: float, right: float):
    return lambda x: np.clip(np.fmax(np.fmin((x - left) / (peak - left),
                                             (right - x) / (right - peak)),
                                     np.zeros_like(x)), 0, 1)


def centroid(ar: np.ndarray, domain: np.ndarray) -> float:
    norm = np.sum(ar)
    if norm == 0:
        return 0
    return np.sum(domain * ar) / norm


class Value:
    def __init__(self, membership: dict[str, callable]):
        self.membership = membership

    def __getattr__(self, item) -> callable:
        return self.membership[item]


class Rule:
    def __init__(self, antecedent: tuple[tuple, callable], consequent: tuple[callable, callable]):
        self.antecedent, self.comb = antecedent
        self.consequent, self.impl = consequent

    def evaluate_antecedent(self, params: tuple) -> float:
        return self.comb([member(p) for member, p in zip(self.antecedent, params)])

    def evaluate(self, params: tuple, domain: np.ndarray) -> np.ndarray:
        return self.impl(self.consequent(domain),
                         np.full_like(domain,
                                      self.evaluate_antecedent(params)))


class System:
    def __init__(self, rules, aggr, domain: np.ndarray):
        self.rules = rules
        self.domain = domain
        self.aggr = aggr

    def __call__(self, parameters: tuple):
        return self.aggr(*[rule.evaluate(parameters, self.domain) for rule in self.rules])


if __name__ == '__main__':
    heading = Value({'left': triangular(-np.pi, -np.pi / 2, 0),
                     'same': triangular(-np.pi / 2, 0, np.pi / 2),
                     'right': triangular(0, np.pi / 2, np.pi)})
    sig = Value({'low': linear(1, 0),
                 'high': linear(0, 1)})

    R1 = Rule(((sig.high, heading.left), min), (heading.left, fprod))
    R2 = Rule(((sig.high, heading.right), min), (heading.right, fprod))
    R3 = Rule(((sig.low, heading.same), max), (heading.same, fprod))

    domain = np.linspace(-np.pi, np.pi, 25)

    system = System([R1, R2, R3], fsum, domain)

    membership = system((.7, np.pi/2))

    x_centroid = centroid(membership, domain)

    import matplotlib.pyplot as plt
    plt.plot(domain, membership)
    plt.fill(domain, membership, alpha=0.5)
    plt.vlines(x_centroid, 0, np.max(membership), 'k', '--')
    plt.show()
