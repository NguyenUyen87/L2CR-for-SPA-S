"""Wilcoxon signed-rank test (two-sided, normal approximation with tie correction) and Holm correction."""
import math


def wilcoxon(a, b):
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    tie_term = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = r
        c = j - i + 1
        tie_term += c ** 3 - c
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48)
    if sigma == 0:
        return 1.0
    z = max(0.0, abs(w_plus - mu) - 0.5) / sigma
    return math.erfc(z / math.sqrt(2))


def holm(pvals):
    """Holm-adjusted p-values, in the order of the input."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    run = 0.0
    m = len(pvals)
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj
