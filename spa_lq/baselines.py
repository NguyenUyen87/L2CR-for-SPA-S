"""Baselines taken from the literature (implemented as described there).

1. HUNGARIAN: the heuristic used in the Hungarian higher-education scheme,
   Biro, Fleiner, Irving, Manlove, TCS 411 (2010), Section 2.4:
   "a generalised version of the Gale-Shapley algorithm is used several times.
   After each run, one college is closed and the Gale-Shapley algorithm is
   called again for the reduced instance until all the remaining colleges
   achieve their lower quotas. The college to be closed is selected from those
   colleges that have not reached their lower quota. Moreover, the chosen
   college is one for which the ratio of the number of assigned applicants to
   the lower quota is minimum."
   The generalised GS for SPA-S is SPA-student (Abraham, Irving, Manlove 2007).
   Ties in the ratio are broken by the smallest project index.

2. IP: exact model for MAX-PS (maximum size pairwise stable matching,
   Biro et al. 2010 Sec. 2.4) generalised to SPA-S:
   - matching + SPA stability constraints (4.5)-(4.16): Olaosebikan, PhD thesis,
     Univ. Glasgow 2020, Sec. 4.5 (super-stability == stability for strict
     preferences);
   - lower quota constraint (5): Agoston, Biro, Szanto, ORP 5 (2018);
   - open/closed project variables y_j (|M(p_j)| = 0 or l_j <= |M(p_j)| <= c_j):
     Arulselvan et al., Algorithmica 80 (2018), problem WMLQ;
   - blocking-pair constraints are switched off for closed projects by adding
     (1 - y_j) to the right-hand side (pairwise stability = condition (a) only).
"""
import time

from .core import Episode


def run_hungarian(inst):
    e = Episode(inst, auto_close=False)
    while not e.done:
        e.close(min(e.deficient, key=lambda p: (e.cp[p] / inst.lq[p], p)))
    return e


def solve_ip(inst, time_limit=60.0, threads=1):
    import highspy
    n, m, k = inst.n, inst.m, inst.k
    lec, c, d, lq, rank = inst.lec, inst.up, inst.ul, inst.lq, inst.rank
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("threads", threads)

    x = {}
    for i in range(n):
        for j in inst.pref[i]:
            x[i, j] = h.addBinary()
    y = [h.addBinary() for _ in range(m)]
    by_proj = [[] for _ in range(m)]
    for (i, j) in x:
        by_proj[j].append(i)
    P = [[] for _ in range(k)]
    for j in range(m):
        P[lec[j]].append(j)
    L = [sorted({i for j in P[kk] for i in by_proj[j]}, key=lambda i: rank[kk][i])
         for kk in range(k)]

    def S(js):
        return h.qsum(js) if js else 0

    load_p = [S([x[i, j] for i in by_proj[j]]) for j in range(m)]
    load_l = [S([x[i, j] for j in P[kk] for i in by_proj[j]]) for kk in range(k)]

    for i in range(n):                                   # (4.5)
        h.addConstr(S([x[i, j] for j in inst.pref[i]]) <= 1)
    for j in range(m):                                   # (4.6) + LQ (5) + open/closed
        h.addConstr(load_p[j] <= c[j] * y[j])
        h.addConstr(load_p[j] >= lq[j] * y[j])
    for kk in range(k):                                  # (4.7)
        h.addConstr(load_l[kk] <= d[kk])

    alpha = [h.addBinary() for _ in range(m)]
    gamma = [h.addBinary() for _ in range(m)]
    beta = [h.addBinary() for _ in range(k)]
    eta = [h.addBinary() for _ in range(k)]
    for j in range(m):
        h.addConstr(c[j] * alpha[j] + load_p[j] >= c[j])           # (4.8)
        h.addConstr(c[j] * gamma[j] - load_p[j] >= 1 - c[j])       # (4.14)
    for kk in range(k):
        h.addConstr(d[kk] * beta[kk] + load_l[kk] >= d[kk])        # (4.9)
        h.addConstr(d[kk] * eta[kk] - load_l[kk] >= 1 - d[kk])     # (4.11)

    # delta_{i,k} (4.12): students in L_k ordered by rank, D_{i,k} = prefix
    delta = {}
    for kk in range(k):
        prefix = []
        for i in L[kk]:
            dv = h.addBinary()
            delta[i, kk] = dv
            better = S([x[i2, j] for i2 in prefix for j in P[kk] if (i2, j) in x])
            h.addConstr(d[kk] * dv - load_l[kk] + better >= 0)
            prefix.append(i)

    for (i, j), xij in x.items():
        kk = lec[j]
        lst = inst.pref[i]
        theta = 1 - xij - S([x[i, j2] for j2 in lst[:lst.index(j)]])
        slack = 1 - y[j]
        h.addConstr(theta + alpha[j] + beta[kk] - slack <= 2)                    # (4.10)
        h.addConstr(theta + alpha[j] + eta[kk] + delta[i, kk] - slack <= 3)      # (4.13)
        lam = h.addBinary()
        better = S([x[i2, j] for i2 in by_proj[j] if rank[kk][i2] < rank[kk][i]])
        h.addConstr(c[j] * lam - load_p[j] + better >= 0)                         # (4.15)
        h.addConstr(theta + gamma[j] + lam - slack <= 2)                          # (4.16)

    t = time.time()
    h.maximize(S(list(x.values())))
    dt = time.time() - t
    info = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus())
    asg = [-1] * n
    ok = info.primal_solution_status == 2   # feasible solution available
    if ok:
        vals = h.vals(list(x.values()))
        for (key, v) in zip(x.keys(), vals):
            if v > 0.5:
                asg[key[0]] = key[1]
    return {
        "asg": asg if ok else None,
        "obj": sum(1 for a in asg if a != -1) if ok else None,
        "bound": info.mip_dual_bound,
        "status": status,
        "time": dt,
    }
