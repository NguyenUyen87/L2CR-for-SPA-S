"""Closing heuristics, preprocessing and sacrifice search.

Constructive baselines
  HUN-16      Hungarian rule with random tie-breaking, best of 16 runs
  ABM+HUN     Lemma 6/7 preprocessing of Agoston, Biro, McBride (JOCO 2016), then HUN
  CASCADE     hand-crafted closing rule (fill, next-deficient, no-next features)
  CEM         linear closing policy learned with the cross-entropy method
Time-budgeted improvement (same budget T for all)
  HUN-MS(T)   random-restart HUN (random tie-breaking) until the budget is spent
  SAC-HUN(T)  sacrifice local search, solutions completed by HUN
  L2CR(T)     sacrifice local search started from and completed by the learned policy (ours)
"""
import math
import random
import time

from .core import Episode, count_coalitions, solve
from . import policy as P


# ------------------------------------------------------------------ helpers
def coal(e):
    return count_coalitions(e.inst, e.is_open, e.asg, e.cl)


def finish(e, chooser):
    while not e.done:
        e.close(chooser(e))
    return e


def hun_choice(e):
    return min(e.deficient, key=lambda p: P.hungarian_key(e, p))


def hun_random_choice(rnd):
    def ch(e):
        ratios = [(e.cp[p] / e.inst.lq[p], p) for p in e.deficient]
        mn = min(r for r, _ in ratios)
        return rnd.choice([p for r, p in ratios if r == mn])
    return ch


def policy_choice(w):
    def ch(e):
        P._members(e)
        best, bs = None, None
        for p in e.deficient:
            sc = sum(a * b for a, b in zip(w, P.features(e, p)))
            if bs is None or sc > bs:
                best, bs = p, sc
        return best
    return ch


# ------------------------------------------------------------------ constructive baselines
def run_hun16(inst, seed=0, runs=16):
    rnd = random.Random(seed)
    best = None
    for r in range(runs):
        ch = hun_choice if r == 0 else hun_random_choice(rnd)
        e = finish(Episode(inst), ch)
        if best is None or e.matched() > best.matched():
            best = e
    return best


CASCADE_W = [0.0] * P.N_FEAT
CASCADE_W[1], CASCADE_W[7], CASCADE_W[9] = -1.0, 1.0, -1.0


def run_cascade(inst):
    return finish(Episode(inst), policy_choice(CASCADE_W))


def abm_must_close(inst):
    """Iterative Lemma 6/7 of Agoston, Biro, McBride (2016): X = projects that reach
    their lower quota with all non-closed projects open (they must stay open in
    every stable solution); a project c outside X that does not reach its lower
    quota even when all other projects outside X are closed must be closed."""
    closed = set()
    while True:
        is_open = [p not in closed for p in range(inst.m)]
        _, cp, _ = solve(inst, is_open)
        X = {p for p in range(inst.m) if is_open[p] and cp[p] >= inst.lq[p]}
        rest = [p for p in range(inst.m) if is_open[p] and p not in X]
        newly = set()
        for c in rest:
            op = [p in X or p == c for p in range(inst.m)]
            _, cp2, _ = solve(inst, op)
            if cp2[c] < inst.lq[c]:
                newly.add(c)
        if not newly:
            return closed
        closed |= newly


def run_abm_hun(inst):
    e = Episode(inst)
    pre = abm_must_close(inst)
    if pre:
        e.close(sorted(pre))
    return finish(e, hun_choice)


def train_cem(instances, iters=15, pop=20, elite=4, batch=6, seed=0, log=print):
    """Cross-entropy method over the linear policy (greedy roll-outs, mean matched ratio)."""
    rnd = random.Random(seed)
    mu = [0.0] * P.N_FEAT
    mu[1] = -2.0
    sd = [1.5] * P.N_FEAT
    for it in range(1, iters + 1):
        batch_inst = rnd.sample(instances, batch)
        cands = []
        for _ in range(pop):
            th = [m + s * rnd.gauss(0, 1) for m, s in zip(mu, sd)]
            sc = sum(finish(Episode(i), policy_choice(th)).matched() / i.n for i in batch_inst) / batch
            cands.append((sc, th))
        cands.sort(key=lambda t: -t[0])
        el = [th for _, th in cands[:elite]]
        mu = [sum(t[j] for t in el) / elite for j in range(P.N_FEAT)]
        sd = [max(0.05, math.sqrt(sum((t[j] - mu[j]) ** 2 for t in el) / elite)) for j in range(P.N_FEAT)]
        if it % 5 == 0:
            log(f"CEM iter {it}: best {cands[0][0]:.4f}")
    return mu


# ------------------------------------------------------------------ time-budgeted improvement
def run_hun_ms(inst, budget, seed=0):
    t0 = time.perf_counter()
    rnd = random.Random(seed)
    best = finish(Episode(inst), hun_choice)
    while time.perf_counter() - t0 < budget:
        e = finish(Episode(inst), hun_random_choice(rnd))
        if e.matched() > best.matched():
            best = e
    return best


def sacrifice_candidates(e):
    """Estimate the gain of closing an open (non-deficient) project p in the final
    solution e: students of p whose next project q was closed while partially
    filled can rescue q if they cover its deficit at closing time."""
    inst = e.inst
    members = [[] for _ in range(inst.m)]
    for s, a in enumerate(e.asg):
        if a != -1:
            members[a].append(s)
    out = []
    for p in range(inst.m):
        if not e.is_open[p]:
            continue
        flow = {}
        lost = 0
        for s in members[p]:
            lst = inst.pref[s]
            nxt = lst[lst.index(p) + 1:] if p in lst else []
            q = nxt[0] if nxt else None
            if q is None:
                lost += 1
            elif e.is_open[q]:
                if e.cp[q] >= inst.up[q]:
                    lost += 1          # will displace someone; count as a loss
            else:
                flow[q] = flow.get(q, 0) + 1
        rescued = 0
        for q, f in flow.items():
            c = e.closed_cnt.get(q, 0)
            if c + f >= inst.lq[q]:     # q (possibly empty when closed) would reach its lower quota
                rescued += c
            else:
                lost += f
        out.append((rescued - lost, p))
    out.sort(key=lambda t: (-t[0], t[1]))
    return [p for _, p in out]


def run_sacrifice(inst, budget, base="rl", w=None, max_coal=None, start=None, pre_closed=None):
    """Sacrifice local search: S = set of projects closed although they meet their
    lower quota; a solution is evaluated by closing S first and completing with
    the base policy. First-improvement over candidates ranked by
    sacrifice_candidates; accept if size increases and #coalitions <= max_coal."""
    t0 = time.perf_counter()
    ch = policy_choice(w) if base == "rl" else hun_choice
    # keep empty projects open in the pre-state: students released by a
    # sacrificed project may fill a project that was empty initially
    pre = Episode(inst, auto_close=False)
    if pre_closed:
        pre.close(sorted(pre_closed))

    def complete(state):
        f = state.copy()
        f.auto_close = True
        f._refresh()
        return finish(f, ch)

    cur = start if start is not None else complete(pre)
    if max_coal is None:
        max_coal = coal(cur)
    S = []
    tried = set()
    evals = 0
    while time.perf_counter() - t0 < budget:
        improved = False
        for p in sacrifice_candidates(cur):
            if time.perf_counter() - t0 >= budget:
                break
            key = (tuple(sorted(S)), p)
            if key in tried:
                continue
            tried.add(key)
            g = pre.copy()
            g.close(p)
            f = complete(g)
            evals += 1
            if f.matched() > cur.matched() and coal(f) <= max_coal:
                S.append(p)
                pre = pre.copy()
                pre.close(p)
                cur = f
                improved = True
                break
        if not improved:
            break
    cur.sac_evals = evals
    cur.sacrificed = len(S)
    return cur
