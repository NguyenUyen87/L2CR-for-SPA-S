"""Learned linear closing rule: features, softmax policy, REINFORCE training."""
import math
import random

from .core import Episode


def hungarian_key(e, p):
    """Biro et al. (2010) Sec. 2.4: minimum |M(p)| / l_p (ties: smallest index)."""
    return (e.cp[p] / e.inst.lq[p], p)



N_FEAT = 12


def features(e, p):
    inst = e.inst
    l = inst.lec[p]
    members = [s for s in e.asg_members[p]]
    nd = nf = nn = nfree = 0
    for s in members:
        lst = inst.pref[s]
        nxt = None
        for q in lst[lst.index(p) + 1:]:
            if e.is_open[q]:
                nxt = q
                break
        if nxt is None:
            nn += 1
        elif e.cp[nxt] < inst.lq[nxt]:
            nd += 1
        elif e.cp[nxt] >= inst.up[nxt]:
            nf += 1
        else:
            nfree += 1
    c = max(1, len(members))
    mean_app = inst.n * 4.5 / inst.m
    return [
        1.0,
        e.cp[p] / inst.up[p],
        (inst.lq[p] - e.cp[p]) / inst.lq[p],
        inst.lq[p] / inst.up[p],
        inst.applicants[p] / mean_app,
        inst.first_choice[p] / (inst.n / inst.m),
        e.cl[l] / inst.ul[l],
        nd / c,
        nf / c,
        nn / c,
        nfree / c,
        e.cp[p] / 6.0,
    ]


def _members(e):
    mem = [[] for _ in range(e.inst.m)]
    for s, a in enumerate(e.asg):
        if a != -1:
            mem[a].append(s)
    e.asg_members = mem


def _probs(w, feats):
    scores = [sum(wi * fi for wi, fi in zip(w, f)) for f in feats]
    mx = max(scores)
    ex = [math.exp(x - mx) for x in scores]
    z = sum(ex)
    return [x / z for x in ex]


def run_policy(inst, w, sample=False, rnd=None, record=False):
    e = Episode(inst)
    traj = []
    while not e.done:
        _members(e)
        cands = e.deficient
        feats = [features(e, p) for p in cands]
        pr = _probs(w, feats)
        if sample:
            x, acc, i = rnd.random(), 0.0, 0
            for i, pi in enumerate(pr):
                acc += pi
                if acc >= x:
                    break
        else:
            i = max(range(len(pr)), key=pr.__getitem__)
        if record:
            traj.append((feats, pr, i))
        e.close(cands[i])
    return (e, traj) if record else e


def run_policy_best_of(inst, w, k=16, seed=0):
    rnd = random.Random(seed)
    best = run_policy(inst, w)
    calls = best.calls
    for _ in range(k - 1):
        e = run_policy(inst, w, sample=True, rnd=rnd)
        calls += e.calls
        if e.matched() > best.matched():
            best = e
    best.calls = calls
    return best


def train(instances, iters=120, samples=6, batch=6, lr=0.1, seed=0, log=print):
    """REINFORCE with shared per-instance baseline (POMO-style) + Adam."""
    rnd = random.Random(seed)
    w = [0.0] * N_FEAT
    w[1] = -2.0  # warm start ~ greedy min-count
    m_, v_ = [0.0] * N_FEAT, [0.0] * N_FEAT
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(1, iters + 1):
        grad = [0.0] * N_FEAT
        tot_r = 0.0
        for inst in rnd.sample(instances, batch):
            runs = [run_policy(inst, w, sample=True, rnd=rnd, record=True)
                    for _ in range(samples)]
            rs = [e.matched() / inst.n for e, _ in runs]
            base = sum(rs) / len(rs)
            tot_r += base
            for (e, traj), r in zip(runs, rs):
                adv = r - base
                if adv == 0:
                    continue
                for feats, pr, i in traj:
                    for j in range(N_FEAT):
                        g = feats[i][j] - sum(pk * f[j] for pk, f in zip(pr, feats))
                        grad[j] += adv * g
        for j in range(N_FEAT):  # gradient ascent
            g = -grad[j] / (batch * samples)
            m_[j] = b1 * m_[j] + (1 - b1) * g
            v_[j] = b2 * v_[j] + (1 - b2) * g * g
            mh = m_[j] / (1 - b1 ** it)
            vh = v_[j] / (1 - b2 ** it)
            w[j] -= lr * mh / (math.sqrt(vh) + eps)
        if it % 10 == 0:
            log(f"iter {it:4d}  mean matched ratio {tot_r / batch:.4f}")
    return w
