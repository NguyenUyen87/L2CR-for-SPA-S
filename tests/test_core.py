"""Correctness checks (run with `python -m pytest tests` or `python tests/test_core.py`)."""
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spa_lq.core import Instance, generate, solve, SPAState, Episode, blocking_pairs, count_coalitions
from spa_lq import methods as M
from spa_lq.baselines import run_hungarian
from spa_lq import policy as P


def count_coalitions_reference(inst, is_open, asg, cl):
    """Plain implementation of the coalition definition (paper, Def. 3.4)."""
    pos = [dict((p, i) for i, p in enumerate(lst)) for lst in inst.pref]
    by_lec = [[] for _ in range(inst.k)]
    for s, p in enumerate(asg):
        if p != -1:
            by_lec[inst.lec[p]].append(s)
    cnt = 0
    for p in range(inst.m):
        if is_open[p]:
            continue
        need, l = inst.lq[p], inst.lec[p]
        cand = sorted(inst.rank[l][s] for s in range(inst.n)
                      if p in pos[s] and (asg[s] == -1 or pos[s][p] < pos[s][asg[s]]))
        if len(cand) < need:
            continue
        C = cand[:need]
        inside = sum(1 for s in by_lec[l] if inst.rank[l][s] in set(C))
        below = sum(1 for s in by_lec[l] if inst.rank[l][s] > C[-1])
        if inst.ul[l] - cl[l] + inside + below >= need:
            cnt += 1
    return cnt


def test_oracle_is_stable():
    for seed in range(30):
        inst = generate(150, seed=seed, lq_ratio=0.6)
        rnd = random.Random(seed)
        is_open = [rnd.random() < 0.8 for _ in range(inst.m)]
        asg, cp, cl = solve(inst, is_open)
        assert all(a == -1 or is_open[a] for a in asg)
        assert all(cp[p] <= inst.up[p] for p in range(inst.m))
        assert all(cl[l] <= inst.ul[l] for l in range(inst.k))
        assert blocking_pairs(inst, is_open, asg) == 0


def test_warm_start_equals_recomputation():
    """Proposition 4.1 / Algorithm 2: warm-started closures == from-scratch SPA-student."""
    for seed in range(60):
        rnd = random.Random(seed)
        inst = generate(rnd.choice([30, 60, 120]), seed=seed, lq_ratio=rnd.choice([0.4, 0.6, 0.8]),
                        zipf=rnd.choice([0.5, 1.0, 1.5]))
        st = SPAState(inst)
        order = list(range(inst.m)); rnd.shuffle(order)
        for p in order[: inst.m // 2]:
            st.close(p)
            asg, cp, cl = solve(inst, st.is_open)
            assert (asg, cp, cl) == (st.asg, st.cp, st.cl)
            assert st.matched == sum(a != -1 for a in asg)


def test_fast_coalition_counter():
    for seed in range(30):
        inst = generate(random.Random(seed).choice([100, 300]), seed=seed, lq_ratio=0.8)
        e = run_hungarian(inst)
        assert count_coalitions(inst, e.is_open, e.asg, e.cl) == count_coalitions_reference(inst, e.is_open, e.asg, e.cl)


def test_hungarian_autoclose_equivalence():
    """Closing empty projects at once leaves the output of HUN unchanged."""
    for seed in range(50):
        inst = generate(random.Random(seed).choice([100, 300]), seed=seed, lq_ratio=0.6)
        a = run_hungarian(inst)
        b = M.finish(Episode(inst), M.hun_choice)
        assert a.asg == b.asg


def test_all_methods_return_pairwise_stable_matchings():
    w = json.load(open(ROOT / "weights" / "rl.json"))
    inst = generate(300, seed=3, lq_ratio=0.8)
    cap = M.coal(run_hungarian(inst))
    for e in (run_hungarian(inst), M.run_abm_hun(inst), M.run_cascade(inst), M.run_hun16(inst),
              M.finish(Episode(inst), M.policy_choice(w)), P.run_policy_best_of(inst, w, 4),
              M.run_sacrifice(inst, 2, "rl", w, max_coal=cap), M.run_sacrifice(inst, 2, "hun", max_coal=cap)):
        assert e.done and e.result()["bp"] == 0


def _example_b():
    lec, up, lq, ul = [0, 1, 0, 1], [2, 3, 3, 3], [2, 3, 3, 2], [5, 6]
    pref = [[1, 3, 0], [2], [1, 3, 0], [0, 2, 1], [0, 1], [2, 3], [1, 2, 3]]
    l1, l2 = [2, 7, 6, 1, 3, 5, 4], [5, 3, 7, 4, 6, 1]
    rank = [[99] * 7, [99] * 7]
    for pos, s in enumerate(l1): rank[0][s - 1] = pos
    for pos, s in enumerate(l2): rank[1][s - 1] = pos
    return Instance(7, 4, 2, pref, lec, up, lq, ul, rank, [0] * 4, [0] * 4)


def test_sacrifice_example():
    """Instance I_b of the paper: only a sacrifice reaches 7 matched students."""
    inst = _example_b()
    w = json.load(open(ROOT / "weights" / "rl.json"))
    assert run_hungarian(inst).matched() == 5
    assert M.run_sacrifice(inst, 2, "hun", max_coal=10 ** 9).matched() == 7
    assert M.run_sacrifice(inst, 2, "rl", w, max_coal=10 ** 9).matched() == 7


def test_ip_matches_enumeration():
    try:
        import highspy  # noqa: F401
    except ImportError:
        return  # optional dependency
    from spa_lq.baselines import solve_ip
    for seed in range(5):
        inst = generate(27, seed=seed, lq_ratio=0.6)          # 9 projects
        best = 0
        for mask in range(1 << inst.m):
            op = [(mask >> j) & 1 == 1 for j in range(inst.m)]
            asg, cp, _ = solve(inst, op)
            if all(not op[j] or cp[j] >= inst.lq[j] for j in range(inst.m)):
                best = max(best, sum(a != -1 for a in asg))
        assert solve_ip(inst, 30)["obj"] == best


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
