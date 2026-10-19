"""Experiments 2 and 3 (Tables "constructive rules" and "time-budgeted methods").

    python experiments/large.py N COUNT BUDGET [rev]

e.g. the paper uses
    python experiments/large.py 1000 10 30
    python experiments/large.py 2000 10 60
    python experiments/large.py 5000 10 120
    python experiments/large.py 10000 5 240

COUNT instances per rho in {0.6, 0.8} (seeds 0..COUNT-1). Results are appended to
results/large.jsonl. Several workers may run the same N in parallel (pass "rev" to
one of them): each instance is claimed through a lock file before it is solved.
All methods of an instance run sequentially in one process; budgeted methods share
the same wall-clock budget. Coalition caps: #coalitions of HUN for SAC-HUN and L2CR,
#coalitions of ABM+HUN for ABM+L2CR.
"""
import os
import sys
import time

from _common import RESULTS, append, done_keys, load_weights
from spa_lq.core import generate, Episode
from spa_lq import methods as M
from spa_lq import policy as P
from spa_lq.baselines import run_hungarian

OUT = RESULTS / "large.jsonl"
CLAIMS = RESULTS / ".claims"


def with_pre(inst, pre, chooser):
    e = Episode(inst)
    if pre:
        e.close(sorted(pre))
    return M.finish(e, chooser)


if __name__ == "__main__":
    n, cnt, T = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    w, w_cem = load_weights("rl"), load_weights("cem")
    CLAIMS.mkdir(exist_ok=True)
    done = done_keys(OUT, lambda r: (r["n"], r["lq"], r["seed"]))
    order = [(lq, s) for lq in (0.6, 0.8) for s in range(cnt)]
    if len(sys.argv) > 4 and sys.argv[4] == "rev":
        order.reverse()
    for lq, s in order:
        if (n, lq, s) in done:
            continue
        try:
            os.close(os.open(CLAIMS / f"{n}_{lq}_{s}", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        inst = generate(n, seed=s, lq_ratio=lq, zipf=1.0)
        r = {"n": n, "lq": lq, "seed": s, "budget": T}

        def rec(name, f, extra_time=0.0):
            t = time.perf_counter(); e = f(); dt = time.perf_counter() - t + extra_time
            res = e.result(); assert res["bp"] == 0
            r[name] = res["matched"]; r[name + "_t"] = dt
            r[name + "_coal"] = res["coalitions"]; r[name + "_coalhr"] = res["coalitions_hr"]
            r[name + "_open"] = res["open"]
            return e

        rec("HUN", lambda: run_hungarian(inst))
        cap = r["HUN_coal"]
        rec("HUN-16", lambda: M.run_hun16(inst))
        t0 = time.perf_counter(); pre = M.abm_must_close(inst); t_pre = time.perf_counter() - t0
        rec("ABM+HUN", lambda: with_pre(inst, pre, M.hun_choice), t_pre)
        rec("CASCADE", lambda: M.run_cascade(inst))
        rec("CEM", lambda: M.finish(Episode(inst), M.policy_choice(w_cem)))
        rec("RL", lambda: M.finish(Episode(inst), M.policy_choice(w)))
        rec("ABM+RL", lambda: with_pre(inst, pre, M.policy_choice(w)), t_pre)
        rec("RL-16", lambda: P.run_policy_best_of(inst, w, 16))
        rec("HUN-MS", lambda: M.run_hun_ms(inst, T))
        e = rec("SAC-HUN", lambda: M.run_sacrifice(inst, T, "hun", max_coal=cap))
        r["SAC-HUN_evals"] = e.sac_evals
        e = rec("L2CR", lambda: M.run_sacrifice(inst, T, "rl", w, max_coal=cap))
        r["L2CR_evals"] = e.sac_evals; r["L2CR_sacrificed"] = e.sacrificed
        e = rec("ABM+L2CR", lambda: M.run_sacrifice(inst, T, "rl", w, max_coal=r["ABM+HUN_coal"], pre_closed=pre), t_pre)
        r["ABM+L2CR_sacrificed"] = e.sacrificed
        append(OUT, r)
        print(n, lq, s, {k: r[k] for k in ("HUN", "ABM+HUN", "RL", "L2CR", "ABM+L2CR")}, flush=True)
