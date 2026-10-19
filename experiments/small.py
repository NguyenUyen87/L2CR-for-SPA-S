"""Experiment 1 (Table "small instances"): heuristics vs. the exact IP optimum.

    python experiments/small.py               # appends to results/small.jsonl (resumable)

n in {40, 60, 80}, rho in {0.6, 0.8}, seeds 0-14. The IP is solved with HiGHS
(time limit 120 s, one thread); budgeted heuristics use 5 s.
"""
import time

from _common import RESULTS, append, done_keys, load_weights
from spa_lq.core import generate, Episode
from spa_lq import methods as M
from spa_lq import policy as P
from spa_lq.baselines import run_hungarian, solve_ip

OUT = RESULTS / "small.jsonl"

if __name__ == "__main__":
    w, w_cem = load_weights("rl"), load_weights("cem")
    done = done_keys(OUT, lambda r: (r["n"], r["lq"], r["seed"]))
    for lq in (0.6, 0.8):
        for n in (40, 60, 80):
            for s in range(15):
                if (n, lq, s) in done:
                    continue
                inst = generate(n, seed=s, lq_ratio=lq, zipf=1.0)
                ip = solve_ip(inst, 120)
                r = {"n": n, "lq": lq, "seed": s, "opt": ip["obj"], "status": ip["status"],
                     "bound": ip["bound"], "ip_time": ip["time"]}

                def rec(name, f):
                    t = time.perf_counter(); e = f(); r[name + "_t"] = time.perf_counter() - t
                    res = e.result(); assert res["bp"] == 0
                    r[name] = res["matched"]; r[name + "_coal"] = res["coalitions"]

                h = run_hungarian(inst); cap = M.coal(h)
                rec("HUN", lambda: h)
                rec("HUN-16", lambda: M.run_hun16(inst))
                rec("ABM+HUN", lambda: M.run_abm_hun(inst))
                rec("CASCADE", lambda: M.run_cascade(inst))
                rec("CEM", lambda: M.finish(Episode(inst), M.policy_choice(w_cem)))
                rec("RL", lambda: M.finish(Episode(inst), M.policy_choice(w)))
                rec("RL-16", lambda: P.run_policy_best_of(inst, w, 16))
                rec("HUN-MS", lambda: M.run_hun_ms(inst, 5))
                rec("SAC-HUN", lambda: M.run_sacrifice(inst, 5, "hun", max_coal=cap))
                rec("L2CR", lambda: M.run_sacrifice(inst, 5, "rl", w, max_coal=cap))
                rec("L2CR-free", lambda: M.run_sacrifice(inst, 5, "rl", w, max_coal=10 ** 9))
                append(OUT, r)
                print(n, lq, s, "opt", r["opt"], {k: r[k] for k in ("HUN", "ABM+HUN", "L2CR")}, flush=True)
