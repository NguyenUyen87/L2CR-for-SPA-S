"""Experiment 4 (Figure "size versus stability"): epsilon-constraint on the coalition count.

    python experiments/tradeoff.py 1000 5 30
    python experiments/tradeoff.py 2000 5 60

rho = 0.8, seeds 0..COUNT-1. For each cap in {coal(ABM+HUN), midpoint, coal(HUN),
2*coal(HUN), no cap} it runs L2CR and ABM+L2CR. Appends to results/tradeoff.jsonl.
"""
import sys

from _common import RESULTS, append, done_keys, load_weights
from spa_lq.core import generate, Episode
from spa_lq import methods as M
from spa_lq.baselines import run_hungarian

OUT = RESULTS / "tradeoff.jsonl"

if __name__ == "__main__":
    n, cnt, T = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    w = load_weights("rl")
    done = done_keys(OUT, lambda r: (r["n"], r["seed"]))
    for s in range(cnt):
        if (n, s) in done:
            continue
        inst = generate(n, seed=s, lq_ratio=0.8, zipf=1.0)
        h = run_hungarian(inst); ch = M.coal(h)
        a = M.run_abm_hun(inst); ca = M.coal(a)
        rl = M.finish(Episode(inst), M.policy_choice(w))
        r = {"n": n, "seed": s, "HUN": [h.matched(), ch], "ABM+HUN": [a.matched(), ca],
             "RL": [rl.matched(), M.coal(rl)], "L2CR": {}, "ABM+L2CR": {}}
        pre = M.abm_must_close(inst)
        for cap in sorted({ca, (ca + ch) // 2, ch, 2 * ch, 10 ** 6}):
            e = M.run_sacrifice(inst, T, "rl", w, max_coal=cap)
            r["L2CR"][str(cap)] = [e.matched(), M.coal(e)]
            e = M.run_sacrifice(inst, T, "rl", w, max_coal=cap, pre_closed=pre)
            r["ABM+L2CR"][str(cap)] = [e.matched(), M.coal(e)]
        append(OUT, r)
        print(r, flush=True)
