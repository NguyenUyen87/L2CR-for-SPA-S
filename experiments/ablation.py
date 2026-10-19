"""Experiment 5 (Table "feature-group ablation").

    python experiments/ablation.py            # writes results/ablation.txt

Retrains RL with feature groups removed (same training set/budget as train_rl.py,
3 training seeds per variant) and evaluates against HUN on 30 validation instances
(n=500, mixed rho and z, seeds 30000+).
"""
import random
import statistics as st

from _common import RESULTS
from spa_lq.core import generate
from spa_lq import policy as P
from spa_lq.baselines import run_hungarian
from spa_lq.stats import wilcoxon

GROUPS = {"quota/load": [1, 2, 3, 6, 11], "popularity": [4, 5], "cascade": [7, 8, 9, 10]}
VARIANTS = {"full": [], "-popularity": ["popularity"], "-cascade": ["cascade"],
            "quota/load only": ["popularity", "cascade"]}

if __name__ == "__main__":
    orig = P.features
    train = [generate(200, seed=10000 + i, lq_ratio=0.6, zipf=1.0) for i in range(40)]
    rnd = random.Random(7)
    val = [generate(500, seed=30000 + i, lq_ratio=rnd.choice([0.6, 0.8]), zipf=rnd.choice([0.5, 1.0, 1.5]))
           for i in range(30)]
    H = [run_hungarian(i).matched() for i in val]
    lines = []
    for name, drop in VARIANTS.items():
        mask = [1.0] * P.N_FEAT
        for g in drop:
            for j in GROUPS[g]:
                mask[j] = 0.0
        P.features = lambda e, p, m=mask: [f * mj for f, mj in zip(orig(e, p), m)]
        rels, pooled = [], []
        for ts in range(3):
            w = P.train(train, iters=120, seed=ts, log=lambda *a: None)
            if mask[1] == 0.0:
                w[1] = 0.0
            R = [P.run_policy(i, w).matched() for i in val]
            rels.append(st.mean(100 * (r - h) / h for r, h in zip(R, H)))
            pooled += R
        p = wilcoxon(pooled, H * 3)
        line = (f"{name:16s} RL vs HUN on validation: mean {st.mean(rels):+.2f}% (per training seed: "
                f"{', '.join(f'{x:+.2f}' for x in rels)})  pooled p={p:.4f}")
        print(line, flush=True)
        lines.append(line)
    P.features = orig
    open(RESULTS / "ablation.txt", "w").write("\n".join(lines) + "\n")
