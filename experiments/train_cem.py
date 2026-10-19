"""Train the same linear closing rule with the cross-entropy method (baseline CEM).

    python experiments/train_cem.py           # writes weights/cem.json

Same training set as train_rl.py; 40 iterations x 20 candidates x 6 instances = 4,800 episodes.
"""
import json
import time

from _common import WEIGHTS
from spa_lq.core import generate
from spa_lq import methods as M

if __name__ == "__main__":
    train = [generate(200, seed=10000 + i, lq_ratio=0.6, zipf=1.0) for i in range(40)]
    t = time.time()
    w = M.train_cem(train, iters=40, pop=20, elite=4, batch=6, seed=0)
    print(f"training time {time.time() - t:.0f}s, weights {[round(x, 2) for x in w]}")
    json.dump(w, open(WEIGHTS / "cem.json", "w"))
