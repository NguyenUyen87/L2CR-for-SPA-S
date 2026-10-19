"""Train the linear closing rule with REINFORCE (paper, Algorithm 4).

    python experiments/train_rl.py            # writes weights/rl.json

Training set: 40 instances, n=200, rho=0.6, z=1 (seeds 10000-10039).
"""
import json
import time

from _common import WEIGHTS
from spa_lq.core import generate
from spa_lq import policy as P

if __name__ == "__main__":
    train = [generate(200, seed=10000 + i, lq_ratio=0.6, zipf=1.0) for i in range(40)]
    t = time.time()
    w = P.train(train, iters=120, samples=6, batch=6, lr=0.1, seed=0)
    print(f"training time {time.time() - t:.0f}s, weights {[round(x, 2) for x in w]}")
    json.dump(w, open(WEIGHTS / "rl.json", "w"))
