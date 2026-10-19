# L2CR: Learning to Close and Rescue

Sacrifice search for Student-Project Allocation with lower quotas.

Code and data for the paper

> *Learning to close and rescue: Sacrifice search for Student-Project Allocation with lower quotas* (submitted to the Asia-Pacific Journal of Operational Research).

**L2CR** (Learning to Close and Rescue) combines a learned rule that decides which deficient projects to *close* with *sacrifice* moves that close projects meeting their lower quota so that their students *rescue* other projects.

The problem (**MAX-PS-SPA-S-LQ**): students rank projects, lecturers rank students, projects have upper **and lower** quotas, and a project that does not reach its lower quota is closed. We look for a maximum matching that is stable for the open projects (pairwise stable), and report the number of blocking coalitions formed by closed projects as a second objective.

## Contents

```
spa_lq/                 library (pure Python, no dependencies)
  core.py               instance generator, SPA-student oracle with warm start (Alg. 2),
                        blocking-pair checker, coalition counter, closing framework (Alg. 1)
  baselines.py          Hungarian heuristic of Biro et al. (2010) and the exact IP model (HiGHS)
  methods.py            ABM preprocessing (Alg. 3), HUN-16, HUN-MS, CASCADE, CEM,
                        sacrifice search / L2CR (Alg. 5)
  policy.py             features, linear softmax closing rule, REINFORCE training (Alg. 4)
  stats.py              Wilcoxon signed-rank test, Holm correction
weights/                trained closing rules used in the paper (rl.json, cem.json)
experiments/            scripts that produce every table and figure
results/                raw results reported in the paper (JSON lines)
tests/                  correctness checks
```

## Installation

```bash
git clone <this repository>
cd spa-lq-sacrifice-search
pip install -r requirements.txt     # only needed for the IP model
```

Python 3.9 or later. The results in the paper were produced with Python 3.14 and HiGHS 1.15.1 on an Apple M1.

## Quick start

```python
from spa_lq.core import generate
from spa_lq.baselines import run_hungarian
from spa_lq import methods as M
import json

inst = generate(1000, seed=0, lq_ratio=0.8)          # 1000 students, l_j = round(0.8 c_j)
w = json.load(open("weights/rl.json"))

hun = run_hungarian(inst)                              # practical rule
abm = M.run_abm_hun(inst)                              # ABM preprocessing + HUN
l2cr = M.run_sacrifice(inst, 30, "rl", w, max_coal=M.coal(hun))                  # L2CR, 30 s
pre = M.abm_must_close(inst)
abm_l2cr = M.run_sacrifice(inst, 30, "rl", w, max_coal=M.coal(abm), pre_closed=pre)  # ABM+L2CR

for name, e in [("HUN", hun), ("ABM+HUN", abm), ("L2CR", l2cr), ("ABM+L2CR", abm_l2cr)]:
    print(name, e.result())   # matched students, coalitions, open projects, blocking pairs (always 0)
```

## Reproducing the paper

Run all commands from the repository root. Scripts append to `results/` and can be interrupted and restarted.

| Paper | Command | Approx. time |
|---|---|---|
| Correctness checks | `python tests/test_core.py` | 2 min |
| Learned rules (`weights/`) | `python experiments/train_rl.py`<br>`python experiments/train_cem.py` | 10–25 s each |
| Exp. 1: small instances vs. IP | `python experiments/small.py` | 2–3 h |
| Exp. 2–3: constructive and budgeted methods | `python experiments/large.py 1000 10 30`<br>`python experiments/large.py 2000 10 60`<br>`python experiments/large.py 5000 10 120`<br>`python experiments/large.py 10000 5 240` | 1–5 h per size |
| Exp. 4: size vs. stability | `python experiments/tradeoff.py 1000 5 30`<br>`python experiments/tradeoff.py 2000 5 60` | 1 h |
| Exp. 5: feature ablation | `python experiments/ablation.py` | 15 min |
| All tables and figure data | `python experiments/make_tables.py [OUTPUT_DIR]` | seconds |

`make_tables.py` reads `results/` and writes the LaTeX tables and pgfplots data (default `tables/`). With the files shipped in `results/` it reproduces the tables of the paper exactly. Retraining reproduces `weights/rl.json` and `weights/cem.json` exactly.

To use several cores for one size, start a second worker in reverse order, e.g. `python experiments/large.py 5000 10 120 rev`; instances are claimed through lock files in `results/.claims/`. Note that budgeted methods depend on wall-clock time, so results vary slightly with the machine and its load.

## Methods

| Name | Description | Source |
|---|---|---|
| HUN | close the deficient project with minimum \|M(p)\|/l_p | Biró, Fleiner, Irving, Manlove, TCS 2010 |
| HUN-16 | HUN plus 15 runs with random tie-breaking, best of 16 | |
| ABM+HUN | Lemma 6/7 preprocessing, then HUN | Ágoston, Biró, McBride, JOCO 2016 |
| CASCADE | hand-crafted linear closing rule | |
| CEM / RL | linear closing rule learned by the cross-entropy method / REINFORCE | |
| ABM+RL, RL-16 | RL after ABM preprocessing; best of 16 RL trajectories | |
| HUN-MS | random-restart HUN within the time budget | |
| SAC-HUN | sacrifice search completed by HUN | |
| L2CR | Learning to Close and Rescue: sacrifice search completed by RL | this paper |
| ABM+L2CR | ABM preprocessing, then L2CR | this paper |
| IP | exact model (Olaosebikan 2020; Ágoston et al. 2018; Arulselvan et al. 2018) | |

## Result files

* `results/small.jsonl`: one line per instance (`n`, `lq` = rho, `seed`, IP `opt`/`status`/`bound`/`ip_time`, and for every method its number of matched students, `<method>_coal`, `<method>_t`).
* `results/large.jsonl`: one line per instance (`n`, `lq`, `seed`, `budget`, and per method matched students, `_coal` (SPA coalition count), `_coalhr` (HR count), `_open`, `_t` in seconds).
* `results/tradeoff.jsonl`: per instance, `[matched, coalitions]` for HUN, ABM+HUN, RL and, per coalition cap, for L2CR and ABM+L2CR.
* `results/ablation.txt`: feature-group ablation.

Instances are generated on the fly from `(n, seed, lq_ratio, zipf)` with `spa_lq.core.generate`, so no instance files are needed.

## Citation

Please cite the paper if you use this code (reference to be added after publication).
