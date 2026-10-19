"""SPA-S with project lower quotas and closable projects (SPA-S-LQ): core.

Core pieces:
  - Instance generator (skewed popularity, correlated lecturer preferences)
  - Oracle: SPA-student algorithm (Abraham, Irving, Manlove 2007) on the set of
    open projects -> student-optimal stable matching, O(L log L)
  - Independent stability checker (blocking pairs, SPA-S definition)
  - Blocking-coalition counter for closed projects (simplified adaptation of
    Biro, Fleiner, Irving, Manlove 2010; see count_coalitions docstring)
  - Episode: state = set of open projects; action = close one deficient project
"""
import heapq
import random
from dataclasses import dataclass


@dataclass
class Instance:
    n: int                 # students
    m: int                 # projects
    k: int                 # lecturers
    pref: list             # pref[s] = list of projects, most preferred first
    lec: list              # lec[p] = lecturer offering p
    up: list               # upper quota of project
    lq: list               # lower quota of project
    ul: list               # upper quota (capacity) of lecturer
    rank: list             # rank[l][s] = position of s in l's list (0 = best)
    applicants: list       # applicants[p] = number of students listing p
    first_choice: list     # first_choice[p] = number of students ranking p first


def generate(n, proj_per_student=1 / 3, proj_per_lec=3, list_len=(3, 6),
             zipf=1.0, lq_ratio=0.6, lec_cap_ratio=0.8, seed=0):
    rnd = random.Random(seed)
    m = max(2, int(n * proj_per_student))
    k = max(1, m // proj_per_lec)
    lec = [p % k for p in range(m)]
    rnd.shuffle(lec)
    up = [rnd.randint(2, 6) for _ in range(m)]
    lq = [max(1, round(lq_ratio * u)) for u in up]
    ul = []
    for l in range(k):
        ps = [p for p in range(m) if lec[p] == l]
        tot = sum(up[p] for p in ps)
        ul.append(max([up[p] for p in ps] + [round(lec_cap_ratio * tot)]) if ps else 1)
    w = [1.0 / (i + 1) ** zipf for i in range(m)]
    rnd.shuffle(w)
    pref = []
    for _ in range(n):
        L = rnd.randint(*list_len)
        chosen, ws, items = [], w[:], list(range(m))
        for _ in range(min(L, m)):
            x = rnd.random() * sum(ws)
            acc = 0.0
            for idx, wi in enumerate(ws):
                acc += wi
                if acc >= x:
                    break
            chosen.append(items.pop(idx))
            ws.pop(idx)
        pref.append(chosen)
    score = [rnd.gauss(0, 1) for _ in range(n)]
    rank = []
    for _ in range(k):
        order = sorted(range(n), key=lambda s: -(score[s] + 0.5 * rnd.gauss(0, 1)))
        r = [0] * n
        for pos, s in enumerate(order):
            r[s] = pos
        rank.append(r)
    applicants = [0] * m
    first_choice = [0] * m
    for lst in pref:
        for p in lst:
            applicants[p] += 1
        if lst:
            first_choice[lst[0]] += 1
    return Instance(n, m, k, pref, lec, up, lq, ul, rank, applicants, first_choice)


def solve(inst, is_open):
    """SPA-student on open projects. Deletions are encoded as rank cutoffs
    (cutp/cutl only decrease), so each student's pointer only moves forward."""
    n, m, k = inst.n, inst.m, inst.k
    lec, up, ul, rank, pref = inst.lec, inst.up, inst.ul, inst.rank, inst.pref
    INF = n + 1
    ptr = [0] * n
    asg = [-1] * n
    cp = [0] * m
    cl = [0] * k
    hp = [[] for _ in range(m)]
    hl = [[] for _ in range(k)]
    cutp = [INF] * m
    cutl = [INF] * k

    def worst_p(p):
        h = hp[p]
        while asg[h[0][1]] != p:
            heapq.heappop(h)
        return h[0]

    def worst_l(l):
        h = hl[l]
        while True:
            a = asg[h[0][1]]
            if a != -1 and lec[a] == l:
                return h[0]
            heapq.heappop(h)

    stack = list(range(n - 1, -1, -1))
    while stack:
        s = stack.pop()
        lst = pref[s]
        i = ptr[s]
        while i < len(lst):
            p = lst[i]
            l = lec[p]
            r = rank[l][s]
            if is_open[p] and r <= cutp[p] and r <= cutl[l]:
                break
            i += 1
        ptr[s] = i
        if i == len(lst):
            continue
        asg[s] = p
        cp[p] += 1
        cl[l] += 1
        heapq.heappush(hp[p], (-r, s))
        heapq.heappush(hl[l], (-r, s))
        if cp[p] > up[p]:
            _, sr = worst_p(p)
            heapq.heappop(hp[p])
            asg[sr] = -1
            cp[p] -= 1
            cl[l] -= 1
            stack.append(sr)
        elif cl[l] > ul[l]:
            _, sr = worst_l(l)
            heapq.heappop(hl[l])
            pt = asg[sr]
            asg[sr] = -1
            cp[pt] -= 1
            cl[l] -= 1
            stack.append(sr)
        if cp[p] == up[p]:
            cutp[p] = min(cutp[p], -worst_p(p)[0])
        if cl[l] == ul[l]:
            cutl[l] = min(cutl[l], -worst_l(l)[0])
    return asg, cp, cl


def _worst_ranks(inst, asg):
    wp = [-1] * inst.m
    wl = [-1] * inst.k
    for s, p in enumerate(asg):
        if p != -1:
            l = inst.lec[p]
            r = inst.rank[l][s]
            wp[p] = max(wp[p], r)
            wl[l] = max(wl[l], r)
    return wp, wl


def blocking_pairs(inst, is_open, asg):
    """Independent checker: number of SPA-S blocking pairs among open projects."""
    cp = [0] * inst.m
    cl = [0] * inst.k
    for p in asg:
        if p != -1:
            cp[p] += 1
            cl[inst.lec[p]] += 1
    wp, wl = _worst_ranks(inst, asg)
    bad = 0
    for s in range(inst.n):
        for p in inst.pref[s]:
            if p == asg[s]:
                break
            if not is_open[p]:
                continue
            l = inst.lec[p]
            r = inst.rank[l][s]
            if cp[p] > inst.up[p] or cl[l] > inst.ul[l]:
                return 10 ** 9  # capacity violated
            if cp[p] < inst.up[p] and cl[l] < inst.ul[l]:
                bad += 1
            elif cp[p] < inst.up[p]:
                if (asg[s] != -1 and inst.lec[asg[s]] == l) or r < wl[l]:
                    bad += 1
            elif r < wp[p]:
                bad += 1
    return bad


def _apps(inst):
    if not hasattr(inst, "_apps"):
        inst._apps = [[] for _ in range(inst.m)]
        inst._pos = []
        for s, lst in enumerate(inst.pref):
            inst._pos.append({p: i for i, p in enumerate(lst)})
            for p in lst:
                inst._apps[p].append(s)
    return inst._apps, inst._pos


def count_coalitions(inst, is_open, asg, cl, hr=False):
    """Number of closed projects forming a blocking coalition (paper, Def. 3.4).
    hr=True drops condition (b) on lecturer capacity (the "HR count")."""
    apps, pos = _apps(inst)
    by_lec_ranks = None
    cnt = 0
    for p in range(inst.m):
        if is_open[p]:
            continue
        need = inst.lq[p]
        l = inst.lec[p]
        cand = []
        for s in apps[p]:
            a = asg[s]
            if a == -1 or pos[s][p] < pos[s][a]:
                cand.append(inst.rank[l][s])
        if len(cand) < need:
            continue
        if hr:
            cnt += 1
            continue
        if by_lec_ranks is None:
            by_lec_ranks = [[] for _ in range(inst.k)]
            for s, a in enumerate(asg):
                if a != -1:
                    by_lec_ranks[inst.lec[a]].append(inst.rank[inst.lec[a]][s])
        cand.sort()
        C = cand[:need]
        cmax = C[-1]
        cset = set(C)
        inside = sum(1 for r in by_lec_ranks[l] if r in cset)
        below = sum(1 for r in by_lec_ranks[l] if r > cmax)
        if inst.ul[l] - cl[l] + inside + below >= need:
            cnt += 1
    return cnt


class SPAState:
    """SPA-student with warm start. Closing p = p rejects all its students and
    the proposal sequence resumes from the current state (deletions are kept as
    rank cutoffs). Equality with a from-scratch run is checked in tests."""

    def __init__(self, inst, is_open=None, rebuilds=0):
        n, m, k = inst.n, inst.m, inst.k
        self.inst = inst
        self.rebuilds = rebuilds
        self.is_open = list(is_open) if is_open else [True] * m
        self.ptr = [0] * n
        self.asg = [-1] * n
        self.cp = [0] * m
        self.cl = [0] * k
        self.hp = [[] for _ in range(m)]
        self.hl = [[] for _ in range(k)]
        self.cutp = [n + 1] * m
        self.cutl = [n + 1] * k
        self.matched = 0
        self._run(list(range(n - 1, -1, -1)))

    def copy(self):
        c = SPAState.__new__(SPAState)
        c.inst = self.inst
        c.is_open = self.is_open[:]
        c.ptr = self.ptr[:]
        c.asg = self.asg[:]
        c.cp = self.cp[:]
        c.cl = self.cl[:]
        c.hp = [h[:] for h in self.hp]
        c.hl = [h[:] for h in self.hl]
        c.cutp = self.cutp[:]
        c.cutl = self.cutl[:]
        c.matched = self.matched
        c.rebuilds = self.rebuilds
        return c

    def _worst_p(self, p):
        h, asg = self.hp[p], self.asg
        while asg[h[0][1]] != p:
            heapq.heappop(h)
        return h[0]

    def _worst_l(self, l):
        h, asg, lec = self.hl[l], self.asg, self.inst.lec
        while True:
            a = asg[h[0][1]]
            if a != -1 and lec[a] == l:
                return h[0]
            heapq.heappop(h)

    def close(self, p):
        if not self.is_open[p]:
            return
        self.is_open[p] = False
        l = self.inst.lec[p]
        if self.cutl[l] <= self.inst.n:
            # l has been full: some students were rejected by l (not by a
            # project) and may now be accepted at another project of l, i.e.
            # they can move UP their lists -> resuming is unsound, re-solve.
            self.rebuilds += 1
            self.__init__(self.inst, self.is_open, self.rebuilds)
            return
        freed = {s for _, s in self.hp[p] if self.asg[s] == p}
        self.hp[p] = []
        for s in freed:
            self.asg[s] = -1
            self.cp[p] -= 1
            self.cl[l] -= 1
            self.matched -= 1
        self._run(sorted(freed, reverse=True))

    def _run(self, stack):
        inst = self.inst
        lec, up, ul, rank, pref = inst.lec, inst.up, inst.ul, inst.rank, inst.pref
        ptr, asg, cp, cl = self.ptr, self.asg, self.cp, self.cl
        hp, hl, cutp, cutl, is_open = self.hp, self.hl, self.cutp, self.cutl, self.is_open
        while stack:
            s = stack.pop()
            lst = pref[s]
            i = ptr[s]
            while i < len(lst):
                p = lst[i]
                l = lec[p]
                r = rank[l][s]
                if is_open[p] and r <= cutp[p] and r <= cutl[l]:
                    break
                i += 1
            ptr[s] = i
            if i == len(lst):
                continue
            asg[s] = p
            cp[p] += 1
            cl[l] += 1
            self.matched += 1
            heapq.heappush(hp[p], (-r, s))
            heapq.heappush(hl[l], (-r, s))
            if cp[p] > up[p]:
                _, sr = self._worst_p(p)
                heapq.heappop(hp[p])
                asg[sr] = -1
                cp[p] -= 1
                cl[l] -= 1
                self.matched -= 1
                stack.append(sr)
            elif cl[l] > ul[l]:
                _, sr = self._worst_l(l)
                heapq.heappop(hl[l])
                pt = asg[sr]
                asg[sr] = -1
                cp[pt] -= 1
                cl[l] -= 1
                self.matched -= 1
                stack.append(sr)
            if cp[p] == up[p]:
                cutp[p] = min(cutp[p], -self._worst_p(p)[0])
            if cl[l] == ul[l]:
                cutl[l] = min(cutl[l], -self._worst_l(l)[0])


class Episode:
    """State of the outer problem: which projects are open.
    auto_close: projects with no students are closed immediately (used by the
    RL/lookahead methods; the Hungarian heuristic closes them itself first,
    since their ratio |M(p)|/l_p = 0 is minimal)."""

    def __init__(self, inst, auto_close=True):
        self.inst = inst
        self.st = SPAState(inst)
        self.auto_close = auto_close
        self.calls = 0
        self.closed_cnt = {}
        self._refresh()

    def copy(self):
        e = Episode.__new__(Episode)
        e.inst, e.auto_close, e.calls = self.inst, self.auto_close, 0
        e.st = self.st.copy()
        e.deficient = self.deficient[:]
        e.closed_cnt = dict(self.closed_cnt)
        return e

    @property
    def asg(self):
        return self.st.asg

    @property
    def cp(self):
        return self.st.cp

    @property
    def cl(self):
        return self.st.cl

    @property
    def is_open(self):
        return self.st.is_open

    def _refresh(self):
        inst, st = self.inst, self.st
        if self.auto_close:
            for p in range(inst.m):
                if st.is_open[p] and st.cp[p] == 0 and inst.lq[p] > 0:
                    st.is_open[p] = False   # no students -> nothing to re-run
                    self.closed_cnt[p] = 0
        self.deficient = [p for p in range(inst.m)
                          if st.is_open[p] and st.cp[p] < inst.lq[p]]

    def close(self, ps):
        for p in ([ps] if isinstance(ps, int) else ps):
            self.closed_cnt[p] = self.st.cp[p]
            self.st.close(p)
            self.calls += 1
        self._refresh()

    @property
    def done(self):
        return not self.deficient

    def matched(self):
        return self.st.matched

    def safe(self):
        """Students not sitting in a deficient project (potential function)."""
        return self.st.matched - sum(self.st.cp[p] for p in self.deficient)

    def result(self):
        assert self.done
        return {
            "matched": self.matched(),
            "coalitions": count_coalitions(self.inst, self.is_open, self.asg, self.cl),
            "coalitions_hr": count_coalitions(self.inst, self.is_open, self.asg, self.cl, hr=True),
            "open": sum(self.is_open),
            "bp": blocking_pairs(self.inst, self.is_open, self.asg),
        }
