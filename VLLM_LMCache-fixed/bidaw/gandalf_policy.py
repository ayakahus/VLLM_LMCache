# SPDX-License-Identifier: Apache-2.0
"""Online full-attention placement using the normalized costs in Gandalf §3.3."""
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, replace
from threading import Lock


@dataclass(frozen=True)
class Decision:
    key: object
    action: str
    victims: tuple = ()
    drop_cost: float = 0.0
    write_cost: float = 0.0
    replacement_cost: float = 0.0
    drop_score: float | None = None
    move_score: float | None = None


class History:
    def __init__(self, horizon):
        self.horizon = horizon
        self.lock = Lock()
        self.clock = 0
        self.last = {}
        self.seen = OrderedDict()
        self.gaps = deque(maxlen=8192)
        self.parents = {}

    def observe(self, session, request_id):
        with self.lock:
            if request_id in self.seen:
                return
            self.seen[request_id] = None
            if len(self.seen) > 8192:
                self.seen.popitem(last=False)
            self.clock += 1
            if session in self.last:
                self.gaps.append(self.clock - self.last[session])
            self.last[session] = self.clock

    def bind(self, prefix):
        with self.lock:
            parent = None
            for key in prefix:
                self.parents.setdefault(key, parent)
                parent = key

    def snapshot(self, keys):
        with self.lock:
            parents = {key: self.parents[key] for key in keys if key in self.parents}
            ages = {session: max(1, self.clock - last) for session, last in self.last.items()}
            completed = Counter(self.gaps)
        censored = Counter(ages.values())
        estimates = {}
        for age in set(ages.values()):
            limit = age + self.horizon
            returns = sum(n for gap, n in completed.items() if age < gap <= limit)
            observed = sum(n for gap, n in completed.items() if gap > age)
            observed += sum(n for idle, n in censored.items() if idle >= limit)
            estimates[age] = (returns + 1) / (observed + 2)
        return parents, {session: (age, estimates[age]) for session, age in ages.items()}


def choose(dram, eligible, disk, pinned, capacity, history, recompute_price,
           write_price, alpha):
    """Return one eviction decision. SSD victims are planned before any deletion."""
    resident = dram.keys() | disk.keys()
    parents, activity = history.snapshot(resident)
    children = defaultdict(set)
    for key, parent in parents.items():
        children[parent].add(key)

    def leaf_cost(key, size):
        age, reuse = activity.get(key.cache_salt, (1, 0.5))
        cost = reuse * recompute_price
        return cost, cost / (size * age)

    def plan_victims(key, needed):
        if needed <= 0:
            return (), 0.0, 0.0
        protected = {key, *ancestors(key)}
        remaining = disk.keys() - pinned - protected
        victims = []
        removed = set()
        while remaining:
            leaves = [item for item in remaining
                      if ssd_children[item] <= removed and (item in dram or not dram_children[item])]
            if not leaves:
                break
            victim = min(leaves, key=lambda item: (ssd_costs[item][1], order[item]))
            remaining.remove(victim)
            removed.add(victim)
            victims.append(victim)
            needed -= disk[victim]
            if needed <= 0:
                # Eq. (8) prices the cheapest victim. Reclaim still reserves
                # enough space for the entire object and removes all planned leaves.
                return (tuple(victims), sum(ssd_costs[item][0] for item in victims),
                        min(ssd_costs[item][1] for item in victims))
        return None

    def ancestors(key):
        while parents.get(key) is not None:
            key = parents[key]
            yield key

    candidates = [key for key in dram if key in eligible]
    dram_children = {key: children[key] & dram.keys() for key in resident}
    ssd_children = {key: children[key] & disk.keys() for key in resident}
    drops = [key for key in candidates
             if not dram_children[key] and (key in disk or not ssd_children[key])]

    best_drop = None
    if drops:
        key = min(drops, key=lambda item: leaf_cost(item, dram[item])[1])
        cost, score = leaf_cost(key, dram[key])
        action = "release_duplicate" if key in disk else "drop"
        # A duplicate competes on its DRAM reuse value, then releases for zero writes.
        best_drop = Decision(key, action, drop_cost=cost, drop_score=score)

    ssd_costs = {key: (0.0, 0.0) if key in dram else leaf_cost(key, size)
                 for key, size in disk.items()}
    order = {key: index for index, key in enumerate(disk)}
    used = sum(disk.values())
    best_move = None
    for key in candidates:
        size = dram[key]
        if (key in disk or key not in parents or size > capacity
                or any(p not in disk for p in ancestors(key))):
            continue
        plan = plan_victims(key, used + size - capacity)
        if plan is None:
            continue
        victims, replacement, replacement_score = plan
        age, _ = activity.get(key.cache_salt, (1, 0.5))
        write_cost = size * write_price(size)
        score = write_cost / (size * age) + replacement_score
        if best_move is None or score < best_move.move_score:
            best_move = Decision(key, "move", victims, write_cost=write_cost,
                                 replacement_cost=replacement, move_score=score)

    if best_move is None:
        return best_drop
    if best_drop is None:
        return best_move
    if alpha * best_move.move_score < best_drop.drop_score:
        return replace(best_move, drop_score=best_drop.drop_score)
    return replace(best_drop, move_score=best_move.move_score)
