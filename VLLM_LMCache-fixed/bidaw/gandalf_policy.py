# SPDX-License-Identifier: Apache-2.0
"""Online full-attention placement for Gandalf (§3.3), using candidate scans."""
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class Decision:
    key: object
    action: str
    victims: tuple = ()
    drop_cost: float = 0.0
    write_cost: float = 0.0
    replacement_cost: float = 0.0


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

    def leaf_value(key):
        return activity.get(key.cache_salt, (1, 0.5))[1] * recompute_price

    def value(key):
        age, reuse = activity.get(key.cache_salt, (1, 0.5))
        # Losing an interior chunk also loses the reachable suffix behind it.
        reachable = 0
        pending = [key]
        while pending:
            item = pending.pop()
            reachable += 1
            pending.extend(children[item])
        return reuse * reachable * recompute_price, age

    def leaf(key, other_tier):
        return key in other_tier or not children[key]

    def plan_victims(key, needed):
        if needed <= 0:
            return (), 0.0
        protected = {key}
        for parent in ancestors(key):
            protected.add(parent)
        remaining = disk.keys() - pinned - protected
        victims = []
        removed = set()
        lost = 0.0
        for _ in range(len(remaining)):
            leaves = [item for item in remaining if item in dram or children[item] <= removed]
            if not leaves:
                break
            victim = min(leaves, key=lambda item: (
                (0 if item in dram else leaf_value(item)) / (disk[item] * scores[item][1]),
                order[item],
            ))
            remaining.remove(victim)
            if victim not in dram:
                removed.add(victim)
                lost += leaf_value(victim)
            victims.append(victim)
            needed -= disk[victim]
            if needed <= 0:
                return tuple(victims), lost
        return None

    def ancestors(key):
        while parents.get(key) is not None:
            key = parents[key]
            yield key

    candidates = [key for key in dram if key in eligible]
    duplicate = next((key for key in candidates if key in disk), None)
    if duplicate is not None:
        return Decision(duplicate, "release_duplicate")

    scores = {key: value(key) for key in resident}
    order = {key: index for index, key in enumerate(disk)}
    used = sum(disk.values())
    best = None
    best_gain = 0.0
    for key in candidates:
        size = dram[key]
        if key not in parents or size > capacity or any(p not in disk for p in ancestors(key)):
            continue
        plan = plan_victims(key, used + size - capacity)
        if plan is None:
            continue
        victims, replacement = plan
        drop_cost, age = scores[key]
        write_cost = size * write_price(size)
        gain = (drop_cost - alpha * (write_cost + replacement)) / (size * age)
        if gain > best_gain:
            best_gain = gain
            best = Decision(key, "move", victims, drop_cost, write_cost, replacement)
    if best is not None:
        return best

    leaves = [key for key in candidates if leaf(key, disk)]
    if not leaves:
        return None
    key = min(leaves, key=lambda item: scores[item][0] / (dram[item] * scores[item][1]))
    return Decision(key, "drop", drop_cost=scores[key][0])
