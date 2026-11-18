"""Exact distinct-object reuse distance for equal-sized KV chunks.

Only the latest position of each identity has a Fenwick-tree weight of one.
Oldest identities may be forgotten to bound memory; their next access is a
censored (None) sample, never a fabricated short distance. L1 eviction does
not remove identities from this history.
"""
from collections import OrderedDict


class ReuseDistance:
    def __init__(self, max_keys=200_000):
        self.max_keys = max(1, max_keys)
        self.last = OrderedDict()
        self.clock = 0
        self.size = 1024
        self.tree = [0] * (self.size + 1)
        self.censored = 0

    def _add(self, pos, delta):
        while pos <= self.size:
            self.tree[pos] += delta
            pos += pos & -pos

    def _sum(self, pos):
        total = 0
        while pos:
            total += self.tree[pos]
            pos -= pos & -pos
        return total

    def _compact(self):
        self.size = max(1024, 2 * (len(self.last) + 1))
        self.tree = [0] * (self.size + 1)
        for pos, key in enumerate(list(self.last), 1):
            self.last[key] = pos
            self._add(pos, 1)
        self.clock = len(self.last)

    def access(self, key):
        if self.clock == self.size:
            self._compact()
        prev = self.last.pop(key, None)
        distance = None
        if prev is not None:
            distance = self._sum(self.clock) - self._sum(prev)
            self._add(prev, -1)
        else:
            self.censored += 1
        self.clock += 1
        self.last[key] = self.clock
        self._add(self.clock, 1)
        if len(self.last) > self.max_keys:
            _, old_pos = self.last.popitem(last=False)
            self._add(old_pos, -1)
        return distance
