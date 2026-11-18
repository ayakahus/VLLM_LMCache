"""Exact weighted reuse distance over distinct identities.

Only the latest position of each identity contributes its weight.
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
        for pos, (key, (_, weight)) in enumerate(self.last.items(), 1):
            self.last[key] = pos, weight
            self._add(pos, weight)
        self.clock = len(self.last)

    def access(self, key, weight=1):
        if self.clock == self.size:
            self._compact()
        prev = self.last.pop(key, None)
        distance = None
        if prev is not None:
            prev_pos, prev_weight = prev
            distance = self._sum(self.clock) - self._sum(prev_pos)
            self._add(prev_pos, -prev_weight)
        else:
            self.censored += 1
        self.clock += 1
        self.last[key] = self.clock, weight
        self._add(self.clock, weight)
        if len(self.last) > self.max_keys:
            _, (old_pos, old_weight) = self.last.popitem(last=False)
            self._add(old_pos, -old_weight)
        return distance
