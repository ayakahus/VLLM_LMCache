# SPDX-License-Identifier: Apache-2.0
"""Online hybrid-cache placement using the normalized costs in Gandalf §3.3."""
from bisect import bisect_left, bisect_right
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, field, replace
from heapq import heappop, heappush
from itertools import accumulate
from threading import RLock


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
    linear_state: str | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Block:
    kind: str
    session: str
    round: int
    group: int
    boundary: int
    window: int
    window_hashes: tuple | list = ()


@dataclass(frozen=True)
class PrefixNode:
    key: object
    parent: "PrefixNode | None"


@dataclass(frozen=True)
class Snapshot:
    parents: dict
    activity: dict
    blocks: dict
    states: dict
    boundaries: dict
    layouts: dict
    full_heads: dict


def _estimate_activity(ages, gaps, horizon):
    gaps = sorted(gaps)
    censored = Counter(ages.values())
    idle_ages = sorted(censored)
    idle_counts = list(accumulate((censored[age] for age in idle_ages), initial=0))
    estimates = {}
    for age in censored:
        limit = age + horizon
        first = bisect_right(gaps, age)
        returns = bisect_right(gaps, limit) - first
        observed = len(gaps) - first + idle_counts[-1] - idle_counts[bisect_left(idle_ages, limit)]
        estimates[age] = (returns + 1) / (observed + 2)
    return {session: (age, estimates[age]) for session, age in ages.items()}


class History:
    def __init__(self, horizon, *, idle_window=None):
        self.horizon = horizon
        self.lock = RLock()
        self.clock = 0
        self.last = {}
        self.seen = OrderedDict()
        self.gaps = deque(maxlen=8192)
        self.parents = {}
        self.full_keys = {}
        self.blocks = {}
        self.rounds = Counter()
        # A session can have several independent linear groups. A sliding
        # window retains its last window of chunks as one live checkpoint.
        self.latest_linear = defaultdict(dict)
        self.layouts = {}
        self._prefix_nodes = {}
        self._session_keys = defaultdict(set)
        self._idle_window = max(8192, 4 * horizon) if idle_window is None else idle_window
        self._next_gc = self._idle_window

    def observe(self, session, request_id):
        with self.lock:
            request = (session, request_id)
            if request in self.seen:
                return self.seen[request]
            self.rounds[session] += 1
            self.seen[request] = self.rounds[session]
            if len(self.seen) > 8192:
                self.seen.popitem(last=False)
            self.clock += 1
            if session in self.last:
                self.gaps.append(self.clock - self.last[session])
            self.last[session] = self.clock
            return self.rounds[session]

    def register_layout(self, model, kinds):
        with self.lock:
            self.layouts[model] = kinds

    def bind(self, prefix, *, round_number=0, group=0):
        with self.lock:
            parent = None
            for boundary, key in enumerate(prefix, start=1):
                self.blocks.setdefault(key, Block("full", key.cache_salt, round_number,
                                                  group, boundary, 0))
                if key not in self.parents:
                    self.parents[key] = parent
                    node = PrefixNode(key, self._prefix_nodes.get(parent))
                    self._prefix_nodes[key] = node
                    self._session_keys[key.cache_salt].add(key)
                    identity = getattr(key, "identity", None)
                    if identity is not None:
                        scope = (identity.model_name, identity.kv_rank, key.cache_salt)
                        self.full_keys.setdefault((scope, identity.chunk_hash), {})[group] = node
                parent = key

    def commit_linear(self, keys):
        written = [(key, key.extra_attributes.get("gandalf_block")) for key in keys]
        written = [(key, Block(**block)) for key, block in written if isinstance(block, dict)]
        if not written:
            return
        # Process the furthest confirmed boundary first within each write completion.
        written.sort(key=lambda pair: (pair[1].round, pair[1].boundary), reverse=True)
        with self.lock:
            for key, block in written:
                self.blocks.setdefault(key, block)
                self._session_keys[block.session].add(key)
                latest = self.latest_linear[block.session]
                group = (key.identity.model_name, key.identity.kv_rank, block.group)
                head, active = latest.get(group, (None, set()))
                if (head is not None and block.round < head.round
                        and key.identity.chunk_hash not in head.window_hashes):
                    continue
                if head is None or (block.round, block.boundary) > (head.round, head.boundary):
                    if head is not None and block.round > head.round:
                        # Carry overlap only when it is on the new round's prefix.
                        active = {k for k in active if k.identity.chunk_hash in block.window_hashes}
                    head = block
                active = {k for k in active if self.blocks[k].boundary > head.boundary - head.window}
                if (block.boundary > head.boundary - head.window
                        and (not head.window_hashes or key.identity.chunk_hash in head.window_hashes)):
                    active.add(key)
                latest[group] = (head, active)

    def _linear_checkpoint(self, key, block):
        group = (key.identity.model_name, key.identity.kv_rank, block.group)
        return self.latest_linear.get(block.session, {}).get(group, (None, ()))

    def linear_state(self, key):
        with self.lock:
            block = self.blocks.get(key)
            if block is None or block.kind != "linear":
                return None
            _, active = self._linear_checkpoint(key, block)
            return "live" if key in active else "superseded"

    def gc_due(self):
        return self.clock >= self._next_gc

    def prune(self, resident):
        with self.lock:
            self._expire_sessions(resident)

    def _expire_sessions(self, resident):
        if self.clock < self._next_gc:
            return
        self._next_gc = self.clock + max(1, self._idle_window // 4)
        expired = {session for session, last in self.last.items()
                   if self.clock - last > self._idle_window and session not in resident}
        for session in expired:
            self.last.pop(session)
            self.rounds.pop(session, None)
            self.latest_linear.pop(session, None)
            for key in self._session_keys.pop(session, ()):
                block = self.blocks.pop(key)
                self.parents.pop(key, None)
                self._prefix_nodes.pop(key, None)
                identity = getattr(key, "identity", None)
                if block.kind == "full" and identity is not None:
                    scope = (identity.model_name, identity.kv_rank, session)
                    self.full_keys.pop((scope, identity.chunk_hash), None)
        for request in tuple(self.seen) if expired else ():
            if request[0] in expired:
                del self.seen[request]

    def eviction_snapshot(self, keys, *, in_flight=()):
        resident = {key.cache_salt for key in keys} | {key.cache_salt for key in in_flight}
        with self.lock:
            self._expire_sessions(resident)
            parents = {key: self.parents[key] for key in keys if key in self.parents}
            last = self.last.copy()
            clock = self.clock
            gaps = tuple(self.gaps)
            blocks = {key: self.blocks[key] for key in keys if key in self.blocks}
            states, boundaries, heads = {}, {}, {}
            for key, block in blocks.items():
                if block.kind == "linear":
                    head, active = self._linear_checkpoint(key, block)
                    live = key in active
                    states[key] = "live" if live else "superseded"
                    if live:
                        boundaries[key] = head.boundary
                        identity = key.identity
                        scope = (identity.model_name, identity.kv_rank, block.session)
                        heads[scope, identity.chunk_hash] = self.full_keys.get(
                            (scope, identity.chunk_hash), {}).copy()
            layouts = self.layouts.copy()
        ages = {session: max(1, clock - touch) for session, touch in last.items()}
        return Snapshot(parents, _estimate_activity(ages, gaps, self.horizon),
                        blocks, states, boundaries, layouts, heads)

    def snapshot(self, keys):
        snapshot = self.eviction_snapshot(keys)
        return snapshot.parents, snapshot.activity


def plan_victims(disk, protected, dram, dram_children, ssd_children,
                 parents, costs, order, needed):
    remaining = disk.keys() - protected
    child_count = {key: len(ssd_children[key]) for key in remaining}
    leaves = []

    def queue_leaf(key):
        if not child_count[key] and (key in dram or not dram_children[key]):
            heappush(leaves, (costs[key][1], order[key], key))

    for key in remaining:
        queue_leaf(key)
    victims = []
    while leaves:
        _, _, victim = heappop(leaves)
        victims.append(victim)
        needed -= disk[victim]
        if needed <= 0:
            return tuple(victims), sum(costs[key][0] for key in victims)
        parent = parents.get(victim)
        if parent in remaining:
            child_count[parent] -= 1
            queue_leaf(parent)
    return None


def plan_evictions(dram, eligible, disk, pinned, capacity, history, recompute_price,
                   write_price, alpha, count, *, in_flight=()):
    """Yield a batch of decisions against one snapshot and an evolving placement."""
    dram, disk, pinned = dict(dram), dict(disk), set(pinned)
    resident = dram.keys() | disk.keys()
    snapshot = history.eviction_snapshot(resident, in_flight=in_flight)
    parents, activity = snapshot.parents, snapshot.activity
    blocks, states, boundaries, layouts = (snapshot.blocks, snapshot.states,
                                         snapshot.boundaries, snapshot.layouts)
    remaining = set(eligible)
    expired = [key for key in dram if key in remaining and states.get(key) == "superseded"]

    children = defaultdict(set)
    for key, parent in parents.items():
        children[parent].add(key)
    dram_children = {key: children[key] & dram.keys() for key in resident}
    ssd_children = {key: children[key] & disk.keys() for key in resident}

    def ancestors(key):
        while parents.get(key) is not None:
            key = parents[key]
            yield key

    # A rollback boundary needs every group at the same prefix hash. Index
    # resident objects once so looking for it does not scan the entire cache.
    groups = defaultdict(dict)
    checkpoints = defaultdict(dict)
    for key, block in (blocks.items() if states else ()):
        identity = getattr(key, "identity", None)
        if identity is None:
            continue
        scope = (identity.model_name, identity.kv_rank, block.session)
        groups[scope][block.group] = block.kind
        checkpoints[scope, identity.chunk_hash][block.group] = key

    def rollback_chunks(key):
        block = blocks[key]
        identity = getattr(key, "identity", None)
        if identity is None:
            return block.boundary
        scope = (identity.model_name, identity.kv_rank, block.session)
        layout = layouts.get(identity.model_name, groups[scope])
        full_groups = [group for group, kind in layout.items() if kind == "full"]
        linear_groups = [group for group, kind in layout.items() if kind == "linear"]
        boundary = boundaries[key]
        if not full_groups:
            return boundary
        node = snapshot.full_heads.get((scope, identity.chunk_hash), {}).get(full_groups[0])
        prefix = []
        while node is not None and node.parent is not None:
            node = node.parent
            prefix.append(node.key.identity.chunk_hash)
        fallback = 0
        linear_runs = Counter()
        for n, chunk_hash in enumerate(reversed(prefix), start=1):
            row = checkpoints.get((scope, chunk_hash), {})
            if any(group not in row or (row[group] not in dram and row[group] not in disk)
                   for group in full_groups):
                break
            for group in linear_groups:
                resident_key = row.get(group)
                linear_runs[group] = (linear_runs[group] + 1
                                      if resident_key in dram or resident_key in disk else 0)
            if all(group in row and linear_runs[group] >= min(n, blocks[row[group]].window)
                   for group in linear_groups):
                fallback = n
        return max(1, boundary - fallback)

    def leaf_cost(key, size):
        state = states.get(key)
        if state == "superseded":
            return 0.0, 0.0
        age, return_probability = activity.get(key.cache_salt, (1, 0.5))
        # Eq. 7 charges for the next read. A live checkpoint has one
        # remaining use under the append-only placement model.
        if state == "live":
            cost = recompute_price * rollback_chunks(key)
        else:
            cost = return_probability * recompute_price
        return cost, cost / (size * age)

    disk_prefix = {}

    def has_disk_ancestors(key):
        unseen = []
        parent = parents.get(key)
        while parent is not None and parent in disk and parent not in disk_prefix:
            unseen.append(parent)
            parent = parents.get(parent)
        complete = parent is None or disk_prefix.get(parent, False)
        disk_prefix.update(dict.fromkeys(unseen, complete))
        return complete

    def invalidate_prefix(key):
        pending = [key]
        while pending:
            affected = pending.pop()
            disk_prefix.pop(affected, None)
            pending.extend(child for child in children[affected] if child in disk)

    used = sum(disk.values())
    order = {key: index for index, key in enumerate(disk)}
    next_order = len(order)
    priority = {key: index for index, key in enumerate(dram)}
    linear = [key for key in dram if key in remaining and states.get(key) == "live"]
    drops = []
    write_costs = {}
    move_candidates = []

    def droppable(key):
        return not dram_children[key] and (key in disk or not ssd_children[key])

    def queue_drop(key):
        if key in remaining and not states.get(key) and droppable(key):
            cost, score = leaf_cost(key, dram[key])
            heappush(drops, (score, priority[key], key, cost))

    for key in dram:
        if key not in remaining or states.get(key) == "superseded":
            continue
        queue_drop(key)
        size = dram[key]
        if size > capacity or (key not in parents and not states.get(key)):
            continue
        age, _ = activity.get(key.cache_salt, (1, 0.5))
        normalization = size * age
        if size not in write_costs:
            write_costs[size] = size * write_price(size)
        cost = write_costs[size]
        move_candidates.append((cost / normalization, priority[key], key, normalization, cost))
    move_candidates.sort()

    def update_disk(key, add):
        parent = parents.get(key)
        invalidate_prefix(key)
        if parent in ssd_children:
            if add:
                ssd_children[parent].add(key)
            else:
                ssd_children[parent].discard(key)
            queue_drop(parent)
        queue_drop(key)

    def best_decision():
        while drops and (drops[0][2] not in remaining or not droppable(drops[0][2])):
            heappop(drops)
        drop = drops[0] if drops else None
        for key in linear:
            if key in remaining:
                cost, score = leaf_cost(key, dram[key])
                candidate = (score, priority[key], key, cost)
                if drop is None or candidate[:2] < drop[:2]:
                    drop = candidate
        best_drop = None
        if drop is not None:
            score, _, key, cost = drop
            action = "release_duplicate" if key in disk else "drop"
            best_drop = Decision(key, action, drop_cost=cost, drop_score=score)

        best_move = None
        plans = {}
        ssd_costs = None
        for write_score, _, key, normalization, write_cost in move_candidates:
            if best_move is not None and write_score >= best_move.move_score:
                break
            if key not in remaining or key in disk or not has_disk_ancestors(key):
                continue
            needed = used + dram[key] - capacity
            if needed > 0:
                if ssd_costs is None:
                    ssd_costs = {k: (0.0, 0.0) if k in dram else leaf_cost(k, size)
                                 for k, size in disk.items()}
                preserve = frozenset(ancestors(key)) - pinned
                protection = frozenset(pinned)
                common = (protection, needed)
                if common not in plans:
                    plans[common] = plan_victims(disk, protection, dram, dram_children,
                                                 ssd_children, parents, ssd_costs, order, needed)
                plan = plans[common]
                if plan is not None and not preserve.isdisjoint(plan[0]):
                    protection |= preserve
                    specific = (protection, needed)
                    if specific not in plans:
                        plans[specific] = plan_victims(
                            disk, protection, dram, dram_children, ssd_children,
                            parents, ssd_costs, order, needed,
                        )
                    plan = plans[specific]
            else:
                plan = ((), 0.0)
            if plan is None:
                continue
            victims, replacement = plan
            score = write_score + replacement / normalization
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

    expired = iter(expired)
    for _ in range(count):
        key = next(expired, None)
        decision = Decision(key, "drop", drop_score=0.0) if key is not None else best_decision()
        if decision is None:
            return
        key = decision.key
        remaining.remove(key)
        size = dram.pop(key)
        parent = parents.get(key)
        if parent in dram_children:
            dram_children[parent].discard(key)
            queue_drop(parent)
        if decision.action == "release_duplicate":
            # A released copy is now the only copy and must survive this batch.
            pinned.add(key)
            pinned.update(ancestors(key))
        elif decision.action == "move":
            for victim in decision.victims:
                used -= disk.pop(victim)
                order.pop(victim)
                update_disk(victim, False)
            disk[key] = size
            order[key] = next_order
            next_order += 1
            used += size
            pinned.add(key)
            pinned.update(ancestors(key))
            update_disk(key, True)
        state = states.get(key)
        yield replace(decision, linear_state=state) if state else decision


def choose(dram, eligible, disk, pinned, capacity, history, recompute_price,
           write_price, alpha):
    """Return one eviction decision without changing the supplied placement."""
    return next(plan_evictions(dram, eligible, disk, pinned, capacity, history,
                               recompute_price, write_price, alpha, 1), None)
