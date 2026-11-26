# SPDX-License-Identifier: Apache-2.0
"""Online hybrid-cache placement using the normalized costs in Gandalf §3.3."""
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, replace
from functools import lru_cache
from heapq import heapify, heappop, heappush
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


@dataclass(frozen=True)
class Block:
    kind: str
    session: str
    round: int
    group: int
    boundary: int
    window: int
    window_hashes: tuple | list = ()


class History:
    def __init__(self, horizon):
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
                    identity = getattr(key, "identity", None)
                    if identity is not None:
                        scope = (identity.model_name, identity.kv_rank, key.cache_salt)
                        self.full_keys[scope, identity.chunk_hash, group] = key
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

    def linear_state(self, key):
        with self.lock:
            block = self.blocks.get(key)
            if block is None or block.kind != "linear":
                return None
            group = (key.identity.model_name, key.identity.kv_rank, block.group)
            _, active = self.latest_linear[block.session][group]
            return "live" if key in active else "superseded"

    def placement_snapshot(self, keys):
        with self.lock:
            blocks = {key: self.blocks[key] for key in keys if key in self.blocks}
            states = {key: self.linear_state(key) for key, block in blocks.items()
                      if block.kind == "linear"}
            boundaries = {}
            for key, state in states.items():
                if state == "live":
                    block = blocks[key]
                    group = (key.identity.model_name, key.identity.kv_rank, block.group)
                    boundaries[key] = self.latest_linear[block.session][group][0].boundary
            return blocks, states, boundaries, self.layouts.copy()

    def full_prefix_hashes(self, scope, chunk_hash, group):
        with self.lock:
            head = self.full_keys.get((scope, chunk_hash, group))
            hashes = []
            while head is not None and (head := self.parents.get(head)) is not None:
                hashes.append(head.identity.chunk_hash)
        return reversed(hashes)

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


def choose(dram, eligible, disk, pinned, capacity, history, recompute_price,
           write_price, alpha):
    """Return one eviction decision. SSD victims are planned before any deletion."""
    resident = dram.keys() | disk.keys()
    with history.lock:
        parents, activity = history.snapshot(resident)
        blocks, states, boundaries, layouts = history.placement_snapshot(resident)
    candidates = [key for key in dram if key in eligible]
    expired = next((key for key in candidates if states.get(key) == "superseded"), None)
    if expired is not None:
        return Decision(expired, "drop", drop_score=0.0)

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
        prefix = history.full_prefix_hashes(scope, identity.chunk_hash, full_groups[0])
        fallback = 0
        linear_runs = Counter()
        for n, chunk_hash in enumerate(prefix, start=1):
            row = checkpoints.get((scope, chunk_hash), {})
            if any(group not in row for group in full_groups):
                break
            for group in linear_groups:
                linear_runs[group] = linear_runs[group] + 1 if group in row else 0
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

    best_drop = None
    move_candidates = []
    write_costs = {}
    used = sum(disk.values())
    for key in candidates:
        size = dram[key]
        state = states.get(key)
        if state or (not dram_children[key] and (key in disk or not ssd_children[key])):
            cost, score = leaf_cost(key, size)
            if best_drop is None or score < best_drop.drop_score:
                action = "release_duplicate" if key in disk else "drop"
                best_drop = Decision(key, action, drop_cost=cost, drop_score=score)
        if (key in disk or size > capacity
                or (not state and (key not in parents or not has_disk_ancestors(key)))):
            continue
        age, _ = activity.get(key.cache_salt, (1, 0.5))
        normalization = size * age
        if size not in write_costs:
            write_costs[size] = size * write_price(size)
        write_cost = write_costs[size]
        move_candidates.append(
            (write_cost / normalization, len(move_candidates), key, normalization, write_cost)
        )

    if any(used + dram[key] > capacity for _, _, key, _, _ in move_candidates):
        ssd_costs = {key: (0.0, 0.0) if key in dram else leaf_cost(key, size)
                     for key, size in disk.items()}
        order = {key: index for index, key in enumerate(disk)}

    @lru_cache(maxsize=32)
    def replacement_plan(protected, needed):
        return plan_victims(
            disk, protected, dram, dram_children, ssd_children, parents, ssd_costs,
            order, needed,
        )

    best_move = None
    heapify(move_candidates)
    while move_candidates:
        write_score, _, key, normalization, write_cost = heappop(move_candidates)
        # Replacement costs are nonnegative, so no later candidate can win.
        if best_move is not None and write_score >= best_move.move_score:
            break
        needed = used + dram[key] - capacity
        plan = ((), 0.0) if needed <= 0 else replacement_plan(
            frozenset(pinned.union(ancestors(key))), needed,
        )
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
