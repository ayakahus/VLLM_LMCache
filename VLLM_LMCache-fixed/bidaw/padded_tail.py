"""Wire representation and server-side zero padding for a 256-token tail."""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

CHUNK_TOKENS = 256
BLOCK_TOKENS = 16
TOKEN_BYTES = 4  # LMCache MP packs token IDs as big-endian uint32.
PADDING_TOKEN = b"\xff" * TOKEN_BYTES  # Outside the model's vocabulary.
TAIL_VALID_TOKENS = "bidaw_tail_valid_tokens"
TAIL_PARENT_REQUEST = "bidaw_tail_parent_request"
PADDED_TAIL_STORE_ENABLED = False


@dataclass
class PaddedTail:
    start: int
    end: int
    valid_tokens: int
    token_bytes: bytes
    block_ids: list[int]


def make_padded_tail(tokens: bytes, computed: int, block_ids: list[int]) -> PaddedTail | None:
    """Select the computed suffix and build one ordinary 256-token STORE range."""
    computed = min(max(0, computed), len(tokens) // TOKEN_BYTES)
    valid = computed % CHUNK_TOKENS
    if not valid:
        return None

    start = computed - valid
    tail_blocks = block_ids[start // BLOCK_TOKENS : (computed + BLOCK_TOKENS - 1) // BLOCK_TOKENS]
    if len(tail_blocks) != (valid + BLOCK_TOKENS - 1) // BLOCK_TOKENS or any(
        block <= 0 for block in tail_blocks
    ):
        return None

    padding = CHUNK_TOKENS - valid
    # Use owned block IDs throughout the native transfer. The server zeroes
    # every invalid slot in the staging object before admitting it to L1.
    padded_blocks = tail_blocks + [tail_blocks[-1]] * (CHUNK_TOKENS // BLOCK_TOKENS - len(tail_blocks))
    padded_tokens = tokens[:computed * TOKEN_BYTES] + PADDING_TOKEN * padding
    return PaddedTail(start, start + CHUNK_TOKENS, valid, padded_tokens, padded_blocks)


def valid_padded_key(key, valid) -> bool:
    """Check that the padding marker is confined to the end of one chunk."""
    if not (type(valid) is int and 0 < valid < CHUNK_TOKENS
            and key.start >= 0 and key.start % CHUNK_TOKENS == 0
            and key.end == key.start + CHUNK_TOKENS
            and len(key.token_bytes) == key.end * TOKEN_BYTES):
        return False
    offset = (key.start + valid) * TOKEN_BYTES
    return (key.token_bytes[offset:] == PADDING_TOKEN * (CHUNK_TOKENS - valid)
            and key.token_bytes[offset - TOKEN_BYTES:offset] != PADDING_TOKEN)


def apply_padded_tail_store() -> None:
    """Sanitize padding in the LMCache MP CUDA store path before L1 commit."""
    global PADDED_TAIL_STORE_ENABLED
    from lmcache import lmcache_native
    from lmcache.logging import init_logger
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import LMCacheDrivenTransferModule
    from lmcache.v1.platform.base.cache_context import BaseCacheContext

    original_store = LMCacheDrivenTransferModule.store_with_chunk_mask
    if getattr(original_store, "_bidaw_padded_tail", False):
        PADDED_TAIL_STORE_ENABLED = True
        return
    original_transfer = BaseCacheContext.transfer_object_group
    tail_slots = ContextVar("bidaw_tail_slots", default=None)
    logger = init_logger(__name__)

    @wraps(original_store)
    def store(self, key, instance_id, gpu_block_ids, event_ipc_handle):
        configs = key.request_configs or {}
        if TAIL_VALID_TOKENS not in configs:
            return original_store(self, key, instance_id, gpu_block_ids, event_ipc_handle)
        valid = configs[TAIL_VALID_TOKENS]
        entry = self.get_and_touch_context_entry(instance_id)
        if (self._ctx.chunk_size != CHUNK_TOKENS or not valid_padded_key(key, valid)
                or entry is None or entry.cache_context.device.type != "cuda"):
            logger.warning("Rejecting invalid padded-tail STORE for %s", key.request_id)
            return b"", False, []

        token = tail_slots.set(valid)
        try:
            return original_store(self, key, instance_id, gpu_block_ids, event_ipc_handle)
        finally:
            tail_slots.reset(token)

    @wraps(original_transfer)
    def transfer(self, block_ids_gpu, memory_objs, *, object_group_id, direction, **kwargs):
        valid = tail_slots.get()
        padding_views = []
        if valid is not None and direction == lmcache_native.TransferDirection.D2H:
            group = self.kv_layer_groups_manager.object_groups[object_group_id]
            for obj in memory_objs:
                if obj is None:
                    continue
                for index in range(len(group.kernel_group_indices)):
                    tensor = obj.get_tensor(index)
                    if (tensor is None or tensor.device.type != "cpu"
                            or tensor.ndim != 4 or tensor.shape[2] != CHUNK_TOKENS):
                        raise ValueError("Padded tails require the full-attention DRAM layout")
                    padding_views.append(tensor[:, :, valid:, :])

        original_transfer(self, block_ids_gpu, memory_objs,
                          object_group_id=object_group_id, direction=direction, **kwargs)
        if padding_views:
            # The native transfer has enqueued copies on this stream. Clear the
            # host staging suffix only after those copies and before commit.
            self.stream.synchronize()
            for padding in padding_views:
                padding.zero_()

    store._bidaw_padded_tail = True
    BaseCacheContext.transfer_object_group = transfer
    LMCacheDrivenTransferModule.store_with_chunk_mask = store
    PADDED_TAIL_STORE_ENABLED = True
