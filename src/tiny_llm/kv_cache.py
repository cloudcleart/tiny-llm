from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import mlx.core as mx

from .attention import causal_mask

if TYPE_CHECKING:
    from .paged_kv_cache import PagedKvMetadata


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            The updated keys, updated values, sequence length, and mask. On
            On Week 2 Day 1, the mask is passed through unchanged. Week 3 Day 1
            uses the sequence length and mask to construct a dense batch.
        """

    def release(self):
        pass

    def materialize(self):
        """Evaluate owned K/V storage without changing its logical layout."""
        pass

    def update_and_fetch_paged(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        pass

    def rewind(self, n: int):
        pass


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int | None = None):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.kv_caches: list[TinyKvCache | None] = [None] * max_active_requests
        self.last_batch_bytes = 0
        self.staging_copy_bytes = 0
        self.HD: tuple[int, int] | None = None

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, None, mx.array]:
        """Append active rows and right-align their caches in a dense batch."""
        if keys.ndim != 4 or keys.shape != values.shape:
            raise ValueError("batch K/V must have matching [B, H, L, D] shapes")
        if keys.shape[0] != self.max_active_requests:
            raise ValueError("batch K/V must contain one row per decode slot")
        if mask_length is None:
            mask_length = keys.shape[2]

        updated = []
        source_length = 0
        for slot, cache in enumerate(self.kv_caches):
            if cache is None:
                continue
            request_keys, request_values, length, _ = cache.update_and_fetch(
                keys[slot : slot + 1], values[slot : slot + 1]
            )
            updated.append((slot, request_keys, request_values, length))
            source_length = max(source_length, length)

        batch, heads, _, dim = keys.shape
        shape = (batch, heads, source_length, dim)
        batched_keys = mx.zeros(shape, dtype=keys.dtype)
        batched_values = mx.zeros(shape, dtype=values.dtype)
        batch_mask = mx.full(
            (batch, 1, mask_length, source_length), -mx.inf, dtype=keys.dtype
        )
        copied_bytes = 0
        for slot, request_keys, request_values, length in updated:
            start = source_length - length
            batched_keys[slot, :, start:, :] = request_keys[0, :, :length, :]
            batched_values[slot, :, start:, :] = request_values[0, :, :length, :]
            batch_mask[slot, 0, :, start:] = causal_mask(
                mask_length, length, keys.dtype
            )
            copied_bytes += heads * length * dim * (keys.itemsize + values.itemsize)

        self.last_batch_bytes = batched_keys.nbytes + batched_values.nbytes
        self.staging_copy_bytes += copied_bytes
        # Per-request lengths are represented by the mask, not a shared offset.
        return batched_keys, batched_values, None, batch_mask

    def update_and_fetch_paged(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        from .paged_kv_cache import PagedKvMetadata, TinyKvPagedCache

        if len(keys.shape) != 4 or len(values.shape) != 4:
            raise ValueError("Batched K/V chunks must be 4D [B, H, S, D]")
        if keys.shape != values.shape:
            raise ValueError("Batched K/V chunks must have the same shape")
        B, H, S, D = keys.shape
        if B != self.max_active_requests:
            raise ValueError(f"Expected batch size {self.max_active_requests}, got {B}")
        if self.HD is not None and self.HD != (H, D):
            raise ValueError(f"expect {self.HD} but got {H, D}")

        # Validate the complete active set before any request or allocator is
        # mutated. In particular, mixed pools must fail before row zero appends.
        pool = None
        active_caches = []
        for b in range(B):
            cache = self.kv_caches[b]
            if cache is None:
                continue
            if not isinstance(cache, TinyKvPagedCache):
                raise ValueError("BatchingKvCache contains a non-paged request cache")
            if pool is None:
                pool = cache.pool
            elif pool is not cache.pool:
                raise ValueError("Paged batch caches must share one page pool")
            if self.max_seq_len is not None and cache.offset + S > self.max_seq_len:
                raise ValueError("Paged batch append exceeds max_seq_len")
            cache.validate_append(keys[b : b + 1], values[b : b + 1])
            active_caches.append((b, cache))

        if pool is None:
            raise ValueError("Cannot build paged metadata without active requests")

        # Per-cache rollback cannot undo earlier successful batch rows. Snapshot
        # the shared pool once and save metadata for every active request.
        pool_state = pool._snapshot_state()
        cache_states = [(cache, cache._snapshot_state()) for _, cache in active_caches]
        old_hd = self.HD
        context_lens = [0] * B
        max_pages = 0
        try:
            for b, cache in active_caches:
                cache.update_and_fetch_paged(
                    keys[b : b + 1],
                    values[b : b + 1],
                    mask_length=mask_length,
                    mask=mask,
                )
                context_lens[b] = cache.offset
                max_pages = max(max_pages, cache.num_pages)
            self.HD = (H, D)
        except Exception:
            pool._restore_state(pool_state)
            for cache, state in cache_states:
                cache._restore_state(state)
            self.HD = old_hd
            raise

        # Returning shared pages and a block table avoids dense batch KV copies.
        self.last_batch_bytes = 0

        # Preserve slot order: inactive slots have length zero and -1 page IDs.
        rows = []
        for cache in self.kv_caches:
            if cache is None:
                rows.append([-1] * max_pages)
            else:
                rows.append(cache.page_ids + [-1] * (max_pages - cache.num_pages))

        return PagedKvMetadata(
            key_pages=pool.key_pages,
            value_pages=pool.value_pages,
            block_table=mx.array(rows, dtype=mx.int32),
            context_lens=mx.array(context_lens, dtype=mx.int32),
            page_size=pool.page_size,
            mask=mask,
        )

    def add_request(self, prefilled: TinyKvCache, id: int):
        if not 0 <= id < self.max_active_requests:
            raise IndexError("decode slot out of range")
        if self.kv_caches[id] is not None:
            raise ValueError("decode slot is already occupied")
        self.kv_caches[id] = prefilled

    def remove_request(self, id: int):
        if not 0 <= id < self.max_active_requests:
            raise IndexError("decode slot out of range")
        cache = self.kv_caches[id]
        if cache is not None:
            cache.release()
            self.kv_caches[id] = None


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the full key-value cache and fetch the updated cache.

        Args:
            key: The key states to append. Shape: (B, H, L_new, D).
            value: The value states to append. Shape: (B, H, L_new, D).
            mask_length: The length of the mask (only used in batching mode).
            mask: The mask to use (only used in batching mode).

        Returns:
            The updated key cache, updated value cache, total sequence length,
            and the mask passed through unchanged.
        """
        if self.key_values is None:
            self.key_values = (key, value)
        else:
            cached_key, cached_value = self.key_values
            self.key_values = (
                mx.concat([cached_key, key], axis=2),
                mx.concat([cached_value, value], axis=2),
            )

        self.offset += key.shape[2]
        key_cache, value_cache = self.key_values
        return key_cache, value_cache, self.offset, mask

    def materialize(self):
        if self.key_values is not None:
            # MLX evaluates lazily, so force both cached tensors to be computed.
            mx.eval(*self.key_values)

    def rewind(self, n: int):
        if n <= 0 or self.key_values is None:
            return

        # Clamp the retained sequence length at zero when rewinding past the cache.
        keep_len = max(self.offset - n, 0)
        key, value = self.key_values
        # The sequence dimension is axis 2 for tensors shaped (B, H, L, D).
        self.key_values = (key[:, :, :keep_len, :], value[:, :, :keep_len, :])
        self.offset = keep_len
