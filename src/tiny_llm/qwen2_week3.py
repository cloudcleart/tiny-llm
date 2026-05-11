import mlx.core as mx
from typing import Any


class Qwen2MultiHeadAttention:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Week 3 not yet implemented")


class Qwen2MLP:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Week 3 not yet implemented")


class Qwen2TransformerBlock:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Week 3 not yet implemented")


class Qwen2ModelWeek3:
    def __init__(self, mlx_model: Any, enable_flash_attn: bool = False, page_size: int = 128):
        pass

    def create_kv_cache(self):
        raise NotImplementedError("Week 3 not yet implemented")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Week 3 not yet implemented")
