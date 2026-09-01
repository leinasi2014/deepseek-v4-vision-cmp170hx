# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DeepSeek V4 per-layer rope selection."""

import types

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
    DeepseekV4ScalingRotaryEmbedding,
)
from vllm.models.deepseek_v4.attention import resolve_layer_compress_ratio
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope

NUM_HIDDEN_LAYERS = 4


@pytest.fixture(autouse=True)
def _clear_rope_cache():
    _ROPE_DICT.clear()
    yield
    _ROPE_DICT.clear()


def _config(compress_ratios: list[int]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        compress_ratios=compress_ratios,
        rope_theta=10000.0,
        compress_rope_theta=1000000.0,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 512,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )


@pytest.mark.parametrize(
    "compress_ratios,layer_id,expected_ratio,expected_unscaled",
    [
        ([1, 4, 4, 1], 0, 1, False),
        ([1, 4, 4, 1], 1, 4, False),
        ([0, 4, 4, 1], 0, 1, False),
        ([1, 4, 4, 1, 0], NUM_HIDDEN_LAYERS, 1, True),
        ([1, 4, 4, 1, 4], NUM_HIDDEN_LAYERS, 4, False),
        ([1, 4, 4, 1], NUM_HIDDEN_LAYERS, 1, False),
        ([1, 4, 4, 1], NUM_HIDDEN_LAYERS + 1, 1, False),
    ],
)
def test_resolve_layer_compress_ratio(
    compress_ratios: list[int],
    layer_id: int,
    expected_ratio: int,
    expected_unscaled: bool,
):
    ratio, unscaled = resolve_layer_compress_ratio(_config(compress_ratios), layer_id)
    assert ratio == expected_ratio
    assert unscaled == expected_unscaled
    assert ratio >= 1


def test_unscaled_rope_selects_plain_rotary_embedding(default_vllm_config):
    config = _config([1, 4, 4, 1, 0])
    rope = build_deepseek_v4_rope(
        config,
        head_dim=64,
        rope_head_dim=64,
        max_position_embeddings=config.max_position_embeddings,
        compress_ratio=1,
        use_unscaled_rope=True,
    )
    assert type(rope) is RotaryEmbedding
    assert not isinstance(rope, DeepseekScalingRotaryEmbedding)


def test_scaled_rope_unchanged_without_flag(default_vllm_config):
    config = _config([1, 4, 4, 1])
    rope = build_deepseek_v4_rope(
        config,
        head_dim=64,
        rope_head_dim=64,
        max_position_embeddings=config.max_position_embeddings,
        compress_ratio=1,
    )
    assert isinstance(rope, DeepseekV4ScalingRotaryEmbedding)


def test_attention_init_wires_the_resolver():
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    assert (
        "resolve_layer_compress_ratio" in DeepseekV4Attention.__init__.__code__.co_names
    )


def test_rope_parameters_dict_not_mutated(default_vllm_config):
    config = _config([1, 4, 4, 1, 0])
    snapshot = dict(config.rope_parameters)
    for compress_ratio, use_unscaled in ((1, True), (1, False), (4, False)):
        build_deepseek_v4_rope(
            config,
            head_dim=64,
            rope_head_dim=64,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=compress_ratio,
            use_unscaled_rope=use_unscaled,
        )
        assert config.rope_parameters == snapshot


def test_default_rope_cos_sin_cache_is_fp32_under_bf16_default(
    default_vllm_config,
):
    config = _config([1, 4, 4, 1, 0])
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        rope = build_deepseek_v4_rope(
            config,
            head_dim=64,
            rope_head_dim=64,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=1,
            use_unscaled_rope=True,
        )
    finally:
        torch.set_default_dtype(old_dtype)
    assert rope.cos_sin_cache.dtype == torch.float32
