# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`output_row_alignment`: an unquantised weight's output rows are zero-padded
after loading (Qwen3-Next / Qwen3.5 router: 512 experts + the shared-expert
gate row = 513 -> 528), and the existing output-padding fields hide it: the
forward returns the unpadded width, `out=` is refused, and row views stay
within the real rows.

The tensors are on the CPU and the GEMM is a torch matmul; `linear` itself
imports AITER, which needs the GPU environment to import.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "atom.model_ops.linear imports aiter, which resolves the chip "
        "architecture through rocminfo",
        allow_module_level=True,
    )

from atom.model_ops import linear

N_ROUTED, K = 512, 64


@pytest.fixture
def router(monkeypatch):
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear, "get_tp_group", lambda: group)
    calls = []

    def mm(x, w, bias=None, otype=None, out=None):
        calls.append(tuple(w.shape))
        return (x.float() @ w.float().t()).to(otype or x.dtype)

    monkeypatch.setattr(linear, "tgemm", SimpleNamespace(mm=mm))

    def build(alignment):
        layer = linear.MergedReplicatedLinear(
            K,
            [N_ROUTED, 1],
            bias=False,
            quant_config=None,
            prefix="mlp.gate",
            output_row_alignment=alignment,
        )
        torch.manual_seed(0)
        w = torch.randn(N_ROUTED + 1, K, dtype=torch.bfloat16)
        layer.weight_loader(layer.weight, w[:N_ROUTED], 0)
        layer.weight_loader(layer.weight, w[N_ROUTED:], 1)
        layer.process_weights_after_loading()
        return layer, w, calls

    return build


def test_router_pads_to_528_rows_and_returns_513_columns(router):
    layer, w, calls = router(16)
    assert layer.is_output_padded
    assert layer.weight.shape == (528, K)
    assert torch.equal(layer.weight[: N_ROUTED + 1], w)
    assert torch.all(layer.weight[N_ROUTED + 1 :] == 0)

    x = torch.randn(4, K, dtype=torch.bfloat16)
    y = layer(x)
    assert calls[-1] == (528, K)  # the GEMM runs on the padded weight
    assert y.shape == (4, N_ROUTED + 1) and y.stride(0) == 528
    torch.testing.assert_close(y, (x.float() @ w.float().t()).to(torch.bfloat16))

    # A second post-load pass (e.g. after a weight sync) keeps the padding.
    layer.process_weights_after_loading()
    assert layer.is_output_padded and layer.weight.shape == (528, K)


def test_padded_router_refuses_out_and_row_views_stay_unpadded(router):
    layer, w, _ = router(16)
    assert not layer.supports_out()
    out = torch.empty(4, N_ROUTED + 1, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        layer(torch.randn(4, K, dtype=torch.bfloat16), out=out)

    x = torch.randn(3, K, dtype=torch.bfloat16)
    shared = layer.shard_view(1)  # the shared-expert gate row
    routed = layer.make_row_view(0, N_ROUTED)
    assert not shared.is_output_padded and not routed.is_output_padded
    assert shared(x).shape == (3, 1) and routed(x).shape == (3, N_ROUTED)
    torch.testing.assert_close(
        shared(x), (x.float() @ w[N_ROUTED:].float().t()).to(torch.bfloat16)
    )
    with pytest.raises(AssertionError):
        layer.make_row_view(N_ROUTED, 16)  # reaches into the padding


def test_without_alignment_nothing_changes(router):
    layer, _, calls = router(None)
    assert not layer.is_output_padded
    assert layer.weight.shape == (N_ROUTED + 1, K)
    assert layer.supports_out()
    y = layer(torch.randn(2, K, dtype=torch.bfloat16))
    assert calls[-1] == (N_ROUTED + 1, K) and y.is_contiguous()
