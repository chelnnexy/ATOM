# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Softmax top-k router, optionally with sigmoid-scored fused shared experts.

Drop-in for AITER ``topk_softmax`` (``vllm::moe::topkGatingSoftmax``), one
Triton program (one wave) per token row. Same expert ids and weights up to fp32
rounding whenever at least K routed logits per row are finite; the differences
are listed at the end.

* routed weights: softmax over the first ``N`` logits, top-``K`` by logit
  (lowest expert id wins ties), renormalised over the K winners (or over the
  whole row when ``renormalize`` is False);
* fused shared experts: ``weights[:, K + s] = sigmoid(logits[:, N + s])``.
  The shared-expert *ids* are not written: ``init_aiter_topK_meta_data`` presets
  them once in the shared buffer, exactly as for the AITER kernel;
* padding rows: with ``num_valid_rows`` (``[1]`` int32 on device), rows at or
  past it get routed ids -1 and routed weights 0, the shared columns are
  written as for any row (AITER's ``num_valid_rows`` contract).

Each selection round takes the row maximum, then the lowest still-unselected
expert id holding it (the rule of ``_route_slot`` in
``atom/model_ops/qwen4_exp/ops/moe_decode.py``). Selected experts are tracked
in a mask rather than overwritten with -inf.

Differences from AITER:

* fewer than K finite routed logits in a row: AITER overwrites each pick with
  -inf, so once only -inf remains its next picks are the lowest ids holding
  -inf, which can repeat an expert it already picked. Here the remaining picks
  are the lowest *unselected* ids holding -inf, so a row never repeats an id.
  Their weights are 0 in both.
* NaN logits rank as -inf here; AITER's comparisons skip a NaN except as the
  first value a thread reads, so where a NaN ends up is not defined there.

The row-per-wave layout follows SGLang's ``_router_triton_kernel``
(sglang/kernels/ops/moe/moe_fused_gate.py, Apache-2.0, Copyright the SGLang
team).
"""

import torch
import triton
import triton.language as tl

from atom.utils.custom_register import direct_register_custom_op


@triton.jit
def _topk_softmax_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    num_valid_rows_ptr,
    stride_lm,
    stride_ln,
    stride_wm,
    stride_im,
    N: tl.constexpr,  # routed experts: the softmax / top-k domain
    K: tl.constexpr,  # routed top-k
    NUM_SHARED: tl.constexpr,  # sigmoid-scored shared logits at [N, N + NUM_SHARED)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # >= K + NUM_SHARED
    RENORMALIZE: tl.constexpr,
    HAS_NUM_VALID_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    logits_row = logits_ptr + row.to(tl.int64) * stride_lm

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    x = tl.load(logits_row + offs_n * stride_ln, mask=mask_n, other=-float("inf"))
    x = x.to(tl.float32)
    x = tl.where(x >= -float("inf"), x, -float("inf"))  # NaN (unordered) -> -inf

    offs_k = tl.arange(0, BLOCK_K)
    sel_val = tl.full([BLOCK_K], -float("inf"), tl.float32)
    sel_idx = tl.zeros([BLOCK_K], dtype=tl.int32)
    free = mask_n
    for k in tl.static_range(K):
        best = tl.max(tl.where(free, x, -float("inf")), axis=0)
        idx = tl.min(tl.where(free & (x == best), offs_n, BLOCK_N), axis=0)
        free = free & (offs_n != idx)
        sel_val = tl.where(offs_k == k, best, sel_val)
        sel_idx = tl.where(offs_k == k, idx, sel_idx)

    row_max = tl.max(sel_val, axis=0)
    numer = tl.exp(sel_val - row_max)  # 0 outside the K routed slots
    if RENORMALIZE:
        denom = tl.sum(numer, axis=0)
    else:
        denom = tl.sum(tl.where(mask_n, tl.exp(x - row_max), 0.0), axis=0)
    weights = numer / denom

    if HAS_NUM_VALID_ROWS:
        pad = row >= tl.load(num_valid_rows_ptr)
        sel_idx = tl.where(pad, -1, sel_idx)
        weights = tl.where(pad, 0.0, weights)

    if NUM_SHARED > 0:
        is_shared = (offs_k >= K) & (offs_k < K + NUM_SHARED)
        s_logit = tl.load(
            logits_row + (N + offs_k - K) * stride_ln, mask=is_shared, other=0.0
        )
        weights = tl.where(is_shared, tl.sigmoid(s_logit.to(tl.float32)), weights)

    tl.store(
        weights_ptr + row.to(tl.int64) * stride_wm + offs_k,
        weights,
        mask=offs_k < K + NUM_SHARED,
    )
    tl.store(ids_ptr + row.to(tl.int64) * stride_im + offs_k, sel_idx, mask=offs_k < K)


def triton_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_shared_scored: int = 0,
    num_valid_rows: torch.Tensor | None = None,
) -> None:
    """Fill ``topk_weights[:, :topk + num_shared_scored]`` and ``topk_ids[:, :topk]``.

    ``gating_output`` is ``[M, N + num_shared_scored]`` with any strides;
    outputs may be strided row views (e.g. of the fused-shared buffers) but
    must be unit-stride along columns.
    """
    M, n_total = gating_output.shape
    n_routed = n_total - num_shared_scored
    assert topk_weights.stride(1) == 1 and topk_ids.stride(1) == 1
    assert topk_weights.dtype == torch.float32 and topk_ids.dtype == torch.int32
    assert topk_weights.shape[0] >= M and topk_ids.shape[0] >= M
    assert topk_weights.shape[1] >= topk + num_shared_scored
    assert topk_ids.shape[1] >= topk and topk <= n_routed
    if M == 0:
        return
    _topk_softmax_kernel[(M,)](
        gating_output,
        topk_weights,
        topk_ids,
        num_valid_rows,
        gating_output.stride(0),
        gating_output.stride(1),
        topk_weights.stride(0),
        topk_ids.stride(0),
        N=n_routed,
        K=topk,
        NUM_SHARED=num_shared_scored,
        BLOCK_N=triton.next_power_of_2(n_routed),
        BLOCK_K=triton.next_power_of_2(topk + num_shared_scored),
        RENORMALIZE=renormalize,
        HAS_NUM_VALID_ROWS=num_valid_rows is not None,
        num_warps=1,
    )


def _triton_topk_softmax_fake(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_shared_scored: int = 0,
    num_valid_rows: torch.Tensor | None = None,
) -> None:
    return None


# Opaque to Dynamo (the MoE routing runs inside compiled model forwards).
direct_register_custom_op(
    op_name="atom_triton_topk_softmax",
    op_func=triton_topk_softmax,
    mutates_args=["topk_weights", "topk_ids"],
    fake_impl=_triton_topk_softmax_fake,
)
triton_topk_softmax_op = torch.ops.aiter.atom_triton_topk_softmax
