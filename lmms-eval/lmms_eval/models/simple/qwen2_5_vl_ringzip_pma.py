from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen2_5_vl_ringzip import (
    CompressPlan,
    Qwen2_5_VLRingZip,
)


def _scatter_softmax(
    scores: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    orig_dtype = scores.dtype
    if scores.dtype not in (torch.float32, torch.float64):
        scores = scores.float()

    rest = scores.shape[1:]
    idx_view = index.view(-1, *([1] * len(rest))).expand_as(scores)

    max_per_group = torch.full(
        (dim_size, *rest), float("-inf"),
        device=scores.device, dtype=scores.dtype,
    )
    max_per_group.scatter_reduce_(
        0, idx_view, scores, reduce="amax", include_self=True,
    )
    stable = scores - max_per_group.gather(0, idx_view)
    exp_s = stable.exp()

    sum_per_group = torch.zeros(
        (dim_size, *rest), device=scores.device, dtype=scores.dtype,
    )
    sum_per_group.scatter_add_(0, idx_view, exp_s)

    out = exp_s / sum_per_group.gather(0, idx_view).clamp(min=1e-12)
    return out.to(orig_dtype)


def _scatter_sum(
    values: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    rest = values.shape[1:]
    idx_view = index.view(-1, *([1] * len(rest))).expand_as(values)
    out = torch.zeros(
        (dim_size, *rest), device=values.device, dtype=values.dtype,
    )
    out.scatter_add_(0, idx_view, values)
    return out


class AttnPool1(nn.Module):

    def __init__(self, d: int, n_heads: int = 8):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} not divisible by n_heads={n_heads}")
        self.d = d
        self.h = n_heads
        self.dh = d // n_heads

        self.q = nn.Parameter(torch.zeros(d))
        self.K = nn.Linear(d, d, bias=False)
        self.V = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

        nn.init.zeros_(self.q)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.eye_(self.V.weight)
        nn.init.eye_(self.out.weight)

    def forward(
        self,
        feats: torch.Tensor,
        region_ids: torch.Tensor,
        num_regions: int,
    ) -> torch.Tensor:
        if feats.ndim != 2 or feats.shape[-1] != self.d:
            raise ValueError(
                f"feats must be (N, {self.d}), got {tuple(feats.shape)}"
            )

        N = feats.shape[0]
        K_feat = self.K(feats).view(N, self.h, self.dh)
        V_feat = self.V(feats).view(N, self.h, self.dh)
        q = self.q.view(1, self.h, self.dh)

        scores = (K_feat * q).sum(dim=-1) / math.sqrt(self.dh)
        alpha = _scatter_softmax(scores, region_ids, num_regions)
        weighted = alpha.unsqueeze(-1) * V_feat
        pooled_heads = _scatter_sum(weighted, region_ids, num_regions)
        pooled = pooled_heads.reshape(num_regions, self.d)
        return self.out(pooled)


class _PMACompressHidden:

    def __init__(self, pma: AttnPool1):
        self.pma = pma

    @torch.no_grad()
    def __call__(
        self,
        vis_hidden: torch.Tensor,
        plan: CompressPlan,
    ) -> torch.Tensor:
        if vis_hidden.ndim != 3:
            raise ValueError(
                f"PMA compress_hidden expects (B, N, D), got "
                f"{tuple(vis_hidden.shape)}"
            )
        B, N, D = vis_hidden.shape
        if B != 1:
            raise ValueError(
                f"PMA compress_hidden only supports B=1, got B={B}"
            )
        if N != plan.num_input:
            raise ValueError(
                f"vis_hidden N={N} != plan.num_input={plan.num_input}"
            )

        feats = vis_hidden.squeeze(0)

        target_dtype = self.pma.K.weight.dtype
        cast_back_dtype = feats.dtype
        if feats.dtype != target_dtype:
            feats = feats.to(target_dtype)

        group_ids = plan._assign_cpu.to(feats.device, non_blocking=True)
        pooled = self.pma(feats, group_ids, plan.num_output)

        if pooled.dtype != cast_back_dtype:
            pooled = pooled.to(cast_back_dtype)
        return pooled.unsqueeze(0)


def _hidden_size(config) -> int:
    for attr in ("hidden_size",):
        val = getattr(config, attr, None)
        if val is not None:
            return int(val)
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        val = getattr(text_cfg, "hidden_size", None)
        if val is not None:
            return int(val)
    raise AttributeError(
        "hidden_size missing from config (checked top-level and text_config)."
    )


def _load_pma_state(pma: AttnPool1, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    missing, unexpected = pma.load_state_dict(state, strict=False)
    if missing:
        eval_logger.warning(f"PMA load: missing keys: {missing}")
    if unexpected:
        eval_logger.warning(f"PMA load: unexpected keys: {unexpected}")
    eval_logger.info(f"📦 Loaded PMA state_dict: {path}")


@register_model("qwen2_5_vl_ringzip_pma")
class Qwen2_5_VLRingZipPMA(Qwen2_5_VLRingZip):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        pma_checkpoint: Optional[str] = None,
        pma_n_heads: int = 8,
        pma_dtype: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained, **kwargs)

        hidden_size = _hidden_size(self.model.config)

        if pma_dtype is not None:
            target_dtype = {
                "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                "fp16": torch.float16, "float16": torch.float16,
                "fp32": torch.float32, "float32": torch.float32,
            }[str(pma_dtype).lower()]
        else:
            target_dtype = next(self._model.parameters()).dtype

        pma = AttnPool1(d=hidden_size, n_heads=int(pma_n_heads))

        if pma_checkpoint is not None and str(pma_checkpoint).lower() != "none":
            pma_path = str(pma_checkpoint)
            if os.path.isdir(pma_path):
                candidate = Path(pma_path) / "pma_state_dict.pt"
                if candidate.exists():
                    pma_path = str(candidate)
                else:
                    raise FileNotFoundError(
                        f"pma_state_dict.pt not found under {pma_path}"
                    )
            elif not os.path.exists(pma_path):
                raise FileNotFoundError(f"pma_checkpoint not found: {pma_path}")
            _load_pma_state(pma, pma_path)
        else:
            eval_logger.warning(
                "PMA running with mean-pool-equivalent init — you probably "
                "want to pass pma_checkpoint pointing at a trained state_dict."
            )

        pma.to(device=self.device, dtype=target_dtype)
        pma.eval()
        for p in pma.parameters():
            p.requires_grad = False
        self.pma = pma

        eval_logger.info(
            f"🎯 PMA-1 initialized: d={hidden_size}, heads={pma_n_heads}, "
            f"dtype={target_dtype}, device={self.device}"
        )

        self.compressor.compress_hidden = _PMACompressHidden(self.pma)
        eval_logger.info(
            "🔧 Patched RingZipCompressor.compress_hidden -> PMA-1 pool"
        )
