from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch

StrideSpec = Union[int, Tuple[int, int]]


def _normalize_stride(s: Optional[StrideSpec]) -> Optional[Tuple[int, int]]:
    if s is None:
        return None
    if isinstance(s, int):
        return (s, s)
    sh, sw = s
    return (int(sh), int(sw))

@dataclass
class CompressPlan:
    num_input: int
    num_output: int
    groups: List[List[int]]
    is_grid: bool = False
    grid_h: Optional[int] = None
    grid_w: Optional[int] = None
    input_h: Optional[int] = None
    input_w: Optional[int] = None
    pool_mode: str = "mean"
    stride: Optional[int] = None

    _assign_cpu: Optional[torch.Tensor] = field(default=None, repr=False)
    _group_size_cpu: Optional[torch.Tensor] = field(default=None, repr=False)
    _cached_device: Optional[torch.device] = field(default=None, repr=False)
    _assign_cuda: Optional[torch.Tensor] = field(default=None, repr=False)
    _group_size_cuda: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        if self._assign_cpu is None:
            assign = torch.zeros(self.num_input, dtype=torch.long)
            group_size = torch.zeros(self.num_output, dtype=torch.float32)
            for out_idx, grp in enumerate(self.groups):
                for in_idx in grp:
                    assign[in_idx] = out_idx
                group_size[out_idx] = len(grp)
            self._assign_cpu = assign
            self._group_size_cpu = group_size

    def _get_indices(self, device: torch.device):
        if self._assign_cpu is None:
            return None, None
        if self._cached_device is not None and self._cached_device == device:
            return self._assign_cuda, self._group_size_cuda
        self._assign_cuda = self._assign_cpu.to(device, non_blocking=True)
        self._group_size_cuda = self._group_size_cpu.to(device, non_blocking=True)
        self._cached_device = device
        return self._assign_cuda, self._group_size_cuda

    def apply(
        self,
        x: torch.Tensor,
        seq_dim: int = -2,
    ) -> torch.Tensor:
        ndim = x.ndim
        if seq_dim < 0:
            seq_dim = ndim + seq_dim
        assert x.shape[seq_dim] == self.num_input, (
            f"CompressPlan.apply: expected seq_dim={seq_dim} size={self.num_input}, "
            f"got {x.shape[seq_dim]}. tensor shape={x.shape}"
        )
        return self._apply_groups_mean(x, seq_dim)

    def _apply_groups_mean(self, x: torch.Tensor, seq_dim: int) -> torch.Tensor:
        device = x.device
        dtype = x.dtype

        x = x.movedim(seq_dim, 0)
        rest_shape = x.shape[1:]
        x_flat = x.reshape(self.num_input, -1).float()

        assign, group_size = self._get_indices(device)

        col_indices = torch.arange(self.num_input, device=device)
        indices = torch.stack([assign, col_indices], dim=0)

        values = 1.0 / group_size[assign].float().clamp(min=1)
        W = torch.sparse_coo_tensor(
            indices, values, size=(self.num_output, self.num_input)
        ).coalesce().to_sparse_csr()
        out_flat = torch.sparse.mm(W, x_flat)

        result = out_flat.reshape(self.num_output, *rest_shape).to(dtype)
        return result.movedim(0, seq_dim)

    def apply_long(self, x: torch.Tensor, seq_dim: int = -1) -> torch.Tensor:
        x_float = x.float()
        result = self.apply(x_float, seq_dim=seq_dim)
        return result.long()

class RingZipCompressor:
    def __init__(
        self,
        init_stride: int = 4,
        norm_temperature: Optional[float] = None,
    ):
        self.init_stride = init_stride
        self.norm_temperature = norm_temperature

    def _compute_cb(self, S_B_norm: float, W_B: float, K: int) -> float:
        if K <= 1:
            return 1.0
        return (S_B_norm / W_B) if W_B > 0 else 0.0

    def plan_compression(
        self,
        vis_features: torch.Tensor,
        h_tok: int,
        w_tok: int,
        outer_stride: Optional[StrideSpec] = None,
        first_split_stride: Optional[StrideSpec] = None,
        external_S_init: Optional[torch.Tensor] = None,
        external_W_init: Optional[torch.Tensor] = None,
    ) -> CompressPlan:
        device = vis_features.device
        num_input = h_tok * w_tok
        D = vis_features.shape[-1]

        features = vis_features.reshape(num_input, D).to(torch.float32)
        norms = features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if self.norm_temperature is None:
            features_normed = features / norms
        else:
            alpha = self.norm_temperature / (1.0 + self.norm_temperature)
            features_normed = features / norms.pow(alpha)

        feat_grid = features_normed.reshape(h_tok, w_tok, D)
        norm_grid_2d = feat_grid.norm(dim=-1)

        prefix_vec_gpu = torch.zeros(
            h_tok + 1, w_tok + 1, D,
            device=device, dtype=torch.float32,
        )
        prefix_vec_gpu[1:, 1:] = feat_grid.cumsum(0).cumsum(1)

        prefix_norm_gpu = torch.zeros(
            h_tok + 1, w_tok + 1,
            device=device, dtype=torch.float32,
        )
        prefix_norm_gpu[1:, 1:] = norm_grid_2d.cumsum(0).cumsum(1)

        if external_S_init is not None:
            S_init_gpu = external_S_init.to(
                device=features_normed.device,
                dtype=features_normed.dtype,
            )
        else:
            S_init_gpu = features_normed.sum(dim=0)   # (D,)
        if external_W_init is not None:
            W_init_gpu = external_W_init.to(
                device=norm_grid_2d.device,
                dtype=norm_grid_2d.dtype,
            )
        else:
            W_init_gpu = norm_grid_2d.sum()           # scalar

        outer_pair = _normalize_stride(outer_stride)
        if outer_pair is None:
            outer_pair = (self.init_stride, self.init_stride)
        sh_outer, sw_outer = outer_pair
        first_split_pair = _normalize_stride(first_split_stride)

        num_rows = (h_tok + sh_outer - 1) // sh_outer
        num_cols = (w_tok + sw_outer - 1) // sw_outer
        num_cells = num_rows * num_cols

        bi = torch.arange(num_rows, device=device)
        bj = torch.arange(num_cols, device=device)
        bi_grid, bj_grid = torch.meshgrid(bi, bj, indexing="ij")
        r_starts_gpu = bi_grid.flatten() * sh_outer            # (num_cells,)
        c_starts_gpu = bj_grid.flatten() * sw_outer            # (num_cells,)
        r_ends_gpu = torch.clamp(r_starts_gpu + sh_outer, max=h_tok)
        c_ends_gpu = torch.clamp(c_starts_gpu + sw_outer, max=w_tok)

        S_B_all_gpu = (
            prefix_vec_gpu[r_ends_gpu, c_ends_gpu]
            - prefix_vec_gpu[r_starts_gpu, c_ends_gpu]
            - prefix_vec_gpu[r_ends_gpu, c_starts_gpu]
            + prefix_vec_gpu[r_starts_gpu, c_starts_gpu]
        )   # (num_cells, D)

        W_B_all_gpu = (
            prefix_norm_gpu[r_ends_gpu, c_ends_gpu]
            - prefix_norm_gpu[r_starts_gpu, c_ends_gpu]
            - prefix_norm_gpu[r_ends_gpu, c_starts_gpu]
            + prefix_norm_gpu[r_starts_gpu, c_starts_gpu]
        )   # (num_cells,)

        K_all_gpu = ((r_ends_gpu - r_starts_gpu)
                     * (c_ends_gpu - c_starts_gpu))    # (num_cells,)
        S_B_norms_all_gpu = S_B_all_gpu.norm(dim=-1)    # (num_cells,)

        S_B_all = S_B_all_gpu.cpu()
        W_B_all_list = W_B_all_gpu.cpu().tolist()
        S_B_norms_all_list = S_B_norms_all_gpu.cpu().tolist()
        K_all_list = K_all_gpu.cpu().tolist()
        r_starts = r_starts_gpu.cpu().tolist()
        r_ends = r_ends_gpu.cpu().tolist()
        c_starts = c_starts_gpu.cpu().tolist()
        c_ends = c_ends_gpu.cpu().tolist()
        S_init = S_init_gpu.cpu()
        W_init = W_init_gpu.item()

        prefix_vec_cpu = None
        prefix_norm_cpu = None

        def _ensure_prefix_cpu():
            nonlocal prefix_vec_cpu, prefix_norm_cpu
            if prefix_vec_cpu is None:
                prefix_vec_cpu = prefix_vec_gpu.cpu()
                prefix_norm_cpu = prefix_norm_gpu.cpu()


        groups: List[List[int]] = []
        group_coherences: List[float] = []
        trace: List = []

        S_global = S_init.clone()
        W_global = W_init
        K_global = num_input
        C_global = self._compute_cb(S_init.norm().item(), W_init, K_global)

        for cell_idx in range(num_cells):
            r_s = r_starts[cell_idx]
            r_e = r_ends[cell_idx]
            c_s = c_starts[cell_idx]
            c_e = c_ends[cell_idx]
            K = K_all_list[cell_idx]
            W_B = W_B_all_list[cell_idx]
            S_B_norm = S_B_norms_all_list[cell_idx]

            if K <= 1:
                groups.append([r_s * w_tok + c_s])
                group_coherences.append(1.0)
                trace.append({
                    "r_start": r_s, "c_start": c_s,
                    "r_end": r_e, "c_end": c_e,
                    "stride_h": sh_outer, "stride_w": sw_outer,
                    "coherence": 1.0, "decision": "keep",
                })
                continue

            C_B = self._compute_cb(S_B_norm, W_B, K)

            S_B = S_B_all[cell_idx]
            pooled_norm = S_B_norm / K
            S_after = S_global - ((K - 1) / K) * S_B
            W_after = W_global - W_B + pooled_norm
            K_after = K_global - (K - 1)
            C_after = self._compute_cb(S_after.norm().item(), W_after, K_after)

            if C_after < C_global:
                indices = []
                for r in range(r_s, r_e):
                    row_base = r * w_tok
                    for c in range(c_s, c_e):
                        indices.append(row_base + c)
                groups.append(indices)
                group_coherences.append(C_B)

                S_global = S_after
                W_global = W_after
                C_global = C_after
                K_global = K_after

                trace.append({
                    "r_start": r_s, "c_start": c_s,
                    "r_end": r_e, "c_end": c_e,
                    "stride_h": sh_outer, "stride_w": sw_outer,
                    "coherence": C_B, "decision": "pool",
                })
            else:
                _ensure_prefix_cpu()
                state = {"S": S_global, "W": W_global, "C": C_global,
                         "K": K_global}
                trace.append({
                    "r_start": r_s, "c_start": c_s,
                    "r_end": r_e, "c_end": c_e,
                    "stride_h": sh_outer, "stride_w": sw_outer,
                    "coherence": C_B, "decision": "split",
                })
                self._split_and_recurse(
                    prefix_vec_cpu, prefix_norm_cpu, w_tok,
                    r_s, c_s, r_e, c_e,
                    r_e - r_s, c_e - c_s,
                    groups, group_coherences,
                    state, sh_outer, sw_outer,
                    trace=trace,
                    next_split_stride=first_split_pair,
                )
                S_global = state["S"]
                W_global = state["W"]
                C_global = state["C"]
                K_global = state["K"]

        num_output = len(groups)
        group_sizes_list = [len(g) for g in groups]
        flat_in_indices = [in_idx for g in groups for in_idx in g]
        if len(flat_in_indices) != num_input:
            raise RuntimeError(
                f"CompressPlan build: sum of group sizes ({len(flat_in_indices)}) "
                f"!= num_input ({num_input}). This indicates a bug in "
                f"plan_compression."
            )
        in_t = torch.tensor(flat_in_indices, dtype=torch.long)
        group_sizes_t = torch.tensor(group_sizes_list, dtype=torch.long)
        out_t = torch.repeat_interleave(
            torch.arange(num_output, dtype=torch.long),
            group_sizes_t,
        )

        assign_cpu = torch.empty(num_input, dtype=torch.long)
        assign_cpu[in_t] = out_t
        group_size_cpu = group_sizes_t.to(torch.float32)

        plan = CompressPlan(
            num_input=num_input,
            num_output=num_output,
            groups=groups,
            is_grid=False,
            input_h=h_tok,
            input_w=w_tok,
            _assign_cpu=assign_cpu,
            _group_size_cpu=group_size_cpu,
        )
        plan.group_coherences = group_coherences
        plan.recursion_trace = trace
        return plan

    @staticmethod
    def _query_block(
        prefix_vec: torch.Tensor,
        prefix_norm: torch.Tensor,
        r_start: int, c_start: int,
        r_end: int, c_end: int,
    ) -> Tuple[torch.Tensor, float]:
        S_B = (prefix_vec[r_end, c_end]
               - prefix_vec[r_start, c_end]
               - prefix_vec[r_end, c_start]
               + prefix_vec[r_start, c_start])
        W_B = (prefix_norm[r_end, c_end]
               - prefix_norm[r_start, c_end]
               - prefix_norm[r_end, c_start]
               + prefix_norm[r_start, c_start]).item()
        return S_B, W_B

    def _process_block(
        self,
        prefix_vec: torch.Tensor,
        prefix_norm: torch.Tensor,
        w_tok: int,
        r_start: int, c_start: int,
        r_end: int, c_end: int,
        groups: List[List[int]],
        group_coherences: List[float],
        state: dict,
        stride_h: int,
        stride_w: int,
        trace: Optional[List] = None,
        next_split_stride: Optional[Tuple[int, int]] = None,
    ):
        block_h = r_end - r_start
        block_w = c_end - c_start
        K = block_h * block_w

        if K <= 1:
            indices = [r_start * w_tok + c_start]
            groups.append(indices)
            group_coherences.append(1.0)
            if trace is not None:
                trace.append({
                    'r_start': r_start, 'c_start': c_start,
                    'r_end': r_end, 'c_end': c_end,
                    'stride_h': stride_h, 'stride_w': stride_w,
                    'coherence': 1.0, 'decision': 'keep',
                })
            return

        S_B, W_B = self._query_block(
            prefix_vec, prefix_norm, r_start, c_start, r_end, c_end)
        C_B = self._compute_cb(S_B.norm().item(), W_B, K)

        S_global = state['S']
        W_global = state['W']
        C_global = state['C']
        K_global = state.get('K', None)

        S_after = S_global - ((K - 1) / K) * S_B
        pooled_norm = S_B.norm().item() / K
        W_after = W_global - W_B + pooled_norm
        K_after = (K_global - (K - 1)) if K_global is not None else None
        if K_after is not None:
            C_after = self._compute_cb(S_after.norm().item(), W_after, K_after)
        else:
            C_after = (S_after.norm().item() / W_after) if W_after > 0 else C_global

        if C_after < C_global:
            indices = []
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    indices.append(r * w_tok + c)
            groups.append(indices)
            group_coherences.append(C_B)

            state['S'] = S_after
            state['W'] = W_after
            state['C'] = C_after
            if K_after is not None:
                state['K'] = K_after

            if trace is not None:
                trace.append({
                    'r_start': r_start, 'c_start': c_start,
                    'r_end': r_end, 'c_end': c_end,
                    'stride_h': stride_h, 'stride_w': stride_w,
                    'coherence': C_B, 'decision': 'pool',
                })
        else:
            if trace is not None:
                trace.append({
                    'r_start': r_start, 'c_start': c_start,
                    'r_end': r_end, 'c_end': c_end,
                    'stride_h': stride_h, 'stride_w': stride_w,
                    'coherence': C_B, 'decision': 'split',
                })
            self._split_and_recurse(
                prefix_vec, prefix_norm, w_tok,
                r_start, c_start, r_end, c_end,
                block_h, block_w,
                groups, group_coherences,
                state, stride_h, stride_w,
                trace=trace,
                next_split_stride=next_split_stride,
            )

    def _split_and_recurse(
        self,
        prefix_vec: torch.Tensor,
        prefix_norm: torch.Tensor,
        w_tok: int,
        r_start: int, c_start: int,
        r_end: int, c_end: int,
        block_h: int, block_w: int,
        groups: List[List[int]],
        group_coherences: List[float],
        state: dict,
        stride_h: int,
        stride_w: int,
        trace: Optional[List] = None,
        next_split_stride: Optional[Tuple[int, int]] = None,
    ):
        if next_split_stride is not None:
            ns_h, ns_w = next_split_stride
            new_sh = max(1, min(ns_h, block_h))
            new_sw = max(1, min(ns_w, block_w))
        else:
            sh = min(stride_h, block_h)
            sw = min(stride_w, block_w)
            new_sh = max(1, sh // 2)
            new_sw = max(1, sw // 2)

        if (next_split_stride is None
                and new_sh == sh and new_sw == sw):
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    groups.append([r * w_tok + c])
                    group_coherences.append(1.0)
                    if trace is not None:
                        trace.append({
                            'r_start': r, 'c_start': c,
                            'r_end': r + 1, 'c_end': c + 1,
                            'stride_h': new_sh, 'stride_w': new_sw,
                            'coherence': 1.0, 'decision': 'keep',
                        })
            return

        sub_rows = (block_h + new_sh - 1) // new_sh
        sub_cols = (block_w + new_sw - 1) // new_sw

        for si in range(sub_rows):
            for sj in range(sub_cols):
                sr = r_start + si * new_sh
                sc = c_start + sj * new_sw
                sr_end = min(sr + new_sh, r_end)
                sc_end = min(sc + new_sw, c_end)

                self._process_block(
                    prefix_vec, prefix_norm, w_tok,
                    sr, sc, sr_end, sc_end,
                    groups, group_coherences,
                    state,
                    new_sh, new_sw,
                    trace=trace,
                    next_split_stride=None,
                )

    def compress_hidden(
        self,
        vis_hidden: torch.Tensor,
        plan: CompressPlan,
    ) -> torch.Tensor:
        return plan.apply(vis_hidden, seq_dim=-2)

    def compress_positions(
        self,
        vis_pos_ids: torch.Tensor,
        plan: CompressPlan,
    ) -> torch.Tensor:
        return plan.apply_long(vis_pos_ids, seq_dim=-1)

    def compress_kv(
        self,
        vis_k: torch.Tensor,
        vis_v: torch.Tensor,
        plan: CompressPlan,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            plan.apply(vis_k, seq_dim=-2),
            plan.apply(vis_v, seq_dim=-2),
        )
