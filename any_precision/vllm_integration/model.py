"""
Any-Precision vLLM model integration — Phase 1 (single fixed precision).

Implements the vLLM model interface for Qwen3ForCausalLM backed by
AnyPrecisionLinear kernels. Replaces HuggingFace's transformers-based
forward pass with vLLM's PagedAttention, enabling:
  - Continuous batching  (requests join/leave mid-generation)
  - PagedAttention KV cache  (2MB pages, zero fragmentation)
  - CUDA graph capture  (fast decode loop)

Architecture: Qwen3-4B
  - 36 decoder layers
  - hidden=2560, heads=32, kv_heads=8, head_dim=128
  - intermediate=9728, vocab=151936
  - QK-norm per head (Qwen3-specific)
  - GQA (group_size=4)
  - RoPE theta=1_000_000

Precision: set via config.anyprec['active_precision'] (default: max/8-bit)
  To serve at 4-bit: add  "active_precision": 4  to config.json

Usage (after registering plugin):
  vllm serve <model_path> --dtype float16

Requirements: vllm >= 0.6.0
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from any_precision.modules.AnyPrecisionLinear import AnyPrecisionLinear

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.config import CacheConfig
    from vllm.model_executor.sampling_metadata import SamplingMetadata
    from vllm.sequence import IntermediateTensors


# ── Helpers ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Standard RMSNorm operating on the last dimension."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


# ── MLP ───────────────────────────────────────────────────────────────────────

class AnyPrecisionVllmMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, supported_bits: List[int]):
        super().__init__()
        self.gate_proj = AnyPrecisionLinear(hidden_size, intermediate_size, supported_bits, bias=False)
        self.up_proj   = AnyPrecisionLinear(hidden_size, intermediate_size, supported_bits, bias=False)
        self.down_proj = AnyPrecisionLinear(intermediate_size, hidden_size, supported_bits, bias=False)
        self.act_fn    = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ── Attention ─────────────────────────────────────────────────────────────────

class AnyPrecisionVllmAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_position_embeddings: int,
        rope_scaling: Optional[dict],
        supported_bits: List[int],
        layer_idx: int,
        cache_config: Optional["CacheConfig"],
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.scaling      = head_dim ** -0.5

        self.q_proj = AnyPrecisionLinear(hidden_size, num_heads    * head_dim, supported_bits, bias=False)
        self.k_proj = AnyPrecisionLinear(hidden_size, num_kv_heads * head_dim, supported_bits, bias=False)
        self.v_proj = AnyPrecisionLinear(hidden_size, num_kv_heads * head_dim, supported_bits, bias=False)
        self.o_proj = AnyPrecisionLinear(num_heads   * head_dim, hidden_size,  supported_bits, bias=False)

        # Per-head QK normalisation (Qwen3-specific)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        # Lazy-imported vLLM components (not available at import time on install-less runs)
        from vllm.model_executor.layers.rotary_embedding import get_rope
        from vllm.attention import Attention

        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # Detect Volta (SM 7.0) — vLLM 0.6's unified_attention Triton op requires
        # Ampere (SM 8.0+) for MMA layout conversions. Fall back to plain PyTorch
        # SDPA on Volta; PagedAttention is used on Ampere+ hardware.
        import torch.cuda as _cuda
        self._use_vllm_attn = _cuda.get_device_capability()[0] >= 8

        if self._use_vllm_attn:
            # prefix registers this Attention in vLLM's static forward context
            attn_prefix = f"{prefix}.attn" if prefix else f"model.layers.{layer_idx}.self_attn.attn"
            self.attn = Attention(
                num_heads=num_heads,
                head_size=head_dim,
                scale=self.scaling,
                num_kv_heads=num_kv_heads,
                cache_config=cache_config,
                prefix=attn_prefix,
            )
        else:
            self.attn = None  # will use _sdpa_forward on Volta

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        T = hidden_states.shape[0]

        q = self.q_proj(hidden_states)  # [T, nh * hd]
        k = self.k_proj(hidden_states)  # [T, nkv * hd]
        v = self.v_proj(hidden_states)  # [T, nkv * hd]

        # Per-head QK-norm: reshape → norm → flatten back
        q = self.q_norm(q.view(T, self.num_heads,    self.head_dim)).view(T, -1)
        k = self.k_norm(k.view(T, self.num_kv_heads, self.head_dim)).view(T, -1)

        # Rotary position embeddings
        q, k = self.rotary_emb(positions, q, k)

        if self._use_vllm_attn:
            # PagedAttention (Ampere+): manages KV cache internally
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        else:
            # Volta fallback: manual paged KV cache + PyTorch SDPA
            attn_output = self._sdpa_forward(q, k, v, T, kv_cache, attn_metadata)

        return self.o_proj(attn_output)  # [T, hidden]

    def _sdpa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
        kv_cache: torch.Tensor,
        attn_meta,
    ) -> torch.Tensor:
        """
        Pure PyTorch attention with manual paged KV-cache management.
        Used on Volta (SM 7.0) where vLLM's Triton unified_attention crashes.

        vLLM's actual kv_cache shape (XFormers/PagedAttention backend):
          [2, num_blocks, block_size * num_kv_heads * head_size]
          dim-0: 0=key, 1=value
          dim-1: block index
          dim-2: flat (block_size × nkv_heads × head_size)

        We reshape dim-2 to [block_size, nkv * hd] for indexed access.
        """
        stride    = self.num_kv_heads * self.head_dim   # nkv * hd per token slot
        # profile_run sends empty 1-D sentinels; real inference: ndim==3
        has_cache = kv_cache.ndim == 3 and kv_cache.numel() > 0
        if has_cache:
            num_blocks = kv_cache.shape[1]
            block_size = kv_cache.shape[2] // stride      # back-compute block_size

        k_hd = k.view(T, self.num_kv_heads, self.head_dim)  # [T, nkv, hd]
        v_hd = v.view(T, self.num_kv_heads, self.head_dim)

        # ── Write current tokens to paged cache ───────────────────────────
        if has_cache:
            slots = attn_meta.slot_mapping  # [T]
            valid = slots >= 0
            if valid.any():
                s = slots[valid]
                blk   = s // block_size                    # block index per token
                pos   = s % block_size                     # position within block
                # Reshape cache to [2, num_blocks, block_size, stride] for assignment
                kv_k = kv_cache[0].view(num_blocks, block_size, stride)
                kv_v = kv_cache[1].view(num_blocks, block_size, stride)
                kv_k[blk, pos] = k_hd[valid].reshape(-1, stride)
                kv_v[blk, pos] = v_hd[valid].reshape(-1, stride)

        # ── Prefill ────────────────────────────────────────────────────────
        if attn_meta.num_prefill_tokens > 0:
            seq_lens = attn_meta.seq_lens or [T]
            if len(seq_lens) == 1:
                q_ = q.view(1, T, self.num_heads, self.head_dim).transpose(1, 2)
                k_ = k_hd.unsqueeze(0).transpose(1, 2)
                v_ = v_hd.unsqueeze(0).transpose(1, 2)
                out = F.scaled_dot_product_attention(
                    q_, k_, v_, is_causal=True, enable_gqa=True
                )
                return out.transpose(1, 2).reshape(T, -1)
            else:
                parts, offset = [], 0
                for slen in seq_lens:
                    q_ = q[offset:offset+slen].view(1, slen, self.num_heads, self.head_dim).transpose(1, 2)
                    k_ = k_hd[offset:offset+slen].unsqueeze(0).transpose(1, 2)
                    v_ = v_hd[offset:offset+slen].unsqueeze(0).transpose(1, 2)
                    out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True, enable_gqa=True)
                    parts.append(out.transpose(1, 2).reshape(slen, -1))
                    offset += slen
                return torch.cat(parts, dim=0)

        # ── Decode ─────────────────────────────────────────────────────────
        # context_lens_tensor[i] = cached tokens BEFORE this step.
        # We already wrote this step's token → total to read = ctx_len + 1.
        num_seqs = attn_meta.num_decode_tokens

        if not has_cache or attn_meta.block_tables is None or attn_meta.block_tables.shape[1] == 0:
            # Profile run: attend only to current token
            q_ = q.view(num_seqs, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k_ = k_hd.view(num_seqs, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_ = v_hd.view(num_seqs, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=False, enable_gqa=True)
            return out.transpose(1, 2).reshape(num_seqs, -1)

        block_tables = attn_meta.block_tables        # [num_seqs, max_blocks]
        ctx_lens     = attn_meta.context_lens_tensor  # [num_seqs]
        kv_k = kv_cache[0].view(num_blocks, block_size, stride)
        kv_v = kv_cache[1].view(num_blocks, block_size, stride)

        # Batched decode: gather all sequences' KV context in one vectorized op,
        # then do a single SDPA call instead of N sequential kernel launches.
        total_lens = ctx_lens + 1                           # [num_seqs] — include token just written
        max_total  = max(int(total_lens.max().item()), 1)
        max_blks_needed = (max_total + block_size - 1) // block_size
        max_blks_needed = min(max_blks_needed, block_tables.shape[1])
        actual_max      = min(max_total, max_blks_needed * block_size)

        # Gather K and V from paged cache — one index op covers all seqs
        # blk_ids: [num_seqs * max_blks_needed]
        blk_ids   = block_tables[:, :max_blks_needed].reshape(-1)
        # gathered: [num_seqs, max_blks_needed * block_size, stride]
        gathered_k = kv_k[blk_ids].view(num_seqs, max_blks_needed * block_size, stride)
        gathered_v = kv_v[blk_ids].view(num_seqs, max_blks_needed * block_size, stride)

        # Slice to actual max context, then reshape to [num_seqs, nkv, actual_max, hd]
        k_batch = gathered_k[:, :actual_max].view(
            num_seqs, actual_max, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_batch = gathered_v[:, :actual_max].view(
            num_seqs, actual_max, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # q: [num_seqs, nh, 1, hd]
        q_batch = q.view(num_seqs, self.num_heads, 1, self.head_dim)

        # Mask padding positions (-inf) so softmax ignores them
        pos      = torch.arange(actual_max, device=q.device)
        pad_mask = pos.unsqueeze(0) >= total_lens.unsqueeze(1).clamp(max=actual_max)
        attn_mask = torch.zeros(num_seqs, 1, 1, actual_max, device=q.device, dtype=q.dtype)
        attn_mask.masked_fill_(pad_mask.view(num_seqs, 1, 1, actual_max), float('-inf'))

        out = F.scaled_dot_product_attention(
            q_batch, k_batch, v_batch, attn_mask=attn_mask, is_causal=False, enable_gqa=True
        )  # [num_seqs, nh, 1, hd]
        return out.reshape(num_seqs, self.num_heads * self.head_dim)


# ── Decoder layer ─────────────────────────────────────────────────────────────

class AnyPrecisionVllmDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, supported_bits: List[int],
                 cache_config: Optional["CacheConfig"], prefix: str = ""):
        super().__init__()
        attn_prefix = f"{prefix}.self_attn" if prefix else f"model.layers.{layer_idx}.self_attn"
        self.self_attn = AnyPrecisionVllmAttention(
            hidden_size            = config.hidden_size,
            num_heads              = config.num_attention_heads,
            num_kv_heads           = config.num_key_value_heads,
            head_dim               = config.head_dim,
            rms_norm_eps           = config.rms_norm_eps,
            rope_theta             = config.rope_theta,
            max_position_embeddings= config.max_position_embeddings,
            rope_scaling           = config.rope_scaling,
            supported_bits         = supported_bits,
            layer_idx              = layer_idx,
            cache_config           = cache_config,
            prefix                 = attn_prefix,
        )
        self.mlp = AnyPrecisionVllmMLP(
            hidden_size       = config.hidden_size,
            intermediate_size = config.intermediate_size,
            supported_bits    = supported_bits,
        )
        self.input_layernorm          = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ── Full transformer body ─────────────────────────────────────────────────────

class AnyPrecisionVllmModel(nn.Module):
    def __init__(self, config, supported_bits: List[int],
                 cache_config: Optional["CacheConfig"]):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            AnyPrecisionVllmDecoderLayer(config, i, supported_bits, cache_config,
                                         prefix=f"model.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states, kv_caches[i], attn_metadata)

        return self.norm(hidden_states)


# ── Top-level causal LM ───────────────────────────────────────────────────────

class AnyPrecisionQwen3ForCausalLM(nn.Module):
    """
    vLLM-compatible Qwen3ForCausalLM backed by any-precision quantization.

    Precision is read from config.anyprec['active_precision'] at load time.
    Default: max supported precision (8-bit).

    To change precision without reloading, call model.set_precision(bits).
    """

    # Tell vLLM this is the model class for "Qwen3ForCausalLM" checkpoints.
    supported_lora_modules: List[str] = []

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()

        # Unpack vLLM config (vllm >= 0.6)
        from vllm.config import VllmConfig
        assert isinstance(vllm_config, VllmConfig), (
            "AnyPrecisionQwen3ForCausalLM requires vllm >= 0.6 (VllmConfig interface)"
        )
        hf_config   = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        # Resolve supported precisions and active precision from anyprec metadata
        anyprec_cfg  = hf_config.anyprec
        seed         = anyprec_cfg["seed_precision"]
        parent       = anyprec_cfg["parent_precision"]
        supported_bits = list(range(seed, parent + 1))      # [3..8]
        active       = anyprec_cfg.get("active_precision", max(supported_bits))
        if active not in supported_bits:
            raise ValueError(
                f"active_precision={active} not in supported_bits={supported_bits}. "
                f"Update config.anyprec['active_precision'] in config.json."
            )

        self._supported_bits = supported_bits
        self._active_precision = active

        # Build model inside set_current_vllm_config so Attention layers
        # register themselves into vllm_config.compilation_config.static_forward_context
        from vllm.config import set_current_vllm_config
        with set_current_vllm_config(vllm_config):
            self.model   = AnyPrecisionVllmModel(hf_config, supported_bits, cache_config)

        # vLLM's LogitsProcessor requires ParallelLMHead (has .linear_method).
        # At TP=1 this behaves identically to nn.Linear.
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        self.lm_head = ParallelLMHead(hf_config.vocab_size, hf_config.hidden_size, bias=False)
        self.config  = hf_config

        # vLLM infrastructure layers
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.sampler import Sampler
        self.logits_processor = LogitsProcessor(hf_config.vocab_size)
        self.sampler = Sampler()

        # Set the initial precision on all AnyPrecisionLinear layers
        self._ap_linears = [
            m for m in self.modules() if isinstance(m, AnyPrecisionLinear)
        ]
        self.set_precision(active)

    # ── Precision control ─────────────────────────────────────────────────────

    def set_precision(self, precision: int) -> None:
        if precision not in self._supported_bits:
            raise ValueError(
                f"precision={precision} not in {self._supported_bits}"
            )
        for m in self._ap_linears:
            m.set_precision(precision)
        self._active_precision = precision

    @property
    def active_precision(self) -> int:
        return self._active_precision

    # ── vLLM interface ────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: "AttentionMetadata",
        intermediate_tensors: Optional["IntermediateTensors"] = None,
    ) -> Union[torch.Tensor, "IntermediateTensors"]:
        return self.model(input_ids, positions, kv_caches, attn_metadata)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: "SamplingMetadata",
    ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: "SamplingMetadata",
    ):
        return self.sampler(logits, sampling_metadata)

    # ── Weight loading ────────────────────────────────────────────────────────

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """
        Load weights from the any-precision packed checkpoint.

        Handles two tensor categories:
          - Normal fp16: embed_tokens, all RMSNorm weights
          - AP buffers (not nn.Parameter): qweight (int32) + lut{3..8} (fp16)
        """
        try:
            from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        except ImportError:
            def default_weight_loader(param, loaded):
                param.data.copy_(loaded)

        # Build a unified name→tensor map covering parameters AND buffers.
        # AnyPrecisionLinear registers qweight/luts as buffers, not parameters,
        # so named_parameters() alone would miss them.
        all_tensors: dict = {}
        for name, p in self.named_parameters(remove_duplicate=False):
            all_tensors[name] = p
        for name, b in self.named_buffers():
            if name not in all_tensors:
                all_tensors[name] = b

        loaded_names: set = set()
        for name, loaded_weight in weights:
            if name not in all_tensors:
                continue
            tensor = all_tensors[name]
            # dtype guard: qweight is int32 in both checkpoint and buffer
            if loaded_weight.dtype != tensor.dtype:
                loaded_weight = loaded_weight.to(tensor.dtype)
            weight_loader = getattr(tensor, "weight_loader", default_weight_loader)
            weight_loader(tensor, loaded_weight)
            loaded_names.add(name)

        # Weight tying: lm_head.weight == model.embed_tokens.weight
        # The checkpoint stores only embed_tokens; copy it to lm_head.
        if "lm_head.weight" not in loaded_names:
            self.lm_head.weight.data.copy_(self.model.embed_tokens.weight.data)

        # All LUTs (lut3..lut8) are kept loaded by default so the user can
        # call set_precision() at any time (Phase 2 per-request switching).
        # Call prune_to_single_precision() explicitly to free VRAM when only
        # one precision will ever be used.

    def prune_to_single_precision(self, precision: Optional[int] = None) -> None:
        """
        Drop all LUT buffers and qweight bit-planes except for one precision,
        freeing VRAM. After this call, only that precision can be used.

        Args:
            precision: bit-width to keep (default: current active_precision)
        """
        keep = precision if precision is not None else self._active_precision
        if keep not in self._supported_bits:
            raise ValueError(f"precision={keep} not in {self._supported_bits}")
        for m in self._ap_linears:
            m.qweight = m.qweight[:keep].contiguous()
            for bit in list(m.supported_bits):
                if bit != keep:
                    buf_name = f"lut{bit}"
                    if hasattr(m, buf_name):
                        delattr(m, buf_name)
            m.supported_bits = [keep]
            m.precisions = [keep]
        self._supported_bits = [keep]
        self._active_precision = keep
        torch.cuda.empty_cache()
