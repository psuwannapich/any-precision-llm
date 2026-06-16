"""
worker.py — Phase 2 custom ModelRunner and Worker for per-request precision.

Strategy A (precision-grouped execution):
  • If all sequences in the batch share the same precision → single forward pass
    (zero overhead vs Phase 1).
  • If the batch contains mixed precisions → split into per-precision sub-batches,
    run each sub-batch with the correct precision, merge outputs.

Decode split is fully supported.
Prefill split falls back to the dominant precision (sequences of different
precisions that arrive simultaneously for prefill share the same forward pass
with the most-common precision among them).  This is rare in practice and has
no correctness issue beyond the quality of the minority-precision prefill.

Usage:
  bash run_vllm_p2.sh  (sets --worker-cls to this class)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch

from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker import Worker
from vllm.attention.backends.xformers import XFormersMetadata

from .precision_manager import PrecisionManager, DEFAULT_PRECISION


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_request_precision_map(model_input) -> Dict[str, int]:
    """Return {request_id: precision} for every request in this batch."""
    mgr = PrecisionManager.get()
    rids = list(model_input.request_ids_to_seq_ids.keys()) if model_input.request_ids_to_seq_ids else []
    return mgr.lookup_batch(rids)


def _seq_id_to_precision(model_input, req_prec: Dict[str, int]) -> Dict[int, int]:
    """Invert request→precision to seq_id→precision."""
    if not model_input.request_ids_to_seq_ids:
        return {}
    out: Dict[int, int] = {}
    for rid, seq_ids in model_input.request_ids_to_seq_ids.items():
        prec = req_prec.get(rid, DEFAULT_PRECISION)
        for sid in seq_ids:
            out[sid] = prec
    return out


def _decode_seq_order(model_input) -> List[int]:
    """
    Return an ordered list of seq_ids matching the decode token order in the
    flat input_tokens tensor.  For decode, sampling_metadata.seq_groups is
    ordered the same way as the batch.
    """
    sm = model_input.sampling_metadata
    if sm is None:
        return []
    order: List[int] = []
    for sg in sm.seq_groups:
        order.extend(sg.seq_ids)
    return order


def _dominant_precision(prec_list: List[int]) -> int:
    if not prec_list:
        return DEFAULT_PRECISION
    from collections import Counter
    return Counter(prec_list).most_common(1)[0][0]


def _slice_xformers_decode(meta: "XFormersMetadata", indices: List[int]) -> "XFormersMetadata":
    """
    Build a minimal XFormersMetadata for a subset of decode sequences.
    `indices` are positions in the original decode batch (0-based).
    """
    idx_t = torch.tensor(indices, dtype=torch.long, device=meta.slot_mapping.device)
    n = len(indices)
    return XFormersMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=n,
        slot_mapping=meta.slot_mapping[idx_t],
        multi_modal_placeholder_index_maps=None,
        seq_lens=None,
        seq_lens_tensor=torch.zeros(n, dtype=torch.int32, device=meta.slot_mapping.device),
        max_query_len=1,
        max_prefill_seq_len=0,
        max_decode_seq_len=meta.max_decode_seq_len,
        max_decode_query_len=1,
        query_start_loc=torch.arange(n + 1, dtype=torch.int32, device=meta.slot_mapping.device),
        seq_start_loc=torch.arange(n + 1, dtype=torch.int32, device=meta.slot_mapping.device),
        context_lens_tensor=meta.context_lens_tensor[idx_t],
        block_tables=(meta.block_tables[idx_t, :] if meta.block_tables is not None else None),
        use_cuda_graph=False,
    )


# ── custom ModelRunner ────────────────────────────────────────────────────────

class AnyPrecisionModelRunner(ModelRunner):
    """
    ModelRunner subclass that groups decode tokens by precision before
    calling the model's forward pass.
    """

    def execute_model(self, model_input, kv_caches, intermediate_tensors=None, num_steps=1):
        req_prec = _build_request_precision_map(model_input)

        # Fast path: no precision info (internal vLLM calls) or all same precision
        if not req_prec:
            return super().execute_model(model_input, kv_caches, intermediate_tensors, num_steps)

        precisions_in_batch: Set[int] = set(req_prec.values())
        if len(precisions_in_batch) == 1:
            prec = next(iter(precisions_in_batch))
            self.model.set_precision(prec)
            return super().execute_model(model_input, kv_caches, intermediate_tensors, num_steps)

        # Mixed precision batch
        is_decode = (
            model_input.attn_metadata is not None
            and model_input.attn_metadata.num_decode_tokens > 0
            and model_input.attn_metadata.num_prefill_tokens == 0
        )

        if not is_decode:
            # Prefill with mixed precisions: fall back to dominant precision
            sid_to_prec = _seq_id_to_precision(model_input, req_prec)
            dominant = _dominant_precision(list(sid_to_prec.values()))
            self.model.set_precision(dominant)
            return super().execute_model(model_input, kv_caches, intermediate_tensors, num_steps)

        # Mixed-precision decode: split batch, run per-precision, merge
        output = self._execute_decode_mixed(model_input, kv_caches, req_prec)
        return [output] if output is not None else []

    def _execute_decode_mixed(self, model_input, kv_caches, req_prec: Dict[str, int]):
        """
        Split the decode batch by precision, run each sub-batch, merge logits.
        """
        sid_to_prec = _seq_id_to_precision(model_input, req_prec)
        seq_order   = _decode_seq_order(model_input)  # ordered seq_ids

        # Map position index → precision
        pos_to_prec: Dict[int, int] = {}
        for pos, sid in enumerate(seq_order):
            pos_to_prec[pos] = sid_to_prec.get(sid, DEFAULT_PRECISION)

        # Group positions by precision
        prec_to_positions: Dict[int, List[int]] = {}
        for pos, prec in pos_to_prec.items():
            prec_to_positions.setdefault(prec, []).append(pos)

        # We'll collect per-position logits then reassemble
        N = model_input.attn_metadata.num_decode_tokens
        hidden_size = None
        output_per_pos: Dict[int, torch.Tensor] = {}

        meta_orig = model_input.attn_metadata

        for prec, positions in sorted(prec_to_positions.items()):
            self.model.set_precision(prec)

            # Build sub model_input
            idx_t = torch.tensor(positions, dtype=torch.long,
                                 device=model_input.input_tokens.device)

            # Slice token/position tensors
            sub_input = model_input.__class__(
                input_tokens=model_input.input_tokens[idx_t],
                input_positions=model_input.input_positions[idx_t],
                token_types=None,
                seq_lens=None,
                query_lens=None,
                lora_mapping=None,
                lora_requests=set(),
                attn_metadata=_slice_xformers_decode(meta_orig, positions),
                prompt_adapter_mapping=None,
                prompt_adapter_requests=set(),
                multi_modal_kwargs=None,
                request_ids_to_seq_ids={
                    rid: sids
                    for rid, sids in (model_input.request_ids_to_seq_ids or {}).items()
                    if req_prec.get(rid, DEFAULT_PRECISION) == prec
                },
                finished_requests_ids=[],
                virtual_engine=model_input.virtual_engine,
                async_callback=None,
                seq_group_metadata_list=None,
                scheduler_outputs=None,
                sampling_metadata=None,  # filled by super after forward
                is_prompt=False,
            )

            # Run just the forward pass (not sample) for this sub-batch
            with torch.inference_mode():
                from vllm.forward_context import set_forward_context
                from vllm.config import get_current_vllm_config
                try:
                    vllm_cfg = get_current_vllm_config()
                except Exception:
                    vllm_cfg = None

                if vllm_cfg:
                    ctx_mgr = set_forward_context(sub_input.attn_metadata, vllm_cfg)
                else:
                    from contextlib import nullcontext
                    ctx_mgr = nullcontext()

                with ctx_mgr:
                    hidden = self.model(
                        sub_input.input_tokens,
                        sub_input.input_positions,
                        kv_caches,
                        sub_input.attn_metadata,
                    )

            # Record per-position hidden states
            for local_i, global_pos in enumerate(positions):
                output_per_pos[global_pos] = hidden[local_i]
            if hidden_size is None:
                hidden_size = hidden.shape[-1]

        # Reconstruct full hidden tensor in original order
        device = model_input.input_tokens.device
        dtype  = next(iter(output_per_pos.values())).dtype
        hidden_full = torch.stack([output_per_pos[p] for p in range(N)], dim=0)

        # Now run compute_logits + sample on merged hidden states
        # Restore dominant precision for the sampler
        dominant = _dominant_precision(list(pos_to_prec.values()))
        self.model.set_precision(dominant)

        logits = self.model.compute_logits(hidden_full, model_input.sampling_metadata)
        if logits is None:
            return None
        output = self.model.sample(logits, model_input.sampling_metadata)
        return output


# ── custom Worker ─────────────────────────────────────────────────────────────

class AnyPrecisionWorker(Worker):
    """
    Worker subclass that injects AnyPrecisionModelRunner.

    vLLM's Worker.__init__ accepts `model_runner_cls` to swap the runner.
    We override __init__ to inject ours, then delegate everything else.
    """

    def __init__(self, vllm_config, local_rank, rank,
                 distributed_init_method, is_driver_worker=False,
                 model_runner_cls=None):
        # Inject precision-aware runner (wraps the default runner instance)
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
            model_runner_cls=model_runner_cls,
        )
        # Monkey-patch the model_runner class to our subclass in-place
        # (safer than passing a constructor since Worker may have already
        #  applied other wrappers via model_runner_cls).
        self.model_runner.__class__ = AnyPrecisionModelRunner
