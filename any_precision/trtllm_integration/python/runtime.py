"""
runtime.py — load an Any-Precision TensorRT-LLM engine and run generation with
per-request precision selection.

The active bit-width is a process-global read by the plugin in enqueue(), so a
single engine serves every supported precision. Each generate() call sets the
precision before invoking the TRT-LLM runner; mixed-precision *within one batch*
is not supported (split the batch by precision, as the vLLM Phase-2 worker does).
"""

import json
import os
from typing import List, Optional

import torch

from ._plugin_loader import load_plugin, set_precision


class AnyPrecisionTRTRunner:
    def __init__(self, engine_dir: str, tokenizer_dir: Optional[str] = None,
                 use_cpp_runtime: bool = False):
        load_plugin()  # register plugin so the engine deserializes

        meta_path = os.path.join(engine_dir, "anyprec_meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.supported_bits = self.meta["supported_bits"]
        self.default_precision = self.meta.get("default_precision",
                                               self.meta["parent_bits"])

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir or engine_dir, trust_remote_code=True)

        if use_cpp_runtime:
            from tensorrt_llm.runtime import ModelRunnerCpp as Runner
        else:
            from tensorrt_llm.runtime import ModelRunner as Runner
        self.runner = Runner.from_dir(engine_dir=engine_dir)

    def _check_precision(self, precision: Optional[int]) -> int:
        p = precision if precision is not None else self.default_precision
        if p not in self.supported_bits:
            raise ValueError(
                f"precision {p} not in supported {self.supported_bits}")
        return p

    @torch.no_grad()
    def generate(self, prompt: str, precision: Optional[int] = None,
                 max_new_tokens: int = 128, temperature: float = 0.0,
                 top_p: float = 1.0) -> str:
        p = self._check_precision(precision)
        set_precision(p)  # <-- read by every AnyPrecisionLinear plugin

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        outputs = self.runner.generate(
            [input_ids.cuda()],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        out_ids = outputs[0][0][len(input_ids):]
        return self.tokenizer.decode(out_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(self, prompts: List[str], precision: Optional[int] = None,
                       max_new_tokens: int = 128, temperature: float = 0.0,
                       top_p: float = 1.0) -> List[str]:
        """Single-precision batch. For mixed precision, group by precision and
        call once per group."""
        p = self._check_precision(precision)
        set_precision(p)
        batch = [self.tokenizer(x, return_tensors="pt").input_ids[0].cuda()
                 for x in prompts]
        outputs = self.runner.generate(
            batch, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        results = []
        for i, inp in enumerate(batch):
            out_ids = outputs[i][0][len(inp):]
            results.append(self.tokenizer.decode(out_ids, skip_special_tokens=True))
        return results
