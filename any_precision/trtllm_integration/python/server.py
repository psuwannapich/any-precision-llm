"""
server.py — OpenAI-compatible HTTP server for the Any-Precision TensorRT-LLM
engine, with **per-request bit-width selection**.

One engine serves every supported bit-width (3..8); the active precision is a
process-global plugin state, so the client picks it per request. Two ways:

  1. JSON field  "precision": 4   (works via the OpenAI SDK `extra_body=...`)
  2. model-name suffix             "anyprec@4bit"  / "anyprec:4" / "anyprec-4bit"
     (works with strict OpenAI clients that forbid extra fields)

Endpoints (OpenAI-compatible):
  GET  /v1/models
  POST /v1/chat/completions     (streaming + non-streaming)
  POST /v1/completions          (streaming + non-streaming)
  GET  /health

Because the precision is process-global and the TRT context is single-threaded,
all generation is serialized on ONE worker thread (a max_workers=1 executor).
set_precision + generate run as one task on that thread, so concurrent requests
with different precisions never race; they queue. (Throughput could later be
improved by batching same-precision requests, as the PyTorch serve.py does.)

Run:
  ./run_trtllm_serve.sh                       # wraps env + uvicorn
  # or directly:
  python -m any_precision.trtllm_integration.python.server \
      --engine_dir trt_engine --tokenizer_dir <packed ckpt> --port 8000
"""
import argparse
import asyncio
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .runtime import AnyPrecisionTRTRunner
from ._plugin_loader import set_precision

# ---------------------------------------------------------------------------
# Global state (populated in main()).
# ---------------------------------------------------------------------------
RUNNER: Optional[AnyPrecisionTRTRunner] = None
GEN_EXECUTOR: Optional[ThreadPoolExecutor] = None  # single thread -> serializes
MODEL_ID = "anyprec"


# ---------------------------------------------------------------------------
# Request / response schemas (the OpenAI-relevant subset).
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[ChatMessage]
    # custom Any-Precision extension: pick the weight bit-width for this request
    precision: Optional[int] = Field(
        default=None,
        description="Weight bit-width (3..8). Also settable via the model name "
                    "suffix, e.g. 'anyprec@4bit'. Defaults to the engine default.")
    max_tokens: Optional[int] = Field(default=128, ge=1)
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: Union[str, List[str]]
    precision: Optional[int] = None
    max_tokens: Optional[int] = Field(default=128, ge=1)
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    stream: Optional[bool] = False


# ---------------------------------------------------------------------------
# Precision resolution: explicit field > model-name suffix > engine default.
# ---------------------------------------------------------------------------
def _precision_from_model_name(model: str) -> Optional[int]:
    # accept "<name>@4bit", "<name>:4", "<name>-4bit", "<name>@4"
    for sep in ("@", ":"):
        if sep in model:
            tail = model.rsplit(sep, 1)[-1]
            tail = tail.lower().removesuffix("bit").strip()
            if tail.isdigit():
                return int(tail)
    tail = model.lower().rsplit("-", 1)[-1].removesuffix("bit")
    if tail.isdigit():
        return int(tail)
    return None


def resolve_precision(req_precision: Optional[int], model: str) -> int:
    assert RUNNER is not None
    p = req_precision
    if p is None:
        p = _precision_from_model_name(model)
    if p is None:
        p = RUNNER.default_precision
    if p not in RUNNER.supported_bits:
        raise HTTPException(
            status_code=400,
            detail=f"precision {p} not supported; choose one of "
                   f"{sorted(RUNNER.supported_bits)}")
    return p


# ---------------------------------------------------------------------------
# Generation primitives (run on the single generation thread).
# ---------------------------------------------------------------------------
def _encode_chat(messages: List[ChatMessage]) -> torch.Tensor:
    tok = RUNNER.tokenizer
    hf_msgs = [{"role": m.role, "content": m.content} for m in messages]
    if tok.chat_template:
        ids = tok.apply_chat_template(hf_msgs, add_generation_prompt=True,
                                      return_tensors="pt")[0]
    else:  # fallback: concatenate
        text = "".join(f"{m.role}: {m.content}\n" for m in messages) + "assistant:"
        ids = tok(text, return_tensors="pt").input_ids[0]
    return ids


def _sampling_kwargs(temperature: float, top_p: float, top_k: int) -> dict:
    tok = RUNNER.tokenizer
    kw = dict(end_id=tok.eos_token_id,
              pad_id=tok.pad_token_id or tok.eos_token_id)
    if not temperature or temperature <= 0.0:
        kw.update(temperature=1.0, top_k=1, top_p=1.0)   # greedy
    else:
        kw.update(temperature=float(temperature), top_p=float(top_p),
                  top_k=int(top_k) if top_k else 0)
    return kw


def _generate_blocking(prec, input_ids, max_tokens, samp):
    """Full (non-streaming) generation. Runs on the generation thread."""
    set_precision(prec)
    outputs = RUNNER.runner.generate(
        [input_ids.cuda()], max_new_tokens=max_tokens,
        return_dict=True, output_sequence_lengths=True, **samp)
    torch.cuda.synchronize()
    out_ids = outputs["output_ids"][0][0]
    seq_len = int(outputs["sequence_lengths"][0][0])
    gen_ids = out_ids[len(input_ids):seq_len]
    text = RUNNER.tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, len(input_ids), int(gen_ids.numel())


async def _run_full(prec, input_ids, max_tokens, samp):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        GEN_EXECUTOR, _generate_blocking, prec, input_ids, max_tokens, samp)


async def _run_stream(prec, input_ids, max_tokens, samp):
    """Async generator yielding text deltas. The blocking streaming generate
    runs on the generation thread and pushes decoded deltas into a queue. If the
    client disconnects, `cancel` is set and the worker stops generating (within
    one token) so it doesn't strand work on the single generation thread."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()
    prompt_len = len(input_ids)

    def worker():
        try:
            set_precision(prec)
            emitted = ""
            gen = RUNNER.runner.generate(
                [input_ids.cuda()], max_new_tokens=max_tokens, streaming=True,
                return_dict=True, output_sequence_lengths=True, **samp)
            n_gen = 0
            for step in gen:
                if cancel.is_set():
                    break
                out_ids = step["output_ids"][0][0]
                seq_len = int(step["sequence_lengths"][0][0])
                gen_ids = out_ids[prompt_len:seq_len]
                n_gen = int(gen_ids.numel())
                full = RUNNER.tokenizer.decode(gen_ids, skip_special_tokens=True)
                if len(full) > len(emitted):     # emit only the new suffix
                    delta = full[len(emitted):]
                    emitted = full
                    loop.call_soon_threadsafe(queue.put_nowait, ("delta", delta))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", n_gen))
        except Exception as e:  # surface errors to the stream
            loop.call_soon_threadsafe(queue.put_nowait, ("error", repr(e)))

    loop.run_in_executor(GEN_EXECUTOR, worker)
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "delta":
                yield payload
            elif kind == "done":
                return
            else:
                raise RuntimeError(payload)
    finally:
        cancel.set()  # client gone / generator closed -> stop the worker


# ---------------------------------------------------------------------------
# FastAPI app + endpoints.
# ---------------------------------------------------------------------------
app = FastAPI(title="Any-Precision TensorRT-LLM Server")


@app.get("/health")
def health():
    return {"status": "ok", "supported_bits": RUNNER.supported_bits if RUNNER else []}


@app.get("/v1/models")
def list_models():
    bits = RUNNER.supported_bits if RUNNER else []
    data = [{"id": MODEL_ID, "object": "model", "owned_by": "any-precision",
             "supported_precisions": bits, "default_precision":
             RUNNER.default_precision if RUNNER else None}]
    # also advertise per-precision aliases so clients can pick via model name
    for b in bits:
        data.append({"id": f"{MODEL_ID}@{b}bit", "object": "model",
                     "owned_by": "any-precision"})
    return {"object": "list", "data": data}


def _now() -> int:
    return int(time.time())


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if RUNNER is None:
        raise HTTPException(503, "model not loaded")
    prec = resolve_precision(req.precision, req.model)
    input_ids = _encode_chat(req.messages)
    samp = _sampling_kwargs(req.temperature, req.top_p, req.top_k)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model_label = f"{MODEL_ID}@{prec}bit"

    if req.stream:
        async def sse():
            head = {"id": cid, "object": "chat.completion.chunk",
                    "created": _now(), "model": model_label,
                    "choices": [{"index": 0, "delta": {"role": "assistant"},
                                 "finish_reason": None}]}
            yield f"data: {json.dumps(head)}\n\n"
            async for delta in _run_stream(prec, input_ids, req.max_tokens, samp):
                chunk = {"id": cid, "object": "chat.completion.chunk",
                         "created": _now(), "model": model_label,
                         "choices": [{"index": 0, "delta": {"content": delta},
                                      "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            tail = {"id": cid, "object": "chat.completion.chunk",
                    "created": _now(), "model": model_label,
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": "stop"}]}
            yield f"data: {json.dumps(tail)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    text, n_prompt, n_gen = await _run_full(prec, input_ids, req.max_tokens, samp)
    return {
        "id": cid, "object": "chat.completion", "created": _now(),
        "model": model_label, "precision": prec,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen,
                  "total_tokens": n_prompt + n_gen},
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if RUNNER is None:
        raise HTTPException(503, "model not loaded")
    if isinstance(req.prompt, list):
        if len(req.prompt) != 1:
            raise HTTPException(400, "only a single prompt is supported")
        prompt = req.prompt[0]
    else:
        prompt = req.prompt
    prec = resolve_precision(req.precision, req.model)
    input_ids = RUNNER.tokenizer(prompt, return_tensors="pt").input_ids[0]
    samp = _sampling_kwargs(req.temperature, req.top_p, req.top_k)
    cid = f"cmpl-{uuid.uuid4().hex[:24]}"
    model_label = f"{MODEL_ID}@{prec}bit"

    if req.stream:
        async def sse():
            async for delta in _run_stream(prec, input_ids, req.max_tokens, samp):
                chunk = {"id": cid, "object": "text_completion",
                         "created": _now(), "model": model_label,
                         "choices": [{"index": 0, "text": delta,
                                      "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    text, n_prompt, n_gen = await _run_full(prec, input_ids, req.max_tokens, samp)
    return {
        "id": cid, "object": "text_completion", "created": _now(),
        "model": model_label, "precision": prec,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen,
                  "total_tokens": n_prompt + n_gen},
    }


# ---------------------------------------------------------------------------
def main():
    global RUNNER, GEN_EXECUTOR, MODEL_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_dir", required=True)
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model_id", default="anyprec")
    ap.add_argument("--cpp_runtime", action="store_true")
    args = ap.parse_args()

    MODEL_ID = args.model_id
    # all CUDA work happens on this one thread -> consistent context + serialized
    GEN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="apgen")
    print(f"[server] loading engine from {args.engine_dir} ...")
    GEN_EXECUTOR.submit(_load_runner, args).result()
    print(f"[server] ready. model='{MODEL_ID}' precisions={RUNNER.supported_bits} "
          f"default={RUNNER.default_precision}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _load_runner(args):
    """Load the runner ON the generation thread so the CUDA context is owned by
    the same thread that later runs generate()."""
    global RUNNER
    RUNNER = AnyPrecisionTRTRunner(
        engine_dir=args.engine_dir, tokenizer_dir=args.tokenizer_dir,
        use_cpp_runtime=args.cpp_runtime)


if __name__ == "__main__":
    main()
