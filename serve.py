"""
Any-Precision LLM — Production-Optimized Server (Option 1a)

Optimizations over naive HuggingFace serving:

  1. Dynamic batching  — requests with the same precision are grouped and
                         executed as a single batched GPU call, directly
                         improving throughput proportional to batch size.
  2. Async request queue — no asyncio.Lock; a dedicated engine loop drains
                           the queue and builds batches, enabling proper
                           request timeouts and backpressure.
  3. Batched streaming  — every request in a batch streams token-by-token
                          via per-request asyncio.Queue fed by a custom
                          HF BaseStreamer running in the generation thread.
  4. torch.compile (opt-in via --compile) — ~10-30% free speedup on the
                          attention/norm layers outside the custom kernels.
  5. /v1/metrics        — live throughput, latency, queue depth.

Endpoints:
  GET  /health
  GET  /v1/models
  GET  /v1/info
  GET  /v1/metrics
  POST /v1/chat/completions   (streaming + non-streaming, custom "precision" field)

Usage:
  python serve.py [--model_path PATH] [--host HOST] [--port PORT]
                  [--max_batch_size N] [--batch_wait_ms N]
                  [--request_timeout_s N] [--compile]
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List, Literal, Optional, Union

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s | %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_MODEL = (
    "/mnt/scratch/users/psuwannapichat/any-precision-llm/cache/packed/"
    "anyprec-(Qwen3-4B)-w8_orig3-gc1-c4_s100_blk512"
)

# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    precision: Optional[int] = Field(
        default=None,
        description="Bit-width (3–8). Defaults to model max precision.",
    )
    max_tokens: Optional[int] = Field(default=512, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Batched streaming
# ──────────────────────────────────────────────────────────────────────────────

class AsyncBatchStreamer(BaseStreamer):
    """
    Custom HF BaseStreamer that fans out generated tokens to per-request
    asyncio queues, enabling streaming for every request inside a batch.

    Queue messages:  ("token", str)  |  ("done", None)  |  ("error", str)

    Contract with transformers generate():
      - put() first call: prompt ids [batch_size, prompt_len]  → skipped
      - put() each step:  next token  [batch_size]             → decoded & routed
      - end():            generation finished                  → sends ("done", None)
    """

    def __init__(self, tokenizer, batch_size: int,
                 queues: list,  # List[Optional[asyncio.Queue]]
                 loop: asyncio.AbstractEventLoop):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.queues = queues
        self.loop = loop
        self._skip_prompt = True

    def _send(self, queue, msg):
        asyncio.run_coroutine_threadsafe(queue.put(msg), self.loop)

    def put(self, value: torch.Tensor):
        if self._skip_prompt:
            self._skip_prompt = False
            return  # first call is the prompt; skip

        # value: [batch_size] — one new token per sequence
        if value.dim() > 1:
            value = value[:, -1]

        for i, token_id in enumerate(value.tolist()):
            q = self.queues[i]
            if q is None:
                continue
            tid = int(token_id)
            if tid == self.tokenizer.eos_token_id:
                continue  # end() will handle the "done" signal
            text = self.tokenizer.decode([tid], skip_special_tokens=True)
            if text:
                self._send(q, ("token", text))

    def end(self):
        for q in self.queues:
            if q is not None:
                self._send(q, ("done", None))


# ──────────────────────────────────────────────────────────────────────────────
# Pending request
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class PendingRequest:
    request_id: str
    precision: int
    input_ids: torch.Tensor          # [1, prompt_len]  on CPU
    prompt_len: int
    max_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    stop_token_ids: List[int]
    stream: bool
    enqueued_at: float = dataclasses.field(default_factory=time.perf_counter)

    # Delivery mechanism — exactly one of these is set
    result_future: Optional[asyncio.Future] = None   # non-streaming
    token_queue: Optional[asyncio.Queue] = None      # streaming


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self.requests_total = 0
        self.requests_failed = 0
        self.tokens_generated = 0
        self.batches_processed = 0
        self._latencies: List[float] = []  # last 1000 request latencies (s)
        self._lock = threading.Lock()

    def record_batch(self, n_requests: int, n_tokens: int):
        with self._lock:
            self.requests_total += n_requests
            self.tokens_generated += n_tokens
            self.batches_processed += 1

    def record_latency(self, latency_s: float):
        with self._lock:
            self._latencies.append(latency_s)
            if len(self._latencies) > 1000:
                self._latencies.pop(0)

    def record_error(self):
        with self._lock:
            self.requests_failed += 1

    def snapshot(self, queue_depth: int) -> dict:
        with self._lock:
            lats = self._latencies
            return {
                "requests_total": self.requests_total,
                "requests_failed": self.requests_failed,
                "tokens_generated": self.tokens_generated,
                "batches_processed": self.batches_processed,
                "queue_depth": queue_depth,
                "avg_latency_s": round(sum(lats) / len(lats), 3) if lats else None,
                "p50_latency_s": round(sorted(lats)[len(lats) // 2], 3) if lats else None,
                "p99_latency_s": round(sorted(lats)[int(len(lats) * 0.99)], 3) if lats else None,
            }


# ──────────────────────────────────────────────────────────────────────────────
# Generation engine
# ──────────────────────────────────────────────────────────────────────────────

class GenerationEngine:
    """
    Async engine that processes incoming requests from a queue, groups them
    into batches by (precision, do_sample), and dispatches batched GPU calls.

    Throughput vs. latency knobs:
      max_batch_size  — hard cap on requests per batch  (more = better throughput)
      batch_wait_ms   — window to wait for a full batch (more = better batching,
                        worse per-request latency)
    """

    def __init__(self, model, tokenizer, device: str,
                 max_batch_size: int = 8,
                 batch_wait_ms: float = 20.0,
                 request_timeout_s: float = 60.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.batch_wait_ms = batch_wait_ms
        self.request_timeout_s = request_timeout_s
        self.metrics = Metrics()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._loop: asyncio.AbstractEventLoop = None

    # ── Public API ──

    async def submit(self, req: PendingRequest) -> PendingRequest:
        """Enqueue a request. Raises HTTPException(503) if queue is full."""
        try:
            self._queue.put_nowait(req)
        except asyncio.QueueFull:
            self.metrics.record_error()
            raise HTTPException(status_code=503,
                                detail="Server overloaded — request queue full.")
        return req

    async def run(self):
        """Main engine loop — runs as an asyncio Task during server lifetime."""
        self._loop = asyncio.get_running_loop()
        logging.info("GenerationEngine started.")
        while True:
            batch = await self._collect_batch()
            if batch:
                await self._loop.run_in_executor(None, self._run_batch_sync, batch)

    # ── Batch collection ──

    async def _collect_batch(self) -> List[PendingRequest]:
        """
        Wait for the first request, then collect more for up to batch_wait_ms
        or until max_batch_size is reached.  All requests in a batch share the
        same (precision, do_sample) key — others go back to the queue.
        """
        first = await self._queue.get()

        # Expire requests that have been waiting too long
        if time.perf_counter() - first.enqueued_at > self.request_timeout_s:
            self._deliver_error(first, TimeoutError("Request timed out in queue."))
            return []

        batch = [first]
        key = (first.precision, first.do_sample)

        deadline = asyncio.get_event_loop().time() + self.batch_wait_ms / 1000.0
        overflow: List[PendingRequest] = []

        while len(batch) < self.max_batch_size:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                req = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(min(remaining, 0.005))
                continue

            if (req.precision, req.do_sample) == key:
                if time.perf_counter() - req.enqueued_at > self.request_timeout_s:
                    self._deliver_error(req, TimeoutError("Request timed out in queue."))
                else:
                    batch.append(req)
            else:
                overflow.append(req)

        # Return different-key requests to the front of the queue
        for req in reversed(overflow):
            try:
                self._queue.put_nowait(req)
            except asyncio.QueueFull:
                self._deliver_error(req, RuntimeError("Queue full while requeueing."))

        return batch

    # ── Batch execution (runs in thread pool, not event loop) ──

    def _run_batch_sync(self, batch: List[PendingRequest]):
        batch_size = len(batch)
        precision = batch[0].precision
        t_start = time.perf_counter()

        # ── 1. Left-pad inputs to uniform length ──
        max_len = max(r.prompt_len for r in batch)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        padded_ids, masks = [], []
        for r in batch:
            pad = max_len - r.prompt_len
            padded_ids.append(F.pad(r.input_ids, (pad, 0), value=pad_id))
            masks.append(F.pad(torch.ones_like(r.input_ids), (pad, 0), value=0))

        input_ids = torch.cat(padded_ids, dim=0).to(self.device)
        attention_mask = torch.cat(masks, dim=0).to(self.device)

        # ── 2. Generation parameters (batch-level) ──
        max_new_tokens = max(r.max_tokens for r in batch)
        do_sample = batch[0].do_sample
        # Use the first request's temperature/top_p for the whole batch.
        # (Per-sequence sampling via logit processors is a future improvement.)
        temperature = batch[0].temperature if do_sample else 1.0
        top_p = batch[0].top_p if do_sample else 1.0

        stop_ids = list({
            tid for r in batch for tid in r.stop_token_ids
        }) + [self.tokenizer.eos_token_id]

        # ── 3. Streaming setup ──
        token_queues = [r.token_queue for r in batch]  # None for non-streaming
        streamer = None
        if any(q is not None for q in token_queues):
            streamer = AsyncBatchStreamer(
                self.tokenizer, batch_size, token_queues, self._loop
            )

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_id,
            eos_token_id=stop_ids,
            precision=precision,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )
        if streamer is not None:
            gen_kwargs["streamer"] = streamer

        # ── 4. Generate ──
        try:
            with torch.no_grad():
                output_ids = self.model.generate(**gen_kwargs)
        except Exception as exc:
            logging.error(f"Generation error (batch_size={batch_size}): {exc}")
            for r in batch:
                self._deliver_error(r, exc)
            self.metrics.record_error()
            return

        # ── 5. Deliver non-streaming results; count tokens for all ──
        total_new = 0
        eos = self.tokenizer.eos_token_id
        for i, r in enumerate(batch):
            gen_ids = output_ids[i][max_len:].tolist()
            if eos in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(eos)]
            total_new += len(gen_ids)
            if r.result_future is not None:
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                self._loop.call_soon_threadsafe(
                    r.result_future.set_result, (text, len(gen_ids))
                )

        # ── 6. Metrics ──
        elapsed = time.perf_counter() - t_start
        self.metrics.record_batch(batch_size, total_new)
        for r in batch:
            self.metrics.record_latency(elapsed)
        logging.info(
            f"Batch done: size={batch_size} prec={precision} "
            f"elapsed={elapsed:.2f}s new_tok={total_new}"
        )

    # ── Error delivery ──

    def _deliver_error(self, req: PendingRequest, exc: Exception):
        self.metrics.record_error()
        if req.result_future is not None and not req.result_future.done():
            self._loop.call_soon_threadsafe(req.result_future.set_exception, exc)
        if req.token_queue is not None:
            asyncio.run_coroutine_threadsafe(
                req.token_queue.put(("error", str(exc))), self._loop
            )


# ──────────────────────────────────────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────────────────────────────────────

g_model = None
g_tokenizer = None
g_engine: Optional[GenerationEngine] = None
g_args = None
g_device = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global g_model, g_tokenizer, g_engine

    warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    args = g_args
    logging.info(f"Loading tokenizer from {args.model_path}")
    g_tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    # Left-padding is required for correct batched causal generation
    g_tokenizer.padding_side = "left"
    if g_tokenizer.pad_token is None:
        g_tokenizer.pad_token = g_tokenizer.eos_token

    logging.info(f"Loading AnyPrecisionForCausalLM from {args.model_path}")
    from any_precision import AnyPrecisionForCausalLM
    t0 = time.perf_counter()
    g_model = AnyPrecisionForCausalLM.from_quantized(
        args.model_path, trust_remote_code=True
    )
    g_model.eval()
    if torch.cuda.is_available():
        g_model.model.to("cuda")
    load_time = time.perf_counter() - t0

    if args.compile:
        logging.info("Compiling model with torch.compile (mode=reduce-overhead)...")
        try:
            g_model.model = torch.compile(
                g_model.model, mode="reduce-overhead", dynamic=True
            )
            logging.info("torch.compile done.")
        except Exception as e:
            logging.warning(f"torch.compile failed ({e}), continuing without.")

    logging.info(
        f"Model ready in {load_time:.1f}s | "
        f"precisions {g_model.precisions} | "
        f"VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB"
    )

    g_engine = GenerationEngine(
        model=g_model,
        tokenizer=g_tokenizer,
        device=g_device,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        request_timeout_s=args.request_timeout_s,
    )
    engine_task = asyncio.create_task(g_engine.run())

    logging.info(
        f"Engine started | max_batch={args.max_batch_size} "
        f"wait={args.batch_wait_ms}ms timeout={args.request_timeout_s}s"
    )
    yield

    engine_task.cancel()
    logging.info("Server shut down.")


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Any-Precision LLM Server (1a)", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _model_name() -> str:
    return os.path.basename(g_args.model_path.rstrip("/"))


def _resolve_precision(requested: Optional[int]) -> int:
    if requested is None:
        return max(g_model.precisions)
    if requested not in g_model.precisions:
        raise HTTPException(
            status_code=400,
            detail=f"Precision {requested} not available. "
                   f"Supported: {sorted(g_model.precisions)}",
        )
    return requested


def _build_prompt(messages: List[ChatMessage]) -> str:
    dicts = [{"role": m.role, "content": m.content} for m in messages]
    try:
        return g_tokenizer.apply_chat_template(
            dicts, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return g_tokenizer.apply_chat_template(
            dicts, tokenize=False, add_generation_prompt=True
        )


def _build_pending(req: ChatCompletionRequest, precision: int) -> PendingRequest:
    prompt = _build_prompt(req.messages)
    enc = g_tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = enc["input_ids"]  # [1, prompt_len]

    stop_ids = []
    if req.stop:
        for s in ([req.stop] if isinstance(req.stop, str) else req.stop):
            ids = g_tokenizer(s, add_special_tokens=False).input_ids
            if ids:
                stop_ids.append(ids[0])

    do_sample = (req.temperature or 0.0) > 0.0

    p = PendingRequest(
        request_id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        precision=precision,
        input_ids=input_ids,
        prompt_len=input_ids.shape[1],
        max_tokens=req.max_tokens,
        do_sample=do_sample,
        temperature=req.temperature if do_sample else 1.0,
        top_p=req.top_p if do_sample else 1.0,
        stop_token_ids=stop_ids,
        stream=req.stream,
    )
    loop = asyncio.get_event_loop()
    if req.stream:
        p.token_queue = asyncio.Queue()
    else:
        p.result_future = loop.create_future()
    return p


def _sse(completion_id: str, model_name: str, delta: dict,
         finish_reason: Optional[str]) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_response(pending: PendingRequest) -> AsyncIterator[str]:
    """Consume the token queue and yield SSE chunks."""
    model_name = _model_name()
    cid = pending.request_id
    q = pending.token_queue

    yield _sse(cid, model_name, {"role": "assistant"}, None)

    finish = "stop"

    while True:
        try:
            kind, value = await asyncio.wait_for(q.get(), timeout=120.0)
        except asyncio.TimeoutError:
            finish = "error"
            break

        if kind == "done":
            break
        if kind == "error":
            logging.error(f"Stream error for {cid}: {value}")
            finish = "error"
            break
        if kind == "token":
            yield _sse(cid, model_name, {"content": value}, None)

    yield _sse(cid, model_name, {}, finish)
    yield "data: [DONE]\n\n"


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": g_model is not None}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _model_name(),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "any-precision-llm",
            "supported_precisions": sorted(g_model.precisions),
        }]
    }


@app.get("/v1/info")
async def model_info():
    gpu_info = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu_info = {
            "gpu": p.name,
            "vram_total_gb": round(p.total_memory / 1e9, 1),
            "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        }
    return {
        "model": _model_name(),
        "model_path": g_args.model_path,
        "supported_precisions": sorted(g_model.precisions),
        "default_precision": max(g_model.precisions),
        "device": g_device,
        "engine": {
            "max_batch_size": g_args.max_batch_size,
            "batch_wait_ms": g_args.batch_wait_ms,
            "request_timeout_s": g_args.request_timeout_s,
            "compiled": g_args.compile,
        },
        **gpu_info,
    }


@app.get("/v1/metrics")
async def metrics():
    return g_engine.metrics.snapshot(queue_depth=g_engine._queue.qsize())


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if g_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    precision = _resolve_precision(req.precision)
    pending = _build_pending(req, precision)
    await g_engine.submit(pending)

    # ── Streaming ──
    if req.stream:
        return StreamingResponse(
            _stream_response(pending),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Non-streaming — await the future ──
    try:
        text, n_new = await asyncio.wait_for(
            pending.result_future,
            timeout=g_args.request_timeout_s + 5.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timed out.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "id": pending.request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name(),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": pending.prompt_len,
            "completion_tokens": n_new,
            "total_tokens": pending.prompt_len + n_new,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Any-Precision LLM server (1a)")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max_batch_size", type=int, default=8,
                   help="Max requests per GPU batch (default 8)")
    p.add_argument("--batch_wait_ms", type=float, default=20.0,
                   help="Max ms to wait for a full batch (default 20)")
    p.add_argument("--request_timeout_s", type=float, default=120.0,
                   help="Max seconds a request waits in queue (default 120)")
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile (reduce-overhead mode)")
    p.add_argument("--log_level", default="info",
                   choices=["debug", "info", "warning", "error"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    g_args = args
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
