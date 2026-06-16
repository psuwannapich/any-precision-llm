"""
p2_server.py — Phase 2 OpenAI-compatible server with per-request precision.

Wraps vLLM's AsyncLLMEngine with a thin FastAPI layer that:
  1. Accepts an extra "precision" field on POST /v1/chat/completions
  2. Registers (request_id → precision) in PrecisionManager before generating
  3. Cleans up the registry entry when generation finishes

Run with:
  bash run_vllm_p2.sh

API extensions (all other fields are standard OpenAI):
  POST /v1/chat/completions
    {
      "model": "...",
      "messages": [...],
      "precision": 4,      ← 3/4/5/6/7/8; default = max (8-bit)
      "max_tokens": 128,
      ...
    }

  GET /v1/precisions        ← returns {"supported": [3,4,5,6,7,8]}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from typing import AsyncIterator, List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

from .precision_manager import PrecisionManager, DEFAULT_PRECISION

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="AnyPrecision vLLM Server (Phase 2)")
g_engine: Optional[AsyncLLMEngine] = None
g_supported_bits: List[int] = []
g_model_name: str = ""


# ── Request / Response models (manual JSON, avoids pydantic version issues) ──

def _parse_request(body: dict) -> tuple:
    """Parse ChatCompletion-style body → (messages, sampling_params, precision)."""
    messages = body.get("messages", [])
    precision = int(body.get("precision", DEFAULT_PRECISION))

    sp = SamplingParams(
        temperature  = body.get("temperature", 1.0),
        top_p        = body.get("top_p", 1.0),
        top_k        = body.get("top_k", -1),
        max_tokens   = body.get("max_tokens", 128),
        stop         = body.get("stop", None) or [],
        presence_penalty  = body.get("presence_penalty", 0.0),
        frequency_penalty = body.get("frequency_penalty", 0.0),
    )
    return messages, sp, precision


def _apply_chat_template(messages: list, tokenizer, enable_thinking: bool = False) -> str:
    """Format messages into a single prompt string."""
    try:
        # enable_thinking=False disables Qwen3's <think> block (faster, cleaner output)
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Fallback for tokenizers that don't support enable_thinking
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _make_response(request_id: str, prompt_toks: int, output, model_name: str) -> dict:
    text = output.text
    finish = output.finish_reason or "stop"
    comp_toks = len(output.token_ids)
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "total_tokens": prompt_toks + comp_toks,
        },
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": g_model_name, "object": "model"}]}


@app.get("/v1/precisions")
async def precisions():
    return {"supported": g_supported_bits, "default": DEFAULT_PRECISION}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages, sampling_params, precision = _parse_request(body)
    stream = bool(body.get("stream", False))
    enable_thinking = bool(body.get("enable_thinking", False))

    if precision not in g_supported_bits:
        return JSONResponse(
            {"error": f"precision={precision} not in {g_supported_bits}"},
            status_code=400,
        )

    tokenizer = await g_engine.get_tokenizer()
    prompt    = _apply_chat_template(messages, tokenizer, enable_thinking=enable_thinking)
    req_id    = str(uuid.uuid4())

    # Register precision BEFORE generating
    mgr = PrecisionManager.get()
    mgr.register(req_id, precision)

    try:
        if stream:
            return StreamingResponse(
                _stream(req_id, prompt, sampling_params, tokenizer),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(
                await _complete(req_id, prompt, sampling_params, tokenizer)
            )
    except Exception as e:
        mgr.unregister(req_id)
        raise


async def _complete(req_id: str, prompt: str,
                    sp: SamplingParams, tokenizer) -> dict:
    mgr = PrecisionManager.get()
    prompt_toks = len(tokenizer.encode(prompt))
    final_output = None
    try:
        async for req_output in g_engine.generate(prompt, sp, request_id=req_id):
            final_output = req_output
    finally:
        mgr.unregister(req_id)

    out = final_output.outputs[0]
    return _make_response(req_id, prompt_toks, out, g_model_name)


async def _stream(req_id: str, prompt: str,
                  sp: SamplingParams, tokenizer) -> AsyncIterator[bytes]:
    mgr = PrecisionManager.get()
    prompt_toks = len(tokenizer.encode(prompt))
    prev_len = 0
    try:
        async for req_output in g_engine.generate(prompt, sp, request_id=req_id):
            out   = req_output.outputs[0]
            delta = out.text[prev_len:]
            prev_len = len(out.text)
            chunk = {
                "id": f"chatcmpl-{req_id}",
                "object": "chat.completion.chunk",
                "model": g_model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta} if delta else {},
                    "finish_reason": out.finish_reason,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    finally:
        mgr.unregister(req_id)


# ── Server startup ────────────────────────────────────────────────────────────

async def _lifespan(engine_args_str: str, port: int, worker_cls: str):
    global g_engine, g_supported_bits, g_model_name

    from vllm import AsyncEngineArgs
    import json as _json

    eargs_dict = _json.loads(engine_args_str)
    eargs = AsyncEngineArgs(**eargs_dict)

    # Inject custom worker class (Phase 2 precision-grouped execution)
    eargs.worker_cls = worker_cls

    g_engine      = AsyncLLMEngine.from_engine_args(eargs)
    g_model_name  = eargs.model

    # Read supported precisions from config
    import json as j
    cfg_path = eargs.model + "/config.json"
    try:
        cfg = j.load(open(cfg_path))
        ap  = cfg.get("anyprec", {})
        seed, parent = ap.get("seed_precision", 3), ap.get("parent_precision", 8)
        g_supported_bits = list(range(seed, parent + 1))
    except Exception:
        g_supported_bits = [3, 4, 5, 6, 7, 8]

    print(f"[Phase 2] Server ready — precisions: {g_supported_bits}")
    print(f"[Phase 2] Worker: {worker_cls}")

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--port",   type=int, default=8001)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--enforce_eager", action="store_true", default=True)
    parser.add_argument("--worker_cls",
                        default="any_precision.vllm_integration.worker.AnyPrecisionWorker")

    args = parser.parse_args()

    import json as _j
    engine_args = {
        "model":                    args.model,
        "tokenizer":                args.model,
        "trust_remote_code":        True,
        "dtype":                    args.dtype,
        "max_model_len":            args.max_model_len,
        "gpu_memory_utilization":   args.gpu_memory_utilization,
        "max_num_seqs":             args.max_num_seqs,
        "enforce_eager":            args.enforce_eager,
    }

    asyncio.run(_lifespan(_j.dumps(engine_args), args.port, args.worker_cls))


if __name__ == "__main__":
    main()
