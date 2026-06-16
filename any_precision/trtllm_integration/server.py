"""
server.py — minimal FastAPI server for the Any-Precision TensorRT-LLM backend.

Mirrors the vLLM Phase-2 server: each request carries a `precision` field, the
server selects the bit-width per request and runs the single shared engine.

Run:
  python -m any_precision.trtllm_integration.server \
      --engine_dir ./trt_engine --tokenizer_dir /path/to/anyprec-Qwen3-4B \
      --port 8002

Request:
  POST /generate {"prompt": "...", "precision": 4, "max_new_tokens": 128}
"""

import argparse
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from .python.runtime import AnyPrecisionTRTRunner

app = FastAPI(title="Any-Precision TensorRT-LLM server")
_runner: Optional[AnyPrecisionTRTRunner] = None


class GenerateRequest(BaseModel):
    prompt: str
    precision: Optional[int] = None
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


@app.get("/health")
def health():
    return {"status": "ok",
            "supported_bits": _runner.supported_bits if _runner else None}


@app.post("/generate")
def generate(req: GenerateRequest):
    text = _runner.generate(
        req.prompt, precision=req.precision, max_new_tokens=req.max_new_tokens,
        temperature=req.temperature, top_p=req.top_p)
    return {"text": text, "precision": req.precision or _runner.default_precision}


def main():
    global _runner
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_dir", required=True)
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--cpp_runtime", action="store_true")
    args = ap.parse_args()

    _runner = AnyPrecisionTRTRunner(
        args.engine_dir, tokenizer_dir=args.tokenizer_dir,
        use_cpp_runtime=args.cpp_runtime)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
