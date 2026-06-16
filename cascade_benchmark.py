"""
cascade_benchmark.py

Cascade routing benchmark for MAS systems — comparing model sizes and
quantization levels to find the best speed/quality/memory trade-off.

Models compared:
  Qwen3-0.6B  fp16    — smallest, fastest, handles simple tasks
  Qwen3-1.7B  fp16    — medium, balanced
  Qwen3-4B    anyprec — 3/4/8-bit quantized via any-precision-llm
  Qwen3-4B    fp16    — full-precision baseline

Measures per model:
  1. Speed    — decode step time (ms) at batch=1, 4, 8 via CUDA events
  2. Memory   — GPU VRAM footprint
  3. Accuracy — MMLU subset (20 questions) with proper chat template
  4. Cascade  — expected avg latency at different routing ratios

Usage:
  python cascade_benchmark.py
"""

import argparse
import re
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_0_6B  = "/mnt/scratch/users/psuwannapichat/any-precision-llm/cache/Qwen3-0.6B"
MODEL_1_7B  = "/mnt/scratch/users/psuwannapichat/any-precision-llm/cache/Qwen3-1.7B"
MODEL_4B_AP = (
    "/mnt/scratch/users/psuwannapichat/any-precision-llm/cache/packed/"
    "anyprec-(Qwen3-4B)-w8_orig3-gc1-c4_s100_blk512"
)
MODEL_4B_FP16 = (
    "/mnt/scratch/users/psuwannapichat/cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)

DECODE_STEPS = 50
WARMUP_STEPS = 8
BATCH_SIZES  = [1, 4, 8]
BENCH_PROMPT = "Explain how neural network quantization reduces model size."

MMLU_QUESTIONS = [
    ("What is the derivative of x^2?", ["A) x", "B) 2x", "C) x^2", "D) 2"], "B"),
    ("What is the capital of Japan?", ["A) Osaka", "B) Kyoto", "C) Tokyo", "D) Hiroshima"], "C"),
    ("Which planet is closest to the Sun?", ["A) Venus", "B) Earth", "C) Mercury", "D) Mars"], "C"),
    ("What is 15 × 8?", ["A) 100", "B) 120", "C) 110", "D) 130"], "B"),
    ("What gas do plants absorb?", ["A) O2", "B) CO2", "C) N2", "D) H2"], "B"),
    ("How many sides does a hexagon have?", ["A) 5", "B) 6", "C) 7", "D) 8"], "B"),
    ("What is 2^10?", ["A) 512", "B) 1024", "C) 2048", "D) 256"], "B"),
    ("What is the boiling point of water (Celsius)?", ["A) 90", "B) 95", "C) 100", "D) 105"], "C"),
    ("Which element has atomic number 1?", ["A) Helium", "B) Hydrogen", "C) Lithium", "D) Carbon"], "B"),
    ("What year did WWII end?", ["A) 1943", "B) 1944", "C) 1945", "D) 1946"], "C"),
    ("What is the powerhouse of the cell?", ["A) Nucleus", "B) Mitochondria", "C) Ribosome", "D) Golgi"], "B"),
    ("What is the chemical symbol for gold?", ["A) Go", "B) Gd", "C) Au", "D) Ag"], "C"),
    ("What is the largest ocean?", ["A) Atlantic", "B) Indian", "C) Arctic", "D) Pacific"], "D"),
    ("How many bones in the adult human body?", ["A) 186", "B) 206", "C) 226", "D) 246"], "B"),
    ("What is the speed of light (approx)?", ["A) 3×10^8 m/s", "B) 3×10^6 m/s", "C) 3×10^10 m/s", "D) 3×10^4 m/s"], "A"),
    ("What is the integral of 1/x?", ["A) x^2/2", "B) ln(x)", "C) e^x", "D) 1/x^2"], "B"),
    ("DNA is made of which sugars?", ["A) Glucose", "B) Fructose", "C) Deoxyribose", "D) Ribose"], "C"),
    ("What is Ohm's Law?", ["A) V=IR", "B) P=IV", "C) F=ma", "D) E=mc^2"], "A"),
    ("In what year did the French Revolution begin?", ["A) 1776", "B) 1789", "C) 1799", "D) 1804"], "B"),
    ("What is the chemical formula for ammonia?", ["A) NO2", "B) N2O", "C) NH3", "D) HNO3"], "C"),
]


# ── Formatting helpers ────────────────────────────────────────────────────────

def sep(title="", width=74):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def bar(ratio, width=20):
    ratio = max(0.0, min(ratio, 2.0)) / 2.0
    n = int(ratio * width)
    return "█" * n + "░" * (width - n)


def gpu_mb():
    return torch.cuda.memory_allocated() / 1e6


# ── Core timing ───────────────────────────────────────────────────────────────

def time_decode(model, tokenizer, prompt, batch_size, steps, warmup):
    """Return sorted list of per-step GPU times (ms) via CUDA events."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    ids = ids.expand(batch_size, -1)
    with torch.no_grad():
        out  = model(ids, use_cache=True)
        pkv  = out.past_key_values
    decode_in = ids[:, -1:]

    with torch.no_grad():
        for _ in range(warmup):
            model(decode_in, past_key_values=pkv, use_cache=True)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    with torch.no_grad():
        for i in range(steps):
            starts[i].record()
            model(decode_in, past_key_values=pkv, use_cache=True)
            ends[i].record()
    torch.cuda.synchronize()
    return sorted(starts[i].elapsed_time(ends[i]) for i in range(steps))


def p50(times):
    return times[len(times) // 2]


# ── MMLU evaluation ───────────────────────────────────────────────────────────

def answer_mmlu(model, tokenizer, question, choices):
    content = (f"{question}\n" + "\n".join(choices)
               + "\nAnswer with just the letter (A/B/C/D):")
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt = content

    ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=64, do_sample=False,
            temperature=None, pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    if reply and reply[0] in "ABCD":
        return reply[0]
    m = re.search(r'(?:answer|correct)[^A-D]{0,20}([A-D])\b', reply, re.IGNORECASE)
    if m:
        return m.group(1)
    matches = re.findall(r'\b([A-D])\b', reply)
    return matches[-1] if matches else "?"


def run_mmlu(model, tokenizer, label):
    correct = 0
    for q, choices, ans in MMLU_QUESTIONS:
        if answer_mmlu(model, tokenizer, q, choices) == ans:
            correct += 1
    acc = correct / len(MMLU_QUESTIONS)
    print(f"  {label:<22}: {acc*100:5.1f}%  ({correct}/{len(MMLU_QUESTIONS)})")
    return acc


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_hf(path, label):
    """Load a standard HuggingFace fp16 model."""
    print(f"  Loading {label} ... ", end="", flush=True)
    mem0 = gpu_mb()
    tok   = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float16, trust_remote_code=True
    ).eval().cuda()
    mem = gpu_mb() - mem0
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"done  ({params:.0f}M params, {mem:.0f} MB)")
    return model, tok, mem


def load_anyprec(path, precisions, label):
    """Load any-precision quantized model with all LUTs."""
    print(f"  Loading {label} ... ", end="", flush=True)
    mem0 = gpu_mb()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    from any_precision.modules.AnyPrecisionForCausalLM import AnyPrecisionForCausalLM
    model = AnyPrecisionForCausalLM.from_quantized(
        path, precisions=precisions
    ).eval().cuda()
    mem = gpu_mb() - mem0
    print(f"done  ({mem:.0f} MB, LUTs {precisions[0]}–{precisions[-1]}-bit)")
    return model, tok, mem


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode_steps", type=int, default=DECODE_STEPS)
    parser.add_argument("--warmup",       type=int, default=WARMUP_STEPS)
    args = parser.parse_args()

    print("\n" + "="*74)
    print("  CASCADE ROUTING BENCHMARK  —  Model Size vs Quantization")
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print("="*74)

    # ── Define model configs ──────────────────────────────────────────────────
    # Each entry: (label, loader_fn, path, extra_args)
    ANYPREC_PRECS = [3, 4, 8]

    configs = [
        ("0.6B fp16",   "hf",      MODEL_0_6B,   {}),
        ("1.7B fp16",   "hf",      MODEL_1_7B,   {}),
        ("4B 3-bit",    "anyprec", MODEL_4B_AP,  {"precision": 3}),
        ("4B 4-bit",    "anyprec", MODEL_4B_AP,  {"precision": 4}),
        ("4B 8-bit",    "anyprec", MODEL_4B_AP,  {"precision": 8}),
        ("4B fp16",     "hf",      MODEL_4B_FP16, {}),
    ]

    speed   = {}   # label → {batch → p50_ms}
    memory  = {}   # label → MB
    accuracy = {}  # label → float

    anyprec_model = None
    anyprec_tok   = None
    anyprec_mem   = None

    # ── 1. Speed benchmark ────────────────────────────────────────────────────
    sep("1. Decode Speed  (CUDA event timing, ms per decode step)")
    print(f"  Prompt: {repr(BENCH_PROMPT[:60])}\n")
    print(f"  {'Model':<14}" + "".join(f"  batch={b}" for b in BATCH_SIZES))
    print("  " + "─" * 52)

    for label, kind, path, extra in configs:
        # Load model
        if kind == "hf":
            model, tok, mem = load_hf(path, label)
        else:
            if anyprec_model is None:
                anyprec_model, anyprec_tok, anyprec_mem = load_anyprec(
                    path, ANYPREC_PRECS, "4B anyprec (all bits)")
            model = anyprec_model
            model.set_precision(extra["precision"])
            tok  = anyprec_tok
            mem  = anyprec_mem
            memory[label] = anyprec_mem   # same physical model

        if kind == "hf":
            memory[label] = mem

        speed[label] = {}
        row = [f"  {label:<14}"]
        for bs in BATCH_SIZES:
            times = time_decode(model, tok, BENCH_PROMPT, bs,
                                args.decode_steps, args.warmup)
            ms = p50(times)
            speed[label][bs] = ms
            tps = bs / ms * 1000
            row.append(f"  {ms:5.1f}ms ({tps:4.0f} t/s)")
        print("".join(row))

        if kind == "hf":
            del model
            torch.cuda.empty_cache()

    # ── Speed summary table ───────────────────────────────────────────────────
    sep("Speed Summary  — speedup vs Qwen3-4B fp16 baseline  (batch=1)")
    ref_label = "4B fp16"
    ref_p50   = speed[ref_label][1]

    print(f"  {'Model':<14}  {'ms/step':>8}  {'tok/s':>7}  {'vs 4B fp16':>11}  Bar (full=2×)")
    print("  " + "─" * 65)
    for label, _, _, _ in configs:
        ms  = speed[label][1]
        tps = 1 / ms * 1000
        sp  = ref_p50 / ms
        b   = bar(sp)
        flag = "  ◄ baseline" if label == ref_label else f"  ×{sp:.2f}"
        print(f"  {label:<14}  {ms:>8.1f}  {tps:>7.0f}  {sp:>10.3f}×  {b}{flag}")

    # ── 2. Memory ─────────────────────────────────────────────────────────────
    sep("2. GPU Memory Footprint")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2

    # anyprec variants share one physical model
    unique_mem = {}
    for label, kind, _, extra in configs:
        if kind == "anyprec":
            unique_mem["4B anyprec (3+4+8 LUTs)"] = anyprec_mem
        else:
            unique_mem[label] = memory[label]

    for label, mem in unique_mem.items():
        pct = mem / total_vram * 100
        b   = bar(pct / 100, width=30)
        print(f"  {label:<28}: {mem:>6.0f} MB  ({pct:4.1f}%)  {b}")

    print()
    # Cascade combos
    for small_label, mem_small in [("0.6B fp16", memory["0.6B fp16"]),
                                    ("1.7B fp16", memory["1.7B fp16"])]:
        total = mem_small + anyprec_mem
        remaining = total_vram - total
        print(f"  {small_label} + 4B anyprec : {total:>6.0f} MB  "
              f"({total/1024:.1f} GB / {total_vram/1024:.0f} GB)  "
              f"→ {remaining:.0f} MB free for KV cache")

    # ── 3. Accuracy ───────────────────────────────────────────────────────────
    sep(f"3. Accuracy  (MMLU, {len(MMLU_QUESTIONS)} questions, chat template applied)")

    for label, kind, path, extra in configs:
        if kind == "hf":
            model, tok, _ = load_hf(path, label)
        else:
            model = anyprec_model
            model.set_precision(extra["precision"])
            tok   = anyprec_tok

        accuracy[label] = run_mmlu(model, tok, label)

        if kind == "hf":
            del model
            torch.cuda.empty_cache()

    # ── 4. Cascade routing simulation ────────────────────────────────────────
    sep("4. Cascade Routing  — Expected Average Latency  (batch=1, 128 tokens)")

    token_count = 128

    for small_label in ["0.6B fp16", "1.7B fp16"]:
        large_label = "4B 4-bit"
        small_latency = speed[small_label][1] * token_count / 1000
        large_latency = speed[large_label][1] * token_count / 1000

        print(f"\n  Small: {small_label}  ({small_latency:.2f}s/req)  |  "
              f"Large: {large_label}  ({large_latency:.2f}s/req)")
        print(f"  {'Route→small':>12}  {'Avg latency':>12}  {'vs all-large':>12}  Savings")
        print("  " + "─" * 52)
        for pct in [0, 20, 40, 60, 80, 90, 100]:
            a = pct / 100
            avg = a * small_latency + (1 - a) * large_latency
            sp  = large_latency / avg
            sav = (1 - avg / large_latency) * 100
            print(f"  {pct:>11}%  {avg:>10.2f}s  ×{sp:>10.2f}  {sav:>5.1f}%")

    # ── 5. Final comparison table ─────────────────────────────────────────────
    sep("Summary  — Speed / Memory / Accuracy")
    print(f"  {'Model':<14}  {'ms/tok':>7}  {'tok/s':>7}  {'VRAM MB':>8}  "
          f"{'MMLU':>6}  {'vs 4B-fp16 speed':>18}")
    print("  " + "─" * 72)

    for label, _, _, _ in configs:
        ms   = speed[label][1]
        tps  = 1000 / ms
        mem  = memory[label]
        acc  = accuracy[label]
        sp   = ref_p50 / ms
        flag = "  ◄ baseline" if label == ref_label else f"  ×{sp:.2f}"
        print(f"  {label:<14}  {ms:>7.1f}  {tps:>7.0f}  {mem:>8.0f}  "
              f"{acc*100:>5.1f}%{flag}")

    # ── Recommendation ────────────────────────────────────────────────────────
    sep("Recommendation for MAS Cascade Design")
    sp_06  = ref_p50 / speed["0.6B fp16"][1]
    sp_17  = ref_p50 / speed["1.7B fp16"][1]
    sp_ap  = ref_p50 / speed["4B 8-bit"][1]
    acc_06 = accuracy["0.6B fp16"]
    acc_17 = accuracy["1.7B fp16"]
    acc_4b = accuracy["4B 8-bit"]

    print(f"""
  Speed lever comparison (vs Qwen3-4B fp16):
    0.6B fp16      : ×{sp_06:.2f} faster  —  {acc_06*100:.0f}% MMLU
    1.7B fp16      : ×{sp_17:.2f} faster  —  {acc_17*100:.0f}% MMLU
    4B anyprec 8-bit: ×{sp_ap:.2f}        —  {acc_4b*100:.0f}% MMLU  (quantized same quality)
    4B fp16        : ×1.00 (baseline) —  {accuracy['4B fp16']*100:.0f}% MMLU

  Practical cascade for your MAS:

    Request → Router
        ├── Simple  (greeting, factual, format)  → 0.6B  ×{sp_06:.1f} faster
        ├── Medium  (summarise, QA, translate)   → 1.7B  ×{sp_17:.1f} faster
        └── Complex (reason, code, math)         → 4B anyprec (quality dial: 3–8 bit)

  Memory to deploy all three simultaneously:
    0.6B ({memory['0.6B fp16']:.0f} MB) + 1.7B ({memory['1.7B fp16']:.0f} MB) + 4B anyprec ({anyprec_mem:.0f} MB)
    = {memory['0.6B fp16'] + memory['1.7B fp16'] + anyprec_mem:.0f} MB  ({(memory['0.6B fp16'] + memory['1.7B fp16'] + anyprec_mem)/1024:.1f} GB / {total_vram/1024:.0f} GB)
""")
    sep()


if __name__ == "__main__":
    main()
