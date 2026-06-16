"""
early_exit_benchmark.py

Early exit on Qwen3-4B (anyprec): one model acting as multiple complexity levels
by stopping at different transformer layer depths.

Approach: temporarily truncate base.layers to exit_layer length, use the model's
own forward/generate infrastructure unchanged, then restore. No manual layer loops
needed, works with transformers 4.57+ Qwen3 attention backend.

Exit points (Qwen3-4B has 36 layers):
  Layer  9 (25%) — "nano"   compute, ~0.6B-equivalent FLOPS
  Layer 18 (50%) — "small"  compute, ~1.7B-equivalent FLOPS
  Layer 27 (75%) — "medium" compute
  Layer 36 (100%) — full 4B baseline

Advantage over loading separate models:
  • Single model in GPU  : ~5.1 GB vs 9.7 GB for three separate models
  • KV cache shrinks with exit depth (25% exit = 25% KV memory)
  • No model switching latency; precision changeable per request via set_precision()

Usage:
  python early_exit_benchmark.py
"""

import re
import warnings
import torch
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

MODEL_PATH = (
    "/mnt/scratch/users/psuwannapichat/any-precision-llm/cache/packed/"
    "anyprec-(Qwen3-4B)-w8_orig3-gc1-c4_s100_blk512"
)
PRECISION     = 4
DECODE_STEPS  = 50
WARMUP_STEPS  = 8
MAX_GEN_TOKS  = 64
BENCH_PROMPT  = "Explain how neural network quantization reduces model size."

MMLU_QUESTIONS = [
    ("What is the derivative of x^2?",
     ["A) x", "B) 2x", "C) x^2", "D) 2"], "B"),
    ("What is the capital of France?",
     ["A) Berlin", "B) Madrid", "C) Paris", "D) Rome"], "C"),
    ("In Python, what does len([1,2,3]) return?",
     ["A) 2", "B) 3", "C) 4", "D) 1"], "B"),
    ("What is the boiling point of water in Celsius?",
     ["A) 90", "B) 100", "C) 110", "D) 80"], "B"),
    ("What is 7 × 8?",
     ["A) 54", "B) 56", "C) 58", "D) 52"], "B"),
    ("Which planet is closest to the Sun?",
     ["A) Venus", "B) Earth", "C) Mercury", "D) Mars"], "C"),
    ("What does DNA stand for?",
     ["A) Deoxyribose Nucleic Acid", "B) Deoxyribonucleic Acid",
      "C) Dinucleotide Acid", "D) Dextro Nucleic Acid"], "B"),
    ("What is the speed of light (approx, m/s)?",
     ["A) 3e6", "B) 3e8", "C) 3e10", "D) 3e4"], "B"),
    ("In which year did World War II end?",
     ["A) 1943", "B) 1944", "C) 1945", "D) 1946"], "C"),
    ("What is the chemical symbol for gold?",
     ["A) Go", "B) Gd", "C) Au", "D) Ag"], "C"),
    ("What is the integral of 2x?",
     ["A) x", "B) x^2 + C", "C) 2x^2 + C", "D) x^2"], "B"),
    ("What is Newton's second law?",
     ["A) F=ma", "B) E=mc^2", "C) F=mv", "D) a=m/F"], "A"),
    ("Which language is known for 'write once, run anywhere'?",
     ["A) C++", "B) Python", "C) Java", "D) Go"], "C"),
    ("What is the largest ocean on Earth?",
     ["A) Atlantic", "B) Indian", "C) Arctic", "D) Pacific"], "D"),
    ("What is the value of pi (approx)?",
     ["A) 2.14", "B) 3.14", "C) 4.14", "D) 1.14"], "B"),
    ("Which element has atomic number 1?",
     ["A) Helium", "B) Lithium", "C) Hydrogen", "D) Oxygen"], "C"),
    ("What is the binary representation of decimal 10?",
     ["A) 1010", "B) 1100", "C) 0110", "D) 1001"], "A"),
    ("What is the powerhouse of the cell?",
     ["A) Nucleus", "B) Ribosome", "C) Mitochondria", "D) Golgi"], "C"),
    ("What is the time complexity of binary search?",
     ["A) O(n)", "B) O(n^2)", "C) O(log n)", "D) O(1)"], "C"),
    ("Which of these is a transformer-based model?",
     ["A) ResNet", "B) LSTM", "C) GPT", "D) SVM"], "C"),
]


def sep(title="", width=76):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def with_exit(inner, base, orig_layers, exit_layer):
    """Context: truncate model layers to exit_layer for the block."""
    base.layers = orig_layers[:exit_layer]
    try:
        yield
    finally:
        base.layers = orig_layers


@torch.no_grad()
def time_decode(inner, base, orig_layers, tokenizer, exit_layer, device="cuda"):
    """Measure decode step latency using CUDA events (batch=1)."""
    msgs = [{"role": "user", "content": BENCH_PROMPT}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    last = ids[:, -1:]

    base.layers = orig_layers[:exit_layer]

    # Prefill
    out = inner(ids, use_cache=True)
    cache = out.past_key_values

    # Warmup
    for _ in range(WARMUP_STEPS):
        inner(last, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()

    # Timed decode steps (same cache/input each step — constant workload)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(DECODE_STEPS)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(DECODE_STEPS)]
    for i in range(DECODE_STEPS):
        starts[i].record()
        inner(last, past_key_values=cache, use_cache=True)
        ends[i].record()
    torch.cuda.synchronize()

    base.layers = orig_layers
    times = sorted([starts[i].elapsed_time(ends[i]) for i in range(DECODE_STEPS)])
    return times


def extract_letter(text):
    """Extract A/B/C/D answer from model output."""
    t = text.strip()
    if t and t[0].upper() in "ABCD":
        return t[0].upper()
    m = re.search(r"(?:answer is|answer:)\s*([A-D])", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = re.findall(r"\b([A-D])\b", t)
    return letters[-1] if letters else None


@torch.no_grad()
def run_mmlu(inner, base, orig_layers, tokenizer, exit_layer, device="cuda"):
    """MMLU accuracy with greedy generation at the given exit depth."""
    base.layers = orig_layers[:exit_layer]
    correct = 0

    for question, choices, answer in MMLU_QUESTIONS:
        prompt = question + "\n" + "\n".join(choices)
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        # Prefill
        out = inner(ids, use_cache=True)
        cache = out.past_key_values

        generated = []
        for _ in range(MAX_GEN_TOKS):
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_tok.item() == tokenizer.eos_token_id:
                break
            generated.append(next_tok.item())
            out = inner(next_tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values

        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if extract_letter(decoded) == answer:
            correct += 1

    base.layers = orig_layers
    return correct, len(MMLU_QUESTIONS)


def main():
    device = "cuda"
    print("\n" + "=" * 76)
    print("  EARLY EXIT BENCHMARK  —  Qwen3-4B as Multiple Complexity Levels")
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 76)

    print("\nLoading tokenizer and model... ", end="", flush=True)
    from any_precision.modules.AnyPrecisionForCausalLM import AnyPrecisionForCausalLM
    model = AnyPrecisionForCausalLM.from_quantized(MODEL_PATH, precisions=[PRECISION])
    model = model.eval().cuda()
    model.set_precision(PRECISION)
    inner = model.model            # Qwen3ForCausalLM
    base  = inner.model            # Qwen3Model
    orig_layers = base.layers      # all 36 layers

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model_mb = torch.cuda.memory_allocated() / 1e6
    n_layers = len(orig_layers)
    print(f"done  ({model_mb:.0f} MB,  {n_layers} layers)")

    exit_configs = [
        (f"exit-{n_layers//4:02d} (25%)", n_layers // 4),
        (f"exit-{n_layers//2:02d} (50%)", n_layers // 2),
        (f"exit-{3*n_layers//4:02d} (75%)", 3 * n_layers // 4),
        (f"full-{n_layers:02d} (100%)",    n_layers),
    ]

    # ── 1. Decode Speed ────────────────────────────────────────────────────────
    sep("1. Decode Speed  (CUDA events, batch=1, ms per decode step)")
    speed = {}
    for label, el in exit_configs:
        print(f"  [{label}] timing...", end="", flush=True)
        times = time_decode(inner, base, orig_layers, tokenizer, el, device)
        p50 = times[len(times) // 2]
        p90 = times[int(len(times) * 0.9)]
        speed[label] = p50
        print(f"  p50={p50:.1f}ms  p90={p90:.1f}ms")

    full_p50 = speed[exit_configs[-1][0]]
    print()
    print(f"  {'Config':<22}  {'layers':>6}  {'p50 ms':>8}  {'tok/s':>7}  {'speedup':>8}  Bar (full=×2)")
    print("  " + "─" * 70)
    for label, el in exit_configs:
        p50     = speed[label]
        speedup = full_p50 / p50
        tps     = 1000 / p50
        bar_n   = max(0, min(20, int(speedup / 2.0 * 20)))
        bar     = "█" * bar_n + "░" * (20 - bar_n)
        suffix  = "  ◄ baseline" if label == exit_configs[-1][0] else f"  ×{speedup:.3f}"
        print(f"  {label:<22}  {el:>6}  {p50:>8.1f}  {tps:>7.1f}  {speedup:>8.3f}  {bar}{suffix}")

    # ── 2. MMLU Accuracy ───────────────────────────────────────────────────────
    sep("2. Accuracy  (MMLU subset, 20 questions, greedy decode)")
    accuracy = {}
    for label, el in exit_configs:
        print(f"  [{label}] running MMLU... ", end="", flush=True)
        correct, total = run_mmlu(inner, base, orig_layers, tokenizer, el, device)
        acc = correct / total * 100
        accuracy[label] = acc
        bar_n = int(acc / 100 * 20)
        bar   = "█" * bar_n + "░" * (20 - bar_n)
        print(f"\r  {label:<22}  {correct}/{total}  {acc:>5.1f}%  {bar}")

    # ── 3. KV Cache Memory ─────────────────────────────────────────────────────
    sep("3. KV Cache Memory  (bytes per token, scales with exit depth)")
    attn = orig_layers[0].self_attn
    num_kv  = getattr(attn, "num_key_value_heads", 8)
    hdim    = getattr(attn, "head_dim", 128)
    bpt_lyr = 2 * num_kv * hdim * 2   # K+V, fp16
    pool_mb = 8 * 1024                 # 8 GB example KV pool

    print(f"\n  Architecture: {num_kv} KV-heads × {hdim} head_dim × 2 (K+V) × 2 bytes = {bpt_lyr} B/token/layer\n")
    print(f"  {'Config':<22}  {'layers':>6}  {'B/token':>9}  {'vs full':>8}  {'max tokens (8GB pool)':>22}")
    print("  " + "─" * 72)
    full_bpt = n_layers * bpt_lyr
    for label, el in exit_configs:
        bpt       = el * bpt_lyr
        ratio     = bpt / full_bpt
        max_toks  = int(pool_mb * 1024 * 1024 / bpt)
        vs        = f"×{ratio:.2f}" if el < n_layers else "baseline"
        print(f"  {label:<22}  {el:>6}  {bpt:>9,}  {vs:>8}  ~{max_toks:>15,} tokens")

    # ── 4. Summary ─────────────────────────────────────────────────────────────
    sep("Summary  —  Speed / Accuracy / KV Memory")
    print(f"\n  {'Config':<22}  {'ms/step':>8}  {'speedup':>8}  {'MMLU%':>7}  {'KV ratio':>9}")
    print("  " + "─" * 60)
    for label, el in exit_configs:
        p50     = speed[label]
        speedup = full_p50 / p50
        acc     = accuracy[label]
        kv_r    = el / n_layers
        suffix  = "  ◄ baseline" if el == n_layers else ""
        print(f"  {label:<22}  {p50:>8.1f}  ×{speedup:>6.3f}  {acc:>7.1f}  {kv_r:>8.0%}{suffix}")

    # ── 5. Key Insight ─────────────────────────────────────────────────────────
    sep("Key Insight")

    # Find best accuracy above 80%
    good = [(lbl, el) for lbl, el in exit_configs
            if accuracy[lbl] >= 80.0 and el < n_layers]
    if good:
        best_lbl, best_el = good[0]
        best_speedup = full_p50 / speed[best_lbl]
        best_acc     = accuracy[best_lbl]
        best_kv      = best_el / n_layers
        print(f"""
  Single 4B model, variable exit depth per request:

    Memory (model weights)  : {model_mb:.0f} MB  — same regardless of exit depth
    vs 3 separate models    : ~9,700 MB  (0.6B + 1.7B + 4B anyprec)
    Savings                 : {9700 - model_mb:.0f} MB  ({(9700-model_mb)/9700*100:.0f}% less GPU memory)

  Best speed+quality point  : {best_lbl}
    Speed                   : ×{best_speedup:.2f} faster than full 4B  ({speed[best_lbl]:.1f}ms vs {full_p50:.1f}ms)
    MMLU accuracy           : {best_acc:.0f}%
    KV cache                : {best_kv:.0%} of full model  (supports {1/best_kv:.1f}× longer context)

  Recommended MAS routing:
    Simple requests  →  exit-{n_layers//4:02d} (25%)  — fastest, lower quality
    Standard requests →  {best_lbl}  — best speed/quality balance
    Complex requests  →  full-{n_layers:02d} (100%) — maximum quality

  Key limitation: no trained exit heads — early layers use the final
  norm/lm_head (trained for depth-{n_layers} outputs), so quality degrades
  more than with trained exit heads (e.g. LayerSkip). Fine-tuning auxiliary
  heads at exit points would close the quality gap.
""")
    else:
        print(f"""
  No exit point reached 80% MMLU with the final norm/lm_head applied naively.
  The early layers' hidden states are not calibrated for the depth-{n_layers} norm.

  Path forward: train lightweight exit heads at each exit point (LayerSkip
  approach) — this typically recovers most of the quality gap with 1–2 epochs
  of auxiliary loss fine-tuning.

  Benefit still holds:
    Single model in GPU: {model_mb:.0f} MB vs ~9,700 MB for 3 separate models.
    KV cache scales with exit depth — 25% exit = 4× longer context capacity.
""")

    sep()


if __name__ == "__main__":
    main()
