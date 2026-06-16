"""
logit_compare.py — compare the TRT engine's next-token logits against the
PyTorch AnyPrecisionForCausalLM reference for the same prompt, to localize the
build_engine.py wiring bug (the AP plugin is already proven correct).
"""
import argparse
import numpy as np
import torch

from ..python._plugin_loader import load_plugin, set_precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--precision", type=int, default=8)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--num_layers", type=int, default=0,
                    help="truncate PyTorch model to N layers (0 = all)")
    args = ap.parse_args()

    load_plugin()
    set_precision(args.precision)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ids = tok(args.prompt, return_tensors="pt").input_ids[0]
    print(f"prompt={args.prompt!r}  tokens={ids.tolist()}")

    # ---- PyTorch reference ----
    from any_precision import AnyPrecisionForCausalLM
    pt = AnyPrecisionForCausalLM.from_quantized(args.model_path,
                                                precisions=[args.precision])
    pt = pt.eval().cuda()
    if args.num_layers:
        # truncate to N decoder layers to localize wiring bugs
        base = pt.model if hasattr(pt, "model") else pt
        dec = base.model if hasattr(base, "model") else base
        dec.layers = dec.layers[:args.num_layers]
        if hasattr(dec, "config"):
            dec.config.num_hidden_layers = args.num_layers
        print(f"[ref] truncated PyTorch to {args.num_layers} layers "
              f"({len(dec.layers)} actual)")
    with torch.no_grad():
        out = pt(ids.unsqueeze(0).cuda(), precision=args.precision)
        ref_logits = out.logits[0, -1].float().cpu()
    ref_top = ref_logits.topk(5)
    print("PyTorch top-5:", [(tok.decode([i]), round(v.item(), 3))
                             for v, i in zip(ref_top.values, ref_top.indices)])
    del pt
    torch.cuda.empty_cache()

    # ---- TRT engine ----
    set_precision(args.precision)
    from tensorrt_llm.runtime import ModelRunner
    runner = ModelRunner.from_dir(engine_dir=args.engine_dir)
    outs = runner.generate([ids.cuda()], max_new_tokens=1, temperature=0.0,
                           end_id=tok.eos_token_id,
                           pad_id=tok.pad_token_id or tok.eos_token_id,
                           output_generation_logits=False,
                           gather_context_logits=True, return_dict=True)
    torch.cuda.synchronize()
    ctx = outs["context_logits"][0]            # [seq, vocab]
    trt_logits = ctx[-1].float().cpu()
    trt_top = trt_logits.topk(5)
    print("TRT     top-5:", [(tok.decode([i]), round(v.item(), 3))
                             for v, i in zip(trt_top.values, trt_top.indices)])

    print(f"argmax  PyTorch={ref_logits.argmax().item()}  TRT={trt_logits.argmax().item()}")
    print(f"logit cosine sim = {torch.nn.functional.cosine_similarity(ref_logits, trt_logits, dim=0).item():.4f}")


if __name__ == "__main__":
    main()
