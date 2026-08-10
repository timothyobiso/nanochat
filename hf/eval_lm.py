"""
lm-eval wrapper that understands patched checkpoints.

Loads via load_router_olmoe when the checkpoint carries a fixed_router spec
(bare `pretrained=<path>` inside lm-eval would silently rebuild an unpatched
model), otherwise standard from_pretrained — so the same command evaluates
baselines, controls, and healed models.

Baseline sanity gate (M5, run BEFORE any swap experiment):
  python -m hf.eval_lm --model allenai/OLMoE-1B-7B-0924 --out baseline.json
Published OLMoE-1B-7B-0924 references (paper, for the ~1pt gate): MMLU 5-shot
~54.1, HellaSwag ~80.0, ARC-C ~62.1, PIQA ~79.8, WinoGrande ~72.3.

Healed checkpoint:
  python -m hf.eval_lm --model /path/to/step_002500/model --out healed.json
"""

import argparse
import json
import os

import torch

DEFAULT_TASKS = "mmlu,hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq"


def load_model(path_or_name, dtype, revision=None):
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM
    from hf.patch_olmoe import ROUTER_CONFIG_KEY, load_router_olmoe

    if os.path.isdir(path_or_name):
        config = OlmoeConfig.from_pretrained(path_or_name)
        if getattr(config, ROUTER_CONFIG_KEY, None) is not None:
            return load_router_olmoe(path_or_name, dtype=dtype)
    return OlmoeForCausalLM.from_pretrained(path_or_name, dtype=dtype, revision=revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="HF name or local (possibly patched) checkpoint dir")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--num-fewshot", type=int, default=None, help="override per-task defaults (MMLU uses 5 via --mmlu-fewshot)")
    parser.add_argument("--mmlu-fewshot", type=int, default=5)
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None, help="examples per task (smoke runs)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = load_model(args.model, dtype, args.revision).to(args.device).eval()
    lm = HFLM(pretrained=model, tokenizer=args.tokenizer, batch_size=args.batch_size, device=args.device)

    task_list = args.tasks.split(",")
    # MMLU wants 5-shot; everything else runs 0-shot unless --num-fewshot forces it
    grouped = [(t, args.mmlu_fewshot if t == "mmlu" else args.num_fewshot) for t in task_list]
    results = {}
    for task, fewshot in grouped:
        out = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=fewshot, limit=args.limit)
        results[task] = out["results"]
        print(task, json.dumps(out["results"].get(task, out["results"]), default=str)[:200])

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "tasks": results}, f, indent=2, default=str)
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
