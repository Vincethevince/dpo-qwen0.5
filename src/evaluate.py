import torch
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import load_ultrafeedback, make_dataloader
from src.dpo import compute_logps, _flash_attn_available

@torch.no_grad()
def evaluate(
    ckpt_dir: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    split:str="test_prefs",
    beta: float = 0.1,
    batch_size: int = 4,
    max_batches: int | None = None,
    max_prompt_len: int = 512,
    max_total_len: int = 1024,
    attn_impl: str = "auto",
    device: str = "cuda",
):
    """Run held-out preference eval. Loads policy from ckpt_dir, ref from base_model"""

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if attn_impl in ("auto", "flash_attention_2"):
        attn_impl = "flash_attention_2" if(
                device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 and _flash_attn_available()
            ) else "sdpa"
    
    print(f"[eval] attn_impl={attn_impl}")

    print(f"[eval] loading policy from {ckpt_dir}")
    policy = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, dtype=torch.bfloat16, attn_implementation=attn_impl
    ).to(device)
    policy.eval()

    print(f"[eval] loading ref from {base_model}")
    ref = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, attn_implementation=attn_impl
    ).to(device)
    ref.eval()

    dataset = load_ultrafeedback(
        tokenizer, split=split, max_prompt_len=max_prompt_len, max_total_len=max_total_len,
    )

    loader = make_dataloader(dataset,tokenizer, batch_size=batch_size, shuffle=False)

    n_pairs = 0
    n_correct = 0
    sum_margin = 0.0
    sum_chosen_r = 0.0
    sum_rejected_r = 0.0

    for i, batch in enumerate(loader):
        batch = {k:v.to(device) for k,v in batch.items()}

        pc = compute_logps(
            policy, batch["chosen_input_ids"],
            batch["chosen_attention_mask"], batch["chosen_response_mask"],
        )

        pr = compute_logps(
            policy, batch["rejected_input_ids"],
            batch["rejected_attention_mask"], batch["rejected_response_mask"],
        )
        
        rc = compute_logps(
            ref, batch["chosen_input_ids"],
            batch["chosen_attention_mask"], batch["chosen_response_mask"],
        )

        rr = compute_logps(
            ref, batch["rejected_input_ids"],
            batch["rejected_attention_mask"], batch["rejected_response_mask"],
        )

        chosen_r = beta * (pc - rc)
        rejected_r = beta * (pr - rr)
        margin = chosen_r - rejected_r

        n_correct += (margin>0).sum().item()
        n_pairs += margin.size(0)
        sum_margin += margin.sum().item()
        sum_chosen_r += chosen_r.sum().item()
        sum_rejected_r += rejected_r.sum().item()

        if (i+1) % 20 ==0:
            print(f"[eval] batch{i+1}: running pref_acc = {n_correct/n_pairs:.3f}", flush=True)

        if max_batches is not None and (i+1) >= max_batches:
            print(f"[eval] hit max_batches={max_batches}, stopping early")
            break

    results = {
        "ckpt": str(ckpt_dir),
        "base_model": base_model,
        "split": split,
        "beta": beta,
        "n_pairs": n_pairs,
        "pref_acc": n_correct / n_pairs,
        "mean_margin": sum_margin / n_pairs,
        "mean_chosen_r": sum_chosen_r / n_pairs,
        "mean_rejected_r": sum_rejected_r / n_pairs,
    }

    print(f"[eval] {json.dumps(results, indent=2)}")

    out_path = Path(ckpt_dir).parent / f"eval_{split}.json"
    with open(out_path,"w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[eval] wrote {out_path}")

    return results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", 
        type=str, 
        required=True,
        help="checkpoint dir, e.g. results/qwen_dpo_b01/checkpoint"
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--split", type=str, default="test_prefs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--attn-impl", 
        type=str, 
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"])
    args = parser.parse_args()

    evaluate(ckpt_dir=args.ckpt, base_model=args.base_model, beta=args.beta,
             split=args.split, batch_size=args.batch_size, max_batches=args.max_batches,
             attn_impl=args.attn_impl
    )

