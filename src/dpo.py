"""DPO loss and training loop for Qwen-0.5B / UltraFeedback"""

import torch
import torch.nn.functional as F
import math
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data import load_ultrafeedback, make_dataloader
import importlib
import json
from pathlib import Path

def dpo_loss(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        beta: float,
):
    """
    DPO loss:
        L = - E[ log sigmoid(beta * ( (logp_pi(y_w) - logp_ref(y_w)) 
                                    - (logp_pi(y_l) - logp_ref(y_l)) ) ) ]

    Returns:
        loss:               scalar
        chosen_rewards:     beta * (logp_pi(y_w) - logp_ref(y_w))      # (B) detached
        rejected_rewards:   beta * (logp_pi(y_l) - logp_ref(y_l))      # (B) detached
    """

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    return loss, chosen_rewards.detach(), rejected_rewards.detach()

def _flash_attn_available() ->bool:
    return importlib.util.find_spec("flash_attn") is not None

def train(
        model_name:str="Qwen/Qwen2.5-0.5B-Instruct",
        beta:float=0.1,
        lr:float=5e-7,
        batch_size:int=2,
        grad_accum:int=8,
        num_epochs:int=1,
        max_steps:int|None=None,
        log_every:int=10,
        seed:int=42,
        max_prompt_len=512,
        max_total_len=1024,
        run_name:str="qwen_dpo_b01",
        out_root:str="results",
        attn_impl:str="auto",
        device:str="cuda",
):
    """Train DPO on UltraFeedback. Policy + frozen ref both = model_name."""
    torch.manual_seed(seed)

    out_dir = Path(out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] writing to {out_dir}")

    with open(out_dir/"config.json","w") as f:
        json.dump({
        "model_name": model_name, "beta": beta, "lr": lr,
          "batch_size": batch_size, "grad_accum": grad_accum,
          "num_epochs": num_epochs, "max_steps": max_steps, "seed": seed,
          "max_prompt_len": max_prompt_len, "max_total_len": max_total_len,
        }, f, indent=2)
    
    metrics_f = open(out_dir / "metrics.jsonl", "w", buffering=1)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if attn_impl in ("auto", "flash_attention_2"):
        attn_impl = "flash_attention_2" if(
                device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 and _flash_attn_available()
            ) else "sdpa"
    
    print(f"[train] attn_impl={attn_impl}")

    print(f"[train] loading policy + ref from {model_name}")
    policy = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation=attn_impl,
    ).to(device)
    policy.train()

    ref = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation=attn_impl,
    ).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    
    dataset = load_ultrafeedback(
        tokenizer, max_prompt_len=max_prompt_len, max_total_len=max_total_len,
    )

    loader = make_dataloader(dataset, tokenizer, batch_size=batch_size,shuffle=True)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    print(f"[train] beta={beta}, lr={lr}, bs={batch_size}, grad_accum={grad_accum},"
          f"effective batch_size={batch_size*grad_accum}")
    
    micro = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        for batch in loader:
            batch = {k: v.to(device) for k,v in batch.items()}

            # policy with grads
            policy_chosen_logps = compute_logps(
                policy, batch["chosen_input_ids"],
                batch["chosen_attention_mask"], batch["chosen_response_mask"],
            )

            policy_rejected_logps = compute_logps(
                policy, batch["rejected_input_ids"],
                batch["rejected_attention_mask"], batch["rejected_response_mask"],
            )

            # ref no grads
            with torch.no_grad():
                ref_chosen_logps = compute_logps(
                    ref, batch["chosen_input_ids"],
                    batch["chosen_attention_mask"], batch["chosen_response_mask"],
                )

                ref_rejected_logps = compute_logps(
                    ref, batch["rejected_input_ids"],
                    batch["rejected_attention_mask"], batch["rejected_response_mask"],
                )
            
            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps, policy_rejected_logps, 
                ref_chosen_logps, ref_rejected_logps, 
                beta,
            )

            # First batch init check:
            if micro==0:
                assert abs(loss.item() - math.log(2)) < 0.05, (
                    f"first batch loss {loss.item():.4f} != log(2) ={math.log(2):.4f};"
                    f"policy/ref init mismatch"
                )
                print(f"[train] init check OK: loss={loss.item():.4f} ~ log(2)")
            
            (loss / grad_accum).backward()

            if micro % log_every == 0:
                margin = (chosen_rewards - rejected_rewards).mean().item()
                acc = (chosen_rewards > rejected_rewards).float().mean().item()
                print(
                    f"epoch={epoch} micro={micro:6d} loss={loss.item():.4f} "
                    f"chosen_r={chosen_rewards.mean().item():+.4f} "
                    f"rejected_r={rejected_rewards.mean().item():+.4f} "
                    f"margin={margin:.3f} acc={acc:.3f} ",
                    flush=True
                )

                row = {
                    "epoch": epoch, "micro": micro,
                    "loss": loss.item(),
                    "chosen_r": chosen_rewards.mean().item(),
                    "rejected_r": rejected_rewards.mean().item(),
                    "margin": margin, "acc": acc,
                }
                metrics_f.write(json.dumps(row)+"\n")
            
            micro += 1
            if micro % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            if max_steps is not None and micro >= max_steps:
                print(f"[train] hit max steps {max_steps}, stopping")
                metrics_f.close()
                ckpt_dir = out_dir / "checkpoint"
                print(f"[train] saving policy to {ckpt_dir}")
                policy.save_pretrained(ckpt_dir)
                return
    
    metrics_f.close()
    ckpt_dir = out_dir / "checkpoint"
    print(f"[train] saving policy to {ckpt_dir}")
    policy.save_pretrained(ckpt_dir)

    print(f"train done")

def compute_logps(model, input_ids, attn_mask, resp_mask):
    """ Teacher-forced sum log-prob over response tokens.
    Returns: 
        (B,) tensor of summed log_probabilities
    """

    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits # (B,T,C)

    logits = logits[:,:-1,:]    # (B, T-1, C)
    targets = input_ids[:,1:]   # (B, T-1)
    loss_mask = resp_mask[:,1:].float()

    logprobs = F.log_softmax(logits, dim=-1)    # (B,T-1, C)
    token_logps = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) # (B,T-1)

    return (token_logps * loss_mask).sum(dim=-1) # (B,)

def _sanity_check_loss():
    # Sanity check 1: zero grad when policy == ref

    p_chosen = torch.tensor([-2.0, -2.5])
    p_rejected = torch.tensor([-2.5, -3.0])

    r_chosen = p_chosen.clone()
    r_rejected = p_rejected.clone()

    loss, c_r, r_r = dpo_loss(p_chosen,p_rejected, r_chosen, r_rejected, beta=0.1)

    print(f"policy == ref: loss = {loss.item():.4f} (expect log(2) = 0.6931)")
    print(f"chosen_reward={c_r.tolist()}    rejected_reward={r_r.tolist()} (expect zeros)")

    
    # Sanity check 2: policy already prefers chosen more than ref does -> loss < log (2)

    p_chosen = torch.tensor([-1.0])     # policy moved chosen higher
    p_rejected = torch.tensor([-4.0])   # policy moved rejected lower

    r_chosen = torch.tensor([-2.0])
    r_rejected = torch.tensor([-3.0])

    loss, c_r, r_r = dpo_loss(p_chosen,p_rejected, r_chosen, r_rejected, beta=0.1)

    print(f"policy > ref: loss = {loss.item():.4f} (expect < 0.6931)")
    print(f"chosen_reward={c_r.tolist()} (expect +0.1)   rejected_reward={r_r.tolist()} (expect -0.1)")

    # Sanity check 3: gradient direction on policy logps
    p_chosen = torch.tensor([-2.0], requires_grad=True)
    p_rejected = torch.tensor([-2.0], requires_grad=True)

    r_chosen = torch.tensor([-2.0])
    r_rejected = torch.tensor([-2.0])

    loss, _, _ = dpo_loss(p_chosen,p_rejected, r_chosen, r_rejected, beta=0.1)
    loss.backward()
    
    print(f"grad chosen={p_chosen.grad.item():.4f} (expect negative -> push UP)")
    print(f"grad rejected={p_rejected.grad.item():.4f} (expect positive -> push Down)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="run loss sanity tests then exit"
    )
    
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model id"
    )

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="qwen_dpo_b01")
    parser.add_argument(
        "--attn-impl", 
        type=str, 
        default="auto", 
        choices=["auto", "flash_attention_2","sdpa","eager"]
    )

    args = parser.parse_args()

    if args.sanity:
        _sanity_check_loss()
    else:
        train(
            model_name=args.model_name, beta=args.beta, lr=args.lr,
            batch_size=args.batch_size, grad_accum=args.grad_accum,
            max_steps=args.max_steps, log_every=args.log_every,
            run_name=args.run_name, attn_impl=args.attn_impl,
        )

