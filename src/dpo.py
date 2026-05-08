"""DPO loss and training loop for Qwen-0.5B / UltraFeedback"""

import torch
import torch.nn.functional as F

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


if __name__ == "__main__":
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

