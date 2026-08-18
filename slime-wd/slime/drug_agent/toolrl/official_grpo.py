"""ToolRL 8cee13e GRPO advantage adapter for Slime.

The official repository subtracts token-level reference KL from the terminal
rule reward *before* summing and normalizing the n responses of a prompt.
Slime's generic GRPO path normalizes rule rewards during rollout and does not
apply ``kl_coef`` in ``get_grpo_returns``. This adapter restores the official
ordering without changing Slime globally.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import torch


def compute_official_8cee13e_advantages(args: Namespace, rollout_data: dict[str, Any]) -> None:
    raw_rewards = rollout_data.get("raw_reward")
    kl = rollout_data.get("kl")
    loss_masks = rollout_data.get("loss_masks")
    if raw_rewards is None or kl is None or loss_masks is None:
        raise ValueError("official ToolRL advantage requires raw_reward, kl, and loss_masks")
    n = int(args.n_samples_per_prompt)
    if n != 4 or len(raw_rewards) % n:
        raise ValueError(f"official ToolRL requires complete n=4 groups, got n={n}, samples={len(raw_rewards)}")

    scores: list[torch.Tensor] = []
    for reward, token_kl, mask in zip(raw_rewards, kl, loss_masks, strict=True):
        local_mask = mask.to(device=token_kl.device, dtype=token_kl.dtype)
        scores.append(
            torch.as_tensor(float(reward), device=token_kl.device, dtype=torch.float32)
            - float(args.kl_coef) * (token_kl.float() * local_mask.float()).sum()
        )

    advantages: list[torch.Tensor] = []
    for group_start in range(0, len(scores), n):
        group = torch.stack(scores[group_start : group_start + n])
        normalized = (group - group.mean()) / (group.std(unbiased=True) + 1e-6)
        for offset, scalar_advantage in enumerate(normalized):
            sample_index = group_start + offset
            advantages.append(torch.ones_like(kl[sample_index], dtype=torch.float32) * scalar_advantage)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [item.clone() for item in advantages]

