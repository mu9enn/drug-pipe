from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def bradley_terry_loss(positive_scores, negative_scores):
    torch, _, _ = _require_ml()
    return -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()


def _require_ml():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("GAD discriminator requires torch and transformers on the GPU worker") from exc
    return torch, AutoModel, AutoTokenizer


class GADDiscriminator:
    def __init__(self, model_path: str, *, lr: float = 1e-5, max_length: int = 4096, device: str = "cuda"):
        torch, AutoModel, AutoTokenizer = _require_ml()
        self.torch = torch
        self.model_path = model_path
        self.max_length = max_length
        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        # The candidate is appended after the state, so preserve the tail when
        # a long trajectory exceeds the discriminator context window.
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # A discriminator only needs hidden states. AutoModel avoids materializing
        # the enormous seq_len x vocabulary LM logits produced by a causal-LM head.
        self.backbone = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device)
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        text_config = getattr(self.backbone.config, "text_config", None)
        hidden_size = hidden_size or getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot determine discriminator hidden size")
        self.score_head = torch.nn.Linear(hidden_size, 1, dtype=dtype, device=self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.score_head.parameters()), lr=lr, weight_decay=0.01
        )
        self.version = 0
        self.running_count = 0
        self.running_mean = 0.0
        self.running_m2 = 0.0

    def _render(self, states: list[list[dict[str, Any]]], candidates: list[str]) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(
                state + [{"role": "assistant", "content": candidate}], tokenize=False, add_generation_prompt=False
            )
            for state, candidate in zip(states, candidates, strict=True)
        ]

    def score(self, states: list[list[dict[str, Any]]], candidates: list[str], *, train: bool = False):
        torch = self.torch
        texts = self._render(states, candidates)
        batch = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.set_grad_enabled(train):
            output = self.backbone(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden = output.hidden_states[-1]
            last_index = batch["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=self.device), last_index]
            return self.score_head(pooled).float().squeeze(-1)

    def score_and_update(
        self,
        states: list[list[dict[str, Any]]],
        teacher_responses: list[str],
        student_responses: list[str],
        *,
        update_steps: int = 1,
        clip_grad: float = 1.0,
        reward_clip: float = 2.0,
    ) -> dict[str, Any]:
        torch = self.torch
        self.backbone.eval()
        self.score_head.eval()
        with torch.no_grad():
            raw_student = self.score(states, student_responses).detach()
        self.backbone.train()
        self.score_head.train()
        before = self.version
        metrics = {}
        for _ in range(update_steps):
            positive = self.score(states, teacher_responses, train=True)
            negative = self.score(states, student_responses, train=True)
            margin = positive - negative
            loss = bradley_terry_loss(positive, negative)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.backbone.parameters()) + list(self.score_head.parameters()), clip_grad
            )
            self.optimizer.step()
            self.version += 1
            metrics = {
                "loss": float(loss.detach()),
                "accuracy": float((margin.detach() > 0).float().mean()),
                "positive_score_mean": float(positive.detach().mean()),
                "negative_score_mean": float(negative.detach().mean()),
                "score_margin": float(margin.detach().mean()),
                "grad_norm": float(grad_norm),
            }
        normalized = self.normalize_batch_and_update(raw_student.cpu().tolist(), clip=reward_clip)
        return {
            "raw_scores": raw_student.cpu().tolist(),
            "normalized_scores": normalized,
            "version_before": before,
            "version_after": self.version,
            "metrics": metrics,
        }

    def normalize_batch_and_update(self, values: list[float], clip: float = 2.0) -> list[float]:
        """Normalize a reward batch against pre-update running statistics.

        Scoring every item against the same snapshot avoids making rewards
        depend on their order inside a GRPO group. The batch is incorporated
        into the running moments only after all normalized scores are built.
        """
        values = [float(value) for value in values]
        if not values:
            return []
        if self.running_count >= 2:
            mean = self.running_mean
            variance = self.running_m2 / (self.running_count - 1)
        else:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        std = max(variance**0.5, 1e-6)
        normalized = [max(-clip, min(clip, (value - mean) / std)) for value in values]
        for value in values:
            self.running_count += 1
            delta = value - self.running_mean
            self.running_mean += delta / self.running_count
            self.running_m2 += delta * (value - self.running_mean)
        return normalized

    def save(self, path: str) -> None:
        torch = self.torch
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out / "backbone")
        self.tokenizer.save_pretrained(out / "backbone")
        torch.save(
            {
                "score_head": self.score_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "version": self.version,
                "running_count": self.running_count,
                "running_mean": self.running_mean,
                "running_m2": self.running_m2,
            },
            out / "gad_state.pt",
        )
        (out / "metadata.json").write_text(json.dumps({"model_path": self.model_path, "version": self.version}, indent=2))

    def load(self, path: str) -> None:
        state = self.torch.load(Path(path) / "gad_state.pt", map_location=self.device, weights_only=False)
        self.score_head.load_state_dict(state["score_head"])
        self.optimizer.load_state_dict(state["optimizer"])
        for key in ("version", "running_count", "running_mean", "running_m2"):
            setattr(self, key, state.get(key, getattr(self, key)))
