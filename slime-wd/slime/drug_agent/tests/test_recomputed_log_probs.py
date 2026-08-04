from __future__ import annotations

import os
import unittest

try:
    import torch
    import torch.distributed as dist

    from slime.utils.ppo_utils import calculate_log_probs_and_entropy
except ModuleNotFoundError:  # The lightweight host test environment has no torch.
    torch = None
    dist = None
    calculate_log_probs_and_entropy = None


@unittest.skipIf(torch is None, "torch is unavailable in the host test environment")
class RecomputedLogProbsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        cls.rank = int(os.environ.get("RANK", "0"))
        if cls.world_size > 1:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl")
            cls.device = torch.device("cuda", local_rank)
            cls.process_group = dist.group.WORLD
        else:
            cls.device = torch.device("cpu")
            cls.process_group = None

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_matches_full_vocab_log_softmax_forward_and_backward(self) -> None:
        torch.manual_seed(17)
        token_count = 37
        global_vocab = 96
        partition_vocab = global_vocab // self.world_size

        source = torch.randn(token_count, global_vocab, device=self.device)
        tokens = torch.randint(0, global_vocab, (token_count,), device=self.device)
        weights = torch.randn(token_count, 1, device=self.device)
        start = self.rank * partition_vocab
        end = start + partition_vocab

        local_logits = source[:, start:end].clone().requires_grad_(True)
        exact_logits = source.clone().requires_grad_(True)
        actual, _ = calculate_log_probs_and_entropy(
            local_logits,
            tokens,
            self.process_group,
            chunk_size=8,
            recompute_log_probs=True,
        )
        expected = torch.log_softmax(exact_logits, dim=-1).gather(-1, tokens.unsqueeze(-1))

        (actual * weights).sum().backward()
        (expected * weights).sum().backward()

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            local_logits.grad,
            exact_logits.grad[:, start:end],
            atol=1e-6,
            rtol=1e-6,
        )

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "BF16 test requires CUDA")
    def test_bf16_logits_use_fp32_tile_math(self) -> None:
        torch.manual_seed(23)
        token_count = 41
        global_vocab = 128
        partition_vocab = global_vocab // self.world_size
        source = torch.randn(token_count, global_vocab, device=self.device).bfloat16()
        tokens = torch.randint(0, global_vocab, (token_count,), device=self.device)
        weights = torch.randn(token_count, 1, device=self.device)
        start = self.rank * partition_vocab
        end = start + partition_vocab

        local_logits = source[:, start:end].clone().requires_grad_(True)
        exact_logits = source.float().clone().requires_grad_(True)
        actual, _ = calculate_log_probs_and_entropy(
            local_logits,
            tokens,
            self.process_group,
            chunk_size=9,
            recompute_log_probs=True,
        )
        expected = torch.log_softmax(exact_logits, dim=-1).gather(-1, tokens.unsqueeze(-1))

        (actual * weights).sum().backward()
        (expected * weights).sum().backward()

        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            local_logits.grad.float(),
            exact_logits.grad[:, start:end],
            atol=6e-3,
            rtol=6e-3,
        )


if __name__ == "__main__":
    unittest.main()
