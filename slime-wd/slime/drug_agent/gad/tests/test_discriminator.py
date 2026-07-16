import unittest


class TestBradleyTerry(unittest.TestCase):
    def test_batch_normalization_is_order_independent(self):
        from drug_agent.gad.discriminator import GADDiscriminator

        discriminator = object.__new__(GADDiscriminator)
        discriminator.running_count = 0
        discriminator.running_mean = 0.0
        discriminator.running_m2 = 0.0
        scores = discriminator.normalize_batch_and_update([-1.0, 1.0])
        self.assertAlmostEqual(scores[0], -scores[1])
        self.assertAlmostEqual(discriminator.running_mean, 0.0)

    def test_loss_and_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is only available on the GPU worker")
        from drug_agent.gad.discriminator import bradley_terry_loss

        positive = torch.tensor([0.0, 0.5], requires_grad=True)
        negative = torch.tensor([0.5, 0.0], requires_grad=True)
        loss = bradley_terry_loss(positive, negative)
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertIsNotNone(positive.grad)
        self.assertLess(float(positive.grad.mean()), 0)


if __name__ == "__main__":
    unittest.main()
