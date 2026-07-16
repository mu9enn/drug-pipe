import unittest

from drug_agent.gad.train_discriminator import _valid_pairs


class TestWarmupPairs(unittest.TestCase):
    def test_duplicate_sample_ids_are_removed(self):
        row = {
            "sample_id": "x",
            "state_messages": [{"role": "user", "content": "task"}],
            "teacher_response": "teacher",
            "student_response": "student",
        }
        self.assertEqual(len(_valid_pairs([row, dict(row)])), 1)


if __name__ == "__main__":
    unittest.main()
