import unittest
from pipeline.cleaning.recover_historical_answers import identity, rank_predictions

class ReconstructionTests(unittest.TestCase):
    def test_interrupt_does_not_become_arbitrary_ranking(self):
        with self.assertRaises(ValueError): rank_predictions(['CC','CO'],{},[])
    def test_numeric_order_and_explicit_uncertainty_tail(self):
        rank, missing=rank_predictions(['CO','CC','CN','CF'],{'CC':-7,'CN':-8},['CC','CN','CO','CC'])
        self.assertEqual(rank,['CN','CC','CO','CF']);self.assertEqual(missing,['CF'])
    def test_no_candidate_order_completion(self):
        self.assertEqual(rank_predictions(['CO','CN','CC'],{},['CO'])[0],['CO','CC','CN'])
    def test_rejects_nan_and_outside_candidate(self):
        for scores in ({'CC':float('nan')},{'CF':-9}):
            with self.assertRaises(ValueError):rank_predictions(['CC','CO'],scores,['CC'])
    def test_no_stereoisomer_rewrite(self):
        self.assertIsNone(identity('C[C@@H](O)F',['C[C@H](O)F']))
    def test_isomeric_equivalence(self):
        self.assertEqual(identity('OCC',['CCO']),'CCO')
