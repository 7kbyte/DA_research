"""CPU invariants for the independent deployment-cost diagnostic."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from replay_study.evaluate import extract_teacher_statistics
source = Path(__file__).resolve().parents[1] / 'scripts/profile_selection_costs.py'
spec = importlib.util.spec_from_file_location('selection_cost_diagnostic', source)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)

class SelectionCostDiagnosticTests(unittest.TestCase):
    def test_base_features_match_joint_without_lm_head(self):
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        torch.manual_seed(811)
        model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=17, hidden_size=16,
            num_hidden_layers=2, num_attention_heads=4, intermediate_size=32,
            max_position_embeddings=32, hidden_dropout=0., attention_dropout=0.)).train()
        data = {'input_ids': torch.tensor([[0,3,4,5,6],[1,2,8,9,10]]),
                'attention_mask': torch.tensor([[1,1,1,0,0],[0,1,1,1,1]])}
        joint = extract_teacher_statistics(model,data,batch_size=1,precision='fp32')
        with patch.object(model.embed_out,'forward',side_effect=AssertionError('LM head executed')):
            features,timing = diagnostic.measure_pass(lambda:diagnostic.feature_only_statistics(
                model,data,batch_size=1,precision='fp32'),'cpu')
        self.assertTrue(model.training)
        torch.testing.assert_close(features['features'],joint['features'],rtol=0,atol=0)
        self.assertEqual(features['prediction_token_counts'].tolist(),[2,3])
        self.assertTrue(diagnostic.compare_tensors(features['features'],joint['features'])['exact_equal'])
        self.assertIsNone(timing['cuda_event_elapsed_seconds'])
        self.assertGreaterEqual(timing['wall_seconds'],0.)
        nll=extract_teacher_statistics(model,data,batch_size=1,include_features=False,precision='fp32')
        torch.testing.assert_close(nll['teacher_nll'],joint['teacher_nll'],rtol=0,atol=0)
        self.assertNotIn('features',nll)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_comparison_reports_errors_without_tolerance_hiding(self):
        row=diagnostic.compare_tensors(torch.tensor([1.,3.]),torch.tensor([1.,2.]))
        self.assertFalse(row['exact_equal'])
        self.assertEqual(row['max_absolute_error'],1.)
        self.assertEqual(row['mean_absolute_error'],.5)
        self.assertAlmostEqual(row['relative_l2_error'],1/(5**.5))
        with self.assertRaisesRegex(ValueError,'shape'):
            diagnostic.compare_tensors(torch.zeros(1),torch.zeros(2))

if __name__=='__main__':unittest.main()
