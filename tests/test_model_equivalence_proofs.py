import importlib.util
from pathlib import Path
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'scripts/make_model_equivalence_proofs.py'
spec=importlib.util.spec_from_file_location('model_proofs',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_full_state_proof_ignores_run_metadata_but_binds_checkpoint_bytes(tmp_path):
    a=tmp_path/'a.pt';b=tmp_path/'b.pt';weights={'weight':torch.tensor([[1.,2.],[3.,4.]])}
    torch.save({'model_state':weights,'config':{'run_label':'B4'}},a)
    torch.save({'model_state':weights,'config':{'run_label':'D'}},b)
    left=module.checkpoint_identity(a);right=module.checkpoint_identity(b)
    assert left['model_state_fingerprint']==right['model_state_fingerprint']
    assert left['checkpoint_sha256']!=right['checkpoint_sha256']
    assert left['model_state_fingerprint']['state_elements']==4
    weights['weight'][0,0]+=1;torch.save({'model_state':weights},b)
    with pytest.raises(ValueError,match='differs from completed audit'):
        module.checkpoint_identity(b,left['model_state_fingerprint'])
