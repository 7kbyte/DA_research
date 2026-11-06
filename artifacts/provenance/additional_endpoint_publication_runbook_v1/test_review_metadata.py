"""Tiny fail-closed tests for the independent, non-live metadata reviewer."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('publication_reviewer',Path(__file__).with_name('review_metadata.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.mark.parametrize('code',[False,True,0.0,'0',None])
def test_rc_must_be_exact_integer_zero(code):
    with pytest.raises(ValueError):m.require_clean_rc({'returncode':code})

@pytest.mark.parametrize('key',['stop_requested','stop_requested_at'])
@pytest.mark.parametrize('value',[None,False,'2026-11-06'])
def test_stop_key_present_is_rejected_regardless_value(key,value):
    with pytest.raises(ValueError):m.require_clean_rc({'returncode':0,key:value})

def test_good_zero_and_scope_hashes():
    m.require_clean_rc({'returncode':0})
    item={'bound':{'config_path':'cfg','summary_path':'sum','input_sha256':{'cfg':'C','sum':'S'}}}
    good={'config_sha256':'C','summary_sha256':'S'}
    m.require_scope_config_hashes(good,item)
    for key in good:
        with pytest.raises(ValueError):m.require_scope_config_hashes({**good,key:'changed'},item)

def test_pending_package_never_emits_acceptance(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'BASE',tmp_path/'empty')
    for name in m.NAMES:
        out=tmp_path/(name+'.json')
        with pytest.raises(ValueError,match='Pending:'):m.review(name,out)
        assert not out.exists()
