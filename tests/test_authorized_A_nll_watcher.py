import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('A_authorized_watcher',Path(__file__).resolve().parents[1]/'scripts/watch_authorized_A_nll_analysis.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    job={'id':'A','outputs':[],'argv':['synthetic']};digest=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
    return {'closure_jobs':[{'job_id':'A','definition':job}]},{'jobs':{'A':{'status':'pending','definition_sha256':digest}}},[job]


def test_watcher_waits_without_score_reads_then_requires_rc0():
    p,s,q=fixture();assert not m.readiness(p,s,q)['ready'];s['jobs']['A'].update(status='completed',returncode=0)
    assert m.readiness(p,s,q)['ready'] and not m.readiness(p,s,q)['score_outputs_parsed_or_hashed']
    s['jobs']['A']['returncode']=1
    with pytest.raises(ValueError):m.readiness(p,s,q)


@pytest.mark.parametrize('change',['master','state','failed','missing_output'])
def test_watcher_fails_closed_on_source_or_completion_error(change):
    p,s,q=fixture()
    if change=='master':q=[]
    elif change=='state':s['jobs']['A']['definition_sha256']='changed'
    elif change=='failed':s['jobs']['A']['status']='failed'
    else:
        q[0]['outputs']=['/does/not/exist/authorized_A_watcher_output'];s['jobs']['A'].update(status='completed',returncode=0)
        s['jobs']['A']['definition_sha256']=hashlib.sha256(json.dumps(q[0],sort_keys=True).encode()).hexdigest()
    with pytest.raises((ValueError,FileNotFoundError)):m.readiness(p,s,q)
