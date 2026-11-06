#!/usr/bin/env python3
"""Temporary lock/false authority/AST checks only; no actual restoration."""
import ast
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREP = ROOT / 'artifacts/provenance/B_native_restore_preparation_v1'

def hashed(path, expected):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    return raw

path = PREP / 'run_B_restore.py'
raw = hashed(path, '6d7245e78e225c3879eaf931b85e7be073b4ba9777d7263b7e0e1c39b1630a02')
spec = importlib.util.spec_from_file_location('independent_B_shared_lock_revision', path)
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
native = r.load_helper('restore_native_final.py', 'independent_B_pinned_native')
whole = r.load_helper('run_whole_child.py', 'independent_B_pinned_PG')
plan_path = PREP / 'bound_plan_v2.json'
plan_sha = '61c19ee07d69936da329846fcb8aa8bd67ea8d8fe1ff8cd6b3084c032c3fa3c7'
plan = json.loads(hashed(plan_path, plan_sha)); r.validate(plan)
assert plan['sequential_restore_lock'] == str(r.QA.parent / '.native_Cfinal_restore.lock')
assert Path(plan['sequential_restore_lock']) == native.SCRATCH / '.native_Cfinal_restore.lock'
old_plan = json.loads(hashed(PREP/'bound_plan.json', '903baac7f57172387bbda7d8f31785d3ac5cd29fe52739b5c856e12d6e74663b'))
comparison = dict(plan); comparison.pop('sequential_restore_lock'); comparison['runner'] = old_plan['runner']
assert comparison == old_plan
authority = PREP / 'ROOT_execution_authority_v2_FALSE_TEMPLATE.json'
args = SimpleNamespace(plan=str(plan_path), plan_sha256=plan_sha,
    authority=str(authority), authority_sha256='c6cbf2c88b5d4f4daa2749f6a9b6606cac94133c84c80afa83ca031858c51d53')
assert json.loads(hashed(authority,args.authority_sha256))['authorized'] is False
try:
    r.authorize(args,native)
except ValueError as err:
    assert 'false template' in str(err)
else:
    raise AssertionError('Actual FALSE template must refuse')

# Exercise only the new wrapper with a temporary stand-in body that raises.
# No true authority is created, resource/TAR routines are never called.
called=[]
with tempfile.TemporaryDirectory(prefix='B_independent_shared_lock_') as tmp:
    r.QA = Path(tmp) / 'never_created_QA'
    r.authorize = lambda *_: ({},{},{})
    def raises_inside_lock(*_):
        with (r.QA.parent / '.native_Cfinal_restore.lock').open('a') as competitor:
            try:
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                called.append('shared_lock_held')
            else:
                raise AssertionError('Lock not held during body')
        raise RuntimeError('toy failure before resource/extraction')
    r.body_locked = raises_inside_lock
    try:
        r.body(None,None)
    except RuntimeError as err:
        assert str(err) == 'toy failure before resource/extraction'
    else:
        raise AssertionError('Expected toy exception')
    with (r.QA.parent / '.native_Cfinal_restore.lock').open('a') as after:
        fcntl.flock(after, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert called == ['shared_lock_held'] and not r.QA.exists()

before = ast.parse((HERE/'retained_e3aa_run_B_restore.py').read_bytes()); after = ast.parse(raw)
old_fn = {n.name:n for n in before.body if isinstance(n,ast.FunctionDef)}
new_fn = {n.name:n for n in after.body if isinstance(n,ast.FunctionDef)}
for name in set(old_fn)-{'body','main'}:
    assert ast.dump(old_fn[name],include_attributes=False) == ast.dump(new_fn[name],include_attributes=False)
old_body = old_fn['body']; locked = new_fn['body_locked']
assert ast.dump(ast.Module(body=old_body.body[1:],type_ignores=[]),include_attributes=False) == \
       ast.dump(ast.Module(body=locked.body,type_ignores=[]),include_attributes=False)
deadline=dt.datetime.fromisoformat(plan['resource_policy']['deadline_utc'])
try:
    whole.admission(plan['resource_policy'], now=deadline-dt.timedelta(seconds=900))
except ValueError:
    pass
else:
    raise AssertionError('Exact 900s remaining must refuse')
assert whole.admission(plan['resource_policy'],now=deadline-dt.timedelta(seconds=901))['cap_seconds']==600

result=dict(status='passed_independent_B_shared_lock_revision_preparation_only',
    current_runner_sha256=hashlib.sha256(raw).hexdigest(), current_plan_sha256=plan_sha,
    actual_FALSE_authority_refused=True, same_native_C_lock_path=True,
    toy_lock_held_and_exception_releases=True, real_QA_created=False,
    old_body_AST_unchanged_inside_lock=True, other_original_functions_AST_unchanged=True,
    exact_900seconds_refused_901_allowed=True, helper_imports_SHA_pinned=True,
    actual_restore_execute_or_extract_called=False, actual_large_payload_or_headers_or_score_bytes_read=0)
with (HERE/'lock_revision_probe.json').open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
