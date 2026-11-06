"""Create independent v3 operational repair from unchanged v2 source text.

This builder never calls a real check/collector or creates a score authority.
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function(source, name):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno])


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Mechanical repair expected exactly one source fragment: ' + old)
    return source.replace(old, new, 1)


def main():
    old_guard = ROOT / 'src/replay_study/supplemental_analysis_guard_v2.py'
    old_cli = ROOT / 'scripts/analyze_supplemental_phase_v2.py'
    if digest(old_guard) != '64c89e37be3d595c239ad2683d72a85cdef6db776374ef6eeae28e23b97c3aea':
        raise ValueError('Retained v2 guard changed')
    if digest(old_cli) != 'a2f613babda5f1fd3b520259515b1acda90703ef42d3c7089c32c7e57497af3a':
        raise ValueError('Retained v2 CLI changed')
    old = old_guard.read_text()
    inspect = function(old, 'inspect')
    inspect = replace_once(inspect, 'def inspect(plan_proof,state_path):', 'def inspect(plan_proof,state_path,repair_proof):')
    inspect = replace_once(inspect, 'plan,v=load_plan(plan_proof);', 'plan,v=load_plan(plan_proof,repair_proof);')
    inspect = replace_once(inspect,
        "        if s.get('status')!='completed' or s.get('returncode')!=0 or s.get('missing_outputs'):pending.append(item['job_id'])",
        """        if s.get('status') not in ('pending','running','completed'):raise ValueError('Scheduler closure has failed/unknown status: '+item['job_id'])
        if 'stop_requested_at' in s or 'stop_requested' in s:raise ValueError('Scheduler closure was stop-requested: '+item['job_id'])
        if s.get('status')=='completed':
            if type(s.get('returncode')) is not int or s['returncode']!=0:raise ValueError('Scheduler completed closure requires integer rc0: '+item['job_id'])
            if s.get('argv')!=definition['argv']:raise ValueError('Scheduler actual wrapper argv differs: '+item['job_id'])
        if s.get('status')!='completed' or s.get('missing_outputs'):pending.append(item['job_id'])""")
    inspect = replace_once(inspect,
        "result={'schema_version':2,'operational_revision':2,",
        "result={'schema_version':3,'operational_revision':3,'operational_repair':repair_proof,")
    inspect = replace_once(inspect,
        "        for name in ('source_audit','source_registry','source_snapshot','producer_source'):\n            if p.get(name):v.proof(p[name])",
        '        weights_sources(v,p)')
    prelude = '''"""Explicit independent operational repair; retained v2 bytes and plans stay valid.

Only the producer source JSON/opaque distinction and stricter scheduler/cache
completion evidence change. Statistics, frozen worker/collector inputs, cache
allowlist and the manifest-scoped first-read journal remain retained v2 policy.
"""
from .supplemental_analysis_guard_v2 import *
from . import supplemental_analysis_guard_v2 as retained


ROOT = Path(__file__).resolve().parents[2]
RETAINED_SOURCES = (
    'src/replay_study/supplemental_analysis_guard_v2.py',
    'scripts/analyze_supplemental_phase_v2.py',
)
ACTIVE_SOURCES = (
    'src/replay_study/supplemental_analysis_guard_v3.py',
    'scripts/analyze_supplemental_phase_v3.py',
)


def validate_repair(plan,repair_proof,verifier):
    repair=verifier.proof(repair_proof)
    if (repair.get('schema_version')!=3 or repair.get('operational_revision')!=3 or
            repair.get('kind')!='supplemental_analysis_guard_operational_bugfix' or
            repair.get('retained_protocol')!=plan['protocol'] or
            repair.get('scientific_or_statistics_changes') is not False or
            repair.get('score_access_authorized') is not False or
            repair.get('first_read_rule')!='exact_retained_v2_plan_manifest_namespace'):
        raise ValueError('Explicit non-authorizing operational repair required')
    for key,names in (('retained_source_sha256',RETAINED_SOURCES),('active_source_sha256',ACTIVE_SOURCES)):
        expected={str(ROOT/name) for name in names}
        sources=repair.get(key)
        if not isinstance(sources,dict) or set(sources)!=expected:raise ValueError('Operational repair source inventory differs: '+key)
        for path,value in sources.items():
            if not isinstance(value,str) or not re.fullmatch(r'[0-9a-f]{64}',value):raise ValueError('Operational repair needs explicit source SHA')
            verifier.bind(path,value)
    return repair


def strict_cache_completion_authorities(verifier):
    """Reject bool/float rc0 before retained cache-hit logic can consider it.

    Only eligible explicit checkpoint receipts are inspected, never checkpoint
    payloads. Their bytes are SHA/stat bound by the same retained verifier.
    Generic receipts still follow the original full-read fallback policy.
    """
    if verifier.cache is None:return
    for path in verifier.checkpoint_paths:
        entry=verifier.cache.entries.get(path)
        if entry is None:continue
        verifier.bind(entry['path'],entry['sha256'])
        receipt=verifier.cache._receipt(path)
        authority=receipt.get('source_authority',{})
        if authority.get('kind')=='completed_training_checkpoint':
            if type(authority.get('returncode')) is not int or authority['returncode']!=0:
                raise ValueError('Historical completed checkpoint requires integer rc0')
            if 'stop_requested_at' in authority or 'stop_requested' in authority:
                raise ValueError('Historical checkpoint authority was stop-requested')


def load_plan(plan_proof,repair_proof):
    # Preserve the retained source checks and exact fresh-prepare identity test.
    plan,verifier=retained.load_plan(plan_proof)
    validate_repair(plan,repair_proof,verifier)
    strict_cache_completion_authorities(verifier)
    return plan,verifier


def weights_sources(verifier,evidence):
    """JSON source evidence has schemas; Python producer source is opaque SHA."""
    for name in ('source_audit','source_registry','source_snapshot'):
        if not evidence.get(name):continue
        value=verifier.proof(evidence[name])
        if name=='source_registry':
            if (not isinstance(value,list) or not value or
                    any(not isinstance(row,dict) or not isinstance(row.get('id'),str) or not row['id'] for row in value) or
                    len({row['id'] for row in value})!=len(value)):
                raise ValueError('Complete-state source registry must be a nonempty unique-ID JSON list')
        else:
            if not isinstance(value,dict) or value.get('status')!='passed':raise ValueError('Complete-state JSON evidence must be passed: '+name)
            if name=='source_audit':
                for key in ('expected_runs','completed_runs'):
                    if key in value and (type(value[key]) is not int or value[key]<1):raise ValueError('Complete-state audit requires integer counts')
                if 'expected_runs' in value and value.get('completed_runs')!=value['expected_runs']:raise ValueError('Complete-state source audit is incomplete')
                if value.get('source_violations'):raise ValueError('Complete-state source audit has violations')
    if evidence.get('producer_source'):
        source=evidence['producer_source']
        if not isinstance(source,dict) or not isinstance(source.get('sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',source['sha256']):
            raise ValueError('Opaque producer source requires explicit SHA256')
        verifier.bind(source['path'],source['sha256'])


def authorize(plan_proof,authority_proof,repair_proof):
    # The old ROOT fields remain mandatory. A previous authority cannot silently
    # authorize a different operational implementation.
    plan,authority=retained.authorize(plan_proof,authority_proof)
    verifier=Verifier()
    validate_repair(plan,repair_proof,verifier)
    if authority.get('operational_repair')!=repair_proof:
        raise PermissionError('New ROOT authority must explicitly bind the operational repair')
    verifier.finish()
    return plan,authority


'''
    new_guard = ROOT / ACTIVE_GUARD
    new_guard.write_text(prelude + inspect + '\n')
    cli_source = old_cli.read_text()
    analyze = function(cli_source, 'analyze')
    analyze = replace_once(analyze,
        'def analyze(plan_proof,authority_proof,state_path,output,first_receipt=None,no_plots=False):',
        'def analyze(plan_proof,authority_proof,state_path,output,repair_proof,first_receipt=None,no_plots=False):')
    analyze = replace_once(analyze,
        'plan,_=guard.authorize(plan_proof,authority_proof);ready=guard.inspect(plan_proof,state_path)',
        'plan,_=guard.authorize(plan_proof,authority_proof,repair_proof);ready=guard.inspect(plan_proof,state_path,repair_proof)')
    analyze = replace_once(analyze,
        "'complete':True,'plan':plan_proof,'authority':authority_proof,'readiness':readiness,",
        "'complete':True,'plan':plan_proof,'authority':authority_proof,'readiness':readiness,'operational_repair':repair_proof,")
    cli_head = '''#!/usr/bin/env python3
"""Retained v2 statistics with an explicit, non-authorizing v3 gate repair."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from replay_study import supplemental_analysis_guard_v3 as guard


'''
    cli_main = '''def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for action in ('check','analyze'):
        child=sub.add_parser(action)
        child.add_argument('--plan',type=Path,required=True);child.add_argument('--plan-sha256',required=True)
        child.add_argument('--state',type=Path,required=True);child.add_argument('--output',type=Path,required=True)
        child.add_argument('--operational-repair',type=Path,required=True);child.add_argument('--operational-repair-sha256',required=True)
        if action=='analyze':
            child.add_argument('--authority',type=Path,required=True);child.add_argument('--authority-sha256',required=True)
            child.add_argument('--first-read-receipt',type=Path);child.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    plan={'path':str(args.plan.resolve()),'sha256':args.plan_sha256}
    repair={'path':str(args.operational_repair.resolve()),'sha256':args.operational_repair_sha256}
    if args.action=='check':result=guard.immutable(args.output,guard.inspect(plan,args.state,repair))
    else:
        authority={'path':str(args.authority.resolve()),'sha256':args.authority_sha256}
        result=analyze(plan,authority,args.state,args.output,repair,args.first_read_receipt,args.no_plots)
    print(json.dumps(result))


if __name__=='__main__':main()
'''
    (ROOT / 'scripts/analyze_supplemental_phase_v3.py').write_text(
        cli_head + function(cli_source, 'write_auxiliary') + '\n\n\n' + analyze + '\n\n\n' + cli_main)


ACTIVE_GUARD = 'src/replay_study/supplemental_analysis_guard_v3.py'


if __name__ == '__main__':
    main()
