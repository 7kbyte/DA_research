#!/usr/bin/env python3
"""One fixed B native-package restore; ROOT must authorize its exact prepared plan."""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE = ROOT/'artifacts/provenance/C_native_final_restore_preparation_v1'
PINS = {'restore_native_final.py':'bfc748c776f18dff86c378c5c55ca72dbc63995cfc40ddfb96eb919486610db1',
        'run_whole_child.py':'2a2d09daac8504b099a5cd9682dff722bbad0519922ba39beedf4e96e9498266',
        'check_restored.py':'6b7df26abef4ce0bd09dabbd7878a5a9585c8681d3cb97115f912dd542d3fa8d'}
RUN = 'bankvar_B4_tr11_gen11_draw1001_k20_b0p05_w0p05'
QA = Path('/tmp/grad-research-20261106/reproduction_qa/B_native_B4_tr11_gen11_draw1001_k20_final_v1')
TAR = Path('/data/202610/7kb/research_delivery/B_all_final_weights_lossless_v1.tar')


def require(ok, message):
    if not ok: raise ValueError(message)


def proof(path):
    p = Path(path).resolve(); require(p.stat().st_size < 32 << 20, 'Small source proof only')
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def load_helper(filename, name):
    path = NATIVE/filename
    require(proof(path)['sha256'] == PINS[filename], 'Frozen reused helper changed')
    spec = importlib.util.spec_from_file_location(name,path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


def validate(plan):
    require(plan.get('status') == 'prepared_not_execution_authority' and plan['run_id'] == RUN and
            plan['model'] == 'pythia-410m' and plan['package'] == 'B_all_final' and plan['kind'] == 'final' and
            Path(plan['qa_directory']) == QA and Path(plan['archive']) == TAR, 'Wrong fixed B single-restore scope')
    selected = plan['selected']
    require(selected['run_id'] == selected['actual_run_id'] == RUN and selected['kind'] == 'final' and
            selected['training_seed'] == selected['generation_seed'] == 11 and selected['selection_seed'] == 1001 and
            selected['bank_chunks'] == 20 and 'alias_of' not in selected, 'Fixed original draw identity differs')
    require(selected['model_fingerprint'] == {'sha256_model_state':'05724eda110e91eb76c8f6d45b54386b374e486662207a266166a0fde0ffb790',
            'state_elements':405334016} and plan['tensor_count'] == 292 and plan['model_raw_bytes'] == 1621336064,
            'Expected complete source model fingerprint differs')
    require(len(plan['members']) == 12 and plan['native_base'] == 'base/pythia-410m-final/model.safetensors', 'Exact twelve-member native closure required')
    require(plan['historical_full_TAR_sha256'] == 'b8f7cb88141ad1b6526f05e92fe34abd68a5f5a13b8219b5d4957ff3035d50d1' and
            plan['archive_stat']['bytes'] == 69578864640 and plan['archive_stat']['inode'] == 411787337, 'Wrong published B archive identity')
    p = plan['resource_policy']
    require(p['max_wall_seconds'] == 600 and p['admission_margin_seconds'] == 300 and p['cpu_threads'] == 2 and
            p['required_scratch_bytes'] == p['required_reclaim_aware_memory_bytes'] == 8 << 30 and
            dt.datetime.fromisoformat(p['deadline_utc']) == dt.datetime(2026,11,7,2,56,32,tzinfo=dt.timezone.utc), 'Protected resource policy differs')


def environment():
    env = os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONUSERBASE'): env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONNOUSERSITE='1')
    return env


def authorize(args, native):
    pp, ap = dict(path=str(Path(args.plan).resolve()),sha256=args.plan_sha256), dict(path=str(Path(args.authority).resolve()),sha256=args.authority_sha256)
    plan, authority = native.small(pp), native.small(ap); validate(plan)
    require(authority.get('authorized') is True and authority.get('authorized_by') == 'root' and authority.get('plan') == pp and
            authority.get('run_id') == RUN and authority.get('scope') == 'one_fixed_B_native_final_CPU_recovery' and
            authority.get('sequential_only') is True and authority.get('runner') == plan['runner'] == proof(__file__) and
            authority.get('checker') == plan['checker'] == proof(NATIVE/'check_restored.py'), 'Exact future ROOT authority required; false template cannot run')
    for reference in plan['evidence'].values(): native.small(reference)
    require(native.signature(TAR) == plan['archive_stat'], 'Current published B TAR stat changed')
    return plan, pp, ap


def body(args, native):
    plan, pp, ap = authorize(args,native)
    # This is the exact lock used by native C execute; authority alone cannot serialize work.
    with (QA.parent/'.native_Cfinal_restore.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        body_locked(plan,pp,ap,native)


def body_locked(plan, pp, ap, native):
    require(not QA.exists() and not QA.is_symlink(), 'Preserve all old QA outputs; fixed NEW directory required')
    admission = native.resource_gate(plan['resource_policy'],QA.parent)
    QA.mkdir();(QA/'logs').mkdir();native.write_new(QA/'bound_plan.json',plan)
    shutil.copyfile(NATIVE/'check_restored.py',QA/'check_restored.py')
    env = environment();start=time.monotonic();su=resource.getrusage(resource.RUSAGE_SELF);cu=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=dict(status='running',model=plan['model'],run_id=RUN,plan=pp,authority=ap,resource_admission=admission,stages=[],
                original_checkpoint_or_live_base_opened=False,whole_source_TAR_rehash=False,new_encoding=False,GPU_used=False)
    def stage(name,argv):
        before=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.monotonic()
        with (QA/'logs'/(name+'.stdout.log')).open('xb') as out,(QA/'logs'/(name+'.stderr.log')).open('xb') as err:
            rc=subprocess.Popen(argv,cwd=QA,env=env,stdout=out,stderr=err).wait()
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        result['stages'].append(dict(stage=name,argv=argv,returncode=rc,wall_seconds=time.monotonic()-wall,
            child_user_seconds=after.ru_utime-before.ru_utime,child_sys_seconds=after.ru_stime-before.ru_stime,
            timer_relationship='Sequential child nested in runner and inclusive process-group timer'))
        require(type(rc) is int and rc==0,'Stage failed: '+name)
    try:
        wall=time.monotonic();cpu=time.process_time();result['extraction']=native.extract_selected(plan,QA)
        result['extraction'].update(wall_seconds=time.monotonic()-wall,process_cpu_seconds=time.process_time()-cpu,
                                    timer_relationship='Parent region nested in whole runner')
        stage('decode',[sys.executable,'-I','tools/scripts/archive_model_weights.py','--threads','2','decode',
              '--archive',plan['selected']['archive_member'],'--base',plan['native_base'],'--output','restored/B4_seed11_final.pt'])
        stage('independent_fingerprint',[sys.executable,'-I','check_restored.py'])
        result['independent_check']=json.loads((QA/'independent_fingerprint.json').read_text())
        require(result['independent_check']['status']=='passed' and native.signature(TAR)==plan['archive_stat'],'Independent check or archive stability failed')
        result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        sa=resource.getrusage(resource.RUSAGE_SELF);ca=resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(runner_body_wall_seconds=time.monotonic()-start,parent_self_user_seconds=sa.ru_utime-su.ru_utime,
            parent_self_sys_seconds=sa.ru_stime-su.ru_stime,children_user_seconds=ca.ru_utime-cu.ru_utime,
            children_sys_seconds=ca.ru_stime-cu.ru_stime,children_peak_RSS_bytes_cumulative=ca.ru_maxrss*1024,
            accounting='Body/stages nested in outer inclusive process-group timer; parent self CPU plus child CPU once',
            claim='One fixed B durable native-package model-only recovery, not all60, original CP recompare, Adam resume or GPU training')
        native.write_new(QA/'actual_restore_receipt.json',result)
    require(result['status']=='passed','Failed QA retained')


def main():
    parent_start=time.monotonic();parent_before=resource.getrusage(resource.RUSAGE_SELF);children_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['execute','_body'])
    for flag in ('plan','plan-sha256','authority','authority-sha256'): parser.add_argument('--'+flag,required=True)
    args=parser.parse_args();native=load_helper('restore_native_final.py','B_restore_native_pure_helpers')
    if args.command=='_body': body(args,native);return
    whole=load_helper('run_whole_child.py','B_restore_existing_process_group_timer')
    plan,_,_=authorize(args,native); admitted=whole.admission(plan['resource_policy'])
    folder=Path(plan['actual_evidence_directory']);require(not folder.exists() and not folder.is_symlink(),'New actual evidence directory required')
    folder.mkdir(parents=True);argv=[sys.executable,'-I',str(Path(__file__).resolve()),'_body','--plan',str(Path(args.plan).resolve()),
        '--plan-sha256',args.plan_sha256,'--authority',str(Path(args.authority).resolve()),'--authority-sha256',args.authority_sha256]
    native.write_new(folder/'launch.json',dict(argv=argv,admission=admitted,runner=proof(__file__),plan_sha256=args.plan_sha256,
        checker=proof(NATIVE/'check_restored.py'),env_policy='isolated -I, no PYTHONPATH, CVD empty, CPU2, offline',publication_performed=False))
    whole.admission(plan['resource_policy'])
    with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        result=whole.run_group(argv,folder,environment(),out,err,admitted['cap_seconds'])
    parent_after=resource.getrusage(resource.RUSAGE_SELF);children_after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(outer_wrapper_parent_wall_seconds=time.monotonic()-parent_start,
        outer_wrapper_self_user_seconds=parent_after.ru_utime-parent_before.ru_utime,
        outer_wrapper_self_sys_seconds=parent_after.ru_stime-parent_before.ru_stime,
        outer_wrapper_children_user_seconds=children_after.ru_utime-children_before.ru_utime,
        outer_wrapper_children_sys_seconds=children_after.ru_stime-children_before.ru_stime,
        outer_wrapper_accounting='Internal main-entry region before CLI/preflight/helper imports through child/group cleanup; excludes Python startup, top-level stdlib imports and final receipt serialization. Nested in the future ROOT external Popen whole-process timer; do not add the two.')
    result.update(admission=admitted,GPU_used=False,new_encoding=False,publication_performed=False)
    native.write_new(folder/'whole_child_timing.json',result)
    raise SystemExit(result['returncode'])


if __name__=='__main__': main()
