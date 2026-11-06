#!/usr/bin/env python3
"""CPU-only operational wrapper around the unchanged v4 preparation API.

Prewarm opaque checkpoint SHA receipts after each scheduler-confirmed run, then
pass a persistent Hashes-compatible object into the original pass_once/build.
Scientific completion, selection and model-equivalence gates are unchanged.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_supplemental_evaluations as original
from schedule_jobs import digest_job
from stat_bound_digest_cache import StatBoundHashes,atomic,small_proof


def absolute(path):
    path=Path(path)
    return (ROOT/path).resolve() if not path.is_absolute() else path.resolve()


def read(path):return json.loads(Path(path).read_bytes())


def initialize(old_path,old_sha):
    if small_proof(old_path)['sha256']!=old_sha:raise ValueError('v4 catalog SHA mismatch')
    cat=copy.deepcopy(read(old_path))
    if cat['preparation_revision']!=4:raise ValueError('Explicit frozen v4 source required')
    old_source=cat['producer_source']
    if small_proof(old_source['path'])!=old_source:raise ValueError('v4 producer changed')
    receipt=ROOT/'docs/supplemental_evaluation_preparation_v4_execution.json'
    frozen=read(receipt)['frozen_scorer_input_sha256']
    for path,wanted in frozen.items():
        if small_proof(absolute(path))['sha256']!=wanted:raise ValueError('Frozen scoring source changed')
    ops=[small_proof(__file__),small_proof(ROOT/'scripts/stat_bound_digest_cache.py'),old_source,
         small_proof(old_path),small_proof(receipt),small_proof(ROOT/'scripts/schedule_jobs.py')]
    for plan in cat['plans']:
        for p in ops:plan['phase']['source_proof_files'][p['path']]=p['sha256']
    descriptors=[];seen=set()
    main_registry=ROOT/'configs/formal_deterministic_registry.json'
    snapshot_path=ROOT/'artifacts/provenance/formal_deterministic_input_snapshot.json'
    sources=[{'registry':small_proof(main_registry),'state_path':str(ROOT/'logs/formal_deterministic_job_state.json'),
              'queue_path':str(ROOT/'configs/formal_deterministic_queue.json'),'config_snapshot':small_proof(snapshot_path)}]
    for plan in cat['plans']:
        proof=plan['phase']['registry']
        if proof['path'] not in {s['registry']['path'] for s in sources}:
            sources.append({'registry':proof,'state_path':str(ROOT/'logs/supplemental_job_state.json'),
                            'queue_path':str(ROOT/'configs/supplemental_v1_queue.json')})
    for source in sources:
        if small_proof(source['registry']['path'])!=source['registry']:raise ValueError('Registry SHA mismatch')
        snapshot=read(source['config_snapshot']['path']) if source.get('config_snapshot') else None
        jobs={j['id']:j for j in read(source['queue_path'])}
        for row in read(source['registry']['path']):
            if row.get('execution_kind')=='alias' or row.get('alias_of'):continue
            if row['id'] in seen:continue
            seen.add(row['id']);cp=absolute(row['config_path'])
            wanted=row.get('config_sha256')
            if not wanted:
                wanted=snapshot['configs'][str(cp.relative_to(ROOT))]['sha256']
            if small_proof(cp)['sha256']!=wanted:raise ValueError('Config SHA differs from frozen authority')
            job=jobs[row['id']]
            descriptors.append({'id':row['id'],'config':{'path':str(cp),'sha256':wanted},
                'summary_path':str(absolute(row['summary_path'])),'output_dir':str(absolute(row['output_dir'])),
                'state_path':source['state_path'],'queue_path':source['queue_path'],
                'job':job,'definition_sha256':digest_job(job)})
    cat.update(preparation_revision=5,producer_source=small_proof(__file__),
        operational_revision={'kind':'stat_bound_complete_file_digest_prewarm','created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'v4_catalog':small_proof(old_path),'v4_producer':old_source,'source_proofs':ops,
            'scientific_scope_or_endpoint_changes':False,'frozen_worker_fresh_input_sha_preserved':True},
        prewarm_sources=sources,prewarm_runs=descriptors)
    return cat


class PendingCheckpoint(FileNotFoundError):pass


class CompletedRuns:
    """Bind opaque files to frozen jobs, rc0, complete summaries and budgets."""
    def __init__(self,catalog,hashes):
        self.catalog=catalog;self.hashes=hashes;self.rows=catalog['prewarm_runs'];self.lookup={}
        for row in self.rows:
            for kind,name in [('final','latest.pt'),('best','checkpoint_best.pt')]:
                key=str(absolute(row['output_dir'])/'checkpoints'/name)
                if key in self.lookup:raise ValueError('Duplicate physical checkpoint in prewarm registry')
                self.lookup[key]=(row,kind)
        self.refresh()

    def refresh(self):
        self.states={};self.jobs={}
        for source in self.catalog['prewarm_sources']:
            self.hashes.proof(source['registry'])
            if source.get('config_snapshot'):self.hashes.proof(source['config_snapshot'])
            for kind,key in [('states','state_path'),('jobs','queue_path')]:
                path=source[key]
                if path not in getattr(self,kind):
                    value=read(path)
                    getattr(self,kind)[path]=value['jobs'] if kind=='states' else {j['id']:j for j in value}
        for row in self.rows:
            current=self.jobs[row['queue_path']].get(row['id'])
            if current!=row['job'] or digest_job(current)!=row['definition_sha256']:
                raise ValueError('Frozen per-run scheduler definition changed: '+row['id'])

    def guard(self,path):
        key=str(absolute(path))
        if key not in self.lookup:raise ValueError('Checkpoint lacks explicit frozen run authority: '+key)
        row,kind=self.lookup[key];record=self.states[row['state_path']].get(row['id'],{})
        if record.get('status')!='completed':raise PendingCheckpoint('Scheduler rc0 pending: '+row['id'])
        if (record.get('returncode')!=0 or record.get('definition_sha256')!=row['definition_sha256'] or
                record.get('missing_outputs') or 'stop_requested_at' in record):
            raise ValueError('Unsuccessful or mismatched scheduler completion: '+row['id'])
        job=row['job'];argv=job['argv']
        if (argv.count('--config')!=1 or absolute(argv[argv.index('--config')+1])!=absolute(row['config']['path']) or
                record.get('argv')!=argv):raise ValueError('Actual completed job does not bind the frozen training config')
        if any(not absolute(p).is_file() for p in job.get('outputs',[])):
            raise PendingCheckpoint('Declared completed output missing: '+row['id'])
        self.hashes.proof(row['config']);cfg=read(row['config']['path']);sp=absolute(row['summary_path'])
        if not sp.is_file():raise PendingCheckpoint('Completed summary missing')
        summary=read(sp);counts=summary.get('counters',{});target=cfg['target_prediction_tokens'];b=Fraction(str(cfg['replay_ratio']))
        if (summary.get('status')!='completed' or summary.get('config')!=cfg or cfg['run_label']!=row['id'] or
                absolute(cfg['output_dir'])!=absolute(row['output_dir']) or counts.get('target_prediction_tokens')!=target or
                counts.get('replay_prediction_tokens')!=target*b.numerator//b.denominator):
            raise ValueError('Completed source config/status/exposure mismatch: '+row['id'])
        metadata=summary if kind=='final' else summary.get('best_target_validation',{})
        if absolute(metadata['checkpoint_path'])!=absolute(path):raise ValueError('Actual selected checkpoint differs from frozen endpoint path')
        if not absolute(path).is_file():raise PendingCheckpoint('Checkpoint not yet present')
        return {'kind':'completed_training_checkpoint','run_id':row['id'],'endpoint':kind,
                'config':row['config'],'summary':{'path':str(sp),'sha256':self.hashes.digest(sp)},
                'state_path':row['state_path'],'job_definition_sha256':row['definition_sha256'],'returncode':0,
                'target_prediction_tokens':target,'replay_prediction_tokens':counts['replay_prediction_tokens']}


def prewarm(gate,hashes,cursor,seconds):
    keys=list(gate.lookup);start=time.monotonic();done=0;pending=0
    while done<len(keys) and (done==0 or time.monotonic()-start<seconds):
        key=keys[cursor%len(keys)];cursor=(cursor+1)%len(keys);done+=1
        try:hashes.digest(key)
        except PendingCheckpoint:pending+=1
        except FileNotFoundError:
            # A not-yet-written path is eligible only when its scheduler is pending.
            row,_=gate.lookup[key];record=gate.states[row['state_path']].get(row['id'],{})
            if record.get('status')=='completed':raise
            pending+=1
    return cursor,{'visited':done,'pending':pending,'seconds':time.monotonic()-start}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initialize-catalog',type=Path);p.add_argument('--v4-catalog',type=Path);p.add_argument('--v4-sha256')
    p.add_argument('--catalog',type=Path);p.add_argument('--catalog-sha256');p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--watch',action='store_true')
    p.add_argument('--poll-seconds',type=float,default=30);p.add_argument('--prewarm-seconds',type=float,default=45)
    a=p.parse_args()
    if a.initialize_catalog:
        if a.initialize_catalog.exists():raise FileExistsError('Use a new catalog namespace')
        cat=initialize(a.v4_catalog,a.v4_sha256);atomic(a.initialize_catalog,cat)
        print(json.dumps({'catalog':small_proof(a.initialize_catalog),'runs':len(cat['prewarm_runs']),'plans':len(cat['plans'])}));return
    if not a.catalog or not a.catalog_sha256 or a.poll_seconds<5 or a.prewarm_seconds<=0:p.error('Explicit catalog SHA and positive preparation timing required')
    a.output_root.mkdir(parents=True,exist_ok=True);a.cache_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
    locks=[]
    for path in [a.output_root/'watcher.lock',a.cache_dir/'writer.lock']:
        stream=path.open('a');fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(stream)
    hashes=StatBoundHashes(directory=a.cache_dir);hashes.proof({'path':str(a.catalog),'sha256':a.catalog_sha256})
    cat=read(a.catalog)
    if cat.get('preparation_revision')!=5:raise ValueError('v5 catalog required')
    for proof in cat['operational_revision']['source_proofs']:hashes.proof(proof)
    gate=CompletedRuns(cat,hashes);hashes.checkpoint_guard=gate.guard;cursor=0
    while True:
        gate.refresh();cursor,warm=prewarm(gate,hashes,cursor,a.prewarm_seconds)
        result=original.pass_once(cat,a.catalog,a.catalog_sha256,a.output_root,hashes,max_freezes=1)
        snapshot=hashes.snapshot(a.output_root/('digest_cache_snapshot_'+str(time.time_ns())+'.json'))
        progress={'schema_version':1,'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'catalog':small_proof(a.catalog),'ready_phase_count':result['ready_phase_count'],'plan_count':len(cat['plans']),
            'prewarm':warm,'counters_this_process':dict(hashes.counters),'cache_snapshot':snapshot,
            'cache_hits_are_current_byte_reads':False,'worker_fresh_full_sha_unchanged':True,
            'gpu_work_started':False,'score_files_parsed':False,'tensor_payloads_deserialized':False}
        atomic(a.output_root/'cache_progress.json',progress)
        print(json.dumps(progress),flush=True)
        if not a.watch or result['all_requested_phases_ready'] or dt.datetime.now(dt.timezone.utc)>=dt.datetime(2026,11,7,3,26,32,tzinfo=dt.timezone.utc):break
        time.sleep(min(a.poll_seconds,60))


if __name__=='__main__':main()
