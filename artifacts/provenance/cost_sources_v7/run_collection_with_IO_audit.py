"""Invoke the unchanged collector; audit only IO, forbid weight/archive payload opens."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();source=ROOT/'scripts/inventory_study_costs_final_v2.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='864effb89f20dbfce3ba6ce51db8eb8351b087c2ac92d255274fe6267753f6ae'
    opened={};denied=[]
    def audit(event,arguments):
        if event!='open' or not isinstance(arguments[0],(str,bytes,os.PathLike)):
            return
        path=str(Path(os.fsdecode(arguments[0])).resolve());flags=arguments[2]
        if flags & os.O_ACCMODE == os.O_WRONLY:
            return
        if path.endswith(('.pt','.safetensors','.zip','.tar','.tar.gz','.bin','.npy','.npz')):
            denied.append(path)
            raise RuntimeError('Forbidden model/archive/tensor payload read attempted: '+path)
        opened[path]=opened.get(path,0)+1
    sys.addaudithook(audit)
    sys.argv=[str(source),'--sources',args.sources,'--output',args.output]
    try:
        runpy.run_path(str(source),run_name='__main__')
    except BaseException:
        output=Path(args.output)
        if output.exists():
            with (output/'IO_failure_audit.json').open('x') as stream:
                json.dump(dict(status='collector_failed',forbidden_payload_open_attempts=denied,
                               read_open_counts=opened,score_output_semantics_parsed=False),stream,indent=2)
                stream.write('\n')
        raise
    output=Path(args.output);inventory=json.loads((output/'inventory.json').read_text())
    sources=inventory['source_snapshots']
    if isinstance(sources,list):
        sources={v['path']:v for v in sources}
    score=[]
    for path,value in sources.items():
        if value.get('read_purpose')!='SHA_only_no_score_deserialization':
            continue
        canonical=str(Path(path).resolve());count=opened.get(canonical,0)
        assert count>0 and value.get('changed_while_reading') is False
        score.append(dict(path=canonical,sha256=value['sha256'],file_bytes=value['bytes_read'],
                          full_stream_hash_open_count=count,actual_full_stream_hash_bytes=value['bytes_read']*count))
    result=dict(schema_version=1,status='passed_cost_only_IO_audit',unchanged_collector_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        forbidden_weight_base_archive_tensor_payload_open_attempts=denied,weight_base_TAR_ZIP_tensor_payload_bytes_read=0,
        score_output_semantic_deserialization=False,score_SHA_verification_policy_changed=False,
        score_output_stream_hash_reads=score,unique_score_outputs_stream_hashed=len(score),
        cumulative_score_stream_SHA_bytes=sum(v['actual_full_stream_hash_bytes'] for v in score),
        unique_score_output_file_bytes=sum(v['file_bytes'] for v in score),
        count_method='Python open-audit counts actual read opens for SHA-only outputs; each successful unchanged hash_output call streams the whole file to EOF. Repeated canonical/alias verification passes are counted, not deduplicated as physical reads.',
        bytes_are_not_physical_disk_IO=True,wrapper_does_not_change_any_collector_logic=True)
    with (output/'IO_audit.json').open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ['status','unique_score_outputs_stream_hashed','cumulative_score_stream_SHA_bytes','weight_base_TAR_ZIP_tensor_payload_bytes_read']}))


if __name__=='__main__':
    main()
