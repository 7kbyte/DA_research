"""Render only the retained, SHA-verified hand-made toy inputs."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=HERE.with_name('behavioral_presentation_preparation_v1')
def proof(path):return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')

def child():
    previous=json.loads((OLD/'synthetic_smoke_COMPLETE.json').read_bytes());results=[]
    assert previous['synthetic_only'] is True and previous['actual_score_or_prediction_files_read'] is False
    for phase in previous['phases']:
        fixture=Path(phase['fixture']['path']);assert proof(fixture)==phase['fixture']
        output=HERE/'synthetic_outputs'/phase['phase']
        subprocess.run([sys.executable,str(ROOT/'scripts/present_behavioral_review.py'),'--synthetic-fixture',str(fixture),'--output',str(output)],cwd=ROOT,check=True)
        results.append({'phase':phase['phase'],'fixture':phase['fixture'],'output_COMPLETE':proof(output/'COMPLETE.json'),
            'nested_presentation_timer':proof(output.with_name(output.name+'_whole_child_timing.json'))})
    write(HERE/'synthetic_smoke_COMPLETE.json',{'complete':True,'synthetic_only':True,'phases':results,
        'actual_score_or_prediction_files_read':False,'original_check_analyze_reviewer_or_inference_called':False,
        'scope':'All five complete phase shapes; retained toy CSVs, compact primary labels, source input final stat checks'})

if __name__=='__main__':
    if '--child' in sys.argv:child()
    else:
        before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter()
        with (HERE/'whole_smoke_stdout.log').open('x') as out,(HERE/'whole_smoke_stderr.log').open('x') as err:
            run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child'],cwd=ROOT,stdout=out,stderr=err,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        write(HERE/'whole_smoke_timing.json',{'returncode':run.returncode,'wall_seconds':time.perf_counter()-start,
            'user_cpu_seconds':after.ru_utime-before.ru_utime,'system_cpu_seconds':after.ru_stime-before.ru_stime,
            'scope':'Entire five-phase synthetic presentation child; nested per-phase presentation timers not additive',
            'source':proof(Path(__file__)),'actual_score_or_prediction_read':False})
        raise SystemExit(run.returncode)
