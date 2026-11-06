import argparse,copy,importlib.util,json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/select_extended_phase.py'
spec=importlib.util.spec_from_file_location('extended_phase',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path

def fixture(tmp_path):
    policy={'torch_version':'frozen','cuda_version':'12.4','precision':'bf16','effective_precision':'cuda_bfloat16_autocast',
        'device_type':'cuda','gpu_name':'H100','attention_implementation':'sdpa','attention_backend_requested':'auto',
        'flash_sdp_enabled':True,'mem_efficient_sdp_enabled':True,'math_sdp_enabled':True,'cudnn_sdp_enabled':True,
        'deterministic_algorithms':True,'deterministic_warn_only':False,'allow_tf32_matmul':True,'allow_tf32_cudnn':True,
        'cudnn_deterministic':True,'cudnn_benchmark':False,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
        'model_weight_dtype':'torch.float32'}
    provenance={'torch_version':'frozen','cuda_version':'12.4','precision':'bf16','device':'cuda:0','gpu':'H100',
        'attention_implementation':'sdpa','attention_backend_policy':'auto','sdpa_backends_enabled':{
            'flash':True,'math':True,'memory_efficient':True,'cudnn':True},
        'deterministic_algorithms':True,'deterministic_warn_only':False,'allow_tf32_matmul':True,'allow_tf32_cudnn':True,
        'cudnn_deterministic':True,'cudnn_benchmark':False,'cublas_workspace_config':':4096:8',
        'model_weight_dtype':'torch.float32'}
    def metrics(nll):return {n:{'nll':nll,'kl':0.,'complete_corpus':True,'valid_prediction_tokens':100}
                             for n in ['target','c4','wikitext']}
    initial=write(tmp_path/'initial.json',{'revision':'pinned','execution_provenance':policy,'corpora':metrics(3.)})
    data=tmp_path/'validation.pt';data.write_bytes(b'frozen validation')
    protocol={'model':'pythia-1b','data_version':'extended_v1','initial_validation_path':str(initial),
        'validation_paths':{'target':str(data)},'validation_sha256':{'target':module.sha(data)},
        'learning_rate_grid_completed':[1e-5,3e-5,1e-4],'target_prediction_tokens':2_000_000,
        'microbatch_size':16,'gradient_accumulation_steps':4,'learning_rate_selection_rule':'min target; ties lower LR'}
    rows=[];jobs=[{'id':'initial'}]
    for index,(lr,nll) in enumerate([(1e-5,2.5),(3e-5,2.),(1e-4,2.7)]):
        run=f'LR{index}';folder=tmp_path/run
        config={'model_revision':'pinned','learning_rate':lr,'microbatch_size':16,'gradient_accumulation_steps':4,
                'seed':101,'precision':'bf16','device':'cuda:0'}
        config_path=write(folder/'config.json',config)
        summary={'config':config,'status':'completed','provenance':copy.deepcopy(provenance),
            'counters':{'target_prediction_tokens':2_000_000},'target_stream':{'seed':101,'trace_hash':'same'},
            'validation_trajectory':[{'target_prediction_tokens':2_000_000,'metrics':metrics(nll)}]}
        summary_path=write(folder/'summary.json',summary)
        rows.append({'id':run,'learning_rate':lr,'label':format(lr,'.0e'),'config_path':str(config_path),
            'config_sha256':module.sha(config_path),'output_dir':str(folder),'summary_path':str(summary_path),
            'checkpoint_path':str(folder/'latest.pt')});jobs.append({'id':run})
    return argparse.Namespace(phase='lr',model='pythia-1b',registry=write(tmp_path/'registry.json',rows),
        protocol=write(tmp_path/'protocol.json',protocol),phase_queue=write(tmp_path/'queue.json',jobs),
        master_state=write(tmp_path/'state.json',{'jobs':{j['id']:{'status':'completed'} for j in jobs}}),
        output=tmp_path/'selection.json')

def test_cpu_selector_joins_completed_training_baseline_and_frozen_protocol(tmp_path):
    args=fixture(tmp_path);module.select(args);selected=json.loads(args.output.read_text())
    assert selected['selected_learning_rate']==3e-5
    assert selected['selection_computation_device']=='cpu_only_no_model_forward'
    assert selected['source_protocol_sha256']==module.sha(args.protocol)

def test_cpu_selector_rejects_actual_execution_policy_mismatch(tmp_path):
    args=fixture(tmp_path);row=json.loads(args.registry.read_text())[0]
    summary=json.loads(Path(row['summary_path']).read_text());summary['provenance']['allow_tf32_matmul']=False
    write(Path(row['summary_path']),summary)
    with pytest.raises(ValueError,match='allow_tf32_matmul'):module.select(args)
    assert not args.output.exists()
