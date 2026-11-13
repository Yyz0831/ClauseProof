"""One deterministic, label-blind inference pass per arm on frozen contracts."""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
import shutil
import time
from contractnli_v3_common import *
import run_expanse_trial as inference

def run(arm):
    out=ROOT/'runs'/NAME/arm;assert not out.exists(),'Never overwrite/repeat an inference arm'
    p=json.loads((EVAL/'protocol.json').read_text());assert file_sha(EVAL/'inputs.jsonl')==p['input_sha256']
    rows=read_rows(EVAL/'inputs.jsonl');assert [r['id'] for r in rows]==p['question_ids'] and len(rows)==306
    cfg=json.loads((ROOT/'configs/scale60-v8-expanse-eval.json').read_text());cfg['label']=arm
    assert cfg['model_id']==MODEL and cfg['revision']==REVISION
    adapter=None;training=None
    sources={
        'v1':(ROOT/'runs/train-contractnli-v1/expanse8b_contractnli_4bit',510),
        'v2':(ROOT/'runs/train-contractnli-v2/expanse8b_contractnli_v2_4bit',1020),
        'v3':(ROOT/'runs/train-contractnli-v3'/ARM,1020),
    }
    if arm in sources:
        source,expected_steps=sources[arm]
        training=json.loads((source/'manifest.json').read_text())
        assert training['training_complete'] and training['status']=='completed' and training['actual_final_global_step']==expected_steps
        assert training['base_model']['model_id']==MODEL and training['base_model']['revision']==REVISION
        adapter=source/'adapter';assert file_sha(adapter/'adapter_model.safetensors')==training['adapter_sha256']
        if arm=='v1':assert training['adapter_sha256']==V1_ADAPTER_SHA
        if arm=='v2':assert training['adapter_sha256']==V2_ADAPTER_SHA
    else:assert arm=='base'
    out.mkdir(parents=True);shutil.copyfile(EVAL/'inputs.jsonl',out/'inputs.jsonl')
    shutil.copyfile(EVAL/'protocol.json',out/'protocol.json')
    v={'status':'preparing','inference_complete':False,'arm':arm,'config':cfg,'started_at_utc':inference.utc_now(),
        'input_sha256':p['input_sha256'],'evaluation_labels_read':False,'automatic_retries':0,
        'adapter_sha256':training['adapter_sha256'] if training else None,
        'code_sha256':{str(q.relative_to(ROOT)):file_sha(q) for q in [Path(__file__),ROOT/'scripts/contractnli_common.py',ROOT/'scripts/contractnli_v3_common.py',ROOT/'scripts/run_expanse_trial.py']}}
    save(out/'manifest.json',v)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
        import torch
        import bitsandbytes as bnb
        from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
        from peft import PeftModel
        assert torch.cuda.is_available();torch.cuda.set_device(0);set_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        kw={'revision':REVISION,'local_files_only':True,'trust_remote_code':False}
        tok=AutoTokenizer.from_pretrained(MODEL,**kw)
        inference.trial_messages=messages;inference.parse_prediction=parse_prediction
        prepared,accounting,prompts=inference.prepare_rows(rows,tok,7936)
        assert all(a['prepared'] and a['fits'] for a in accounting)
        save(out/'token-accounting.json',accounting);save_rows(out/'prompts.jsonl',prompts)
        v.update(status='loading_model',packages={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes','accelerate']});save(out/'manifest.json',v)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        model=AutoModelForCausalLM.from_pretrained(MODEL,**kw,dtype=torch.bfloat16,quantization_config=quant,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
        if adapter:model=PeftModel.from_pretrained(model,adapter,is_trainable=False,autocast_adapter_dtype=True).eval()
        options,support=inference.generation_options(model,tok,cfg)
        v.update(status='running',loaded_model=inference.audit_loaded_model(model,bnb),generation_kwargs=options,generation_support=support,
            gpu=torch.cuda.get_device_name(0));save(out/'manifest.json',v)
        results=[];started=time.monotonic()
        with (out/'predictions.jsonl').open('x') as f:
            for row,cpu,item in prepared:
                result=inference.run_case(row,cpu,item,model,tok,torch,cfg,options)
                results.append(result);f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
                print(json.dumps({'arm':arm,'completed':len(results),'total':len(rows),'status':result['status'],'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
        complete=all(r['generation_attempts']==1 and r['raw_output'] is not None for r in results)
        v.update(status='completed' if complete else 'incomplete',inference_complete=complete,
            completed_at_utc=inference.utc_now(),elapsed_seconds=round(time.monotonic()-started,2),
            status_counts=dict(Counter(r['status'] for r in results)),predictions_sha256=file_sha(out/'predictions.jsonl'),
            hit_token_limit_count=sum(r['hit_token_limit'] for r in results))
        save(out/'manifest.json',v);assert complete,'Inference errors retained; no retries'
    except Exception as e:
        v.update(status='failed',inference_complete=False,error=f'{type(e).__name__}: {e}');save(out/'manifest.json',v);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['base','v1','v2','v3']);run(ap.parse_args().arm)
