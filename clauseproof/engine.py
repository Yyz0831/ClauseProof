"""The frozen Aya Expanse 8B ContractNLI v2 adapter, loaded once on Ubuntu."""
import importlib.metadata
import json
import os
from pathlib import Path
import time
from core import ROOT,MODEL,REVISION,ADAPTER_SHA,INSTRUCTION,RequestError,digest,messages

class AyaEngine:
    def __init__(self,run_dir):
        self.status='loading';self.error=None;self.run_dir=run_dir;self.provenance={}

    def load(self):
        try:
            os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
            import torch
            from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
            from peft import PeftModel
            from run_expanse_trial import generation_options
            from contractnli_common import file_sha
            self.torch=torch;set_seed(42)
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            adapter=ROOT/'runs/train-contractnli-v2/expanse8b_contractnli_v2_4bit/adapter'
            manifest=json.loads((adapter.parent/'manifest.json').read_text())
            if not manifest['training_complete'] or manifest['actual_final_global_step']!=1020:
                raise RuntimeError('Training manifest does not match the completed v2 run')
            if file_sha(adapter/'adapter_model.safetensors')!=ADAPTER_SHA:raise RuntimeError('Adapter checksum mismatch')
            if manifest['base_model']['model_id']!=MODEL or manifest['base_model']['revision']!=REVISION:
                raise RuntimeError('Base model lineage mismatch')
            cfg=json.loads((ROOT/'configs/scale60-v8-expanse-eval.json').read_text())
            kw={'revision':REVISION,'local_files_only':True,'trust_remote_code':False}
            self.tokenizer=AutoTokenizer.from_pretrained(MODEL,**kw)
            quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
            base=AutoModelForCausalLM.from_pretrained(MODEL,**kw,dtype=torch.bfloat16,quantization_config=quant,
                device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
            self.model=PeftModel.from_pretrained(base,adapter,is_trainable=False,autocast_adapter_dtype=True).eval()
            self.options,self.support=generation_options(self.model,self.tokenizer,cfg)
            self.provenance={'model_id':MODEL,'model_revision':REVISION,'adapter':'ContractNLI v2',
                'adapter_sha256':ADAPTER_SHA,'instruction_sha256':digest(INSTRUCTION),
                'server_engine_sha256':file_sha(Path(__file__)),'generation':self.options,
                'quantization':cfg['quantization'],'gpu':torch.cuda.get_device_name(0),
                'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes']}}
            self.run_dir.mkdir(parents=True,exist_ok=True)
            (self.run_dir/'model-manifest.json').write_text(json.dumps(self.provenance,indent=2)+'\n')
            self.status='ready'
        except Exception as exc:
            self.error=type(exc).__name__;self.status='error'
            self.run_dir.mkdir(parents=True,exist_ok=True)
            (self.run_dir/'startup-error.json').write_text(json.dumps({'error':str(exc),'type':type(exc).__name__})+'\n')

    def prepare(self,rows):
        if self.status!='ready':raise RequestError('model_not_ready','Aya v2 is still loading or unavailable. Check the model status.',503)
        prepared=[]
        for row in rows:
            prompt=self.tokenizer.apply_chat_template(messages(row),tokenize=False,add_generation_prompt=True)
            cpu=self.tokenizer(prompt,add_special_tokens=False,return_tensors='pt',truncation=False)
            count=int(cpu['input_ids'].shape[-1])
            if count>7936:
                raise RequestError('over_token_budget',f'This document needs {count:,} input tokens; the limit is 7,936. No text was truncated.',422,input_tokens=count,max_input_tokens=7936)
            prepared.append((cpu,{'input_tokens':count,'prompt_sha256':digest(prompt)}))
        return prepared

    def generate(self,prepared):
        cpu,account=prepared;torch=self.torch;inputs=output=None
        try:
            inputs={k:v.to('cuda:0') for k,v in cpu.items()}
            torch.cuda.synchronize();started=time.perf_counter()
            with torch.inference_mode():output=self.model.generate(**inputs,**self.options)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            ids=output[0,account['input_tokens']:].tolist()
            raw=self.tokenizer.decode(ids,skip_special_tokens=True)
            eos=self.options['eos_token_id'];eos=[eos] if isinstance(eos,int) else (eos or [])
            return raw,{**account,'generated_tokens':len(ids),'generation_seconds':round(elapsed,3),
                'hit_token_limit':len(ids)>=256 and bool(ids) and ids[-1] not in eos,
                'generation_attempts':1,'automatic_retries':0}
        finally:
            del inputs,output
            # No unconditional empty_cache; retain a warm model without touching other processes.
