"""Bind ContractNLI to the proven one-epoch Aya 8B QLoRA engine."""
import json
from pathlib import Path
import train_scale60_offload_v3 as engine
from contractnli_common import ROOT,ARM,MODEL,REVISION,INSTRUCTION,parse_prediction

engine.require(engine.file_sha(ROOT/'scripts/train_scale60_offload_v3.py')=='7253437d20131bc3e507148dd42afb99ef895a75aa6923f1176e1895fad63601','Validated engine changed')
engine.EXPERIMENT='contractnli-v1-sft'
engine.MODELS={ARM:(MODEL,REVISION,'cohere')}
engine.CODE_FILES=list(dict.fromkeys(engine.CODE_FILES+['scripts/train_contractnli.py','scripts/prepare_contractnli.py','scripts/contractnli_common.py']))

def validate_protocol(protocol_path,config_bytes,cfg,data,rows,root=None):
    root=ROOT if root is None else Path(root);raw=Path(protocol_path).read_bytes();p=json.loads(raw);req=engine.require
    req(p['experiment']==engine.EXPERIMENT and p['schema_version']==1,'Wrong experiment')
    req(len(rows)==1020 and p['selected_ids']==[r['id'] for r in rows],'Frozen cohort changed')
    req(cfg['arm']==ARM and cfg['training']['max_steps']==510,'Schedule changed')
    req(p['counts']['labels']=={'Entailment':340,'Contradiction':340,'NotMentioned':340},'Label balance changed')
    req(p['instruction_sha256']==engine.sha(INSTRUCTION.encode()),'Instruction changed')
    req(p['no_truncation'] and p['official_train_only'] and p['semantic_review_rounds']==1,'Policy changed')
    req(engine.sha(config_bytes)==p['config_sha256'] and engine.sha(data)==p['data_sha256'],'Config/data changed')
    for name,expected in p['source_files_sha256'].items():
        req(not any(s in name for s in ['dev.json','test.json','labels.jsonl','predictions']),'Training may not read evaluation labels')
        req(engine.file_sha(root/name)==expected,'Source changed: '+name)
    lb=(root/p['logical_data']['path']).read_bytes();req(engine.sha(lb)==p['logical_data']['sha256'],'Logical data changed')
    logical=[json.loads(l) for l in lb.decode().splitlines()];req(len(logical)==len(rows),'Count mismatch')
    official=json.loads((root/'data/raw/external/contract-nli/train.json').read_text())
    docs={str(d['id']):d for d in official['documents']}
    from contractnli_common import target,source_row,messages
    for source,row in zip(logical,rows):
        req(all(source[k]==row[k] for k in ['id','paper_id','answerable','reference_index','human_review_status']),'Identity mismatch')
        req(source['source_split']=='train','Non-train example')
        gold,end=json.JSONDecoder().raw_decode(row['completion'])
        req(row['completion'][:end]==source['target_json'],'Target changed')
        parsed,error,_=parse_prediction(source['target_json'],{x['id'] for x in source['paragraphs']})
        req(error is None and parsed==gold,'Invalid target')
        doc=docs[row['paper_id']];hid=row['id'].rsplit(':',1)[1]
        req(gold==target(doc['annotation_sets'][0]['annotations'][hid]),'Official label or evidence changed')
        original=source_row(doc,hid,official['labels'][hid]['hypothesis'])
        req(source['messages']==messages(original) and source['paragraphs']==original['paragraphs'],'Full source changed')
        req(row['answerable']==(gold['label']!='NotMentioned'),'Metadata inconsistent')
    eb=(root/p['evaluation_input']['path']).read_bytes();req(engine.sha(eb)==p['evaluation_input']['sha256'],'Holdout changed')
    evaldocs={json.loads(l)['paper_id'] for l in eb.decode().splitlines()}
    req(not evaldocs & {r['paper_id'] for r in rows},'Contract leakage')
    return {'sha256':engine.sha(raw),'arm':ARM,'shared_logical_data_sha256':engine.sha(lb),'same_ordered_shared_ids_verified':len(rows),'evaluation_labels_read':False},logical

def load_runtime_policy(path,protocol_path,root=None):
    root=ROOT if root is None else Path(root);raw=Path(path).read_bytes();p=json.loads(raw)
    original=json.loads((root/'configs/revision-v10-runtime.json').read_text())
    expected={**original,'original_data_protocol_sha256':engine.file_sha(protocol_path),'reason':'Reuse validated activation offloading for an independent ContractNLI pilot.'}
    engine.require(p==expected,'Runtime policy changed');return raw,p

engine.validate_protocol=validate_protocol
engine.load_runtime_policy=load_runtime_policy
if __name__=='__main__':engine.main()
