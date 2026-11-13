"""CPU-only audit and frozen 1020-example ContractNLI pilot preparation."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import statistics
import time
from contractnli_common import *
from prepare_expanse_sft import render_pair

def main():
    assert not (DATA/'protocol.json').exists(), 'Never overwrite a prepared experiment'
    REPORT.mkdir(parents=True,exist_ok=True)
    raw={s:json.loads((RAW/(s+'.json')).read_text()) for s in ('train','dev','test')}
    assert [len(raw[s]['documents']) for s in raw]==[423,61,123]
    assert all(raw[s]['labels']==raw['train']['labels'] for s in raw)
    ids={s:{d['id'] for d in raw[s]['documents']} for s in raw}
    assert not ids['train']&ids['dev'] and not ids['train']&ids['test'] and not ids['dev']&ids['test']
    hashes={s:Counter(normalized_sha(d['text']) for d in raw[s]['documents']) for s in raw}
    # Held-out text is used solely for exact duplicate exclusion; test annotations are not inspected.
    held=set(hashes['dev'])|set(hashes['test']);seen=set();exclusions=[];train_docs=[]
    for doc in sorted(raw['train']['documents'],key=lambda d:ordered_key(d['id'])):
        h=normalized_sha(doc['text'])
        reason='same_normalized_text_as_heldout' if h in held else 'duplicate_training_text' if h in seen else None
        if reason:exclusions.append({'document_id':doc['id'],'reason':reason,'normalized_sha256':h});continue
        seen.add(h);train_docs.append(doc)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,trust_remote_code=False)
    tok.add_bos_token=False;tok.padding_side='right';tok.model_max_length=1000000
    labels=raw['train']['labels'];hypotheses=sorted(labels)
    train_candidates=[];dev_candidates=[];account=[];label_counts={};start=time.monotonic()
    for split,docs in [('train',train_docs),('dev',raw['dev']['documents'])]:
        label_counts[split]=Counter()
        for n,doc in enumerate(docs):
            assert len(doc['annotation_sets'])==1
            annotations=doc['annotation_sets'][0]['annotations'];assert set(annotations)==set(labels)
            sources=[];native=[];golds=[];all_input_fit=True
            for hid in hypotheses:
                row=source_row(doc,hid,labels[hid]['hypothesis']);gold=target(annotations[hid])
                assert parse_prediction(json.dumps(gold),{p['id'] for p in row['paragraphs']})[1] is None
                label_counts[split][gold['label']]+=1
                rendered,_=render_pair(tok,messages(row),json.dumps(gold))
                inputfit=rendered['input_tokens']<=7936;trainfit=rendered['total_tokens']<=8192 and rendered['completion_tokens']<=256
                all_input_fit &= inputfit
                account.append({'split':split,'id':row['id'],'document_id':doc['id'],
                    'input_tokens':rendered['input_tokens'],'completion_tokens':rendered['completion_tokens'],
                    'total_tokens':rendered['total_tokens'],'input_fits':inputfit,'training_fits':trainfit})
                sources.append(row);native.append(rendered);golds.append(gold)
            if split=='dev':
                # Retain whole contracts using input length only, never a gold-label/evidence filter.
                if all_input_fit:dev_candidates.append((doc['id'],sources,golds))
            else:
                for row,rendered,gold in zip(sources,native,golds):
                    if rendered['input_tokens']<=7936 and rendered['total_tokens']<=8192 and rendered['completion_tokens']<=256:
                        train_candidates.append((row,rendered,gold))
            if (n+1)%100==0:print(json.dumps({'stage':'tokenization','split':split,'documents':n+1,'seconds':round(time.monotonic()-start,1)}),flush=True)
    # Balance labels, without duplicating examples, spreading each label over contracts first.
    selected=[]
    for label in LABELS:
        groups=defaultdict(list)
        for item in train_candidates:
            if item[2]['label']==label:groups[item[0]['paper_id']].append(item)
        for items in groups.values():items.sort(key=lambda x:ordered_key(x[0]['id']))
        chosen=[];ordered_docs=sorted(groups,key=ordered_key);level=0
        while len(chosen)<340:
            batch=[groups[d][level] for d in ordered_docs if len(groups[d])>level]
            assert batch,'Insufficient eligible examples for balanced pilot'
            chosen.extend(batch[:340-len(chosen)]);level+=1
        selected.extend(chosen)
    selected.sort(key=lambda x:ordered_key(x[0]['id']))
    assert len(selected)==len({x[0]['id'] for x in selected})==1020
    dev_candidates.sort(key=lambda x:ordered_key(x[0]));dev=dev_candidates[:20]
    assert len(dev)==20,'Insufficient full-length development contracts'
    native_rows=[];logical=[]
    for row,rendered,gold in selected:
        metadata={k:row[k] for k in ['id','paper_id']}
        metadata.update(answerable=gold['label']!='NotMentioned',reference_index=0,human_review_status='pending')
        native_rows.append({**metadata,**rendered})
        logical.append({**metadata,'source_split':'train','hypothesis':row['question'],
            'messages':messages(row),'target_json':json.dumps(gold),
            'paragraphs':row['paragraphs'],'source_text_sha256':normalized_sha(next(d['text'] for d in train_docs if str(d['id'])==row['paper_id']))})
    inputs=[r for _,rows,_ in dev for r in rows]
    golds=[{'id':r['id'],'paper_id':r['paper_id'],**g} for _,rows,gs in dev for r,g in zip(rows,gs)]
    save_rows(DATA/'train.jsonl',native_rows);save_rows(DATA/'logical-data.jsonl',logical)
    save_rows(EVAL/'inputs.jsonl',inputs);save_rows(EVAL/'labels.jsonl',golds)
    save_rows(REPORT/'token-accounting.jsonl',account)
    from train_contractnli import engine
    cfg=engine.build_config(ARM,1020);config=ROOT/'configs/contractnli-v1-training.json';save(config,cfg)
    protocol={'schema_version':1,'experiment':engine.EXPERIMENT,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'selected_ids':[r['id'] for r in native_rows],'counts':{'examples':1020,'contracts':len({r['paper_id'] for r in native_rows}),'labels':dict(Counter(x[2]['label'] for x in selected))},
        'data_sha256':file_sha(DATA/'train.jsonl'),'config_sha256':file_sha(config),
        'logical_data':{'path':str((DATA/'logical-data.jsonl').relative_to(ROOT)),'sha256':file_sha(DATA/'logical-data.jsonl')},
        'evaluation_input':{'path':str((EVAL/'inputs.jsonl').relative_to(ROOT)),'sha256':file_sha(EVAL/'inputs.jsonl')},
        'instruction_sha256':sha(INSTRUCTION.encode()),'no_truncation':True,'official_train_only':True,
        'source_files_sha256':{str(p.relative_to(ROOT)):file_sha(p) for p in [RAW/'train.json',Path(__file__),ROOT/'scripts/contractnli_common.py']},
        'selection':'340 per label; deterministic hash, round-robin across training contracts, no repetitions',
        'evaluation_selection':'20 deterministic hash-selected full-input-fit official dev contracts, all 17 hypotheses; selected before model outputs; no target-based selection',
        'semantic_review_rounds':1,'test_annotations_used':False,'evaluation_is_official_test':False}
    save(DATA/'protocol.json',protocol)
    runtime=json.loads((ROOT/'configs/revision-v10-runtime.json').read_text())
    runtime.update(original_data_protocol_sha256=file_sha(DATA/'protocol.json'),reason='Reuse validated activation offloading for an independent ContractNLI pilot.')
    save(ROOT/'configs/contractnli-v1-runtime.json',runtime)
    save(EVAL/'protocol.json',{'input_sha256':file_sha(EVAL/'inputs.jsonl'),'labels_sha256':file_sha(EVAL/'labels.jsonl'),
        'question_ids':[r['id'] for r in inputs],'contracts':[d for d,_,_ in dev],
        'selection':protocol['evaluation_selection'],'semantic_review_rounds':1,
        'primary_metrics':['label_macro_f1','span_micro_f1_on_gold_non_neutral','label_and_exact_evidence_accuracy'],
        'content_review_plan':'One paired AI review of 18 cases: 6 deterministic cases per gold label, across different contracts within each label, selected before predictions. Not an overall human quality estimate.',
        'review_ids':[]})
    # Deterministic review subset uses label strata but never predictions.
    review=[]
    for label in LABELS:
        used=set()
        for g in sorted([g for g in golds if g['label']==label],key=lambda g:ordered_key(g['id'])):
            if g['paper_id'] in used:continue
            used.add(g['paper_id']);review.append(g['id'])
            if len(used)==6:break
        assert len(used)==6
    ep=json.loads((EVAL/'protocol.json').read_text());ep['review_ids']=review;save(EVAL/'protocol.json',ep)
    summary={'status':'suitable_for_pilot','dataset_url':'https://stanfordnlp.github.io/contract-nli/',
        'archive_sha256':file_sha(RAW/'contract-nli.zip'),'source_hashes':{s:file_sha(RAW/(s+'.json')) for s in raw},
        'official_contract_counts':{s:len(raw[s]['documents']) for s in raw},
        'official_train_pairs':423*17,'official_dev_pairs':61*17,'hypotheses':17,
        'duplicate_exclusions':exclusions,'remaining_training_contracts':len(train_docs),
        'eligible_training_pairs':len(train_candidates),'eligible_dev_contracts':len(dev_candidates),
        'selected_training':protocol['counts'],'selected_eval_contracts':20,'selected_eval_pairs':len(inputs),
        'selected_eval_labels':dict(Counter(g['label'] for g in golds)),
        'input_tokens_train':{'median':statistics.median(r['input_tokens'] for r in native_rows),'max':max(r['input_tokens'] for r in native_rows)},
        'total_training_tokens':sum(r['total_tokens'] for r in native_rows),
        'length_input_budget':7936,'length_total_budget':8192,'generation_budget':256,
        'all_source_text_preserved':True,'evidence_annotations':'Original official sentence/list-item spans, unchanged',
        'normalized_text_overlap_train_dev':len(set(hashes['train'])&set(hashes['dev'])),
        'normalized_text_overlap_train_test':len(set(hashes['train'])&set(hashes['test'])),
        'semantic_near_duplicate_audit_performed':False,'test_annotations_used':False,
        'split_policy':'Official splits preserved; held-out text hashes used only to remove exact normalized duplicates from training',
        'evaluation_scope':'Small development pilot on 20 unseen contracts; not full benchmark or arbitrary legal QA',
        'format':'label + evidence_ids; NotMentioned is different from Contradiction',
        'source_label_counts_after_dedup':{s:dict(c) for s,c in label_counts.items()},
        'preparation_seconds':round(time.monotonic()-start,2)}
    save(REPORT/'data-inspection.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
