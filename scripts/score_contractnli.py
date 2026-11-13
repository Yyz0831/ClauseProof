"""Deterministic scoring of frozen ContractNLI outputs; no generation or AI review."""
from collections import Counter
import json
import math
import re
from contractnli_common import *

def diagnostic_label(raw):
    # Only recover a single explicit label field; do not infer from prose or rewrite outputs.
    hits=re.findall(r'"label"\s*:\s*"(Entailment|Contradiction|NotMentioned)"',raw or '')
    return hits[0] if len(hits)==1 else None

def prf(tp,fp,fn):
    return {'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'tp':tp,'fp':fp,'fn':fn}

def aggregate(decisions):
    n=len(decisions);matrix={g:{p:0 for p in (*LABELS,'Invalid')} for g in LABELS}
    for d in decisions:matrix[d['gold_label']][d['label'] or 'Invalid']+=1
    per={}
    for label in LABELS:
        tp=matrix[label][label];fp=sum(matrix[g][label] for g in LABELS if g!=label)
        fn=sum(matrix[label][p] for p in matrix[label] if p!=label)
        per[label]={**prf(tp,fp,fn),'support':sum(matrix[label].values())}
    supported=[d for d in decisions if d['gold_label']!='NotMentioned']
    neutral=[d for d in decisions if d['gold_label']=='NotMentioned']
    ev=prf(sum(d['evidence_tp'] for d in supported),sum(d['evidence_fp'] for d in supported),sum(d['evidence_fn'] for d in supported))
    invalid=[d for d in decisions if not d['format_valid']]
    return {'examples':n,'format_valid':n-len(invalid),'format_valid_rate':(n-len(invalid))/n,
        'label_correct':sum(d['label_correct'] for d in decisions),'label_accuracy':sum(d['label_correct'] for d in decisions)/n,
        'diagnostic_label_correct':sum(d['diagnostic_label_correct'] for d in decisions),
        'diagnostic_label_accuracy':sum(d['diagnostic_label_correct'] for d in decisions)/n,
        'invalid_format_examples':len(invalid),'invalid_format_with_diagnostic_correct_label':sum(d['diagnostic_label_correct'] for d in invalid),
        'label_macro_f1':sum(s['f1'] for s in per.values())/3,'per_class':per,'confusion_matrix':matrix,
        'evidence_on_gold_supported':{'examples':len(supported),**ev,'exact_sets':sum(d['evidence_exact'] for d in supported)},
        'notmentioned':{'examples':len(neutral),'correct':sum(d['label_correct'] for d in neutral),
            'false_definite':sum(d['diagnostic_label'] in ('Entailment','Contradiction') for d in neutral),
            'false_definite_rate':sum(d['diagnostic_label'] in ('Entailment','Contradiction') for d in neutral)/len(neutral),
            'strict_valid_outputs_with_citations':sum(bool(d['pred_evidence']) for d in neutral)},
        'joint_exact_success':sum(d['joint_exact_success'] for d in decisions),
        'joint_exact_success_rate':sum(d['joint_exact_success'] for d in decisions)/n}

def main():
    assert not (REPORT/'metrics.json').exists(),'Do not overwrite a completed scoring artifact'
    proof=json.loads((REPORT/'remote-artifact-proof.json').read_text())
    for name,h in proof['hashes'].items():assert file_sha(ROOT/name)==h,'Local/remote mismatch: '+name
    p=json.loads((EVAL/'protocol.json').read_text());inputs=read_rows(EVAL/'inputs.jsonl');gold=read_rows(EVAL/'labels.jsonl')
    assert file_sha(EVAL/'inputs.jsonl')==p['input_sha256'] and file_sha(EVAL/'labels.jsonl')==p['labels_sha256']
    assert [r['id'] for r in inputs]==[r['id'] for r in gold]==p['question_ids'] and len(gold)==340
    train=ROOT/'runs/train-contractnli-v1'/ARM;m=json.loads((train/'manifest.json').read_text())
    checks=json.loads((train/'gradient-checks.json').read_text())
    assert m['status']=='completed' and m['training_complete'] and m['actual_final_global_step']==510 and m['actual_final_epoch']==1.
    assert len(checks)==510 and all(c['finite'] and c['any_nonzero'] for c in checks)
    assert m['adapter_final_trainable_sha256']!=m['adapter_trainer_resolved_trainable_sha256'] and m['finite_adapter_weights']
    assert file_sha(train/'adapter/adapter_model.safetensors')==m['adapter_sha256']
    trainrows=read_rows(DATA/'train.jsonl');assert not {r['paper_id'] for r in trainrows}&{r['paper_id'] for r in inputs}
    decisions={};predictions={};manifests={};summaries={}
    for arm in ('base','tuned'):
        directory=ROOT/'runs'/NAME/arm;mf=json.loads((directory/'manifest.json').read_text());manifests[arm]=mf
        preds=read_rows(directory/'predictions.jsonl');predictions[arm]={r['id']:r for r in preds}
        assert mf['status']=='completed' and mf['inference_complete'] and len(preds)==340
        assert [r['id'] for r in preds]==p['question_ids']
        assert mf['input_sha256']==p['input_sha256'] and mf['predictions_sha256']==file_sha(directory/'predictions.jsonl')
        assert all(r['generation_attempts']==1 and r['automatic_retries']==0 and r['raw_output'] is not None for r in preds)
        out=[]
        for row,g,pr in zip(inputs,gold,preds):
            parsed,error,_=parse_prediction(pr['raw_output'],{q['id'] for q in row['paragraphs']})
            assert parsed==pr['prediction'] and error==pr['parse_error'],'Parser changed'
            label=parsed['label'] if parsed else None;ev=set(parsed['evidence_ids']) if parsed else set();ge=set(g['evidence_ids'])
            dl=diagnostic_label(pr['raw_output'])
            out.append({'id':g['id'],'paper_id':g['paper_id'],'gold_label':g['label'],'label':label,
                'format_valid':error is None,'parse_error':error,'diagnostic_label':dl,'diagnostic_label_correct':dl==g['label'],
                'label_correct':label==g['label'],'pred_evidence':sorted(ev),'gold_evidence':sorted(ge),
                'evidence_tp':len(ev&ge),'evidence_fp':len(ev-ge),'evidence_fn':len(ge-ev),
                'evidence_exact':ev==ge,'joint_exact_success':error is None and label==g['label'] and ev==ge,
                'hit_token_limit':pr['hit_token_limit']})
        decisions[arm]=out;summaries[arm]=aggregate(out)
        summaries[arm].update(generation_seconds=mf['elapsed_seconds'],truncated_outputs=sum(r['hit_token_limit'] for r in preds))
    assert manifests['base']['adapter_sha256'] is None and manifests['tuned']['adapter_sha256']==m['adapter_sha256']
    assert {k:v for k,v in manifests['base']['config'].items() if k!='label'}=={k:v for k,v in manifests['tuned']['config'].items() if k!='label'}
    assert manifests['base']['generation_kwargs']==manifests['tuned']['generation_kwargs']
    assert manifests['base']['packages']==manifests['tuned']['packages']
    assert file_sha(ROOT/'runs'/NAME/'base/prompts.jsonl')==file_sha(ROOT/'runs'/NAME/'tuned/prompts.jsonl')
    paired={}
    for metric in ('label_correct','joint_exact_success'):
        c=Counter()
        for b,t in zip(decisions['base'],decisions['tuned']):c['both_pass' if b[metric] and t[metric] else 'improved' if t[metric] else 'regressed' if b[metric] else 'both_fail']+=1
        paired[metric]=dict(c)
    result={'schema_version':1,'experiment':NAME,'evaluation_split':'official dev subset','contracts':20,'examples':340,
        'training_examples':1020,'training_contracts':368,'training_steps':510,'training_minutes':m['wall_seconds']/60,
        'model':MODEL,'base_revision':REVISION,'adapter_sha256':m['adapter_sha256'],
        'definitions':{'label_metrics':'Strict parsed outputs; invalid schema is an Invalid prediction and a false negative for the true class.',
            'diagnostic_label':'Exactly one explicit JSON-style label field in raw output; reporting only, never repairs an output.',
            'evidence':'Span ID set micro P/R/F1 on 202 gold Entailment or Contradiction examples; strict-invalid output has an empty predicted set; independent of predicted label.',
            'joint_exact':'Strict schema valid AND correct label AND predicted evidence set exactly equals official evidence set; stricter than semantic adequacy.',
            'notmentioned_false_definite':'Gold NotMentioned with diagnostic Entailment/Contradiction; invalid/unreadable counted separately, not considered correct.'},
        'arms':summaries,'paired':paired,'raw_outputs_identical':sum(predictions['base'][g['id']]['raw_output']==predictions['tuned'][g['id']]['raw_output'] for g in gold),
        'limits':['One seed and 20 contracts; 340 hypotheses are clustered within contracts.','17 fixed hypotheses; not arbitrary legal QA.','No full official test evaluation.','Exact normalized duplicate exclusion performed; no comprehensive near-duplicate audit.','Semantic audit is a separate single AI pass on 18 preselected pairs; no human verification.']}
    save(REPORT/'metrics.json',result);save(REPORT/'per-example-metrics.json',decisions)
    save(REPORT/'result-verification.json',{'status':'passed','local_remote_hashes_verified':len(proof['hashes']),
        'training_complete':True,'optimizer_steps':510,'finite_nonzero_gradient_steps':510,'finite_saved_adapter':True,
        'adapter_changed_after_initialization':True,'identical_prompts':True,'identical_generation_settings':True,
        'identical_inference_packages':True,'contract_split_overlap':0,'generations_per_arm':340,'generation_attempts_per_case':1,
        'current_live_D_ready':proof['current_live_health']['status']=='ready'})
    byid={r['id']:r for r in inputs};gs={r['id']:r for r in gold};packet=[]
    for qid in p['review_ids']:
        row=byid[qid];g=gs[qid];parts={x['id']:x['text'] for x in row['paragraphs']}
        citations=set(g['evidence_ids'])
        for arm in ('base','tuned'):
            pred=predictions[arm][qid]['prediction']
            if pred:citations.update(pred['evidence_ids'])
        packet.append({'id':qid,'paper_id':row['paper_id'],'hypothesis':row['question'],'gold':g,
            'base_raw':predictions['base'][qid]['raw_output'],'tuned_raw':predictions['tuned'][qid]['raw_output'],
            'evidence_text':{s:parts[s] for s in sorted(citations)},'full_source_in_inputs_jsonl':True})
    save(REPORT/'single-review-packet.json',packet)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
