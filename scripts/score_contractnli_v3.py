"""Score frozen v3 outputs and build one preselected review packet; no generation."""
from collections import Counter, defaultdict
import json
import re
from contractnli_v3_common import *
from score_contractnli import aggregate, diagnostic_label

ARMS = ('base', 'v1', 'v2', 'v3')


def paired(old, new):
    result = {}
    for metric in ('label_correct', 'joint_exact_success'):
        c = Counter()
        for b, t in zip(old, new):
            assert b['id'] == t['id']
            c['both_pass' if b[metric] and t[metric] else 'improved' if t[metric]
              else 'regressed' if b[metric] else 'both_fail'] += 1
        result[metric] = dict(c)
    return result


def decisions_for(inputs, gold, preds, content=False):
    out = []
    for src, g, pr in zip(inputs, gold, preds):
        raw = pr['raw_output'].strip()
        wrapper = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.S) if content else None
        text = wrapper.group(1).strip() if wrapper else raw
        parsed, error, _ = parse_prediction(text, {x['id'] for x in src['paragraphs']})
        if not content:
            assert parsed == pr['prediction'] and error == pr['parse_error'], 'Parser changed'
        label = parsed['label'] if parsed else None
        ev = set(parsed['evidence_ids']) if parsed else set()
        ge = set(g['evidence_ids'])
        dl = label if content else diagnostic_label(raw)
        out.append({'id': g['id'], 'paper_id': g['paper_id'], 'gold_label': g['label'],
            'label': label, 'format_valid': error is None, 'parse_error': error,
            'diagnostic_label': dl, 'diagnostic_label_correct': dl == g['label'],
            'label_correct': label == g['label'], 'pred_evidence': sorted(ev), 'gold_evidence': sorted(ge),
            'evidence_tp': len(ev & ge), 'evidence_fp': len(ev - ge), 'evidence_fn': len(ge - ev),
            'evidence_exact': ev == ge,
            'joint_exact_success': error is None and label == g['label'] and ev == ge,
            'hit_token_limit': pr['hit_token_limit'], 'strict_format_valid': pr['parse_error'] is None,
            'wrapper_removed_for_diagnostic_only': bool(wrapper)})
    return out


def main():
    assert not (REPORT / 'metrics.json').exists(), 'Never overwrite scoring artifacts'
    proof = json.loads((REPORT / 'remote-artifact-proof.json').read_text())
    for name, h in proof['hashes'].items():
        assert file_sha(ROOT / name) == h, 'Local/remote mismatch: ' + name
    protocol = json.loads((EVAL / 'protocol.json').read_text())
    inputs, gold = read_rows(EVAL / 'inputs.jsonl'), read_rows(EVAL / 'labels.jsonl')
    assert file_sha(EVAL / 'inputs.jsonl') == protocol['input_sha256']
    assert file_sha(EVAL / 'labels.jsonl') == protocol['labels_sha256']
    assert [r['id'] for r in inputs] == [r['id'] for r in gold] == protocol['question_ids']
    assert len(gold) == 306 and Counter(g['label'] for g in gold) == {'Entailment':150,'NotMentioned':130,'Contradiction':26}
    train = ROOT / 'runs/train-contractnli-v3' / ARM
    m = json.loads((train / 'manifest.json').read_text())
    checks = json.loads((train / 'gradient-checks.json').read_text())
    assert m['status'] == 'completed' and m['training_complete']
    assert m['actual_final_global_step'] == 1020 and m['actual_final_epoch'] == 1.
    assert len(checks) == 1020 and all(c['finite'] and c['any_nonzero'] for c in checks)
    assert m['adapter_final_trainable_sha256'] != m['adapter_trainer_resolved_trainable_sha256']
    assert m['finite_adapter_weights'] and file_sha(train / 'adapter/adapter_model.safetensors') == m['adapter_sha256']
    trainrows = read_rows(DATA / 'train.jsonl')
    assert len(trainrows) == 2040 and len({r['paper_id'] for r in trainrows}) == 416
    assert not {r['paper_id'] for r in trainrows} & {r['paper_id'] for r in inputs}
    for version in ('v1','v2'):
        old = json.loads((ROOT / f'data/experiments/contractnli-{version}/protocol.json').read_text())
        assert not set(old['contracts']) & set(protocol['contracts'])
    predictions, manifests, strict, content = {}, {}, {}, {}
    for arm in ARMS:
        d = ROOT / 'runs' / NAME / arm
        mf = json.loads((d / 'manifest.json').read_text())
        preds = read_rows(d / 'predictions.jsonl')
        assert mf['status'] == 'completed' and mf['inference_complete'] and len(preds) == 306
        assert [r['id'] for r in preds] == protocol['question_ids']
        assert mf['input_sha256'] == protocol['input_sha256']
        assert mf['predictions_sha256'] == file_sha(d / 'predictions.jsonl')
        assert all(r['generation_attempts'] == 1 and r['automatic_retries'] == 0 and r['raw_output'] is not None for r in preds)
        predictions[arm], manifests[arm] = preds, mf
        strict[arm] = decisions_for(inputs, gold, preds)
        content[arm] = decisions_for(inputs, gold, preds, content=True)
    assert manifests['base']['adapter_sha256'] is None
    assert manifests['v1']['adapter_sha256'] == V1_ADAPTER_SHA
    assert manifests['v2']['adapter_sha256'] == V2_ADAPTER_SHA
    assert manifests['v3']['adapter_sha256'] == m['adapter_sha256']
    for arm in ('v1', 'v2', 'v3'):
        assert {k:v for k,v in manifests['base']['config'].items() if k != 'label'} == {k:v for k,v in manifests[arm]['config'].items() if k != 'label'}
        assert manifests['base']['generation_kwargs'] == manifests[arm]['generation_kwargs']
        assert manifests['base']['packages'] == manifests[arm]['packages']
        assert file_sha(ROOT / 'runs' / NAME / 'base/prompts.jsonl') == file_sha(ROOT / 'runs' / NAME / arm / 'prompts.jsonl')
    for mode, decisions, filename in [('strict', strict, 'metrics.json'), ('content', content, 'content-diagnostics.json')]:
        summaries = {}
        for arm, ds in decisions.items():
            s = aggregate(ds)
            s['generation_seconds'] = manifests[arm]['elapsed_seconds']
            s['truncated_outputs'] = sum(d['hit_token_limit'] for d in ds)
            s['markdown_wrappers_removed'] = sum(d['wrapper_removed_for_diagnostic_only'] for d in ds)
            invalid = [d for d in ds if not d['strict_format_valid']]
            s['strict_invalid_content'] = {'examples':len(invalid),
                'correct_label':sum(d['label_correct'] for d in invalid),
                'correct_label_and_exact_evidence':sum(d['joint_exact_success'] for d in invalid)}
            summaries[arm] = s
        contracts = []
        for doc in protocol['contracts']:
            row = {'contract':str(doc)}
            for arm in ARMS:
                ds = [d for d in decisions[arm] if str(d['paper_id']) == str(doc)]
                assert len(ds) == 17
                row[arm] = {'label_correct':sum(d['label_correct'] for d in ds),
                    'joint_exact_success':sum(d['joint_exact_success'] for d in ds)}
            contracts.append(row)
        result = {'schema_version':1, 'experiment':NAME, 'mode':mode, 'examples':306, 'contracts':18,
            'training_examples':2040, 'training_contracts':416, 'training_steps':1020,
            'training_minutes':m['wall_seconds']/60, 'model':MODEL, 'base_revision':REVISION,
            'adapter_sha256':m['adapter_sha256'],
            'definitions': {'label':'Strict schema-valid output; invalid output is a false negative for the true class.',
                'content_transformation':'Content mode removes only a single enclosing JSON/untyped Markdown fence; no other repair, raw output unchanged.',
                'evidence':'Span set micro P/R/F1 on 176 gold non-neutral examples, independent of predicted label. Invalid output has empty predicted set.',
                'joint_exact':'Schema valid in this mode, correct label and exact official evidence set. Stricter than semantic adequacy.'},
            'arms':summaries, 'paired_v2_v3':paired(decisions['v2'], decisions['v3']), 'paired_v1_v3':paired(decisions['v1'], decisions['v3']),
            'paired_base_v3':paired(decisions['base'], decisions['v3']), 'by_contract':contracts,
            'limitations':['One seed; 18 development contracts, not official test.',
                '306 hypotheses clustered within 18 contracts; 17 fixed hypotheses.',
                'Same 2040 examples and 1020 updates as v2; label mixture and evidence structure both changed, so no single-factor attribution.',
                'One separate AI source review of 18 preselected pairs; no human expert verification.',
                'No exhaustive semantic near-duplicate audit.']}
        save(REPORT / filename, result)
    save(REPORT / 'per-example-metrics.json', strict)
    save(REPORT / 'content-diagnostic-per-example.json', content)
    save(REPORT / 'result-verification.json', {'status':'passed','local_remote_hashes_verified':len(proof['hashes']),
        'training_complete':True,'optimizer_steps':1020,'finite_nonzero_gradient_steps':1020,
        'finite_saved_adapter':True,'adapter_changed_after_initialization':True,
        'identical_prompts':True,'identical_generation_settings':True,'identical_inference_packages':True,
        'train_eval_contract_overlap':0,'previous_v1_v2_eval_contract_overlap':0,'generations_per_arm':306,
        'generation_attempts_per_case':1,'current_live_D_ready':proof['current_live_health']['status']=='ready'})
    srcs = {x['id']:x for x in inputs}; gs = {x['id']:x for x in gold}
    byarm = {a:{x['id']:x for x in ps} for a,ps in predictions.items()}
    packet = []
    for qid in protocol['review_ids']:
        src, g = srcs[qid], gs[qid]
        ids = set(g['evidence_ids'])
        for a in ARMS:
            p = byarm[a][qid]['prediction']
            if p: ids.update(p['evidence_ids'])
        parts = {p['id']:p['text'] for p in src['paragraphs']}
        packet.append({'id':qid,'paper_id':src['paper_id'],'hypothesis':src['question'],'gold':g,
            'raw':{a:byarm[a][qid]['raw_output'] for a in ARMS},
            'evidence_text':{i:parts[i] for i in sorted(ids)},'full_source_in_inputs_jsonl':True})
    save(REPORT / 'single-review-packet.json', packet)
    print(json.dumps({'status':'scoring_complete','experiment':NAME,'arms':ARMS,'questions':len(gold),'review_packet_cases':len(packet)}))


if __name__ == '__main__':
    main()
