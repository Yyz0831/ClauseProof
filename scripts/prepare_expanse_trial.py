"""Freeze 59 previously used development inputs for the quantized Aya trial.

No download, tokenizer, model, GPU, or inference is used here. The seven-field
inference file contains no evaluation labels. Existing files are never replaced.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from build_blind_review import digest, json_bytes, verified_run
from run_baseline import ROOT, INSTRUCTION, messages_for
from serve_live import MODE, LIVE_INSTRUCTION, LIVE_EXAMPLES, live_messages, prepare_document
from summarize_ad_review import validate_public, load_verified_key

EXPANSE_REVISION = '5062468bf9bc0c6035fd64e06274333ec127d980'
EXPANSE_MODEL_ID = 'CohereLabs/aya-expanse-8b'
INPUT_FIELDS = ('id', 'paper_id', 'question', 'title', 'paragraphs', 'context', 'prompt_mode')
CODE_FILES = (
    'scripts/prepare_expanse_trial.py', 'scripts/run_expanse_trial.py',
    'scripts/summarize_expanse_trial.py', 'scripts/run_baseline.py',
    'scripts/serve_live.py', 'scripts/paperproof_eval.py', 'scripts/analyze_run.py',
    'scripts/summarize_ad_review.py', 'scripts/build_blind_review.py',
)
LABEL_FIELDS = ('answerable', 'reference_answers', 'reference_evidence_ids',
                'original_answer_annotations', 'review_status', 'source_split',
                'question_id', 'context_mode')
GROUP_COUNTS = {'review40': 40, 'live_visible': 12,
                'live_opened_heldback': 4, 'live_real_paper': 3}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jsonl_bytes(rows):
    return ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode('utf-8')


def message_hash(row):
    messages = messages_for(row) if row['prompt_mode'] == 'baseline' else live_messages(row)
    return digest(json_bytes(messages))


def validate_inputs(rows, labels):
    require(len(rows) == len(labels) == 59, 'Require all 59 fixed inputs and labels')
    ids = [r['id'] for r in rows]
    require(len(set(ids)) == 59 and ids == [r['id'] for r in labels], 'Input/label IDs differ or repeat')
    require(dict(Counter(r['group'] for r in labels)) == GROUP_COUNTS, 'Wrong fixed group counts')
    require([r['prompt_mode'] for r in rows] == ['baseline'] * 40 + ['live'] * 19,
            'Prompt mode order changed')
    for row, label in zip(rows, labels):
        require(set(row) == set(INPUT_FIELDS), f'Inference field leak/missing field: {row["id"]}')
        for field in ('id', 'paper_id', 'question', 'title', 'context'):
            require(isinstance(row[field], str) and row[field].strip(), f'Empty/invalid {field}: {row["id"]}')
        pids = [p['id'] for p in row['paragraphs']]
        require(pids and len(set(pids)) == len(pids), f'Paragraph IDs repeat/empty: {row["id"]}')
        require(all(set(p) == {'id', 'section', 'text'} and all(isinstance(v, str) for v in p.values())
                    for p in row['paragraphs']), f'Paragraph schema changed: {row["id"]}')
        require(all(label[k] == row[k] for k in ('id', 'paper_id', 'prompt_mode')), 'Label identity mismatch')
        if row['prompt_mode'] == 'baseline':
            require(type(label['answerable']) is bool, 'Original source label is not boolean')
            require(all(e in pids for group in label['reference_evidence_ids'] for e in group),
                    f'Original evidence ID missing: {row["id"]}')
        else:
            require('answerable' not in label and 'reference_answers' not in label,
                    'Live expectation must not masquerade as original dataset gold')
            require('expected' in label, 'Live expectation missing')


def build_payloads(root, *, frozen_at, expanse_model_id, require_trial_code=True):
    """Build bytes deterministically from frozen sources, without model outputs filtering."""
    root = Path(root)
    source_bytes = {}

    def read_bytes(name):
        data = (root/name).read_bytes()
        source_bytes[name] = data
        return data

    def read_json(name):
        return json.loads(read_bytes(name))

    def read_rows(name):
        return [json.loads(line) for line in read_bytes(name).decode().splitlines() if line.strip()]

    require(datetime.fromisoformat(frozen_at).tzinfo is not None, 'Freeze time needs a timezone')
    require(expanse_model_id == EXPANSE_MODEL_ID, 'Use the fixed official Aya Expanse model repository')
    review_root = root/'reports/review-ad-v5'
    reviewed = validate_public(review_root)
    key = load_verified_key(review_root, reviewed)
    for name, data in reviewed['contents'].items():
        source_bytes['reports/review-ad-v5/' + name] = data
    dev_path = 'data/curated/followup-v2/untouched_dev.jsonl'
    dev_rows = read_rows(dev_path)
    dev = {r['id']: r for r in dev_rows}
    require(len(dev) == len(dev_rows), 'Duplicate original development IDs')
    source_sha = digest(source_bytes[dev_path])
    cfg = read_json('configs/baseline.json')
    require(cfg['revision'] == 'af89d219b53ed9b13b8a4645f9c8028973510324', 'Tiny Aya revision changed')
    require(cfg['max_total_tokens'] == 8192 and cfg['max_new_tokens'] == 256
            and cfg['do_sample'] is False and cfg['seed'] == 42, 'Generation configuration changed')
    training = read_json('runs/train-d-followup-v2/manifest.json')
    require(training['training_complete'] is True and training['base_model'] == cfg, 'D lineage mismatch')
    adapter_path = 'runs/train-d-followup-v2/adapter'
    adapter_sha = digest(read_bytes(adapter_path + '/adapter_model.safetensors'))
    require(adapter_sha == training['adapter_sha256'] == '576eac6622ce5bc332912998e42bed51f5c24099888c543856ecca20c009d437',
            'D adapter checksum mismatch')
    adapter_config_sha = digest(read_bytes(adapter_path + '/adapter_config.json'))
    references, old_predictions = {}, {}
    for label, path in [('A', 'runs/compare-a-untouched-v2'), ('D', 'runs/compare-d-untouched-v2')]:
        records, manifest, provenance = verified_run(root/path, dev, source_sha)
        old_predictions[label] = records
        require(manifest['config'] == cfg and manifest.get('quantization') is None, f'{label} is not the original BF16 reference')
        if label == 'A':
            require(manifest.get('adapter') is None, 'Original A must not have an adapter')
        else:
            require(manifest['adapter']['weights_sha256'] == adapter_sha, 'Original D adapter differs')
        for field, script in [('runner_sha256', 'scripts/run_baseline.py'), ('evaluator_sha256', 'scripts/paperproof_eval.py')]:
            require(manifest[field] == digest((root/script).read_bytes()), f'Historical computation changed: {script}')
        for filename in ('manifest.json', 'predictions.jsonl'):
            read_bytes(path + '/' + filename)
        references[label] = {
            'run_dir': path, 'manifest_sha256': digest(source_bytes[path + '/manifest.json']),
            'predictions_sha256': digest(source_bytes[path + '/predictions.jsonl']),
            'source_input_sha256': source_sha, 'dtype': 'bfloat16', 'quantization': None,
            'adapter_sha256': adapter_sha if label == 'D' else None,
            'role': 'Historical BF16 reference only; not a controlled 4-bit size comparison.',
        }
    rows, labels = [], []
    for public, mapping, review in zip(reviewed['packet']['questions'], key['questions'], reviewed['cases']):
        original = dev[mapping['id']]
        require(mapping['review_id'] == public['review_id'] == review['review_id'], 'Review order mismatch')
        require(mapping['paper_id'] == original['paper_id'] and mapping['source_answerable'] == original['answerable'],
                'Original identity/label differs from fixed packet')
        require(all(public[k] == original[k] for k in ('title', 'question', 'paragraphs')), 'Public source differs from original dev')
        row = {field: deepcopy(original[field]) for field in INPUT_FIELDS if field != 'prompt_mode'}
        row['prompt_mode'] = 'baseline'
        rows.append(row)
        labels.append({
            'id': row['id'], 'paper_id': row['paper_id'], 'prompt_mode': 'baseline', 'group': 'review40',
            'review_id': public['review_id'], 'source_path': dev_path, 'source_id': row['id'],
            **{field: deepcopy(original[field]) for field in LABEL_FIELDS},
            'previous_ai_review': {'assessment': deepcopy(review), 'models_by_slot': deepcopy(mapping['models_by_slot']),
                                   'independent_human_review': False, 'human_review_status': 'pending'},
        })
        a, d = old_predictions['A'][row['id']], old_predictions['D'][row['id']]
        require((a['prompt_sha256'], a['input_tokens']) == (d['prompt_sha256'], d['input_tokens']),
                'Historical A/D rendered inputs differ')
    require([r['review_id'] for r in labels] == [f'q{i:03d}' for i in range(1, 41)], 'Fixed 40 order changed')
    require(len({r['paper_id'] for r in rows}) == 40, 'Fixed 40 must remain distinct real papers')
    require(sum(label['answerable'] for label in labels) == 20, 'Preserve the original 20/20 source-label allocation')

    acceptance = read_json('reports/live-v5/acceptance-plan.json')
    visible_review = read_json('reports/live-v5/visible-ai-assessment.json')
    opened_review = read_json('reports/live-v5/heldback-ai-assessment.json')
    require(len(opened_review['cases']) == 4, 'Previously held-back fixtures must already have been opened/reviewed')
    require(opened_review['independent_human_review'] is False, 'Do not claim AI fixture review is human review')
    for filename, group, count in [('live-test-cases.jsonl', 'live_visible', 12),
                                    ('live-test-cases-heldback.jsonl', 'live_opened_heldback', 4)]:
        path = 'reports/live-v5/' + filename
        fixtures = read_rows(path)
        require(digest(source_bytes[path]) == acceptance['files'][path]['sha256'], f'Fixture checksum mismatch: {path}')
        require(len(fixtures) == count, 'Do not filter synthetic fixtures by previous results')
        for fixture in fixtures:
            prepared = prepare_document(fixture['request'])
            require([{k: p[k] for k in ('id', 'text')} for p in prepared['paragraphs']] == fixture['source_paragraphs'],
                    'Live fixture paragraph mapping changed')
            row = {'id': fixture['id'], 'paper_id': 'synthetic:' + fixture['id'],
                   **prepared, 'prompt_mode': 'live'}
            rows.append(row)
            if group == 'live_visible':
                previous = [{'run': run['run'], 'assessment': deepcopy(next(c for c in run['cases'] if c['id'] == fixture['id']))}
                            for run in visible_review['runs']]
            else:
                previous = deepcopy(next(c for c in opened_review['cases'] if c['id'] == fixture['id']))
            labels.append({'id': row['id'], 'paper_id': row['paper_id'], 'prompt_mode': 'live', 'group': group,
                           'source_path': path, 'source_id': fixture['id'], 'origin': fixture['origin'],
                           'category': fixture['category'], 'expected': deepcopy(fixture['expected']),
                           'previous_ai_review': previous, 'independent_human_review': False,
                           'human_review_status': 'pending', 'previously_used_development_material': True})

    smoke_inputs = read_json('reports/live-v5/real-paper-smoke-input.json')
    smoke_plan = read_json('reports/live-v5/real-paper-smoke-plan.json')
    smoke_review = read_json('reports/live-v5/real-paper-smoke-report.json')
    require(digest(source_bytes['reports/live-v5/real-paper-smoke-input.json']) == smoke_plan['input_sha256'],
            'Real-paper request checksum mismatch')
    require(smoke_plan['source_packet_sha256'] == digest(source_bytes['reports/review-ad-v5/human-review-packet.json']),
            'Smoke source packet mismatch')
    require(len(smoke_inputs['cases']) == len(smoke_plan['cases']) == len(smoke_review['cases']) == 3,
            'Keep all three previous real-paper cases')
    key_by_review = {q['review_id']: q for q in key['questions']}
    for index, (item, spec, prior) in enumerate(zip(smoke_inputs['cases'], smoke_plan['cases'], smoke_review['cases']), 1):
        require(item['case_id'] == spec['case_id'] == prior['case_id'], 'Real-paper order changed')
        prepared = prepare_document(item['request'])
        require(digest(json.dumps(prepared['paragraphs'], ensure_ascii=False, sort_keys=True).encode()) == spec['expected_api_paragraphs_sha256'],
                'Real-paper live paragraph mapping changed')
        old_key = key_by_review[item['source_review_id']]
        row = {'id': f'live-v5-real-paper-{index:02d}', 'paper_id': old_key['paper_id'], **prepared, 'prompt_mode': 'live'}
        rows.append(row)
        labels.append({'id': row['id'], 'paper_id': row['paper_id'], 'prompt_mode': 'live', 'group': 'live_real_paper',
                       'source_path': 'reports/live-v5/real-paper-smoke-input.json', 'source_id': item['case_id'],
                       'review_id': item['source_review_id'], 'same_original_question_id': old_key['id'],
                       'expected': deepcopy(spec['expectation']), 'previous_ai_review': deepcopy(prior['judgment']),
                       'independent_human_review': False, 'human_review_status': 'pending',
                       'previously_used_development_material': True})
    validate_inputs(rows, labels)
    inputs_bytes, labels_bytes = jsonl_bytes(rows), jsonl_bytes(labels)
    code_files = [p for p in CODE_FILES if require_trial_code or p not in
                  ('scripts/run_expanse_trial.py', 'scripts/summarize_expanse_trial.py')]
    code_hashes = {p: digest((root/p).read_bytes()) for p in code_files}
    generation = {'max_total_tokens': 8192, 'max_new_tokens': 256, 'max_input_tokens': 7936,
                  'do_sample': False, 'seed': 42, 'automatic_retries': 0, 'batch_size': 1,
                  'attention_implementation': 'sdpa', 'dtype': 'bfloat16'}
    quantization = {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
                    'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_use_double_quant': True}
    models = {
        'tiny_a_4bit': {'model_id': cfg['model_id'], 'revision': cfg['revision'], 'adapter': None},
        'tiny_d_4bit': {'model_id': cfg['model_id'], 'revision': cfg['revision'],
                       'adapter': {'path': adapter_path, 'weights_sha256': adapter_sha,
                                   'config_sha256': adapter_config_sha}},
        'expanse8b_4bit': {'model_id': expanse_model_id, 'revision': EXPANSE_REVISION, 'adapter': None},
    }
    arm_config_files = {'tiny_a_4bit': 'configs/expanse8b-v6-tiny-a.json',
                        'tiny_d_4bit': 'configs/expanse8b-v6-tiny-d.json',
                        'expanse8b_4bit': 'configs/expanse8b-v6-expanse8b.json'}
    for arm, model in models.items():
        name = arm_config_files[arm]
        arm_config = read_json(name)
        expected = {'label': arm, 'model_id': model['model_id'], 'revision': model['revision'],
                    **{k: generation[k] for k in ('dtype', 'attention_implementation', 'max_total_tokens',
                                                   'max_new_tokens', 'do_sample', 'seed')},
                    'quantization': quantization}
        require(arm_config == expected, f'Arm configuration differs from fixed protocol: {arm}')
        model.update(config_file=name, config_sha256=digest(source_bytes[name]))
    protocol = {
        'schema_version': 1, 'experiment': 'expanse8b-v6', 'frozen_at_utc': frozen_at,
        'phase': 'before_any_trial_model_tokenization_or_inference',
        'inputs_file': 'inputs.jsonl', 'inputs_sha256': digest(inputs_bytes),
        'evaluation_labels_file': 'evaluation-labels.jsonl', 'evaluation_labels_sha256': digest(labels_bytes),
        'input_fields': list(INPUT_FIELDS), 'selected_question_ids': [r['id'] for r in rows],
        'groups': {group: {'count': n, 'ids': [l['id'] for l in labels if l['group'] == group]}
                   for group, n in GROUP_COUNTS.items()},
        'counts': {'inputs': 59, 'baseline': 40, 'live': 19, 'original_real_papers': 40,
                   'synthetic_documents': 16, 'repeated_real_paper_live_inputs': 3,
                   'review40_original_answerable': 20, 'review40_original_unanswerable': 20},
        'selection_rule': 'All fixed review40 in q001..q040 order, then all 12 visible fixtures, all 4 already opened former-heldback fixtures, then all 3 previous smoke requests in their frozen order. No filtering by original labels, scores, old or new model outputs.',
        'models': models, 'generation': generation, 'quantization': quantization,
        'prompt_builders': {
            'baseline': {'function': 'run_baseline.messages_for', 'instruction_sha256': digest(INSTRUCTION.encode()),
                         'source_context_unchanged': True},
            'live': {'function': 'serve_live.live_messages', 'preparation': 'serve_live.prepare_document',
                     'mode': MODE, 'instruction_sha256': digest(LIVE_INSTRUCTION.encode()),
                     'examples_sha256': digest(json.dumps(LIVE_EXAMPLES, ensure_ascii=False, sort_keys=True).encode()),
                     'note': 'Existing fixed live examples remain part of the frozen prompt; per-input expectations/labels are excluded.'},
        },
        'logical_messages_sha256': {r['id']: message_hash(r) for r in rows},
        'historical_bf16_references': references,
        'source_files_sha256': {name: digest(data) for name, data in source_bytes.items()},
        'computation_files_sha256': code_hashes,
        'accounting_rules': {
            'preflight': 'Tokenize each complete chat-templated prompt independently with each pinned model tokenizer, truncation=False. Record actual input_tokens, max_input_tokens=7936 and preflight_fits for every one of 59 IDs before generation.',
            'over_budget': 'Do not truncate. Record skipped_over_budget or equivalent explicit failure for every over-budget ID; retain all 59 denominators.',
            'common_subset': 'Intersection of the three actual tokenizer preflight_fits sets only, not status==ok. Subsequent invalid outputs, OOM and runtime failures remain in this common subset and its denominator.',
            'runtime_failure': 'Retain one outcome record per selected ID, including all skips/errors; no automatic retry, replacement or dropping.',
            'raw_output': 'Keep raw text, strict parser result/errors, token counts, prompt hash, elapsed time and CUDA peak memory. Malformed/unknown outputs never become successful refusals.',
            'reporting': 'Report all59 execution counts plus review40 and live19 separately; score original review40 labels as proxies only. Live semantic assessment stays pending until a new source-based AI/human review, never keyword-match correctness.',
            'pairing': 'Report exact ID/order and model-specific preflight lengths. Same logical messages need not have the same token count or chat-template bytes across model families.',
            'historical': 'Historical A/D BF16 results may be displayed only as explicitly different-precision references. New three-arm 4-bit comparison is not a pure causal model-size experiment either.',
        },
        'model_download_performed_by_preparation': False, 'tokenization_performed_by_preparation': False,
        'inference_performed_by_preparation': False, 'prospective_holdout': False,
        'previously_used_development_material': True, 'independent_human_review': False,
        'human_review_status': 'pending', 'winner_rule': 'No winner or factuality improvement is declared from format-only or synthetic counts. Inspect paired semantic failures and resource feasibility descriptively.',
        'limitations': [
            'All 59 records reuse development materials; former held-back fixtures are already opened and are not a new holdout.',
            'The three real-paper live requests repeat papers/questions in review40 with a different prompt; 59 rows are not 59 independent papers.',
            'Original source labels, old AI judgments and synthetic expectations have different meanings; keep them separately identified and unchanged.',
            'NF4/BF16-compute is shared across the new arms, but model family, training, tokenizer and chat template differ; parameter count is not the sole changed factor.',
            'Historical A/D were unquantized BF16; cross-precision differences cannot be attributed solely to model size or the adapter.',
            'Longer inputs can fit different model tokenizers differently; report all fixed cases and the preflight intersection without dropping failed generations.',
        ],
    }
    for name, data in source_bytes.items():
        require((root/name).read_bytes() == data, f'Source changed while building: {name}')
    return {'inputs.jsonl': inputs_bytes, 'evaluation-labels.jsonl': labels_bytes,
            'protocol.json': json_bytes(protocol)}


def verify_materialized(root, output_dir):
    """Replay construction into memory and require every artifact to match bytes."""
    root, output_dir = Path(root), Path(output_dir)
    protocol = json.loads((output_dir/'protocol.json').read_bytes())
    for name, expected in protocol['source_files_sha256'].items():
        require(digest((root/name).read_bytes()) == expected, f'Source hash mismatch: {name}')
    for name, expected in protocol['computation_files_sha256'].items():
        require(digest((root/name).read_bytes()) == expected, f'Code hash mismatch: {name}')
    replay = build_payloads(root, frozen_at=protocol['frozen_at_utc'],
                            expanse_model_id=protocol['models']['expanse8b_4bit']['model_id'])
    for name, data in replay.items():
        require((output_dir/name).read_bytes() == data, f'Non-reproducible artifact: {name}')
    return {name: digest(data) for name, data in replay.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/experiments/expanse8b-v6')
    parser.add_argument('--expanse-model-id', default=EXPANSE_MODEL_ID)
    parser.add_argument('--frozen-at')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps({'status': 'verified', 'artifact_sha256': verify_materialized(ROOT, args.output_dir)}))
        return
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory; frozen experiment files are never overwritten')
    artifacts = build_payloads(ROOT, frozen_at=args.frozen_at or datetime.now(timezone.utc).isoformat(),
                               expanse_model_id=args.expanse_model_id)
    args.output_dir.mkdir(parents=True)
    for name, data in artifacts.items():
        with (args.output_dir/name).open('xb') as handle:
            handle.write(data)
    hashes = verify_materialized(ROOT, args.output_dir)
    print(json.dumps({'status': 'frozen_and_replayed', 'output_dir': str(args.output_dir),
                      'rows': 59, 'baseline': 40, 'live': 19, 'artifact_sha256': hashes}))


if __name__ == '__main__':
    main()
