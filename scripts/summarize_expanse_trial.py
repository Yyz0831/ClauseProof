"""CPU-only summary of the frozen three-arm Expanse development trial.

All selected questions remain in the primary denominators. The common-runnable
view is the tokenizer-preflight intersection, never an output-success subset.
Original review40 labels are lexical/reference proxies; live diagnostic semantic
judgments remain pending in a separate AI review. No keyword grading is used.
"""
from collections import Counter
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_run import declared_answerability
from paperproof_eval import parse_prediction, summarize
from run_baseline import ROOT, messages_for, save_json
from serve_live import live_messages


GENERATED_STATUSES = {'ok', 'invalid_output'}
GROUP_COUNTS = {'review40': 40, 'live_visible': 12,
                'live_opened_heldback': 4, 'live_real_paper': 3}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(rows, description):
    result = {row['id']: row for row in rows}
    require(len(result) == len(rows), f'Duplicate IDs in {description}')
    return result


def distribution(values):
    values = [value for value in values if value is not None]
    require(all(type(value) in {int, float} and math.isfinite(value) and value >= 0
                for value in values), 'Runtime measurements must be finite, nonnegative numbers')
    return {'measured_count': len(values),
            'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'maximum': max(values) if values else None,
            'sum': sum(values) if values else None}


def validate_cohort(inputs, labels, require_full_trial=True):
    indexed(inputs, 'inputs')
    indexed(labels, 'labels')
    require([r['id'] for r in inputs] == [r['id'] for r in labels],
            'Inputs and labels must have identical ordered IDs')
    for source, label in zip(inputs, labels):
        require(all(source[key] == label[key] for key in ['paper_id', 'prompt_mode']),
                f'Input/label identity mismatch: {source["id"]}')
        require(label['group'] in GROUP_COUNTS, 'Unknown trial group')
        if label['group'] == 'review40':
            require(type(label.get('answerable')) is bool, 'review40 requires its original boolean label')
            require(isinstance(label.get('reference_answers'), list)
                    and isinstance(label.get('reference_evidence_ids'), list),
                    'review40 requires original reference lists')
            if label['answerable']:
                require(bool(label['reference_answers']) and bool(label['reference_evidence_ids']),
                        'Positive references must not be empty')
        else:
            require(not {'answerable', 'reference_answers', 'reference_evidence_ids'} & set(label),
                    'Live diagnostics must not be silently promoted to benchmark gold labels')
            require('expected' in label, 'Live diagnostic expectation metadata is missing')
    if require_full_trial:
        require(dict(Counter(row['group'] for row in labels)) == GROUP_COUNTS,
                'Trial must contain the fixed 40+12+4+3 cohort')


def reparse_records(inputs, records):
    source = indexed(inputs, 'inputs')
    predictions = indexed(records, 'predictions')
    require(set(source) == set(predictions), 'Every selected ID must have exactly one record, including skips')
    result = {}
    for question_id, row in source.items():
        original = predictions[question_id]
        tokens, limit, fits = (original[key] for key in
                              ['input_tokens', 'max_input_tokens', 'preflight_fits'])
        require(type(limit) is int and limit > 0 and type(fits) is bool,
                'Actual input budgets and boolean preflight decisions are required')
        if tokens is None:
            require(original['status'] in {'preflight_error', 'setup_failed'} and not fits,
                    'Only a preflight error may have no measured input length')
        else:
            require(type(tokens) is int and tokens > 0 and fits == (tokens <= limit),
                    'preflight_fits must agree with the actual tokenizer count')
        require(isinstance(original.get('status'), str) and original['status'], 'Missing execution status')
        raw = original.get('raw_output', '')
        require(raw is None or isinstance(raw, str), 'Raw output must be text or an explicit missing value')
        current = dict(original)
        current['recorded_status'] = original['status']
        current['prediction'] = None
        current['raw_declared_answerable'] = declared_answerability(raw) if raw is not None else None
        if original['status'] in GENERATED_STATUSES:
            require(fits, 'Over-budget input cannot have a generated status')
            require(isinstance(raw, str), 'Generated statuses require the unaltered raw text')
            pred, error, wrapper = parse_prediction(raw, {p['id'] for p in row['paragraphs']})
            # Cached prediction fields are never trusted as scoring inputs.
            current.update(prediction=pred, parse_error=error, output_wrapper=wrapper,
                           status='invalid_output' if error else 'ok')
        elif not fits:
            require(not raw, 'Skipped over-budget input must not contain an output')
        result[question_id] = current
    return result


def execution_summary(ids, records):
    selected = [records[i] for i in ids]
    valid = [r for r in selected if r['status'] == 'ok' and r['prediction'] is not None]
    generated = [r for r in selected if r['status'] in GENERATED_STATUSES]
    rate = lambda count: count / len(selected) if selected else None
    raw = Counter('true' if r['raw_declared_answerable'] is True else
                  'false' if r['raw_declared_answerable'] is False else 'unknown' for r in selected)
    answers = sum(r['prediction']['answerable'] for r in valid)
    refusals = len(valid) - answers
    return {'questions': len(selected), 'question_ids': ids,
            'status_counts': dict(Counter(r['status'] for r in selected)),
            'recorded_status_counts': dict(Counter(r['recorded_status'] for r in selected)),
            'preflight_runnable_count': sum(r['preflight_fits'] for r in selected),
            'over_token_budget_count': sum(r['input_tokens'] is not None and r['input_tokens'] > r['max_input_tokens'] for r in selected),
            'preflight_error_count': sum(r['status'] == 'preflight_error' for r in selected),
            'unmeasured_input_count': sum(r['input_tokens'] is None for r in selected),
            'generated_output_count': len(generated),
            'generated_output_coverage': rate(len(generated)),
            'valid_output_count': len(valid), 'valid_output_rate': rate(len(valid)),
            'strict_answer_count': answers, 'strict_answer_coverage': rate(answers),
            'strict_legal_refusal_count': refusals,
            'invalid_output_count': sum(r['status'] == 'invalid_output' for r in selected),
            'failed_or_skipped_count': len(selected) - len(generated),
            'raw_declared_answerability': {key: raw[key] for key in ['true', 'false', 'unknown']},
            'raw_unknown_with_generated_output_count': sum(r['raw_declared_answerable'] is None for r in generated),
            'generation_seconds': distribution([r.get('generation_seconds') for r in generated]),
            'reported_attempt_wall_seconds': distribution([r.get('attempt_wall_seconds') for r in selected]),
            'gpu_generation_ms': distribution([r.get('gpu_generation_ms') for r in generated]),
            'reported_peak_allocated_bytes': distribution([r.get('peak_allocated_bytes') for r in selected]),
            'reported_peak_reserved_bytes': distribution([r.get('peak_reserved_bytes') for r in selected]),
            'global_device_peak_used_bytes_sampled': distribution([(r.get('device_memory') or {}).get('peak_device_used_bytes_sampled') for r in selected]),
            'reported_peak_allocated_gib': distribution([r.get('peak_allocated_gib') for r in selected]),
            'reported_peak_reserved_gib': distribution([r.get('peak_reserved_gib') for r in selected]),
            'generated_tokens': distribution([r.get('generated_tokens') for r in generated]),
            'input_tokens': distribution([r['input_tokens'] for r in selected]),
            'hit_token_limit_count': sum(r.get('hit_token_limit') is True for r in generated),
            'runtime_scope': 'Generation measurements include valid and invalid generated outputs; load/preflight are excluded. Memory summarizes available measurements; global device usage includes other processes and its sampled maximum is not an exact process peak.'}


def summarize_view(inputs, labels, models, ids):
    source = indexed(inputs, 'inputs')
    label_index = indexed(labels, 'labels')
    review_ids = [i for i in ids if label_index[i]['group'] == 'review40']
    live_ids = [i for i in ids if label_index[i]['group'] != 'review40']
    # Only benchmark rows are passed to the frozen lexical/reference scorer.
    gold = [{**label_index[i], 'paragraphs': source[i]['paragraphs']} for i in review_ids]
    arms = {}
    for arm, records in models.items():
        metrics, _ = summarize(gold, [records[i] for i in review_ids])
        strata = {}
        for name, answerable in [('positive', True), ('negative', False)]:
            selected = [i for i in review_ids if label_index[i]['answerable'] is answerable]
            strata[name] = execution_summary(selected, records)
        arms[arm] = {'all_selected': execution_summary(ids, records),
                     'review40': {'strict_reference_metrics': metrics, 'by_original_label': strata},
                     'live19': {'execution': execution_summary(live_ids, records),
                                'semantic_ai_review': {'status': 'pending', 'cases': len(live_ids),
                                    'ratings': None, 'automatically_graded': False,
                                    'independent_human_review': False, 'human_review_status': 'pending'}},
                     'groups': {group: execution_summary([i for i in ids if label_index[i]['group'] == group], records)
                                for group in GROUP_COUNTS}}
    return {'questions': len(ids), 'question_ids': ids,
            'group_counts': dict(Counter(label_index[i]['group'] for i in ids)), 'arms': arms}


def summarize_trial(inputs, labels, records_by_arm, require_full_trial=True):
    validate_cohort(inputs, labels, require_full_trial)
    require(len(records_by_arm) == 3, 'Exactly the three newly run 4-bit arms are required')
    models = {arm: reparse_records(inputs, records) for arm, records in records_by_arm.items()}
    ids = [r['id'] for r in inputs]
    common = [i for i in ids if all(records[i]['preflight_fits'] for records in models.values())]
    return {'all_selected': summarize_view(inputs, labels, models, ids),
            'common_runnable': {**summarize_view(inputs, labels, models, common),
                'definition': 'Intersection of actual-tokenizer preflight_fits for all three new 4-bit arms; output format and generation success do not filter this subset.'},
            'metrics_reparsed_from_raw_outputs': True,
            'semantic_ai_review_status': 'pending', 'human_review_status': 'pending',
            'limitations': [
                'Fixed, previously inspected development questions; not a prospective holdout or official benchmark.',
                'All selected questions remain in primary denominators; skipped/failed/invalid outputs receive zero benchmark F1.',
                'Raw true/false are declarations, unknown is separate, and invalid outputs are never successful refusals.',
                'Original review40 annotations are unchanged reference proxies, not semantic truth or prior AI ratings.',
                'The 19 live diagnostics have no automatic semantic score; no keyword matching is used.',
                'Token counts and rendered chat templates may differ across model families; source/messages must remain fixed.',
                'Historical A/D BF16 outputs are contextual references with a quantization difference; they are excluded from this three-arm common subset.',
                'Lexical F1 and citation ID overlap do not establish answer or citation entailment.',
            ]}


def hash_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def message_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def verified_file(path, expected):
    require(hash_file(path) == expected, f'File checksum mismatch: {path}')
    return expected


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def repo_file(root, name):
    path = (root/name).resolve()
    require(path.is_relative_to(root.resolve()), 'Frozen source must be inside the repository')
    return path


def verify_protocol(root, path):
    root, path = Path(root), Path(path)
    protocol = load_json(path)
    require(protocol['experiment'] == 'expanse8b-v6' and protocol['schema_version'] == 1,
            'Unsupported experiment protocol')
    require(set(protocol['models']) == {'tiny_a_4bit', 'tiny_d_4bit', 'expanse8b_4bit'},
            'Protocol must identify exactly the three new 4-bit arms')
    verified = {}
    for category in ['source_files_sha256', 'computation_files_sha256']:
        for name, expected in protocol[category].items():
            # Repository sources are never resolved relative to a temporary run.
            verified[name] = verified_file(repo_file(root, name), expected)
    inputs_path = path.parent/protocol['inputs_file']
    labels_path = path.parent/protocol['evaluation_labels_file']
    verified[str(inputs_path)] = verified_file(inputs_path, protocol['inputs_sha256'])
    verified[str(labels_path)] = verified_file(labels_path, protocol['evaluation_labels_sha256'])
    inputs, labels = load_jsonl(inputs_path), load_jsonl(labels_path)
    validate_cohort(inputs, labels)
    require([row['id'] for row in inputs] == protocol['selected_question_ids'], 'Frozen selected ID/order mismatch')
    require([r['prompt_mode'] for r in inputs] == ['baseline']*40 + ['live']*19, 'Frozen prompt-mode order changed')
    for group, count in GROUP_COUNTS.items():
        expected = {'count': count, 'ids': [r['id'] for r in labels if r['group'] == group]}
        require(protocol['groups'][group] == expected, 'Frozen group membership/order mismatch')
    for row in inputs:
        require(set(row) == set(protocol['input_fields']), 'Inference input schema differs from protocol')
        messages = messages_for(row) if row['prompt_mode'] == 'baseline' else live_messages(row)
        require(sha(message_bytes(messages)) == protocol['logical_messages_sha256'][row['id']],
                f'Frozen logical messages differ: {row["id"]}')
    return protocol, inputs, labels, verified


def verify_inventory(directory, inventory):
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    require(actual == set(inventory), f'Saved artifact inventory differs: {directory}')
    for name, item in inventory.items():
        path = directory/name
        require(path.stat().st_size == item['bytes'], f'Saved artifact size differs: {path}')
        verified_file(path, item['sha256'])


def load_verified_arm(run, arm, protocol, inputs, protocol_sha256):
    run = Path(run)
    manifest = load_json(run/'manifest.json')
    failed = manifest['status'] == 'failed'
    require(manifest['status'] in {'completed', 'failed'} and manifest['prepare_only'] is False,
            'Preparation-only or running arms cannot be summarized as final outcomes')
    require(manifest.get('outcome_records_complete') is True,
            f'{arm} does not have all outcome records; comparison is unavailable')
    binding = manifest.get('protocol') or {}
    require(binding.get('sha256') == protocol_sha256 and binding.get('arm') == arm
            and binding.get('logical_messages_verified') == len(inputs)
            and binding.get('evaluation_labels_read') is False,
            'Run is not bound to the exact frozen source-only protocol')
    verified_file(run/'protocol.json', protocol_sha256)
    if not failed:
        for key in ['model_loaded', 'preflight_complete', 'inference_complete']:
            require(manifest.get(key) is True, f'{arm} is incomplete: {key}')
    require((run/'predictions.jsonl').is_file(), f'{arm} has no complete outcome records; comparison is unavailable')
    require(datetime.fromisoformat(manifest['started_at_utc']) >= datetime.fromisoformat(protocol['frozen_at_utc']),
            'Trial execution predates the frozen protocol')
    require(manifest['selected_question_ids'] == protocol['selected_question_ids'], 'Run selected IDs/order differs')
    require(manifest['input_sha256'] == protocol['inputs_sha256'], 'Run input hash differs from protocol')
    require(manifest['questions'] == len(inputs), 'Run question count differs')
    verified_file(run/'inputs.jsonl', protocol['inputs_sha256'])
    cfg, spec = manifest['config'], protocol['models'][arm]
    require(cfg['label'] == arm and all(cfg[k] == spec[k] for k in ['model_id', 'revision']),
            'Run model/label/revision differs from protocol')
    for key in ['max_total_tokens', 'max_new_tokens', 'do_sample', 'seed', 'attention_implementation']:
        require(type(cfg[key]) is type(protocol['generation'][key]) and cfg[key] == protocol['generation'][key],
                f'Run generation setting differs: {key}')
    expected_quantization = protocol['quantization']
    require(cfg['quantization'] == manifest['quantization'] == expected_quantization,
            'Quantization differs from frozen shared settings')
    require(cfg['dtype'] == protocol['generation']['dtype'], 'Compute dtype differs')
    require(manifest['config_sha256'] == sha(canonical(cfg)), 'Effective config digest differs')
    require(load_json(run/'effective-config.json') == cfg, 'Saved effective config differs')
    verified_file(run/'requested-config.json', manifest['config_source_sha256'])
    require(manifest['config_source_sha256'] == spec['config_sha256'], 'Run used an unfrozen configuration file')
    for key, expected in [('batch_size', 1), ('automatic_retries', 0), ('offload', False), ('truncation', False)]:
        require(type(manifest[key]) is type(expected) and manifest[key] == expected, f'Run contract differs: {key}')
    budget = manifest.get('max_input_tokens', protocol['generation']['max_input_tokens'])
    require(budget == protocol['generation']['max_input_tokens'], 'Run input budget differs')
    if 'max_new_tokens' in manifest:
        require(manifest['max_new_tokens'] == protocol['generation']['max_new_tokens'], 'Run output budget differs')
    if 'generation_kwargs' in manifest:
        require(manifest['generation_kwargs']['max_new_tokens'] == protocol['generation']['max_new_tokens']
                and manifest['generation_kwargs']['do_sample'] is False, 'Effective generation options differ')
    if spec['adapter'] is None:
        require(manifest['adapter'] is None, 'Unexpected adapter in base arm')
    else:
        require(all(manifest['adapter'][key] == spec['adapter'][key] for key in ['weights_sha256', 'config_sha256']),
                'D adapter provenance differs')
    require(set(manifest['code_sha256']) == {'scripts/run_expanse_trial.py', 'scripts/run_baseline.py',
            'scripts/serve_live.py', 'scripts/paperproof_eval.py'}, 'Run computation dependency set differs')
    for name, expected in manifest['code_sha256'].items():
        require(protocol['computation_files_sha256'][name] == expected, 'Run computation differs from frozen protocol')
        verified_file(run/'code'/Path(name).name, expected)
    if 'tokenizer' in manifest:
        verify_inventory(run/'tokenizer', manifest['tokenizer']['files'])
    if 'model_config_files' in manifest:
        verify_inventory(run/'model-config', manifest['model_config_files'])
    if not failed or 'loaded_model' in manifest:
        loaded = manifest['loaded_model']
        require(loaded['is_loaded_in_4bit'] is True and loaded['linear4bit_module_count'] > 0,
                'Loaded model has no verified 4-bit modules')
        require(set(loaded['parameter_device_counts']) == {'cuda:0'}, 'Loaded model used CPU/disk offload')
        require(all(m['quant_type'] == 'nf4' and m['compute_dtype'] == 'torch.bfloat16' and m['nested_quantization']
                    for m in loaded['linear4bit_modules']), 'Loaded model quantization audit differs')
    for name, key in [('token_accounting.json', 'token_accounting_sha256'), ('prompts.jsonl', 'prompts_sha256'),
                      ('predictions.jsonl', 'predictions_sha256'), ('run-summary.json', 'summary_sha256')]:
        verified_file(run/name, manifest[key])
    accounting = load_json(run/'token_accounting.json')
    prompts, records = load_jsonl(run/'prompts.jsonl'), load_jsonl(run/'predictions.jsonl')
    ids = protocol['selected_question_ids']
    for name, values in [('accounting', accounting), ('prompts', prompts), ('predictions', records)]:
        indexed(values, name)
        require([r['id'] for r in values] == ids, f'{name} must cover every frozen ID in order')
    for source, token, prompt, record in zip(inputs, accounting, prompts, records):
        require(token['truncation'] is False, 'Prompt was truncated')
        require(token['source_sha256'] == sha(canonical(source['paragraphs'])), 'Source paragraph hash differs')
        require(token['max_input_tokens'] == budget, 'Per-record token budget differs')
        require(all(token[k] == source[k] for k in ['id', 'paper_id', 'prompt_mode']), 'Preflight source identity differs')
        if token['messages_sha256'] is not None:
            require(token['messages_sha256'] == protocol['logical_messages_sha256'][source['id']]
                    == sha(message_bytes(prompt['messages'])), 'Logical messages differ')
        if token['prompt_sha256'] is not None:
            require(isinstance(prompt['rendered_prompt'], str)
                    and sha(prompt['rendered_prompt'].encode()) == token['prompt_sha256'], 'Rendered prompt hash differs')
        if token['prepared']:
            require(type(token['input_tokens']) is int and token['input_tokens'] > 0
                    and token['fits'] == (token['input_tokens'] <= token['max_input_tokens']), 'Preflight count/fit mismatch')
        else:
            require(token['fits'] is False and record['status'] in {'preflight_error', 'setup_failed'}, 'Preflight failure was dropped')
        for key in ['id', 'paper_id', 'prompt_mode', 'source_sha256', 'messages_sha256',
                    'prompt_sha256', 'input_tokens', 'max_input_tokens']:
            require(record[key] == token[key], f'Prediction/preflight identity differs: {key}')
        require(record['preflight_fits'] is token['fits'], 'Prediction/preflight fit decision differs')
        require(record['automatic_retries'] == 0 and record['generation_attempts'] in [0, 1], 'Retries detected')
        require(record['status'] in GENERATED_STATUSES | {'preflight_error', 'skipped_over_token_budget',
                'cuda_out_of_memory', 'generation_error', 'setup_failed'}, 'Unknown runtime outcome')
        if record['status'] == 'setup_failed':
            require(failed, 'An unexecuted setup failure cannot belong to a completed arm')
        if token['prepared'] and not token['fits']:
            require(record['status'] == 'skipped_over_token_budget', 'Over-budget case must be retained as a skip')
        if record['status'] in {'preflight_error', 'skipped_over_token_budget', 'setup_failed'}:
            require(record['generation_attempts'] == 0 and record['raw_output'] is None, 'Skipped input was generated')
        if record['status'] in GENERATED_STATUSES:
            require(record['generation_attempts'] == 1, 'Generated output lacks a generation attempt')
            parsed, error, wrapper = parse_prediction(record['raw_output'], {p['id'] for p in source['paragraphs']})
            require(record['status'] == ('invalid_output' if error else 'ok')
                    and record['prediction'] == parsed and record['parse_error'] == error
                    and record['output_wrapper'] == wrapper, 'Stored parse result differs from raw output')
    counts = dict(Counter(r['status'] for r in records))
    require(manifest['status_counts'] == counts == load_json(run/'run-summary.json')['status_counts'],
            'Cached execution counts differ from raw records')
    return records, manifest


def historical_references(root, protocol, inputs, labels):
    selected_labels = [r for r in labels if r['group'] == 'review40']
    ids = [r['id'] for r in selected_labels]
    selected_inputs = [r for r in inputs if r['id'] in set(ids)]
    reports = {}
    for arm, spec in protocol['historical_bf16_references'].items():
        run = repo_file(root, spec['run_dir'])
        verified_file(run/'manifest.json', spec['manifest_sha256'])
        verified_file(run/'predictions.jsonl', spec['predictions_sha256'])
        manifest = load_json(run/'manifest.json')
        require(manifest.get('quantization') is None and manifest['config']['dtype'] == 'bfloat16',
                'Historical reference is not unquantized BF16')
        require(manifest['input_sha256'] == spec['source_input_sha256'], 'Historical source input differs')
        all_records = indexed(load_jsonl(run/'predictions.jsonl'), 'historical predictions')
        selected = []
        for i in ids:
            record = dict(all_records[i])
            record['max_input_tokens'] = manifest['max_input_tokens']
            record['preflight_fits'] = record['input_tokens'] <= record['max_input_tokens']
            selected.append(record)
        records = reparse_records(selected_inputs, selected)
        metrics, _ = summarize(selected_labels, [records[i] for i in ids])
        reports[arm] = {'role': spec['role'], 'included_in_new_three_arm_common_subset': False,
                       'precision': 'unquantized bfloat16', 'manifest_sha256': spec['manifest_sha256'],
                       'predictions_sha256': spec['predictions_sha256'],
                       'review40_execution': execution_summary(ids, records),
                       'review40_strict_reference_metrics': metrics}
    return reports


def build_report(root, protocol_path, run_root):
    root, protocol_path, run_root = Path(root), Path(protocol_path), Path(run_root)
    protocol, inputs, labels, verified = verify_protocol(root, protocol_path)
    models, manifests = {}, {}
    for arm in protocol['models']:
        models[arm], manifests[arm] = load_verified_arm(run_root/arm, arm, protocol, inputs, hash_file(protocol_path))
    # Runtime/package identity is required; model-specific tokenizer/config are not equalized.
    environment = {}
    for field in ['packages', 'gpu', 'cuda_runtime', 'batch_size', 'quantization', 'offload']:
        reported = [m[field] for m in manifests.values() if field in m]
        require(all(value == reported[0] for value in reported), f'Unmatched reported three-arm environment: {field}')
        environment[field] = {'reported_arm_count': len(reported), 'all_three_reported': len(reported) == 3}
    result = summarize_trial(inputs, labels, models)
    result.update(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=hash_file(protocol_path), verified_files=verified,
                  summary_script_sha256=hash_file(Path(__file__)),
                  historical_bf16_references=historical_references(root, protocol, inputs, labels),
                  official_test_used=False, prospective_holdout=False, winner_declared=False,
                  arm_execution_status={arm: m['status'] for arm, m in manifests.items()},
                  all_arms_inference_complete=all(m.get('inference_complete') is True for m in manifests.values()),
                  environment_alignment=environment,
                  per_arm_provenance={arm: {'manifest_sha256': hash_file(run_root/arm/'manifest.json'),
                      'predictions_sha256': m['predictions_sha256'], 'config': m['config'],
                      'token_accounting_sha256': m['token_accounting_sha256'], 'prompts_sha256': m['prompts_sha256'],
                      'model_load_seconds': m.get('model_load_seconds'),
                      'load_peak_allocated_bytes': m.get('load_peak_allocated_bytes'),
                      'load_peak_reserved_bytes': m.get('load_peak_reserved_bytes'),
                      'loaded_model': m.get('loaded_model'), 'packages': m.get('packages'), 'gpu': m.get('gpu'),
                      'resolved_config_commit': m.get('resolved_config_commit'),
                      'failed_stage': m.get('failed_stage'), 'error_type': m.get('error_type'), 'error': m.get('error')}
                      for arm, m in manifests.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'data/experiments/expanse8b-v6/protocol.json')
    parser.add_argument('--run-root', type=Path, default=ROOT/'runs/expanse8b-v6')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/expanse8b-v6-summary.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path; previous reports are not overwritten')
    report = build_report(ROOT, args.protocol, args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)
    print(json.dumps({'output': str(args.output), 'questions': report['all_selected']['questions'],
                      'common_runnable': report['common_runnable']['questions'],
                      'semantic_ai_review_status': report['semantic_ai_review_status']}))


if __name__ == '__main__':
    main()
