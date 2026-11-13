"""Re-render exactly D's 128 train examples for the pinned Aya Expanse tokenizer.

This CPU-only preparation never loads model weights or reads evaluation labels.
It proves the original Tiny Aya rendering before using the same logical user
message and target JSON with the Expanse native chat template. Nothing is
truncated, filtered, resampled, corrected, or inferred from evaluation results.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

from paperproof_eval import parse_prediction
from run_baseline import ROOT, messages_for

MODEL_ID = 'CohereLabs/aya-expanse-8b'
MODEL_REVISION = '5062468bf9bc0c6035fd64e06274333ec127d980'
D_DATA = 'data/curated/followup-v2/d_15pct_abstention.jsonl'
D_DATA_SHA = 'b87644f64bdab7a179a059fc63a708fe9d84cae776a908455497163ba8bb744a'
D_MANIFEST = 'runs/train-d-followup-v2/manifest.json'
D_MANIFEST_SHA = '7c66783becfc958c215847040d4b600414d17396c219bfbbea20011b4e669cdb'
D_ADAPTER_SHA = '576eac6622ce5bc332912998e42bed51f5c24099888c543856ecca20c009d437'
TRAIN_SOURCE = 'data/processed/train_candidates.jsonl'
EVAL_INPUT = 'data/experiments/expanse8b-v6/inputs.jsonl'
EVAL_INPUT_SHA = 'eda231ab7c5da0a6d8a47da378f63758200eb1f2132930bf7f5b0eb699488e12'
TRAIN_CONFIG = 'configs/expanse8b-v7-training.json'
ROW_KEYS = {'id', 'paper_id', 'answerable', 'prompt', 'completion',
            'input_tokens', 'completion_tokens', 'total_tokens',
            'reference_index', 'human_review_status'}
MAX_LENGTH = 4096


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def jsonl_bytes(rows):
    return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()


def source_target(row):
    require(type(row['answerable']) is bool, 'Source answerable must be boolean')
    target = {'answerable': row['answerable'],
              'answer': row['reference_answers'][0].strip() if row['answerable'] else '',
              'evidence_ids': row['reference_evidence_ids'][0] if row['answerable'] else []}
    text = json.dumps(target, ensure_ascii=False)
    parsed, error, _ = parse_prediction(text, {p['id'] for p in row['paragraphs']})
    require(error is None and parsed == target, 'Original training target is invalid')
    return text


def render_pair(tokenizer, messages, target):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': target}],
                                          tokenize=False, add_generation_prompt=False)
    require(isinstance(prompt, str) and isinstance(full, str) and full.startswith(prompt),
            'Native chat template prefix differs; no boundary guessing is permitted')
    completion = full[len(prompt):]
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
    full_ids = tokenizer(full, add_special_tokens=False, truncation=False)['input_ids']
    require(full_ids[:len(prompt_ids)] == prompt_ids, 'Tokenizer merged across the prompt/completion boundary')
    require(len(full_ids) > len(prompt_ids), 'Empty supervised completion')
    return {'prompt': prompt, 'completion': completion, 'input_tokens': len(prompt_ids),
            'completion_tokens': len(full_ids) - len(prompt_ids), 'total_tokens': len(full_ids)}, full_ids


def rerender_rows(original_rows, sources, tiny_tokenizer, expanse_tokenizer, max_length=MAX_LENGTH):
    """Pure row transformation, with all selection decisions already fixed by D."""
    require(len({r['id'] for r in original_rows}) == len(original_rows), 'Repeated original IDs')
    rebuilt, accounting = [], []
    expanse_tokenizer.padding_side = 'right'
    expanse_tokenizer.add_bos_token = False
    for original in original_rows:
        require(set(original) == ROW_KEYS, 'Original D row fields changed')
        source = sources[original['id']]
        require(source['source_split'] == 'train', 'A non-training source is forbidden')
        require(source['paper_id'] == original['paper_id'] and source['answerable'] is original['answerable'],
                'Original source identity or answerability changed')
        require(original['reference_index'] == 0 and original['human_review_status'] == 'pending',
                'Original reference choice or human status changed')
        target = source_target(source)
        messages = messages_for(source)
        replay, _ = render_pair(tiny_tokenizer, messages, target)
        require(all(replay[field] == original[field] for field in replay),
                'Original D prompt/completion/token count replay differs: ' + original['id'])
        new, ids = render_pair(expanse_tokenizer, messages, target)
        default_ids = expanse_tokenizer(new['prompt'] + new['completion'], truncation=False)['input_ids']
        require(default_ids == ids, 'Default training tokenization inserts or changes tokens after disabling BOS')
        rebuilt.append({**original, **new})
        original_target, _ = json.JSONDecoder().raw_decode(original['completion'])
        new_target, _ = json.JSONDecoder().raw_decode(new['completion'])
        require(original_target == new_target == json.loads(target), 'Target JSON changed across native templates')
        source_text = {k: source[k] for k in ('id', 'paper_id', 'title', 'question', 'context', 'paragraphs')}
        accounting.append({
            'id': original['id'], 'paper_id': original['paper_id'], 'answerable': original['answerable'],
            'source_text_sha256': digest(json_bytes(source_text)),
            'logical_messages_sha256': digest(json_bytes(messages)),
            'target_json_sha256': digest(target.encode()), 'target_json': target,
            'original_prompt_sha256': digest(original['prompt'].encode()),
            'original_completion_sha256': digest(original['completion'].encode()),
            'expanse_prompt_sha256': digest(new['prompt'].encode()),
            'expanse_completion_sha256': digest(new['completion'].encode()),
            'original_input_tokens': original['input_tokens'],
            'original_completion_tokens': original['completion_tokens'],
            'original_total_tokens': original['total_tokens'],
            'input_tokens': new['input_tokens'], 'completion_tokens': new['completion_tokens'],
            'total_tokens': new['total_tokens'], 'max_sequence_tokens': max_length,
            'fits': new['total_tokens'] <= max_length, 'truncation': False,
            'original_D_replay_passed': True, 'same_logical_message_and_target': True,
            'default_training_tokenization_matches_explicit_ids': True,
            'expected_label_mask': {'prompt_tokens_masked': new['input_tokens'],
                                    'supervised_completion_tokens': new['completion_tokens']},
        })
    require([r['id'] for r in rebuilt] == [r['id'] for r in original_rows], 'Order or membership changed')
    return rebuilt, accounting


def tokenizer_metadata(tokenizer, model_id, revision):
    template = tokenizer.get_chat_template()
    require(isinstance(template, str), 'The selected native chat template must be text')
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    return {'model_id': model_id, 'revision': revision, 'class': type(tokenizer).__name__,
            'vocab_size': tokenizer.vocab_size, 'total_tokens': len(tokenizer),
            'chat_template_sha256': digest(template.encode()),
            'all_native_templates_sha256': digest(json_bytes(tokenizer.chat_template)),
            'native_template_names': sorted(tokenizer.chat_template) if isinstance(tokenizer.chat_template, dict) else None,
            'backend_tokenizer_sha256': digest(backend.to_str().encode()) if backend is not None else None,
            'special_tokens_map': tokenizer.special_tokens_map,
            'bos_token_id': tokenizer.bos_token_id, 'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'default_add_bos_token': getattr(tokenizer, 'add_bos_token', None),
            'default_add_eos_token': getattr(tokenizer, 'add_eos_token', None),
            'local_files_only': True, 'trust_remote_code': False}


def build_payloads(root, tiny_tokenizer, expanse_tokenizer, *, frozen_at):
    root = Path(root)
    files = [D_DATA, D_MANIFEST, TRAIN_SOURCE, EVAL_INPUT, TRAIN_CONFIG,
             'configs/baseline.json', 'configs/frameworks.lock.json',
             'scripts/curate_training.py', 'scripts/run_baseline.py', 'scripts/paperproof_eval.py']
    contents = {name: (root/name).read_bytes() for name in files}
    require(digest(contents[D_DATA]) == D_DATA_SHA, 'The fixed D data checksum changed')
    require(digest(contents[D_MANIFEST]) == D_MANIFEST_SHA, 'The fixed D training manifest changed')
    require(digest(contents[EVAL_INPUT]) == EVAL_INPUT_SHA, 'The fixed 59 evaluation inputs changed')
    old = [json.loads(line) for line in contents[D_DATA].decode().splitlines()]
    old_manifest = json.loads(contents[D_MANIFEST])
    require(old_manifest['training_complete'] is True and old_manifest['data_sha256'] == D_DATA_SHA
            and old_manifest['adapter_sha256'] == D_ADAPTER_SHA, 'D training lineage is not the frozen run')
    require(len(old) == 128 and sum(r['answerable'] for r in old) == 109, 'D must have 109 positive and 19 negative rows')
    require([r['id'] for r in old] == old_manifest['selected_ids'], 'D selected order changed')
    source_rows = [json.loads(line) for line in contents[TRAIN_SOURCE].decode().splitlines()]
    sources = {r['id']: r for r in source_rows}
    require(len(sources) == len(source_rows), 'Duplicate source IDs')
    evaluation = [json.loads(line) for line in contents[EVAL_INPUT].decode().splitlines()]
    require(len(evaluation) == 59 and len({r['id'] for r in evaluation}) == 59, 'Require all fixed evaluation inputs')
    train_papers = {r['paper_id'] for r in old}
    eval_papers = {r['paper_id'] for r in evaluation}
    require(not train_papers & eval_papers, 'Training papers overlap evaluation papers')
    cfg = json.loads(contents[TRAIN_CONFIG])
    require(cfg['base_model']['model_id'] == MODEL_ID and cfg['base_model']['revision'] == MODEL_REVISION,
            'Unexpected training model identity')
    require(cfg['training']['max_sequence_tokens'] == MAX_LENGTH, 'Training context budget changed')
    tiny_cfg = json.loads(contents['configs/baseline.json'])
    require(tiny_cfg == old_manifest['base_model'], 'Original Tiny Aya model configuration changed')
    metadata = {'tiny_source': tokenizer_metadata(tiny_tokenizer, tiny_cfg['model_id'], tiny_cfg['revision']),
                'expanse_native': tokenizer_metadata(expanse_tokenizer, MODEL_ID, MODEL_REVISION)}
    rows, accounting = rerender_rows(old, sources, tiny_tokenizer, expanse_tokenizer)
    overlong = [r for r in accounting if not r['fits']]
    train_bytes, accounting_bytes = jsonl_bytes(rows), jsonl_bytes(accounting)
    require(datetime.fromisoformat(frozen_at).tzinfo is not None, 'Freeze timestamp needs a timezone')
    protocol = {
        'schema_version': 1, 'experiment': 'expanse8b-v7-sft', 'frozen_at_utc': frozen_at,
        'phase': 'after_CPU_tokenizer_preparation_before_8B_training',
        'model': {'model_id': MODEL_ID, 'revision': MODEL_REVISION},
        'data_file': 'train.jsonl', 'data_sha256': digest(train_bytes),
        'selected_ids': [r['id'] for r in rows], 'train_paper_ids': sorted(train_papers),
        'counts': {'examples': 128, 'positive': 109, 'negative': 19},
        'max_sequence_tokens': MAX_LENGTH, 'truncation': False, 'resampling': False,
        'target_relabeling': False, 'overlong_count': len(overlong),
        'overlong_rows': [{'id': r['id'], 'total_tokens': r['total_tokens']} for r in overlong],
        'source_D': {'data_file': D_DATA, 'data_sha256': D_DATA_SHA,
                     'training_manifest_file': D_MANIFEST, 'training_manifest_sha256': D_MANIFEST_SHA,
                     'adapter_sha256': D_ADAPTER_SHA},
        'evaluation_input': {'path': EVAL_INPUT, 'sha256': EVAL_INPUT_SHA,
                             'question_count': 59, 'paper_overlap': 0,
                             'evaluation_paper_ids': sorted(eval_papers), 'labels_read': False},
        'training_config': {'path': TRAIN_CONFIG, 'sha256': digest(contents[TRAIN_CONFIG])},
        'training': cfg['training'], 'lora': cfg['lora'], 'quantization': cfg['quantization'],
        'source_files_sha256': {name: digest(data) for name, data in contents.items()},
        'preparation_code_sha256': digest((root/'scripts/prepare_expanse_sft.py').read_bytes()),
        'tokenizer_metadata': metadata,
        'token_summary': {'original_D_total': sum(r['total_tokens'] for r in old),
                          'original_D_supervised': sum(r['completion_tokens'] for r in old),
                          'expanse_total': sum(r['total_tokens'] for r in rows),
                          'expanse_supervised': sum(r['completion_tokens'] for r in rows),
                          'minimum_total': min(r['total_tokens'] for r in rows),
                          'maximum_total': max(r['total_tokens'] for r in rows)},
        'checks': {'all_128_original_D_strings_and_counts_replayed': True,
                   'all_logical_inputs_and_JSON_targets_unchanged': True,
                   'all_token_prefix_boundaries_verified': True,
                   'all_default_training_tokenization_verified_after_disabling_BOS': True,
                   'actual_trainer_label_mask_check_required_before_training': True,
                   'source_train_split_only': True, 'evaluation_labels_read': False,
                   'model_weights_loaded': False, 'GPU_operations_performed': False},
        'training_tokenizer_adjustment': 'padding_side=right; add_bos_token=False because native rendered prompts already contain BOS; default tokenization was checked against explicit IDs on all 128 rows.',
        'human_review_status': 'pending', 'independent_human_review': False,
        'limitations': ['The same original D labels and reference-index-zero targets are retained, including known quality limitations; E/F edits are not substituted.',
                       'Different native chat templates/tokenizers mean supervised structural tokens and token totals differ, although paper/question/target JSON and ordered examples are identical.',
                       'Paper disjointness prevents train/eval text overlap; these previously inspected evaluation cases are not a new holdout.',
                       'Expected prompt/completion label counts are preparation checks; the actual trainer collator must verify all masks before updates.'],
        'files': {'train.jsonl': digest(train_bytes), 'token-accounting.jsonl': digest(accounting_bytes)},
    }
    return {'train.jsonl': train_bytes, 'token-accounting.jsonl': accounting_bytes,
            'protocol.json': json_bytes(protocol)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/curated/expanse8b-v7')
    parser.add_argument('--preflight-report', type=Path, help='Write only a CPU token/preparation report; do not freeze data')
    parser.add_argument('--frozen-at')
    args = parser.parse_args()
    destination = args.preflight_report or args.output
    if destination.exists():
        raise FileExistsError('Existing artifacts are never overwritten: ' + str(destination))
    from transformers import AutoTokenizer
    tiny_cfg = json.loads((ROOT/'configs/baseline.json').read_text())
    tiny = AutoTokenizer.from_pretrained(tiny_cfg['model_id'], revision=tiny_cfg['revision'],
                                         local_files_only=True, trust_remote_code=False)
    expanse = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                            local_files_only=True, trust_remote_code=False)
    artifacts = build_payloads(ROOT, tiny, expanse,
        frozen_at=args.frozen_at or datetime.now(timezone.utc).isoformat())
    protocol = json.loads(artifacts['protocol.json'])
    result = {'status': 'CPU_preflight_passed' if not protocol['overlong_count'] else 'overlong_rows_block_freeze',
              'counts': protocol['counts'], 'token_summary': protocol['token_summary'],
              'overlong_rows': protocol['overlong_rows'], 'checks': protocol['checks'],
              'artifact_sha256': {name: digest(data) for name, data in artifacts.items()},
              'packages': {name: importlib.metadata.version(name) for name in ['transformers', 'tokenizers', 'huggingface-hub']}}
    if args.preflight_report:
        args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        with args.preflight_report.open('xb') as handle:
            handle.write(json_bytes(result))
    else:
        require(not protocol['overlong_count'], 'One or more complete 8B sequences exceed 4096; report them before deciding next steps')
        args.output.mkdir(parents=True)
        for name, data in artifacts.items():
            with (args.output/name).open('xb') as handle:
                handle.write(data)
        result.update(status='frozen', output=str(args.output))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
