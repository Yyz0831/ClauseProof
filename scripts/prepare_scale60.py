"""Freeze a nested 60%-of-official-train cohort for Tiny Aya and Aya Expanse.

CPU-only preparation. The official development/test membership is not changed;
test contents and evaluation labels are never read. Full paper text and original
targets are retained. Both native tokenizers must fit each complete sequence.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import tarfile

from prepare_data import SHA256 as SOURCE_ARCHIVE_SHA, convert, paragraphs_for, norm, answer_text
from prepare_expanse_sft import (
    D_DATA, D_DATA_SHA, D_MANIFEST, D_MANIFEST_SHA, EVAL_INPUT, EVAL_INPUT_SHA,
    MODEL_ID, MODEL_REVISION, render_pair, source_target, tokenizer_metadata,
)
from run_baseline import ROOT, messages_for

ARMS = ('tiny_d60_4bit', 'expanse8b_60_4bit')
OUTPUT = 'data/curated/scale60-v8'
MAX_LENGTH = 8192
SEED = 42
FRACTION = 0.60
EXPECTED_SOURCE_QUESTIONS = 2593
EXPECTED_ELIGIBLE = 1722


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def jsonl(rows):
    return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()


def read_training_member(path):
    path = Path(path)
    require(digest(path.read_bytes()) == SOURCE_ARCHIVE_SHA, 'Raw archive checksum changed')
    with tarfile.open(path, 'r:gz') as archive:
        return json.load(archive.extractfile('qasper-train-v0.3.json'))


def question_source_index(papers):
    """Mirror the existing convert() eligibility order without a char limit.

    The returned first-valid index is into original answer annotations, while
    reference_index=0 remains the first reference retained by convert().
    """
    result = {}
    for pid, paper in sorted(papers.items()):
        paras = paragraphs_for(paper)
        lookup = {norm(p['text']) for p in paras}
        for question in paper['qas']:
            ident = pid + ':' + question['question_id']
            annotations = [a['answer'] for a in question.get('answers', [])]
            reason, valid_index = None, None
            if not annotations:
                reason = 'no_annotations'
            elif len({a['unanswerable'] for a in annotations}) != 1:
                reason = 'answerability_disagreement'
            elif annotations[0]['unanswerable']:
                valid_index = 0
            elif any(e.strip().startswith('FLOAT SELECTED') for a in annotations for e in a.get('evidence', [])):
                reason = 'annotated_figure_or_table_evidence'
            elif len({a['yes_no'] for a in annotations if a.get('yes_no') is not None}) > 1:
                reason = 'yes_no_disagreement'
            else:
                for index, a in enumerate(annotations):
                    evidence = a.get('evidence', [])
                    if answer_text(a) and evidence and all(norm(e) in lookup for e in evidence):
                        valid_index = index
                        break
                if valid_index is None:
                    reason = 'no_reference_with_matching_text_evidence'
            original = question['answers'][valid_index] if valid_index is not None else None
            require(ident not in result, 'Duplicate source question ID')
            result[ident] = {
                'id': ident, 'paper_id': pid, 'quality_rejection': reason,
                'original_annotation_index': valid_index,
                'original_annotation_id': None if original is None else original.get('annotation_id'),
                'original_question_sha256': digest(encoded(question)),
                'original_annotation_sha256': None if original is None else digest(encoded(original)),
            }
    return result


def choose_nested(eligible_ids, original_ids, target_size, seed=SEED):
    require(len(eligible_ids) == len(set(eligible_ids)), 'Duplicate eligible IDs')
    require(len(original_ids) == len(set(original_ids)), 'Duplicate original IDs')
    require(set(original_ids) <= set(eligible_ids), 'Original D rows no longer fit the joint cohort')
    require(len(original_ids) <= target_size <= len(eligible_ids), 'Requested cohort cannot be filled')
    # Sorted complement makes membership independent of input dictionary order.
    remainder = sorted(set(eligible_ids) - set(original_ids))
    additional = random.Random(seed).sample(remainder, target_size - len(original_ids))
    chosen = list(original_ids) + additional
    random.Random(seed).shuffle(chosen)
    require(len(chosen) == target_size and len(set(chosen)) == target_size, 'Invalid sampled cohort')
    return chosen, additional


def make_logical(source, annotation_info):
    return {
        'id': source['id'], 'paper_id': source['paper_id'],
        'source_split': source['source_split'], 'question_id': source['question_id'],
        'title': source['title'], 'question': source['question'],
        'paragraphs': source['paragraphs'], 'answerable': source['answerable'],
        'messages': messages_for(source), 'target_json': source_target(source),
        'reference_index': 0, 'human_review_status': 'pending',
        'context_mode': source['context_mode'],
        'source_context_sha256': digest(source['context'].encode()),
        **{k: annotation_info[k] for k in ('original_annotation_index', 'original_annotation_id',
             'original_question_sha256', 'original_annotation_sha256')},
    }


def make_native(source, rendered):
    return {
        'id': source['id'], 'paper_id': source['paper_id'],
        'answerable': source['answerable'], **rendered,
        'reference_index': 0, 'human_review_status': 'pending',
    }


def prove_original_D(originals, source_by_id, native_by_arm):
    require(len(originals) == 128 and sum(r['answerable'] for r in originals) == 109,
            'The original D cohort is not the frozen 109+19 cohort')
    for old in originals:
        ident = old['id']
        source = source_by_id[ident]
        require(source['source_split'] == 'train' and source['paper_id'] == old['paper_id'], 'D source identity changed')
        require(old['reference_index'] == 0 and old['human_review_status'] == 'pending', 'D label provenance changed')
        expected = make_native(source, native_by_arm[ARMS[0]][ident])
        require(expected == old, 'Original D native text, counts or target changed: ' + ident)
        for arm in ARMS:
            target, _ = json.JSONDecoder().raw_decode(native_by_arm[arm][ident]['completion'])
            require(target == json.loads(source_target(source)), 'Original D target changed across models')


def build(root, tokenizers, *, frozen_at):
    from train_scale60 import build_config
    root = Path(root)
    require(root.resolve() == ROOT.resolve(), 'Run preparation from the real project copy')
    source_files = [
        'data/raw/qasper-train-dev-v0.3.tgz', D_DATA, D_MANIFEST, EVAL_INPUT,
        'configs/baseline.json', 'configs/frameworks.lock.json',
        'scripts/prepare_data.py', 'scripts/prepare_expanse_sft.py',
        'scripts/run_baseline.py', 'scripts/paperproof_eval.py',
    ]
    source_hashes = {name: digest((root/name).read_bytes()) for name in source_files}
    require(source_hashes[D_DATA] == D_DATA_SHA and source_hashes[D_MANIFEST] == D_MANIFEST_SHA, 'Original D provenance changed')
    require(source_hashes[EVAL_INPUT] == EVAL_INPUT_SHA, 'Existing development inputs changed')
    archive_path = root/'data/raw/qasper-train-dev-v0.3.tgz'
    require(source_hashes['data/raw/qasper-train-dev-v0.3.tgz'] == SOURCE_ARCHIVE_SHA, 'Raw source changed')
    # Parse only the official train member. The tar also contains dev JSON, but
    # its answers are not even deserialized by this training-data preparation.
    train = read_training_member(archive_path)
    source_index = question_source_index(train)
    rows, quality_stats = convert(train, 'train', 10**9)
    require(len(source_index) == quality_stats['source_questions'] == EXPECTED_SOURCE_QUESTIONS,
            'Official train question count differs')
    sources = {r['id']: r for r in rows}
    require(set(sources) == {i for i, x in source_index.items() if x['quality_rejection'] is None},
            'Independent eligibility accounting differs from established quality filters')
    require(dict(Counter(x['quality_rejection'] for x in source_index.values() if x['quality_rejection'])) == quality_stats['rejected'],
            'Rejection counts differ from existing quality rules')
    originals = [json.loads(line) for line in (root/D_DATA).read_text().splitlines()]
    old_manifest = json.loads((root/D_MANIFEST).read_text())
    original_ids = [r['id'] for r in originals]
    require(old_manifest['selected_ids'] == original_ids and old_manifest['training_complete'], 'Original D manifest membership differs')
    evaluation = [json.loads(line) for line in (root/EVAL_INPUT).read_text().splitlines()]
    require(len(evaluation) == 59, 'Existing evaluation cohort changed')
    eval_papers = {r['paper_id'] for r in evaluation}
    require(not set(train) & eval_papers, 'Training and evaluation paper IDs overlap')
    cfg_tiny = json.loads((root/'configs/baseline.json').read_text())
    models = {ARMS[0]: cfg_tiny, ARMS[1]: {'model_id': MODEL_ID, 'revision': MODEL_REVISION}}
    native_by_arm, accounts_by_arm, metadata = {}, {}, {}
    for arm in ARMS:
        tok = tokenizers[arm]
        tok.padding_side = 'right'
        tok.add_bos_token = False
        model = models[arm]
        metadata[arm] = tokenizer_metadata(tok, model['model_id'], model['revision'])
        native, accounts = {}, {}
        for i, source in enumerate(rows, 1):
            rendered, ids = render_pair(tok, messages_for(source), source_target(source))
            default_ids = tok(rendered['prompt'] + rendered['completion'], truncation=False)['input_ids']
            require(default_ids == ids, 'Default training tokenization differs from explicit IDs')
            require(ids[-1] == tok.eos_token_id, 'Native completion does not end with EOS')
            native[source['id']] = rendered
            accounts[source['id']] = {
                **{k: rendered[k] for k in ('input_tokens', 'completion_tokens', 'total_tokens')},
                'fits': rendered['total_tokens'] <= MAX_LENGTH,
                'prompt_sha256': digest(rendered['prompt'].encode()),
                'completion_sha256': digest(rendered['completion'].encode()),
                'full_input_ids_sha256': digest(encoded(ids)),
                'expected_prompt_label_mask_tokens': rendered['input_tokens'],
                'expected_supervised_tokens': rendered['completion_tokens'],
                'default_tokenization_matches_explicit_ids': True,
                'native_prefix_boundary_verified': True,
            }
            if i % 500 == 0:
                print(json.dumps({'stage': 'CPU_token_accounting', 'arm': arm, 'done': i, 'total': len(rows)}), flush=True)
        native_by_arm[arm], accounts_by_arm[arm] = native, accounts
    prove_original_D(originals, sources, native_by_arm)
    eligible = sorted(i for i in sources if all(accounts_by_arm[a][i]['fits'] for a in ARMS))
    require(len(eligible) == EXPECTED_ELIGIBLE, 'Joint 8192-token cohort differs from the inspected feasibility result')
    target_size = round(EXPECTED_SOURCE_QUESTIONS * FRACTION)
    selected, additional = choose_nested(eligible, original_ids, target_size)
    selected_set, eligible_set = set(selected), set(eligible)
    logical = [make_logical(sources[i], source_index[i]) for i in selected]
    require(not {x['paper_id'] for x in logical} & eval_papers, 'Selected cohort paper overlap')
    accounting = []
    for ident in selected:
        row = sources[ident]
        accounting.append({
            'id': ident, 'paper_id': row['paper_id'], 'answerable': row['answerable'],
            'from_original_D': ident in set(original_ids),
            'logical_messages_sha256': digest(encoded(messages_for(row))),
            'target_json_sha256': digest(source_target(row).encode()),
            'source_context_sha256': digest(row['context'].encode()),
            'models': {a: accounts_by_arm[a][ident] for a in ARMS},
        })
    all_accounting = []
    for ident, source_info in source_index.items():
        reason = source_info['quality_rejection']
        state = ('quality_rejected' if reason else 'over_complete_sequence_budget' if ident not in eligible_set
                 else 'selected' if ident in selected_set else 'eligible_not_selected')
        all_accounting.append({**source_info, 'state': state,
            'models': {a: accounts_by_arm[a][ident] for a in ARMS} if reason is None else None})
    counts = {
        'source_papers': len(train), 'source_questions': len(source_index),
        'quality_eligible': len(rows), 'quality_rejected': quality_stats['rejected'],
        'joint_token_eligible': len(eligible), 'joint_overlong': len(rows)-len(eligible),
        'examples': len(selected), 'positive': sum(x['answerable'] for x in logical),
        'negative': sum(not x['answerable'] for x in logical),
        'selected_papers': len({x['paper_id'] for x in logical}),
        'original_D_retained': len(original_ids), 'newly_added': len(additional),
        'eligible_not_selected': len(eligible)-len(selected),
    }
    require(counts['source_questions'] == sum(counts['quality_rejected'].values()) + counts['joint_overlong'] + counts['examples'] + counts['eligible_not_selected'], 'Source accounting does not balance')
    payloads = {'logical-data.jsonl': jsonl(logical), 'token-accounting.jsonl': jsonl(accounting),
                'eligibility-accounting.jsonl': jsonl(all_accounting)}
    configs, per_arm = {}, {}
    for arm in ARMS:
        payloads[arm+'.jsonl'] = jsonl([make_native(sources[i], native_by_arm[arm][i]) for i in selected])
        config_path = f'configs/scale60-v8-{arm}-training.json'
        config = build_config(arm, len(selected), MAX_LENGTH)
        configs[config_path] = encoded(config)
        per_arm[arm] = {
            'model_id': models[arm]['model_id'], 'revision': models[arm]['revision'],
            'data_file': arm+'.jsonl', 'data_sha256': digest(payloads[arm+'.jsonl']),
            'config_file': config_path, 'config_sha256': digest(configs[config_path]),
            'max_sequence_tokens': MAX_LENGTH,
            'token_summary': {
                'input_tokens': sum(accounts_by_arm[arm][i]['input_tokens'] for i in selected),
                'supervised_tokens': sum(accounts_by_arm[arm][i]['completion_tokens'] for i in selected),
                'total_tokens': sum(accounts_by_arm[arm][i]['total_tokens'] for i in selected),
                'min_total_tokens': min(accounts_by_arm[arm][i]['total_tokens'] for i in selected),
                'max_total_tokens': max(accounts_by_arm[arm][i]['total_tokens'] for i in selected),
            },
        }
    protocol = {
        'schema_version': 1, 'experiment': 'scale60-v8-sft', 'frozen_at_utc': frozen_at,
        'phase': 'CPU_data_preparation_before_either_new_model_training',
        'selected_ids': selected, 'train_paper_ids': sorted({x['paper_id'] for x in logical}),
        'counts': counts, 'source_fraction_requested': FRACTION,
        'source_fraction_realized': len(selected)/len(source_index),
        'selection': {
            'method': 'Retain all original D128; sample sorted joint-eligible complement with random.Random(42).sample(...,1428); combine original D order + sample order, then shuffle with a fresh random.Random(42). No label balancing.',
            'seed': SEED, 'nested_original_D_ids': original_ids, 'additional_ids': additional,
            'outcome_dependent_selection': False, 'labels_used_for_stratification': False,
            'candidate_quality_rule': 'Unchanged prepare_data.convert filters with character cap disabled; actual native full-sequence token lengths replace the former character prefilter.',
            'not_uniform_sample_of_all_source_questions': True,
        },
        'logical_data': {'path': OUTPUT+'/logical-data.jsonl', 'sha256': digest(payloads['logical-data.jsonl'])},
        'token_accounting_file': 'token-accounting.jsonl', 'token_accounting_sha256': digest(payloads['token-accounting.jsonl']),
        'eligibility_accounting_file': 'eligibility-accounting.jsonl',
        'per_arm': per_arm, 'max_sequence_tokens': MAX_LENGTH,
        'no_truncation': True, 'no_target_relabeling': True, 'train_eval_paper_overlap': 0,
        'source_D': {'data_file': D_DATA, 'data_sha256': D_DATA_SHA, 'training_manifest_file': D_MANIFEST,
                     'training_manifest_sha256': D_MANIFEST_SHA, 'selected_count': 128,
                     'all_original_native_rows_and_targets_replayed': True},
        'evaluation_input': {'path': EVAL_INPUT, 'sha256': EVAL_INPUT_SHA, 'question_count': len(evaluation),
                             'paper_overlap': 0, 'labels_read': False},
        'source_files_sha256': source_hashes,
        'preparation_code_sha256': digest(Path(__file__).read_bytes()),
        'tokenizer_metadata': metadata,
        'files': {name: digest(data) for name, data in payloads.items()},
        'checks': {'source_train_only': True, 'all_original_D128_retained_unchanged': True,
                   'both_models_same_ordered_logical_rows_and_targets': True,
                   'all_complete_sequences_fit': True, 'all_native_prefix_boundaries_verified': True,
                   'all_default_training_tokenization_verified': True,
                   'evaluation_labels_read': False, 'official_test_read': False,
                   'model_weights_loaded': False, 'GPU_operations_performed': False,
                   'actual_trainer_label_masks_must_be_checked_before_updates': True},
        'human_review_status': 'pending', 'independent_human_review': False,
        'limitations': [
            '1556/2593 is approximately 60.01% of official training questions, selected from text-quality and joint-length eligible questions; not a random representative 60% of all source data.',
            'Original D targets and first-usable reference targets are retained with their annotation limitations; no E/F corrections or automatic relabeling are applied.',
            'Counts of answerable and unanswerable examples are observed after whole-pool sampling, not forced to match the old 15% rejection ratio.',
            'Same examples and target JSON do not imply equal tokenizer lengths, trainable parameter counts, or GPU compute.',
            'Both candidates train from fresh pinned base models and fresh adapters; original D is a retained-data reference, not an initialization checkpoint.',
            'Existing development inputs have already been inspected. They are not new holdout material; formal test contents remain unused.',
        ],
    }
    payloads['protocol.json'] = encoded(protocol)
    return payloads, configs, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-report', type=Path)
    parser.add_argument('--frozen-at')
    args = parser.parse_args()
    destination = args.preflight_report or ROOT/OUTPUT
    require(not destination.exists(), 'Do not overwrite an existing preparation artifact')
    from transformers import AutoTokenizer
    tiny = json.loads((ROOT/'configs/baseline.json').read_text())
    specs = {ARMS[0]: tiny, ARMS[1]: {'model_id': MODEL_ID, 'revision': MODEL_REVISION}}
    tokenizers = {a: AutoTokenizer.from_pretrained(s['model_id'], revision=s['revision'],
        local_files_only=True, trust_remote_code=False) for a, s in specs.items()}
    payloads, configs, protocol = build(ROOT, tokenizers,
        frozen_at=args.frozen_at or datetime.now(timezone.utc).isoformat())
    report = {
        'status': 'CPU_preflight_passed' if args.preflight_report else 'frozen',
        'counts': protocol['counts'], 'source_fraction_realized': protocol['source_fraction_realized'],
        'per_arm': protocol['per_arm'], 'checks': protocol['checks'],
        'artifact_sha256': {name: digest(data) for name, data in payloads.items()},
        'config_sha256': {name: digest(data) for name, data in configs.items()},
        'packages': {name: importlib.metadata.version(name) for name in ['transformers', 'tokenizers', 'huggingface-hub']},
    }
    if args.preflight_report:
        args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        with args.preflight_report.open('xb') as stream:
            stream.write(encoded(report))
    else:
        for name in configs:
            require(not (ROOT/name).exists(), 'Do not overwrite an existing config: '+name)
        destination.mkdir(parents=True)
        for name, data in payloads.items():
            with (destination/name).open('xb') as stream:
                stream.write(data)
        for name, data in configs.items():
            with (ROOT/name).open('xb') as stream:
                stream.write(data)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
