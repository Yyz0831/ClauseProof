"""CuratorKIT-backed, train-only preparation with matched pilot schedules."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from run_baseline import ROOT, digest, messages_for, save_json
from paperproof_eval import parse_prediction


def write_rows(path, rows):
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/curated/pilot-v1')
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--examples', type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    from curatorkit.schema import DataSample
    from curatorkit.gates.schema import SchemaGate
    from curatorkit.normalizers.dedup import ExactDeduplicator
    from transformers import AutoTokenizer
    import curatorkit
    cfg = json.loads((ROOT / 'configs/baseline.json').read_text())
    tok = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['revision'], trust_remote_code=False)
    source = ROOT / 'data/processed/train_candidates.jsonl'
    train = list(map(json.loads, source.read_text().splitlines()))
    dev = list(map(json.loads, (ROOT / 'data/processed/dev_candidates.jsonl').read_text().splitlines()))
    assert not {r['paper_id'] for r in train} & {r['paper_id'] for r in dev}
    samples, rejected, by_id = [], [], {}
    for row in train:
        assert row['source_split'] == 'train'
        target = {'answerable': row['answerable'],
                  'answer': row['reference_answers'][0].strip() if row['answerable'] else '',
                  'evidence_ids': row['reference_evidence_ids'][0] if row['answerable'] else []}
        text = json.dumps(target, ensure_ascii=False)
        _, error, _ = parse_prediction(text, {p['id'] for p in row['paragraphs']})
        if error:
            rejected.append({'id': row['id'], 'reason': 'target_schema:' + error})
            continue
        messages = messages_for(row)
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(messages + [{'role': 'assistant', 'content': text}],
                                       tokenize=False, add_generation_prompt=False)
        assert full.startswith(prompt), 'Chat prefix changed; do not guess label boundaries'
        completion = full[len(prompt):]
        pids = tok(prompt, add_special_tokens=False)['input_ids']
        ids = tok(full, add_special_tokens=False)['input_ids']
        assert ids[:len(pids)] == pids, 'Tokenizer boundary mismatch'
        count = len(ids) - len(pids)
        reason = 'full_sequence_over_budget' if len(ids) > args.max_length else (
            'completion_over_192_tokens' if count > 192 else None)
        if reason:
            rejected.append({'id': row['id'], 'reason': reason, 'tokens': len(ids), 'completion_tokens': count})
            continue
        by_id[row['id']] = dict(id=row['id'], paper_id=row['paper_id'], answerable=row['answerable'],
                               prompt=prompt, completion=completion, input_tokens=len(pids),
                               completion_tokens=count, total_tokens=len(ids),
                               reference_index=0, human_review_status='pending')
        samples.append(DataSample(id=row['id'], source_uri='allenai/qasper@v0.3/train/' + row['paper_id'],
                                  instruction=prompt, output=completion,
                                  metadata={'paper_id': row['paper_id'], 'question_id': row['question_id'],
                                            'source_split': 'train', 'answerable': row['answerable'],
                                            'human_review_status': 'pending'}))
    # This gate checks structure/encoding only. Actual Tiny Aya token lengths were checked above.
    passed, invalid = SchemaGate(min_tokens=1, max_tokens=1_000_000, use_tiktoken=False).run(samples)
    rejected.extend({'id': x.id, 'reason': x.rejection_reason} for x in invalid)
    deduped = ExactDeduplicator().run(passed)
    kept_ids = {x.id for x in deduped}
    rejected.extend({'id': x.id, 'reason': 'curatorkit_exact_duplicate'} for x in passed if x.id not in kept_ids)
    kept = [by_id[x.id] for x in deduped]
    rng = random.Random(42)
    positive = [x for x in kept if x['answerable']]
    negative = [x for x in kept if not x['answerable']]
    rng.shuffle(positive); rng.shuffle(negative)
    n_negative = round(args.examples * 0.30)
    if len(negative) < n_negative or len(positive) < args.examples:
        raise ValueError(f'Not enough distinct candidates: positive={len(positive)}, negative={len(negative)}')
    shared = positive[:args.examples - n_negative]
    c_extra = negative[:n_negative]
    pool = positive[args.examples - n_negative:]
    b_extra = []
    for neg in c_extra:
        nearest = min(pool, key=lambda x: (abs(x['total_tokens'] - neg['total_tokens']), x['id']))
        pool.remove(nearest)
        b_extra.append(nearest)
    b = shared + b_extra
    c = shared + c_extra
    order = list(range(args.examples)); rng.shuffle(order)
    b = [b[i] for i in order]; c = [c[i] for i in order]
    token_b, token_c = sum(x['total_tokens'] for x in b), sum(x['total_tokens'] for x in c)
    if abs(token_b - token_c) / max(token_b, token_c) > 0.02:
        raise ValueError('Token budgets differ by over 2%; review selection before training')
    write_rows(args.output / 'candidates.jsonl', kept)
    write_rows(args.output / 'rejections.jsonl', rejected)
    write_rows(args.output / 'curatorkit_provenance.jsonl', [x.model_dump(mode='json') for x in deduped])
    write_rows(args.output / 'b_answerable_only.jsonl', b)
    write_rows(args.output / 'c_with_abstention.jsonl', c)
    report = {
        'source_sha256': digest(source.read_bytes()), 'model_revision': cfg['revision'],
        'curatorkit_version': curatorkit.__version__,
        'frameworks': json.loads((ROOT / 'configs/frameworks.lock.json').read_text()),
        'input_candidates': len(train), 'kept': len(kept),
        'kept_by_answerability': dict(Counter(str(x['answerable']) for x in kept)),
        'rejections': dict(Counter(x['reason'] for x in rejected)),
        'max_sequence_tokens': args.max_length, 'truncation': False,
        'train_dev_paper_overlap': 0, 'first_reference_selected_deterministically': True,
        'human_review_status': 'pending', 'synthetic_labels_created': False,
        'pilot_examples_each': args.examples, 'shared_positive_examples': len(shared),
        'B': {'positive': len(b), 'negative': 0, 'total_tokens': token_b,
              'supervised_tokens': sum(x['completion_tokens'] for x in b)},
        'C': {'positive': len(shared), 'negative': len(c_extra), 'total_tokens': token_c,
              'supervised_tokens': sum(x['completion_tokens'] for x in c)},
        'relative_total_token_gap': abs(token_b - token_c) / max(token_b, token_c),
        'notes': ['Pilot ablation: both groups have the same JSON/evidence format; C replaces 30% of positive examples with natural unanswerable examples.',
                  'This tests the abstention-data mixture, not evidence supervision in isolation.',
                  'Total token budgets are matched approximately; supervised answer-token counts differ and are reported.',
                  'No dev/test examples are used for training or deduplication.',
                  'SchemaGate is not a factuality judge; original labels remain subject to human review.'],
        'files': {p.name: digest(p.read_bytes()) for p in sorted(args.output.glob('*.jsonl'))},
    }
    save_json(args.output / 'manifest.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
