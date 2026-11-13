"""Supplement strict schema scores with raw, declared answerability counts.

These are diagnostic self-reports, not judgments that the answer is supported.
Original predictions and strict metrics are never changed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from run_baseline import digest


def declared_answerability(raw):
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate key')
            value[key] = item
        return value
    raw = raw.strip()
    if raw.startswith('```json\n') and raw.endswith('\n```'):
        raw = raw[8:-4]
    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
        if isinstance(value, dict) and type(value.get('answerable')) is bool:
            return value['answerable']
    except (ValueError, TypeError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    data = args.data or Path(manifest['input_path'])
    if digest(data.read_bytes()) != manifest['input_sha256']:
        raise ValueError('Input checksum mismatch')
    if digest((args.run_dir / 'predictions.jsonl').read_bytes()) != manifest['predictions_sha256']:
        raise ValueError('Predictions checksum mismatch')
    gold = {r['id']: r for r in map(json.loads, data.read_text().splitlines())}
    predictions = [json.loads(line) for line in (args.run_dir / 'predictions.jsonl').read_text().splitlines()]
    groups = {True: Counter(), False: Counter()}
    errors = Counter()
    for result in predictions:
        value = declared_answerability(result.get('raw_output', ''))
        groups[gold[result['id']]['answerable']][str(value)] += 1
        if result.get('parse_error'):
            error = result['parse_error']
            errors['invalid_json' if 'line ' in error else error] += 1
    report = {
        'predictions_sha256': manifest['predictions_sha256'],
        'diagnostic_script_sha256': digest(Path(__file__).read_bytes()),
        'declared_answerability_on_answerable': dict(groups[True]),
        'declared_answerability_on_unanswerable': dict(groups[False]),
        'schema_error_counts': dict(errors),
        'peak_allocated_gib': max(p.get('peak_allocated_gib', 0) for p in predictions),
        'mean_generation_seconds': statistics.mean(p['generation_seconds'] for p in predictions if 'generation_seconds' in p),
        'notes': [
            'True/False are the model answerable field, extracted even when other schema fields are invalid.',
            'None means no unambiguous boolean could be parsed; no guess from truncated text.',
            'A False field can conflict with nonempty answer text; inspect raw predictions.',
            'These diagnostic counts are not an entailment or human correctness assessment.',
            'metrics.json retains the original strict output-contract scoring.'
        ],
    }
    output = args.run_dir / 'diagnostics.json'
    if output.exists():
        raise FileExistsError('Existing diagnostics will not be overwritten')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
