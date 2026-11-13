"""Aggregate existing v3 scores only; no model calls or semantic re-review."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/contractnli-v3'


def read(name):
    return json.loads((REPORT / name).read_text())


def ratio(n, d):
    return n / d if d else None


def aggregate(rows):
    tp = sum(r['evidence_tp'] for r in rows)
    fp = sum(r['evidence_fp'] for r in rows)
    fn = sum(r['evidence_fn'] for r in rows)
    return {
        'n': len(rows),
        'label_correct': sum(r['label_correct'] for r in rows),
        'joint_exact': sum(r['joint_exact_success'] for r in rows),
        'evidence_exact': sum(r['evidence_exact'] for r in rows),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': ratio(tp, tp + fp),
        'recall': ratio(tp, tp + fn),
        'f1': ratio(2 * tp, 2 * tp + fp + fn),
        'cases_missing_gold_spans': sum(r['evidence_fn'] > 0 for r in rows),
        'cases_with_extra_spans': sum(r['evidence_fp'] > 0 for r in rows),
        'mean_predicted_spans': ratio(sum(len(r['pred_evidence']) for r in rows), len(rows)),
    }


def main():
    scored = read('content-diagnostic-per-example.json')
    v2, v3 = scored['v2'], scored['v3']
    assert len(v2) == len(v3) == 306
    assert [r['id'] for r in v2] == [r['id'] for r in v3]
    assert all(r['strict_format_valid'] for r in v2 + v3)
    groups = {
        'all': lambda r: True,
        'gold_non_neutral': lambda r: r['gold_label'] != 'NotMentioned',
        'gold_single_span': lambda r: len(r['gold_evidence']) == 1,
        'gold_multi_span': lambda r: len(r['gold_evidence']) >= 2,
    }
    for label in ['Entailment', 'Contradiction', 'NotMentioned']:
        groups[label] = lambda r, label=label: r['gold_label'] == label
    subsets = {}
    for name, select in groups.items():
        left, right = [r for r in v2 if select(r)], [r for r in v3 if select(r)]
        pair = {}
        for field in ['label_correct', 'joint_exact_success']:
            pair[field] = {
                'improved': sum(not a[field] and b[field] for a, b in zip(left, right)),
                'regressed': sum(a[field] and not b[field] for a, b in zip(left, right)),
                'both_pass': sum(a[field] and b[field] for a, b in zip(left, right)),
                'both_fail': sum(not a[field] and not b[field] for a, b in zip(left, right)),
            }
        subsets[name] = {'v2': aggregate(left), 'v3': aggregate(right), 'paired': pair}
    contracts = read('content-diagnostics.json')['by_contract']
    contract_summary = {}
    for field in ['label_correct', 'joint_exact_success']:
        contract_summary[field] = {
            'improved': sum(r['v3'][field] > r['v2'][field] for r in contracts),
            'regressed': sum(r['v3'][field] < r['v2'][field] for r in contracts),
            'unchanged': sum(r['v3'][field] == r['v2'][field] for r in contracts),
        }
    contradiction_by_hypothesis = {}
    for hypothesis in sorted({r['id'].split(':')[-1] for r in v2}):
        rows2 = [r for r in v2 if r['id'].endswith(':' + hypothesis)]
        rows3 = [r for r in v3 if r['id'].endswith(':' + hypothesis)]
        contradiction_by_hypothesis[hypothesis] = {
            'gold_contradiction': sum(r['gold_label'] == 'Contradiction' for r in rows2),
            **{arm: {
                'true_positive': sum(r['gold_label'] == r['label'] == 'Contradiction' for r in rows),
                'false_positive': sum(r['gold_label'] != 'Contradiction' and r['label'] == 'Contradiction' for r in rows),
            } for arm, rows in [('v2', rows2), ('v3', rows3)]},
        }
    result = {
        'experiment': 'contractnli-v3',
        'method': 'Deterministic aggregation of existing paired scores; no new generation or semantic review.',
        'subsets': subsets,
        'contract_summary': contract_summary,
        'by_contract': contracts,
        'contradiction_by_hypothesis': contradiction_by_hypothesis,
        'format': {arm: dict(Counter('strict_valid' if r['strict_format_valid'] else 'strict_invalid' for r in rows)) for arm, rows in scored.items()},
        'limits': [
            'Single/multi groups are defined by official gold span count, not predicted span count.',
            'Evidence precision/recall/F1 uses pooled span counts; primary evidence results exclude gold NotMentioned.',
            'Gold NotMentioned empty evidence can raise aggregate joint accuracy; do not use all-row evidence-exact as citation quality.',
            '18 contracts and 26 contradictions; 17 hypotheses within each contract are clustered, not independent contracts.',
        ],
    }
    output = REPORT / 'targeted-diagnostics.json'
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'subsets': subsets, 'contract_summary': contract_summary, 'contradiction_by_hypothesis': contradiction_by_hypothesis}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
