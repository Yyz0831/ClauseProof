"""Pilot metrics, not a score on the complete official QASPER benchmark.

Answer normalization and max-over-references follow the QASPER/SQuAD convention:
https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py
Paragraph metrics here use our preserved paragraph IDs instead of paragraph text.
"""
from collections import Counter
import json
import re
import statistics
import string


def normalize_answer(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_f1(predicted, reference):
    predicted = normalize_answer(predicted).split()
    reference = normalize_answer(reference).split()
    shared = sum((Counter(predicted) & Counter(reference)).values())
    return 2 * shared / (len(predicted) + len(reference)) if shared else 0.0


def evidence_f1(predicted, reference):
    predicted, reference = set(predicted), set(reference)
    if not predicted and not reference:
        return 1.0
    return 2 * len(predicted & reference) / (len(predicted) + len(reference))


def parse_prediction(text, paragraph_ids):
    """Strict JSON/schema; a single Markdown code fence is allowed and recorded."""
    clean = text.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', clean, flags=re.DOTALL)
    wrapper = 'markdown_fence' if fenced else 'none'
    if fenced:
        clean = fenced.group(1)
    def unique_keys(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('duplicate_key:' + key)
            out[key] = value
        return out
    try:
        value = json.loads(clean, object_pairs_hook=unique_keys)
        if not isinstance(value, dict) or set(value) != {'answerable', 'answer', 'evidence_ids'}:
            raise ValueError('wrong_keys_or_non_object')
        if type(value['answerable']) is not bool or not isinstance(value['answer'], str):
            raise ValueError('wrong_answer_types')
        ids = value['evidence_ids']
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
            raise ValueError('wrong_evidence_type')
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate_evidence_ids')
        if any(x not in paragraph_ids for x in ids):
            raise ValueError('unknown_evidence_id')
        if value['answerable'] and (not value['answer'].strip() or not ids):
            raise ValueError('answer_without_text_or_evidence')
        if not value['answerable'] and (value['answer'].strip() or ids):
            raise ValueError('refusal_with_answer_or_evidence')
        return value, None, wrapper
    except (ValueError, TypeError) as exc:
        return None, str(exc), wrapper


def score_one(gold, result):
    pred = result.get('prediction') if result.get('status') == 'ok' else None
    valid = pred is not None
    answered = valid and pred['answerable']
    refused = valid and not pred['answerable']
    return {
        'id': gold['id'], 'gold_answerable': gold['answerable'],
        'status': result.get('status', 'missing'), 'valid_prediction': valid,
        'answered': answered, 'refused': refused,
        'answer_f1': max(answer_f1(pred['answer'], ref) for ref in gold['reference_answers'])
                     if answered and gold['answerable'] else 0.0,
        'answer_exact_match': max(float(normalize_answer(pred['answer']) == normalize_answer(ref))
                                  for ref in gold['reference_answers'])
                              if answered and gold['answerable'] else 0.0,
        'evidence_f1': max(evidence_f1(pred['evidence_ids'], ref)
                           for ref in gold['reference_evidence_ids']) if valid else 0.0,
        'generation_seconds': result.get('generation_seconds'),
        'generated_tokens': result.get('generated_tokens'),
        'hit_token_limit': result.get('hit_token_limit', False),
    }


def summarize(gold_rows, results):
    indexed = {r['id']: r for r in results}
    if len(indexed) != len(results):
        raise ValueError('Duplicate prediction IDs')
    if set(indexed) - {g['id'] for g in gold_rows}:
        raise ValueError('Predictions contain unknown question IDs')
    scores = [score_one(g, indexed.get(g['id'], {})) for g in gold_rows]
    positive = [s for s in scores if s['gold_answerable']]
    negative = [s for s in scores if not s['gold_answerable']]
    mean = lambda values: statistics.mean(values) if values else None
    rate = lambda rows, key: mean([float(r[key]) for r in rows])
    summary = {
        'questions': len(scores), 'answerable_questions': len(positive),
        'unanswerable_questions': len(negative),
        'status_counts': dict(Counter(s['status'] for s in scores)),
        'valid_output_rate': rate(scores, 'valid_prediction'),
        'answerable_answer_f1': mean([s['answer_f1'] for s in positive]),
        'answerable_answer_exact_match': mean([s['answer_exact_match'] for s in positive]),
        'answerable_evidence_f1': mean([s['evidence_f1'] for s in positive]),
        'all_questions_evidence_f1': mean([s['evidence_f1'] for s in scores]),
        'false_refusal_rate_on_answerable': rate(positive, 'refused'),
        'answer_rate_on_unanswerable': rate(negative, 'answered'),
        'correct_refusal_rate_on_unanswerable': rate(negative, 'refused'),
        'invalid_or_failed_rate_on_answerable': mean([float(not s['valid_prediction']) for s in positive]),
        'invalid_or_failed_rate_on_unanswerable': mean([float(not s['valid_prediction']) for s in negative]),
        'mean_generation_seconds': mean([s['generation_seconds'] for s in scores if s['generation_seconds'] is not None]),
        'hit_token_limit_count': sum(s['hit_token_limit'] for s in scores),
        'notes': [
            'Stratified development pilot, not the full QASPER test benchmark.',
            'All selected questions remain in denominators; invalid/failed/skipped outputs score zero F1.',
            'Invalid/failed outputs are reported separately, not counted as successful refusals.',
            'Answering an unanswerable-labeled item is a proxy for unsupported answering, not an entailment judgment.',
            'Evidence ID validity does not establish that citations support the answer.',
            'Reference annotations and review_status are unchanged; human review is pending.'
        ],
    }
    return summary, scores
