"""Build an offline, initially unrated blind packet from the frozen 40-question sample.

Run only after both inference manifests are complete. Share the HTML or packet,
never the separate key/provenance files. Existing artifacts are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TOKEN = '__PAPERPROOF_BLIND_PACKET__'
RATING_OPTIONS = {
    'question_answerable': ['yes', 'no', 'uncertain'],
    'answer_support': ['supported', 'partial', 'unsupported', 'refusal', 'uncertain'],
    'citation_support': ['supported', 'partial', 'unsupported', 'not_applicable', 'uncertain'],
    'attribution_error': ['yes', 'no', 'uncertain'],
}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {row['id']: row for row in rows}
    if not rows or len(rows) != len(by_id):
        raise ValueError(f'Empty data or duplicate question IDs: {path}')
    return by_id


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Provenance timestamps must include a timezone')
    return result


def citation_display(raw):
    """Extract only citations for navigation; retain the complete original output.

    A display parse is not a correctness judgement. Unknown IDs stay visible.
    """
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    text = raw.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4]
    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (TypeError, ValueError):
        return {'citation_ids': [], 'citations_detected': False}
    ids = value.get('evidence_ids') if isinstance(value, dict) else None
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        return {'citation_ids': [], 'citations_detected': False}
    return {'citation_ids': ids, 'citations_detected': True}


def verified_run(run_dir, source, source_sha):
    manifest_path = run_dir / 'manifest.json'
    prediction_path = run_dir / 'predictions.jsonl'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('inference_complete') is not True:
        raise ValueError(f'Inference is not complete: {run_dir}')
    if manifest.get('input_sha256') != source_sha:
        raise ValueError(f'Input checksum mismatch: {run_dir}')
    prediction_sha = digest(prediction_path.read_bytes())
    if manifest.get('predictions_sha256') != prediction_sha:
        raise ValueError(f'Prediction checksum mismatch: {run_dir}')
    predictions = load_rows(prediction_path)
    if set(predictions) != set(source) or manifest.get('selected_question_ids') != list(source):
        raise ValueError(f'Prediction/manifest question IDs differ from the full source: {run_dir}')
    instruction = manifest.get('instruction')
    if not isinstance(instruction, str) or digest(instruction.encode()) != manifest.get('instruction_sha256'):
        raise ValueError(f'Instruction checksum mismatch: {run_dir}')
    for qid, row in predictions.items():
        if row.get('question') != source[qid]['question']:
            raise ValueError(f'Question text mismatch: {qid}')
        # Schema-invalid generations are evidence for the reviewer, not missing data.
        if row.get('status') not in {'ok', 'invalid_output'} or not isinstance(row.get('raw_output'), str):
            raise ValueError(f'Missing generated output: {qid}')
        if not re.fullmatch(r'[0-9a-f]{64}', row.get('prompt_sha256', '')):
            raise ValueError(f'Missing prompt checksum: {qid}')
        if type(row.get('input_tokens')) is not int or row['input_tokens'] < 1:
            raise ValueError(f'Invalid input token count: {qid}')
    provenance = {
        'run_path': str(run_dir),
        'manifest_sha256': digest(manifest_path.read_bytes()),
        'predictions_sha256': prediction_sha,
        'input_sha256': source_sha,
        'instruction_sha256': manifest['instruction_sha256'],
        'adapter': manifest.get('adapter'),
    }
    return predictions, manifest, provenance


def blank_ratings(packet):
    return {
        'schema_version': 1,
        'packet_id': packet['packet_id'],
        'reviewer_name': '',
        'reviewer_attestation': False,
        'human_review_complete': False,
        'review_status': 'not_started',
        'completed_questions': 0,
        'questions': [{
            'review_id': item['review_id'],
            'question_answerable': '',
            'notes': '',
            'responses': [{
                'slot': response['slot'],
                'answer_support': '',
                'citation_support': '',
                'attribution_error': '',
                'notes': '',
            } for response in item['responses']],
        } for item in packet['questions']],
    }


def build_payloads(selection_path, data_path, e_run, f_run):
    """Validate all inputs and return public packet, private key, ratings, provenance."""
    selection_path, data_path, e_run, f_run = map(resolve, [selection_path, data_path, e_run, f_run])
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    source_sha = digest(data_path.read_bytes())
    if selection.get('input_sha256') != source_sha:
        raise ValueError('Frozen selection input checksum differs from source data')
    if selection.get('phase') != 'before_F_training_and_inference':
        raise ValueError('Selection was not frozen before the candidate experiment')
    selected = selection.get('question_ids', [])
    if len(selected) != 40 or len(set(selected)) != 40:
        raise ValueError('The frozen sample must contain exactly 40 distinct questions')
    if type(selection.get('seed')) is not int:
        raise ValueError('The frozen selection must specify an integer seed')
    source = load_rows(data_path)
    if not set(selected) <= set(source):
        raise ValueError('A frozen question is missing from source data')
    if len({source[qid]['paper_id'] for qid in selected}) != 40:
        raise ValueError('The frozen sample must contain 40 different papers')
    if any(type(source[qid].get('answerable')) is not bool for qid in selected):
        raise ValueError('Source answerability must be a boolean')
    positive = sum(source[qid]['answerable'] for qid in selected)
    if positive != 20 or selection.get('positive') != 20 or selection.get('negative') != 20:
        raise ValueError('The frozen source-label allocation must remain 20/20')
    predictions, manifests, run_provenance = {}, {}, {}
    for label, run_dir in [('E', e_run), ('F', f_run)]:
        predictions[label], manifests[label], run_provenance[label] = verified_run(run_dir, source, source_sha)
    if timestamp(selection['frozen_at_utc']) >= timestamp(manifests['F']['started_at_utc']):
        raise ValueError('The selection must predate candidate inference')
    if manifests['E']['instruction_sha256'] != manifests['F']['instruction_sha256']:
        raise ValueError('The two runs used different instructions')
    for qid in source:
        left, right = predictions['E'][qid], predictions['F'][qid]
        if (left['prompt_sha256'], left['input_tokens']) != (right['prompt_sha256'], right['input_tokens']):
            raise ValueError(f'The two runs used different input prompts: {qid}')

    questions, keys = [], []
    for index, qid in enumerate(selected, 1):
        row = source[qid]
        paragraphs = [{key: paragraph[key] for key in ('id', 'section', 'text')} for paragraph in row['paragraphs']]
        if not paragraphs or len({p['id'] for p in paragraphs}) != len(paragraphs):
            raise ValueError(f'Empty or duplicate source paragraphs: {qid}')
        if not all(isinstance(value, str) for p in paragraphs for value in p.values()):
            raise ValueError(f'Invalid source paragraph fields: {qid}')
        # Independent per-question coin flip; seed and model mapping stay private.
        flip = hashlib.sha256(f"blind-v1:{selection['seed']}:{qid}".encode()).digest()[0] & 1
        order = ['F', 'E'] if flip else ['E', 'F']
        responses = []
        for slot, label in zip(('A', 'B'), order):
            raw = predictions[label][qid]['raw_output']
            responses.append({'slot': slot, 'raw_output': raw, **citation_display(raw)})
        review_id = f'q{index:03d}'
        questions.append({'review_id': review_id, 'title': row['title'], 'question': row['question'],
                          'paragraphs': paragraphs, 'responses': responses})
        keys.append({'review_id': review_id, 'id': qid, 'paper_id': row['paper_id'],
                     'source_answerable': row['answerable'], 'models_by_slot': dict(zip(('A', 'B'), order))})
    packet_id = digest(json_bytes(questions))
    packet = {
        'schema_version': 1,
        'packet_id': packet_id,
        'review_status': 'pending',
        'instructions': '先读题目与所供论文全文，再分别判断两个匿名回答。只根据所供正文，勿查外部资料；引用段落也须读上下文。所有判断默认留空，不确定时请选择“不确定”。',
        'source_note': '这里展示模型收到的全部论文文本与段落编号；原数据未提供的图片、公式或表格不会自动补全。证据不足时可以选择“不确定”。',
        'rating_options': RATING_OPTIONS,
        'questions': questions,
    }
    key = {'schema_version': 1, 'packet_id': packet_id, 'private_unblinding_key': True,
           'do_not_share_with_reviewers': True, 'randomization_seed': selection['seed'], 'questions': keys}
    provenance = {
        'schema_version': 1, 'packet_id': packet_id,
        'private_provenance': True, 'do_not_share_with_reviewers': True,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection': {'path': str(selection_path), 'sha256': digest(selection_path.read_bytes()),
                      'frozen_at_utc': selection['frozen_at_utc']},
        'source': {'path': str(data_path), 'sha256': source_sha},
        'runs': run_provenance, 'question_count': 40, 'paper_count': 40,
        'verification': {'full_input_hash_match': True, 'all_prompt_hashes_match': True,
                         'all_prediction_hashes_match': True, 'selection_predates_candidate_inference': True},
        'human_review_status': 'pending', 'independent_human_review': False,
        'limitations': 'Balanced targeted development sample, not an independent test or prevalence-weighted accuracy estimate. Blind order does not establish independent human review.',
    }
    return packet, key, blank_ratings(packet), provenance


def render_html(packet, template):
    if template.count(TOKEN) != 1:
        raise ValueError('Blind review template must contain exactly one packet placeholder')
    encoded = json.dumps(packet, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    encoded = encoded.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    return template.replace(TOKEN, encoded)


def write_artifacts(packet, key, ratings, provenance, output_dir, html_path, template_path):
    output_dir, html_path, template_path = map(resolve, [output_dir, html_path, template_path])
    template = template_path.read_text(encoding='utf-8')
    artifacts = {
        output_dir / 'human-review-packet.json': json_bytes(packet),
        output_dir / 'human-review-key.json': json_bytes(key),
        output_dir / 'blank-human-ratings.json': json_bytes(ratings),
        html_path: render_html(packet, template).encode('utf-8'),
    }
    provenance = dict(provenance)
    provenance['builder_sha256'] = digest(Path(__file__).read_bytes())
    provenance['template_sha256'] = digest(template_path.read_bytes())
    provenance['artifacts'] = {str(path): {'sha256': digest(content)} for path, content in artifacts.items()}
    artifacts[output_dir / 'human-review-provenance.json'] = json_bytes(provenance)
    if len({path.resolve() for path in artifacts}) != 5:
        raise ValueError('Output paths must be distinct')
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite review artifacts: ' + ', '.join(existing))
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as output:
            output.write(content)
    return list(artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', default='reports/attribution-v4/human-review-selection.json')
    parser.add_argument('--data', default='data/curated/followup-v2/untouched_dev.jsonl')
    parser.add_argument('--e-run', default='runs/compare-e-cleaning-v3')
    parser.add_argument('--f-run', default='runs/compare-f-attribution-v4')
    parser.add_argument('--output-dir', default='reports/attribution-v4')
    parser.add_argument('--html', default='demo/human-review-v4.html')
    parser.add_argument('--template', default='demo/blind-review-template.html')
    args = parser.parse_args()
    payloads = build_payloads(args.selection, args.data, args.e_run, args.f_run)
    paths = write_artifacts(*payloads, args.output_dir, args.html, args.template)
    print(json.dumps({'created': [str(path) for path in paths], 'human_review_status': 'pending'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
