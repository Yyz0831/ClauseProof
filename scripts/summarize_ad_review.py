"""Validate completed anonymous AI reviews before unblinding A/D descriptions.

This is a retrospective selected development sample, never independent human
evidence or a statistical estimate of a fine-tuning improvement. Human files and
the original reviews are read-only. --validate-only never opens private files.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from build_blind_review import blank_ratings, citation_display, json_bytes
from paperproof_eval import parse_prediction

ROOT = Path(__file__).resolve().parents[1]
GROUP_SIZES = {'a': 13, 'b': 13, 'c': 14}
ENUMS = {
    'source_question_status': ['answerable', 'unsupported', 'ambiguous'],
    'answer_support': ['supported', 'partial', 'unsupported', 'refusal', 'uncertain'],
    'citation_support': ['supported', 'partial', 'unsupported', 'not_applicable', 'uncertain'],
    'attribution_error': ['yes', 'no', 'uncertain'],
}
LIMITATIONS = [
    '40 题来自 40 篇回顾性选定、反复使用的开发论文；不是前瞻保留测试或独立基准。',
    '同一助手系统的匿名 AI 审查可能有先前项目暴露，不能代替独立人工审核；人工评分仍全部留空。',
    '这些是本次选定样本的描述性计数，不能据此统计推断微调提升、总体事实正确率或宣布 D 胜出。',
    'refusal 只描述审查者读到的行为；合法空拒答由原始输出严格解析，格式失败不会转成拒答。',
    '支持文字、引用支持、归属错误和严格格式是不同维度；不把支持文字自动视作可用回答。',
    'source_question_status 是本次 AI 判断，不覆盖原始数据标签；所有源标签、原审查和人工文件保持不变。',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=unique_object)


def text_or_list(value, location):
    require(isinstance(value, str) or (isinstance(value, list)
            and all(isinstance(item, str) for item in value)), f'Invalid text/list: {location}')


def check_ai_only(value, location):
    require(value.get('reviewer_type') == 'AI_assistant', f'Not an AI review: {location}')
    require(value.get('independent_human_review') is False, f'Human-independence claim: {location}')
    require(value.get('human_review_status') == 'pending', f'Human status changed: {location}')
    require(value.get('prospective_holdout', False) is False, f'Holdout claim: {location}')


def validate_public(review_dir):
    """Finish all public validation before any private mapping can be read."""
    review_dir = Path(review_dir)
    contents = {}

    def read(name):
        data = (review_dir / name).read_bytes()
        contents[name] = data
        return decode(data)

    plan = read('ai-review-plan.json')
    packet = read('human-review-packet.json')
    ratings = read('blank-human-ratings.json')
    check_ai_only(plan, 'plan')
    plan_hash = sha(contents['ai-review-plan.json'])
    require(plan.get('public_packet_sha256') == sha(contents['human-review-packet.json']),
            'Public packet SHA mismatch')
    require(plan.get('packet_id') == packet.get('packet_id') == sha(json_bytes(packet['questions'])),
            'Packet identity mismatch')
    require(ratings == blank_ratings(packet), 'Human ratings are not the original blank template')
    questions = packet['questions']
    ids = [q['review_id'] for q in questions]
    require(ids == [f'q{i:03d}' for i in range(1, 41)], 'Expected exactly 40 ordered public questions')
    for key, values in ENUMS.items():
        require(plan['rubric'].get(key) == values, f'Unexpected rubric enum: {key}')
    expected_inputs = {f'ai-review-input-{group}.jsonl' for group in GROUP_SIZES}
    require(set(plan['input_files']) == expected_inputs, 'Unexpected input assignment files')
    cases, joined, groups, quote_count = [], [], {}, 0
    for group, size in GROUP_SIZES.items():
        input_name = f'ai-review-input-{group}.jsonl'
        data = (review_dir / input_name).read_bytes()
        contents[input_name] = data
        rows = [decode(line) for line in data.decode('utf-8').splitlines() if line.strip()]
        spec = plan['input_files'][input_name]
        require(spec['sha256'] == sha(data), f'Input SHA mismatch: {group}')
        require(spec['questions'] == len(rows) == size, f'Input count mismatch: {group}')
        require(spec['review_ids'] == [r['review_id'] for r in rows], f'Input order mismatch: {group}')
        require(spec['paragraphs'] == sum(len(r['paragraphs']) for r in rows), f'Paragraph count: {group}')
        require(spec['fulltext_characters'] == sum(len(p['text']) for r in rows for p in r['paragraphs']),
                f'Source character count: {group}')
        name = f'ai-review-{group}.json'
        report = read(name)
        require(all(k in report for k in plan['required_report_fields']), f'Missing report field: {group}')
        check_ai_only(report, group)
        require(report['input_sha256'] == spec['sha256'] and report['plan_sha256'] == plan_hash,
                f'Review provenance mismatch: {group}')
        require(report.get('packet_id', packet['packet_id']) == packet['packet_id'], f'Review packet: {group}')
        require(report.get('review_status', 'completed') in ('complete', 'completed'),
                f'Review not complete: {group}')
        require(isinstance(report['cases'], list) and len(report['cases']) == size,
                f'Review case count mismatch: {group}')
        require([c['review_id'] for c in report['cases']] == spec['review_ids'], f'Review IDs/order: {group}')
        for optional in ('expected_cases', 'reviewed_cases'):
            require(report.get(optional, size) == size, f'Incomplete {optional}: {group}')
        for case, row in zip(report['cases'], rows):
            location = case['review_id']
            require(all(k in case for k in plan['required_case_fields']), f'Missing case field: {location}')
            require(case['source_question_status'] in ENUMS['source_question_status'], f'Source enum: {location}')
            require(isinstance(case['reason'], str) and case['reason'].strip(), f'Empty case reason: {location}')
            text_or_list(case['caveats'], location)
            paragraphs = row['paragraphs']
            pids = [p['id'] for p in paragraphs]
            require(pids and len(set(pids)) == len(pids), f'Empty/duplicate source IDs: {location}')
            require(all(isinstance(p[k], str) for p in paragraphs for k in ('id', 'section', 'text')),
                    f'Invalid source paragraph: {location}')
            require(case['reviewed_paragraph_ids'] == pids, f'Incomplete paragraph coverage: {location}')
            require([r['slot'] for r in row['responses']] == ['A', 'B'], f'Public response slots: {location}')
            require([r['slot'] for r in case['responses']] == ['A', 'B'], f'Review response slots: {location}')
            for response, original in zip(case['responses'], row['responses']):
                require(all(k in response for k in plan['required_response_fields']), f'Missing response field: {location}')
                for key in ('answer_support', 'citation_support', 'attribution_error'):
                    require(response[key] in ENUMS[key], f'Invalid {key}: {location}')
                require(isinstance(response['reason'], str) and response['reason'].strip(), f'Empty response reason: {location}')
                text_or_list(response['format_limitations'], location)
                for key in ('raw_output', 'citation_ids', 'citations_detected'):
                    if key in response:
                        require(response[key] == original[key], f'Review rewrote {key}: {location}')
                require(isinstance(original['raw_output'], str), f'Non-text raw output: {location}')
                display = citation_display(original['raw_output'])
                require(all(original[k] == display[k] for k in display), f'Public citation display mismatch: {location}')
            source = {p['id']: p['text'] for p in paragraphs}
            for obj in [case] + case['responses']:
                require(isinstance(obj['evidence'], list), f'Invalid evidence: {location}')
                for evidence in obj['evidence']:
                    require(set(evidence) == {'paragraph_id', 'quote'}, f'Evidence schema: {location}')
                    require(evidence['paragraph_id'] in source and isinstance(evidence['quote'], str)
                            and evidence['quote'] and evidence['quote'] in source[evidence['paragraph_id']],
                            f'Non-exact evidence quote: {location}')
                    quote_count += 1
        groups[group] = {'review_sha256': sha(contents[name]), 'input_sha256': spec['sha256'],
                         'questions': size, 'responses': 2 * size,
                         'paragraphs': spec['paragraphs'], 'fulltext_characters': spec['fulltext_characters']}
        cases.extend(report['cases'])
        joined.extend(rows)
    require(joined == questions, 'AI inputs do not exactly reproduce the complete public packet')
    require(len(cases) == 40 and sum(len(c['responses']) for c in cases) == 80, 'Incomplete 40/80 coverage')
    return {'plan': plan, 'packet': packet, 'cases': cases, 'contents': contents, 'groups': groups,
            'validation': {'status': 'passed', 'questions': 40, 'responses': 80,
                'paragraphs': sum(g['paragraphs'] for g in groups.values()),
                'fulltext_characters': sum(g['fulltext_characters'] for g in groups.values()),
                'exact_quotes_validated': quote_count, 'human_ratings_blank': True}}


def load_verified_key(review_dir, validated):
    """Called only after every public review has passed validate_public."""
    root = Path(review_dir)
    protocol_bytes = (root / 'private/protocol.json').read_bytes()
    protocol = decode(protocol_bytes)
    key_bytes = (root / 'private/unblinding-key.json').read_bytes()
    key = decode(key_bytes)
    packet_id = validated['packet']['packet_id']
    require(protocol.get('packet_id') == key.get('packet_id') == packet_id, 'Private packet identity mismatch')
    require(key.get('private_unblinding_key') is True, 'Not an unblinding key')
    # Original manifests may contain another checkout's absolute path. Match the
    # known artifact suffix uniquely; never follow arbitrary manifest paths.
    for name, data in {**{k: v for k, v in validated['contents'].items()
                         if not k.startswith('ai-review-') or k.endswith('.jsonl') or k == 'ai-review-plan.json'},
                       'private/unblinding-key.json': key_bytes}.items():
        hashes = [v for k, v in protocol['artifacts_sha256'].items()
                  if k == name or k.endswith('/' + name)]
        require(len(hashes) == 1 and hashes[0] == sha(data), f'Frozen artifact SHA mismatch: {name}')
    rows = key['questions']
    require([q['review_id'] for q in rows] == [q['review_id'] for q in validated['packet']['questions']],
            'Unblinding key IDs/order mismatch')
    require(len({q['paper_id'] for q in rows}) == 40 and len({q['id'] for q in rows}) == 40,
            'Expected 40 distinct source papers/questions')
    for row in rows:
        mapping = row['models_by_slot']
        require(set(mapping) == {'A', 'B'} and set(mapping.values()) == {'A', 'D'}, 'Invalid model-slot mapping')
        require(type(row['source_answerable']) is bool, 'Invalid preserved source label')
    validated['contents']['private/protocol.json'] = protocol_bytes
    validated['contents']['private/unblinding-key.json'] = key_bytes
    return key


def count_values(rows, key, values):
    counter = Counter(r[key] for r in rows)
    return {value: counter[value] for value in values}


def summarize(validated, key):
    pairs, by_model = [], {'A': [], 'D': []}
    for source, case, mapping in zip(validated['packet']['questions'], validated['cases'], key['questions']):
        pair = {'review_id': case['review_id'], 'source_question_status': case['source_question_status'], 'models': {}}
        pids = {p['id'] for p in source['paragraphs']}
        for original, response in zip(source['responses'], case['responses']):
            label = mapping['models_by_slot'][response['slot']]
            parsed, error, wrapper = parse_prediction(original['raw_output'], pids)
            state = ('answered' if parsed['answerable'] else 'refused') if parsed else 'invalid_output'
            entry = {k: response[k] for k in ('answer_support', 'citation_support', 'attribution_error')}
            entry.update(review_id=case['review_id'], anonymous_slot=response['slot'],
                         source_question_status=case['source_question_status'], strict_state=state,
                         parse_error=error, wrapper=wrapper, raw_output_sha256=sha(original['raw_output'].encode()),
                         reviewer_format_notes=response['format_limitations'],
                         strict_supported_answer_and_citation=(state == 'answered'
                             and response['answer_support'] == 'supported'
                             and response['citation_support'] == 'supported'
                             and response['attribution_error'] == 'no'))
            pair['models'][label] = entry
            by_model[label].append(entry)
        pairs.append(pair)
    models = {}
    for model, rows in by_model.items():
        models[model] = {
            'responses': len(rows),
            **{name: count_values(rows, name, ENUMS[name])
               for name in ('answer_support', 'citation_support', 'attribution_error')},
            'strict_state': count_values(rows, 'strict_state', ['answered', 'refused', 'invalid_output']),
            'format_invalid_count': sum(r['strict_state'] == 'invalid_output' for r in rows),
            'format_error_counts': dict(Counter(r['parse_error'] for r in rows if r['parse_error'])),
            'markdown_fence_count': sum(r['wrapper'] == 'markdown_fence' for r in rows),
            'strict_supported_answer_and_citation_count': sum(r['strict_supported_answer_and_citation'] for r in rows),
            'by_AI_source_question_status': {
                status: {'questions': sum(r['source_question_status'] == status for r in rows),
                         'strict_state': count_values([r for r in rows if r['source_question_status'] == status],
                                                      'strict_state', ['answered', 'refused', 'invalid_output'])}
                for status in ENUMS['source_question_status']},
        }
    transitions = {}
    for metric in ('answer_support', 'citation_support', 'attribution_error', 'strict_state',
                   'strict_supported_answer_and_citation'):
        buckets = {}
        for pair in pairs:
            a, d = pair['models']['A'][metric], pair['models']['D'][metric]
            token = f'{a} -> {d}'
            buckets.setdefault(token, []).append(pair['review_id'])
        transitions[metric] = {k: {'count': len(v), 'review_ids': v} for k, v in sorted(buckets.items())}
    code_hashes = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in
                   [Path(__file__), ROOT/'scripts/paperproof_eval.py', ROOT/'scripts/build_blind_review.py']}
    return {
        'schema_version': 1, 'packet_id': validated['packet']['packet_id'],
        'reviewer_type': 'AI_assistant', 'independent_human_review': False, 'human_review_status': 'pending',
        'review_status': 'completed', 'prospective_holdout': False, 'population_inference_permitted': False,
        'winner_declared': False, 'unblinding_after_all_public_reviews_validated': True,
        'validation': validated['validation'], 'groups': validated['groups'],
        'source_files_sha256': {k: sha(v) for k, v in validated['contents'].items()},
        'computation_files_sha256': code_hashes,
        'source_question_status': dict(Counter(c['source_question_status'] for c in validated['cases'])),
        'models': models, 'paired_A_to_D': transitions, 'pairs': pairs,
        'definitions': {
            'source_files_sha256': 'Paths are relative to the review directory containing this summary.',
            'format_invalid_count': 'Existing paperproof_eval.parse_prediction fails on the unchanged raw response; the parser allows one recorded Markdown fence.',
            'reviewer_format_notes': 'Original string/list retained; nonempty notes are not counted as format failures because notes may state that JSON is valid.',
            'strict_supported_answer_and_citation_count': 'Strict answered output AND AI supported answer AND AI supported citations AND no attribution error. Descriptive only.',
            'paired_A_to_D': 'Transitions by named model, not anonymous slot; categories are not a total ranking and no inferential tests are run.',
            'by_AI_source_question_status': 'New AI source judgment, never original gold/source label; unsupported+strict refused describes this AI judgment only.',
        },
        'limitations': LIMITATIONS,
    }


def markdown(summary):
    a, d = summary['models']['A'], summary['models']['D']
    lines = ['# A/D 匿名 AI 审查汇总', '',
        '已验证三组全部 40 题、80 个响应后解盲。以下均为固定样本内的描述性计数；A 为原模型，D 为微调工作参照。', '',
        '| 维度 | 类别 | A | D |', '| --- | --- | ---: | ---: |']
    for field, title in [('answer_support', '答案支持'), ('citation_support', '引用支持'), ('attribution_error', '归属错误')]:
        for value in ENUMS[field]:
            lines.append(f'| {title} | {value} | {a[field][value]} | {d[field][value]} |')
    for field in ('answered', 'refused', 'invalid_output'):
        lines.append(f'| 严格原始输出解析 | {field} | {a["strict_state"][field]} | {d["strict_state"][field]} |')
    lines.append(f'| 联合条件 | 合法回答、答案/引用均支持、无归属错误 | {a["strict_supported_answer_and_citation_count"]} | {d["strict_supported_answer_and_citation_count"]} |')
    lines += ['', '严格解析格式失败与审查者对可见文字的判断分别统计；裸 false、截断或 false 带答案都不会修复为合法拒答。', '',
              '## 按本次 AI 对问题的判断分组', '',
              '| 来源问题判断 | 题数 | A 合法回答 / 合法拒答 / 格式失败 | D 合法回答 / 合法拒答 / 格式失败 |',
              '| --- | ---: | --- | --- |']
    for status in ENUMS['source_question_status']:
        left = a['by_AI_source_question_status'][status]
        right = d['by_AI_source_question_status'][status]
        a_counts = ' / '.join(str(left['strict_state'][k]) for k in ('answered', 'refused', 'invalid_output'))
        d_counts = ' / '.join(str(right['strict_state'][k]) for k in ('answered', 'refused', 'invalid_output'))
        lines.append(f'| {status} | {left["questions"]} | {a_counts} | {d_counts} |')
    lines += ['', '这里使用本次 AI 的来源判断，未修改或替代原数据标签；ambiguous 不并入不可回答。', '',
              '## 配对变化', '', '| 维度 | A → D | 题数 | 匿名题号 |', '| --- | --- | ---: | --- |']
    for metric, transitions in summary['paired_A_to_D'].items():
        for transition, record in transitions.items():
            if transition.split(' -> ')[0] != transition.split(' -> ')[1]:
                lines.append(f'| {metric} | {transition} | {record["count"]} | {", ".join(record["review_ids"])} |')
    lines += ['', '全部不变项及每题 A/D 对应关系见 JSON；类别没有统一好坏排序，不把每种变化都解释为改善。', '', '## 限制', '']
    lines.extend('- ' + value for value in summary['limitations'])
    validation = summary['validation']
    lines += ['', f'来源校验：{validation["paragraphs"]} 段、{validation["exact_quotes_validated"]} 条精确引文；输入、计划、公共包、三组审查与解盲键的 SHA-256 见 JSON。人工评分文件保持留空。', '']
    return '\n'.join(lines)


def build_summary(review_dir):
    validated = validate_public(review_dir)
    key = load_verified_key(review_dir, validated)
    result = summarize(validated, key)
    for name, original in validated['contents'].items():
        require((Path(review_dir)/name).read_bytes() == original, f'Input changed during summary: {name}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, default=ROOT/'reports/review-ad-v5')
    parser.add_argument('--validate-only', action='store_true', help='Public checks only; never open private files or write outputs')
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_public(args.review_dir)['validation'], ensure_ascii=False))
        return
    outputs = [args.review_dir/'ai-review-summary.json', args.review_dir/'ai-review-summary.md']
    if any(p.exists() for p in outputs):
        raise FileExistsError('Summary already exists; preserve it and use --validate-only for public rechecks')
    result = build_summary(args.review_dir)
    for path, content in zip(outputs, [json_bytes(result), markdown(result).encode('utf-8')]):
        with path.open('xb') as handle:
            handle.write(content)
    print(json.dumps({'status': 'completed', 'questions': 40, 'responses': 80,
                      'outputs': [str(p) for p in outputs]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
