"""Build a source-only demo corpus and a clearly separate saved experiment view."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'assets'

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path): return [json.loads(x) for x in path.read_text().splitlines()]
def save(name, data):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')

def main():
    rawpath = ROOT/'data/raw/external/contract-nli/dev.json'
    raw = read(rawpath)
    protocol = read(ROOT/'data/experiments/contractnli-v3/protocol.json')
    docs = {str(d['id']):d for d in raw['documents']}
    ids = sorted(map(str, protocol['contracts']), key=int)
    labels = [{'id':k, 'title':v['short_description'], 'hypothesis':v['hypothesis']}
              for k,v in sorted(raw['labels'].items(), key=lambda kv:int(kv[0].split('-')[1]))]
    # First four numeric IDs, chosen without inspecting model correctness.
    examples = []
    for n, did in enumerate(ids[:4], 1):
        d=docs[did]
        examples.append({'id':did, 'title':f'Sample NDA {n:02d}', 'text':d['text'],
            'spans':d['spans'], 'source':'ContractNLI official development split',
            'source_document_id':did, 'source_url':d.get('url'),
            'source_sha256':hashlib.sha256(d['text'].encode()).hexdigest(),
            'note':'Previously used development document. This live demo is not a new evaluation.'})
    assert not any('annotation_sets' in d for d in examples)
    save('catalog.json', {'hypotheses':labels, 'examples':examples,
        'example_selection':'First four numeric document IDs from the frozen contractnli-v3 dev cohort; no correctness-based selection.',
        'source_sha256':sha(rawpath)})

    report=ROOT/'reports/contractnli-v3'
    strict=read(report/'metrics.json'); content=read(report/'content-diagnostics.json')
    assert strict['examples']==306 and strict['contracts']==18
    data={'experiment':'ContractNLI · shared development cohort',
        'examples':306, 'contracts':18, 'statements_per_contract':17,
        'evidence_questions':176, 'labels':{'Entailment':150,'Contradiction':26,'NotMentioned':130},
        'selected_model':'v2', 'model':'CohereLabs/aya-expanse-8b',
        'modes':{'strict':strict['arms'],'content':content['arms']},
        'models':{},'cases':[],
        'limitations':[
            '18 development contracts, 306 statement decisions. Questions within a contract are correlated.',
            'All these documents have been used for model selection. This is not a held-out test score.',
            'Evidence F1 measures overlap with official span IDs, not whether the citation is semantically sufficient.',
            'One training seed; data size and composition changed together. These are whole-recipe comparisons.',
            'The v2/v3 source review covered 18 selected pairs once, by AI, not a legal expert or independent judge.',
            'The live paste-text feature uses generated span boundaries. Its accuracy has not been established by this benchmark.'
        ],
        'source_files':{f'reports/contractnli-v3/{n}':sha(report/n) for n in ['metrics.json','content-diagnostics.json','targeted-diagnostics.json','single-review.json']}}
    for arm in ['base','v1','v2','v3']:
        if arm=='base':
            data['models'][arm]={'title':'Original Aya','training_examples':0,'training_steps':0,'note':'No task adapter'}
        else:
            names={'v1':'expanse8b_contractnli_4bit','v2':'expanse8b_contractnli_v2_4bit','v3':'expanse8b_contractnli_v3_4bit'}
            m=read(ROOT/f'runs/train-contractnli-{arm}'/names[arm]/'manifest.json')
            data['models'][arm]={'title':f'Aya {arm}', 'training_examples':m['selected_examples'],
                'training_steps':m['actual_final_global_step'], 'adapter_sha256':m['adapter_sha256'],
                'note':{'v1':'Initial balanced subset','v2':'Expanded missing-information and evidence coverage','v3':'More explicit contradiction contrasts'}[arm]}
    inp=rows(ROOT/'data/experiments/contractnli-v3/inputs.jsonl')
    gold={r['id']:r for r in rows(ROOT/'data/experiments/contractnli-v3/labels.jsonl')}
    preds={arm:{r['id']:r for r in rows(ROOT/f'runs/contractnli-v3/{arm}/predictions.jsonl')} for arm in data['models']}
    for row in inp:
        g=gold[row['id']]
        case={'id':row['id'],'document_id':row['paper_id'],'hypothesis':row['question'],
            'reference':{'label':g['label'],'evidence_ids':g['evidence_ids']},'outputs':{}}
        referenced=set(g['evidence_ids'])
        for arm, records in preds.items():
            r=records[row['id']];pr=r['prediction']
            case['outputs'][arm]={'raw':r['raw_output'],'prediction':pr,'format_valid':not r['parse_error'],
                'parse_error':r['parse_error'],'label_correct':bool(pr and pr['label']==g['label']),
                'joint_exact':bool(pr and pr['label']==g['label'] and set(pr['evidence_ids'])==set(g['evidence_ids']))}
            if pr: referenced.update(pr['evidence_ids'])
        case['source_spans']=[p for p in row['paragraphs'] if p['id'] in referenced]
        data['cases'].append(case)
    save('experiments.json',data)
    print(json.dumps({'examples':len(examples),'hypotheses':len(labels),'saved_cases':len(data['cases'])}))

if __name__=='__main__': main()
