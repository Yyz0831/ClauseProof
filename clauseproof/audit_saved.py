"""Run AuditKIT on frozen ContractNLI outputs, without inference or an AI judge."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from contractnli_common import parse_prediction,file_sha,read_rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import auditkit as ak
    from auditkit.metric import Metric
    from auditkit.sample import Sample
    from auditkit.score import Score
    from auditkit.types import Direction
    inputs=read_rows(ROOT/'data/experiments/contractnli-v3/inputs.jsonl')
    labels=read_rows(ROOT/'data/experiments/contractnli-v3/labels.jsonl')
    protocol=json.loads((ROOT/'data/experiments/contractnli-v3/protocol.json').read_text())
    assert file_sha(ROOT/'data/experiments/contractnli-v3/inputs.jsonl')==protocol['input_sha256']
    assert file_sha(ROOT/'data/experiments/contractnli-v3/labels.jsonl')==protocol['labels_sha256']
    assert [r['id'] for r in inputs]==[r['id'] for r in labels]==protocol['question_ids']
    assert len(inputs)==306
    source={r['id']:r for r in inputs};gold={r['id']:r for r in labels}
    expected=json.loads((ROOT/'reports/contractnli-v3/metrics.json').read_text())['arms']
    mapping={'strict_format':'format_valid_rate','strict_label':'label_accuracy','strict_joint':'joint_exact_success_rate'}
    class ContractMetric(Metric):
        direction=Direction.MAXIMIZE
        def __init__(self,name):self.name=name
        def identity(self):return {'name':self.name,'implementation_sha256':file_sha(__file__),
            'parser_sha256':file_sha(ROOT/'scripts/contractnli_common.py')}
        def score(self,sample,output,context=None):
            p,error,_=parse_prediction(output,set(sample.metadata['span_ids']))
            g=sample.metadata['reference']
            correct=bool(p and p['label']==g['label'])
            exact=bool(correct and set(p['evidence_ids'])==set(g['evidence_ids']))
            value={'strict_format':error is None,'strict_label':correct,'strict_joint':exact}[self.name]
            return Score(name=self.name,value=float(value),direction=self.direction,metadata={'parse_error':error})
    summary={'scope':'306 saved development decisions across 18 contracts; no new predictions or semantic review.',
        'auditkit_version':importlib.metadata.version('auditkit'),
        'auditkit_evaluate_source_sha256':file_sha(inspect.getfile(ak.evaluate)),
        'adapter_training_backend':'AlignTune TRLSFTTrainer',
        'training_manifest_sha256':file_sha(ROOT/'runs/train-contractnli-v2/expanse8b_contractnli_v2_4bit/manifest.json'),
        'input_sha256':protocol['input_sha256'],'labels_sha256':protocol['labels_sha256'],'arms':{}}
    for arm in ['base','v1','v2','v3']:
        folder=ROOT/f'runs/contractnli-v3/{arm}'
        manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['inference_complete'] and manifest['status']=='completed'
        assert manifest['input_sha256']==protocol['input_sha256']
        assert manifest['predictions_sha256']==file_sha(folder/'predictions.jsonl')
        records=read_rows(folder/'predictions.jsonl')
        assert len(records)==306 and {r['id'] for r in records}==set(source)
        samples=[Sample(id=r['id'],input=source[r['id']]['question'],target=json.dumps(gold[r['id']]),
            actual_output=r['raw_output'],metadata={'reference':gold[r['id']],
                'span_ids':[p['id'] for p in source[r['id']]['paragraphs']]}) for r in records]
        result=ak.evaluate(samples,model='precomputed',scorers=[ContractMetric(k) for k in mapping])
        result.metadata.update({'arm':arm,'source_manifest_sha256':file_sha(folder/'manifest.json'),
            'prediction_sha256':manifest['predictions_sha256'],'scope':summary['scope'],
            'note':'These three metrics validate format/reference parity only. Timing is scoring latency, not inference.'})
        data=result.to_dict()
        assert not data['errors'],data['errors']
        stats={k:data['stats'][k]['mean'] for k in mapping}
        for key,original_key in mapping.items():assert abs(stats[key]-expected[arm][original_key])<1e-12
        result.save(str(args.output/f'{arm}-auditkit.json'))
        summary['arms'][arm]={'fingerprint':result.fingerprint,'stats':stats,'errors':len(data['errors']),
            'prediction_sha256':manifest['predictions_sha256'],'parity':'passed'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
