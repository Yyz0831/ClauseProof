"""Validation, immutable source snapshots and strict output handling."""
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'scripts'))
from contractnli_common import messages, parse_prediction, source_row, MODEL, REVISION, INSTRUCTION

MAX_BODY_BYTES=800_000
MAX_TEXT_CHARS=160_000
MAX_SPANS=1000
ADAPTER_SHA='9ca2062d142aefbaee856056e8ead60033a5954f670e1e7d69ff33bd6b2445b0'

class RequestError(Exception):
    def __init__(self,code,message,status=400,**details):
        super().__init__(message)
        self.code,self.message,self.status,self.details=code,message,status,details

def digest(text): return hashlib.sha256(text.encode()).hexdigest()
def load_catalog(): return json.loads((HERE/'assets/catalog.json').read_text())

def prepare_request(payload,catalog):
    if not isinstance(payload,dict) or set(payload)-{'example_id','text','title','hypothesis_ids'}:
        raise RequestError('invalid_fields','The request contains unsupported fields.')
    chosen=payload.get('hypothesis_ids')
    hypotheses={h['id']:h for h in catalog['hypotheses']}
    if not isinstance(chosen,list) or not chosen or len(chosen)>17 or not all(isinstance(x,str) for x in chosen):
        raise RequestError('invalid_selection','Select between 1 and 17 statements to check.')
    if len(set(chosen))!=len(chosen) or not set(chosen)<=set(hypotheses):
        raise RequestError('invalid_selection','The statement selection is invalid.')
    if 'example_id' in payload:
        if 'text' in payload: raise RequestError('ambiguous_document','Choose one document source.')
        if not isinstance(payload['example_id'],str):raise RequestError('invalid_example','Unknown sample document.')
        example=next((d for d in catalog['examples'] if d['id']==payload['example_id']),None)
        if not example:raise RequestError('invalid_example','Unknown sample document.')
        text=example['text'];spans=example['spans'];title=example['title']
        origin={'kind':'official_example','dataset':'ContractNLI','split':'dev','document_id':example['id'],
            'span_policy':'Official ContractNLI spans','note':example['note']}
    else:
        text=payload.get('text');title=payload.get('title') or 'Untitled agreement'
        if not isinstance(text,str) or not isinstance(title,str):raise RequestError('invalid_text','Provide contract text and a document title.')
        if len(title)>180:raise RequestError('title_too_long','The title must be at most 180 characters.')
        if len(text)>MAX_TEXT_CHARS:raise RequestError('document_too_large','This document exceeds the text limit. No text was truncated.',413)
        if '\x00' in text+title:raise RequestError('invalid_character','The document contains an unsupported null character.')
        text=text.replace('\r\n','\n').replace('\r','\n')
        if len(text.strip())<30:raise RequestError('empty_document','Paste at least 30 characters of contract text.')
        spans=[(m.start(),m.end()) for m in re.finditer(r'[^\n]+',text) if m.group().strip()]
        if len(spans)>MAX_SPANS:raise RequestError('too_many_spans','The document has too many text lines. No text was truncated.',413)
        origin={'kind':'user_text','span_policy':'Each non-empty line is a citation span; no semantic clause parser.',
            'note':'Custom text is outside the saved benchmark. English plain text only; document completeness is not verified.'}
    doc={'id':digest(text)[:16],'text':text,'spans':spans}
    selected=[h for h in catalog['hypotheses'] if h['id'] in chosen]
    records=[source_row(doc,h['id'],h['hypothesis']) for h in selected]
    source_spans=[{**p,'start':span[0],'end':span[1]}
                  for p,span in zip(records[0]['paragraphs'],spans)]
    snapshot={'title':title,'text':text,'source_sha256':digest(text),'origin':origin,
        'paragraphs':source_spans,'hypotheses':selected}
    return snapshot,records

def interpret(raw,paragraphs):
    parsed,error,_=parse_prediction(raw,{p['id'] for p in paragraphs})
    by_id={p['id']:p for p in paragraphs}
    return {'status':'invalid_output' if error else 'valid','prediction':parsed,
        'raw_output':raw,'parse_error':error,
        'citations':[by_id[x] for x in parsed['evidence_ids']] if parsed else []}
