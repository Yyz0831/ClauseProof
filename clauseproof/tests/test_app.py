"""Functional boundary checks; the test engine never appears in the live app."""
import copy
import http.client
import json
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import RequestError,load_catalog,prepare_request,interpret,digest
from server import Application,Handler,ThreadingHTTPServer

class ControlledEngine:
    status='ready';error=None;provenance={'adapter':'test-only'}
    def __init__(self,fail=False):
        self.entered=threading.Event();self.release=threading.Event();self.fail=fail;self.calls=0
    def prepare(self,rows):return rows
    def generate(self,item):
        self.calls+=1;self.entered.set()
        if not self.release.wait(3):raise TimeoutError('test gate')
        if self.fail and self.calls==2:raise RuntimeError('controlled failure')
        return '{"label":"NotMentioned","evidence_ids":[]}',{'generation_attempts':1}

def terminal(app,jid):
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        result=app.get(jid)
        if result['status']!='running':return result
        time.sleep(.005)
    raise AssertionError('worker did not finish')

class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.catalog=load_catalog()
        self.payload={'example_id':self.catalog['examples'][0]['id'],'hypothesis_ids':['nda-1','nda-2']}
    def test_live_corpus_has_no_gold_annotations(self):
        for d in self.catalog['examples']:
            self.assertFalse({'annotations','annotation_sets','choice','evidence_ids'}&set(d))
        self.assertEqual(len(self.catalog['hypotheses']),17)
    def test_unicode_snapshot_and_context_preserve_source(self):
        text='\n  NDA 📝 — confidential information.\n\n  Second clause with conditions.\nThird clause.\n '
        doc,rows=prepare_request({'text':text,'hypothesis_ids':['nda-3']},self.catalog)
        self.assertEqual(doc['text'],text)
        self.assertEqual(doc['source_sha256'],digest(text))
        for p in doc['paragraphs']:
            self.assertEqual(text[p['start']:p['end']],p['text'])
        import re
        self.assertEqual(re.sub(r'\n\[s\d+\] ','',rows[0]['context']),text)
    def test_reject_invalid_requests(self):
        bad=[{},[],{**self.payload,'hypothesis_ids':[]},{**self.payload,'hypothesis_ids':['nda-1','nda-1']},
             {**self.payload,'hypothesis_ids':['nda-999']},{**self.payload,'model':'other'},
             {**self.payload,'text':'ambiguous'},{'text':'short','hypothesis_ids':['nda-1']},
             {'text':'x'*160001,'hypothesis_ids':['nda-1']},{'text':'line\n'*1001,'hypothesis_ids':['nda-1']}]
        for payload in bad:
            with self.subTest(payload=str(payload)[:90]):
                with self.assertRaises(RequestError):prepare_request(payload,self.catalog)
    def test_raw_format_failures_are_not_repaired(self):
        pars=[{'id':'s0000','text':'A clause'}]
        for raw in ['```json\n{"label":"NotMentioned","evidence_ids":[]}\n```',
                    '{"label":"NotMentioned","evidence_ids":["s0000"]}',
                    '{"label":"Entailment","evidence_ids":["s9999"]}',
                    '{"label":"Entailment","label":"NotMentioned","evidence_ids":[]}']:
            result=interpret(raw,pars)
            self.assertEqual(result['raw_output'],raw);self.assertEqual(result['status'],'invalid_output')
            self.assertIsNone(result['prediction']);self.assertEqual(result['citations'],[])
    def test_cancel_concurrency_and_snapshot(self):
        engine=ControlledEngine();app=Application(engine);payload=copy.deepcopy(self.payload)
        job=app.start(payload);self.assertTrue(engine.entered.wait(1))
        with self.assertRaises(RequestError) as ctx:app.start(payload)
        self.assertEqual(ctx.exception.status,409)
        payload['hypothesis_ids'].clear();job['document']['text']='mutated copy'
        app.cancel(job['id']);engine.release.set();result=terminal(app,job['id'])
        self.assertEqual(result['status'],'cancelled');self.assertEqual(result['completed'],1)
        self.assertNotEqual(result['document']['text'],'mutated copy')
        self.assertEqual(len(result['document']['hypotheses']),2)
    def test_failure_retains_completed_outputs(self):
        engine=ControlledEngine(fail=True);engine.release.set();app=Application(engine)
        result=terminal(app,app.start(self.payload)['id'])
        self.assertEqual(result['status'],'failed');self.assertEqual(result['completed'],1)
        self.assertEqual(result['error']['type'],'RuntimeError');self.assertEqual(engine.calls,2)
    def test_expired_job_not_accessible(self):
        engine=ControlledEngine();engine.release.set();app=Application(engine)
        jid=app.start(self.payload)['id'];terminal(app,jid)
        app.jobs[jid]['started_monotonic']-=1801
        with self.assertRaises(RequestError) as ctx:app.get(jid)
        self.assertEqual(ctx.exception.status,410)
    def test_retention_cleanup_removes_stale_documents(self):
        engine=ControlledEngine();engine.release.set();app=Application(engine)
        jid=app.start(self.payload)['id'];terminal(app,jid)
        app.jobs[jid]['started_monotonic']-=1801;app.expire()
        self.assertNotIn(jid,app.jobs)
    def test_saved_case_arithmetic_matches_metrics(self):
        data=json.loads((Path(__file__).resolve().parents[1]/'assets/experiments.json').read_text())
        self.assertEqual(len(data['cases']),306)
        self.assertEqual(len({c['document_id'] for c in data['cases']}),18)
        for arm,m in data['modes']['strict'].items():
            n=sum(c['outputs'][arm]['label_correct'] for c in data['cases'])
            j=sum(c['outputs'][arm]['joint_exact'] for c in data['cases'])
            self.assertAlmostEqual(n/306,m['label_accuracy'])
            self.assertAlmostEqual(j/306,m['joint_exact_success_rate'])
    def test_http_guards_and_routes(self):
        engine=ControlledEngine();server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        port=server.server_address[1];host=f'127.0.0.1:{port}'
        server.allowed_hosts={host};server.app=Application(engine)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        def request(path,method='GET',body=None,headers=None):
            c=http.client.HTTPConnection('127.0.0.1',port,timeout=2)
            c.request(method,path,body=body,headers=headers or {});r=c.getresponse();status=r.status;content=r.read();c.close();return status,content
        try:
            self.assertEqual(request('/')[0],200)
            self.assertEqual(request('/experiments')[0],200)
            self.assertEqual(request('/../../core.py')[0],404)
            self.assertEqual(request('/api/health',headers={'Host':'evil.example'})[0],403)
            self.assertEqual(request('/api/jobs','POST','{}',{'Origin':'https://evil.example','Content-Type':'application/json'})[0],403)
            self.assertEqual(request('/api/jobs','POST','{}',{'Content-Type':'text/plain'})[0],415)
            self.assertEqual(request('/api/jobs','POST','{',{'Content-Type':'application/json'})[0],400)
            self.assertEqual(request('/api/jobs/no-such-id')[0],404)
        finally:server.shutdown();server.server_close();worker.join()

if __name__=='__main__':unittest.main(verbosity=2)
