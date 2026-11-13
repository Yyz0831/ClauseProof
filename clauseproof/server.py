"""Local ClauseProof application. Contract content and job results stay in RAM."""
import argparse
import copy
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import uuid
from core import HERE,ROOT,MAX_BODY_BYTES,RequestError,load_catalog,prepare_request,interpret
from engine import AyaEngine

def now():return datetime.now(timezone.utc).isoformat()

class Application:
    def __init__(self,engine):
        self.engine=engine;self.catalog=load_catalog()
        self.jobs={};self.lock=threading.Lock();self.gpu_slot=threading.Lock()
    def health(self):
        return {'status':self.engine.status,'model':'Aya Expanse 8B','adapter':'ContractNLI v2',
            'busy':self.gpu_slot.locked(),'error':self.engine.error,'provenance':self.engine.provenance,
            'max_input_tokens':7936,'max_new_tokens':256,'document_logging':False,
            'results_retention_minutes':30,'scope':'Local prototype · English NDA statement checking'}
    def expire(self):
        with self.lock:
            cutoff=time.monotonic()-1800
            self.jobs={k:v for k,v in self.jobs.items() if v['status']=='running' or v['started_monotonic']>=cutoff}
    def start(self,payload):
        snapshot,rows=prepare_request(payload,self.catalog)
        if not self.gpu_slot.acquire(blocking=False):raise RequestError('busy','Another review is running. Wait for it to finish or stop it first.',409)
        try:
            prepared=self.engine.prepare(rows)
            jid=uuid.uuid4().hex
            job={'id':jid,'status':'running','created_at_utc':now(),'started_monotonic':time.monotonic(),
                 'document':snapshot,'total':len(rows),'completed':0,'results':[],
                 'model':copy.deepcopy(self.engine.provenance),'cancel_requested':False,'error':None}
            with self.lock:
                cutoff=time.monotonic()-1800
                self.jobs={k:v for k,v in self.jobs.items() if v['started_monotonic']>=cutoff or v['status']=='running'}
                while len(self.jobs)>=5:
                    key=next((k for k,v in self.jobs.items() if v['status']!='running'),None)
                    if key is None:raise RequestError('busy','The review queue is full.',409)
                    del self.jobs[key]
                self.jobs[jid]=job
            threading.Thread(target=self._work,args=(jid,rows,prepared),daemon=True).start()
            return self.get(jid)
        except Exception:
            self.gpu_slot.release();raise
    def _work(self,jid,rows,prepared):
        try:
            for row,item in zip(rows,prepared):
                with self.lock:
                    if self.jobs[jid]['cancel_requested']:break
                raw,diagnostics=self.engine.generate(item)
                result={'hypothesis_id':row['id'].rsplit(':',1)[-1],'hypothesis':row['question'],
                    **interpret(raw,row['paragraphs']),'diagnostics':diagnostics}
                with self.lock:
                    self.jobs[jid]['results'].append(result);self.jobs[jid]['completed']+=1
            with self.lock:
                j=self.jobs[jid]
                j['status']='completed' if j['completed']==j['total'] else 'cancelled'
        except Exception as exc:
            with self.lock:
                self.jobs[jid]['status']='failed'
                self.jobs[jid]['error']={'code':'generation_failed','message':'Generation failed. Completed results are preserved; no automatic retry.', 'type':type(exc).__name__}
        finally:
            with self.lock:
                j=self.jobs[jid];j['ended_at_utc']=now();j['elapsed_seconds']=round(time.monotonic()-j['started_monotonic'],2)
            self.gpu_slot.release()
    def get(self,jid):
        with self.lock:
            j=self.jobs.get(jid)
            if not j:raise RequestError('job_not_found','This review is unavailable or has expired. Run a new review.',404)
            if j['status']!='running' and time.monotonic()-j['started_monotonic']>1800:
                del self.jobs[jid];raise RequestError('job_expired','This review has expired after 30 minutes.',410)
            return copy.deepcopy({k:v for k,v in j.items() if k!='started_monotonic'})
    def cancel(self,jid):
        with self.lock:
            if jid not in self.jobs:raise RequestError('job_not_found','Review not found.',404)
            if self.jobs[jid]['status']=='running':self.jobs[jid]['cancel_requested']=True
        return self.get(jid)

class Handler(BaseHTTPRequestHandler):
    server_version='ClauseProof/0.1'
    def log_message(self,*args):pass
    def reply(self,status,body,kind='application/json; charset=utf-8'):
        if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
    def failure(self,e):self.reply(e.status,{'error':{'code':e.code,'message':e.message,**e.details}})
    def guard_host(self):
        host=self.headers.get('Host','')
        if host not in self.server.allowed_hosts:raise RequestError('invalid_host','This prototype accepts local access only.',403)
        origin=self.headers.get('Origin')
        if origin and origin!='http://'+host:raise RequestError('invalid_origin','Cross-origin requests are not allowed.',403)
    def do_GET(self):
        try:
            self.guard_host();path=urlsplit(self.path).path;app=self.server.app
            static={'/':'index.html','/experiments':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path in static:
                p=HERE/'web'/static[path]
                kind={'html':'text/html','js':'application/javascript','css':'text/css'}[p.suffix[1:]]+'; charset=utf-8'
                self.reply(200,p.read_bytes(),kind)
            elif path=='/api/health':self.reply(200,app.health())
            elif path=='/api/catalog':
                self.reply(200,{'hypotheses':app.catalog['hypotheses'],'examples':[
                    {k:v for k,v in d.items() if k not in ['text','spans']} for d in app.catalog['examples']]})
            elif path.startswith('/api/examples/'):
                ex=next((x for x in app.catalog['examples'] if x['id']==path.split('/')[-1]),None)
                if not ex:raise RequestError('not_found','Sample not found.',404)
                self.reply(200,ex)
            elif path=='/api/experiments':self.reply(200,(HERE/'assets/experiments.json').read_bytes())
            elif path.startswith('/api/jobs/'):
                self.reply(200,app.get(path.split('/')[-1]))
            else:raise RequestError('not_found','Page not found.',404)
        except RequestError as e:self.failure(e)
    def do_POST(self):
        try:
            self.guard_host()
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise RequestError('content_type','Send JSON.',415)
            try:size=int(self.headers.get('Content-Length','0'))
            except ValueError:raise RequestError('invalid_length','Invalid request length.')
            if size<=0 or size>MAX_BODY_BYTES:raise RequestError('body_size','Request is empty or too large.',413)
            self.connection.settimeout(30)
            try:payload=json.loads(self.rfile.read(size))
            except (ValueError,UnicodeDecodeError):raise RequestError('invalid_json','Invalid request JSON.')
            path=urlsplit(self.path).path
            if path=='/api/jobs':self.reply(202,self.server.app.start(payload))
            elif path.startswith('/api/jobs/') and path.endswith('/cancel'):
                self.reply(200,self.server.app.cancel(path.split('/')[-2]))
            else:raise RequestError('not_found','Endpoint not found.',404)
        except RequestError as e:self.failure(e)
        except (TimeoutError,ConnectionError):pass
        except Exception:self.reply(500,{'error':{'code':'server_error','message':'The request could not be processed.'}})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8791)
    ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--no-model',action='store_true')
    args=ap.parse_args()
    engine=AyaEngine(args.run_dir)
    if args.no_model:engine.status='offline'
    else:threading.Thread(target=engine.load,daemon=True).start()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.allowed_hosts={f'127.0.0.1:{args.port}',f'localhost:{args.port}'};server.app=Application(engine)
    stop=threading.Event()
    def expire_results():
        while not stop.wait(30):server.app.expire()
    threading.Thread(target=expire_results,daemon=True).start()
    print(json.dumps({'status':'serving','url':f'http://127.0.0.1:{args.port}','document_logging':False}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:stop.set();server.server_close()

if __name__=='__main__':main()
