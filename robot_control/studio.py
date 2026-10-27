"""Loopback model browser with isolated runtime workers and serialized GPU jobs."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, unquote
from . import model_library as lib

JOBS=lib.ROOT/'runs/studio/jobs'
GPU=threading.Lock()
POOL=concurrent.futures.ThreadPoolExecutor(max_workers=2)
GPU_POOL=concurrent.futures.ThreadPoolExecutor(max_workers=1)
PROCESSES={}
STOP=threading.Event()
WORKER=r'''
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
root,runtime,jobfile=sys.argv[1:]
j=json.loads(Path(jobfile).read_text())
spec=importlib.util.spec_from_file_location('visualizer',Path(root)/'scripts/visualize.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
sys.path.insert(0,runtime)
if runtime!=root:
 import robot_control.runtime.g1_env as ge
 ge.ROOT=Path(root)
import torch,hashlib
torch.set_num_threads(1)
if runtime==root:
 cp=torch.load(j['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
 for name,expected in cp.get('source_sha256',{}).items():
  path=Path(root)/name
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('模型运行代码已变化：'+name)
 del cp
if j['kind']=='render':
 v.render(SimpleNamespace(checkpoint=Path(j['checkpoint']),out=Path(j['output']),motion=[j['motion']],device=j['device'],seconds=j['seconds'],width=j['width'],height=j['height']))
else:
 if runtime==root:
  spec=importlib.util.spec_from_file_location('evaluator',Path(root)/'scripts/evaluate.py')
  ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
  ev.run_single(['--checkpoint',j['checkpoint'],'--out',str(Path(j['output'])/'evaluation.json'),'--device',j['device']])
 else:
  from robot_control.runtime.evaluation_worker import load,evaluate
  cp,env,actor=load(Path(j['checkpoint']))
  if j['device'].startswith('cuda'):
   from robot_control.runtime.gpu_evaluation import evaluate
   actor=actor.to(j['device']);result=evaluate(env,actor,device=j['device'])
  else:result=evaluate(env,actor)
  (Path(j['output'])/'evaluation.json').write_text(json.dumps(result))
'''

def training_active():
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            c=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if ('scripts/train.py ' in c or 'robot_control.runtime.train ' in c) and 'python' in c:return True
        except (OSError,ProcessLookupError):pass
    return False

def jobs():
    values=[lib.read(p) for p in JOBS.glob('*/job.json')]
    for j in values:
        j['progress']=lib.read(JOBS/j['id']/'output/progress.json')
    return sorted(values,key=lambda x:x['created_at'],reverse=True)

def update(j,**kw):
    j.update(kw);lib.atomic(JOBS/j['id']/'job.json',j)

def execute(j):
    lock=GPU if j['device'].startswith('cuda') or j['kind']=='render' else threading.Lock()
    try:
        with lock:
            while (j['device'].startswith('cuda') or j['kind']=='render') and training_active():
                update(j,status='waiting_training')
                if STOP.wait(5):raise RuntimeError('服务已停止')
            m=lib.get_model(j['model_id'])
            if lib.digest(lib.policy_path(m))!=m['sha256']:raise ValueError('权重校验失败')
            update(j,status='running',started_at=lib.now())
            with (JOBS/j['id']/'worker.log').open('w') as log:
                proc=subprocess.Popen([sys.executable,'-u','-c',WORKER,str(lib.ROOT),str(lib.runtime(m)),str(JOBS/j['id']/'job.json')],cwd=lib.ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'MUJOCO_GL':'egl','OMP_NUM_THREADS':'1'})
                PROCESSES[j['id']]=proc
                update(j,pid=proc.pid)
                code=proc.wait();PROCESSES.pop(j['id'],None)
                if code:raise RuntimeError('工作进程失败，请查看任务日志')
            if j['kind']=='evaluate':
                result=lib.read(Path(j['output'])/'evaluation.json')
                lib.record_evaluation(m['id'],result,j['device'],'full')
                update(j,summary=lib.summary(result))
            else:
                report=lib.read(Path(j['output'])/'render_report.json')
                if report['sha256']!=m['sha256']:raise ValueError('视频权重指纹不匹配')
                update(j,result=report['results'][0],video=f"/media/{j['id']}/{j['motion']}.mp4",poster=f"/media/{j['id']}/{j['motion']}_poster.jpg")
            update(j,status='complete',finished_at=lib.now())
    except Exception as e:update(j,status='failed',error=str(e),finished_at=lib.now())

CREATE_LOCK=threading.Lock()
def create(body):
    m=lib.get_model(body.get('model_id'));kind=body.get('kind','render');device=body.get('device','cuda:0')
    if kind not in ('render','evaluate') or device not in ('cpu','cuda:0'):raise ValueError('无效任务类型或设备')
    motion=body.get('motion','dance1_subject2');seconds=float(body.get('seconds',20));height=int(body.get('height',720))
    motions=lib.catalog()['motions'][m['robot']]
    match=next((x for x in motions if x['id']==motion),None)
    if kind=='render' and not match:raise ValueError('动作不存在')
    if not 0<=seconds<=3600 or height not in (720,1080):raise ValueError('时长或清晰度无效')
    if seconds==0 and match:seconds=match['seconds']
    key=hashlib.sha256(json.dumps([m['sha256'],kind,device,motion,seconds,height]).encode()).hexdigest()
    with CREATE_LOCK:
        for old in jobs():
            if old['key']==key and old['status'] not in ('failed','interrupted'):return old
        jid=uuid.uuid4().hex[:16];folder=JOBS/jid;folder.mkdir(parents=True);(folder/'output').mkdir()
        j=dict(id=jid,key=key,model_id=m['id'],robot=m['robot'],kind=kind,device=device,motion=motion,seconds=seconds,height=height,width=height*16//9,status='queued',created_at=lib.now(),checkpoint=str(lib.policy_path(m)),output=str(folder/'output'))
        update(j)
        pool=GPU_POOL if device.startswith('cuda') or kind=='render' else POOL
        pool.submit(execute,j);return j

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def json(self,value,status=200):
        data=json.dumps(value,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def local_host(self):
        return urlsplit('http://'+self.headers.get('Host','')).hostname in ('127.0.0.1','localhost')
    def do_POST(self):
        if not self.local_host():return self.json({'error':'只允许本机访问'},403)
        try:
            origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=self.headers.get('Host'):raise ValueError('只允许同源请求')
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<8192:raise ValueError('请求大小无效')
            b=json.loads(self.rfile.read(size));p=urlsplit(self.path).path
            if p=='/api/jobs':return self.json(create(b),202)
            if p=='/api/scan':return self.json(lib.import_candidates())
            if p.startswith('/api/models/') and p.endswith('/approval'):
                if not isinstance(b.get('approved'),bool):raise ValueError('必须明确选择确认状态')
                return self.json(lib.approve(p.split('/')[3],b['approved'],b.get('note','')))
            self.json({'error':'接口不存在'},404)
        except (ValueError,KeyError,TypeError) as e:self.json({'error':str(e)},400)
        except Exception as e:self.json({'error':str(e)},500)
    def do_GET(self):
        if not self.local_host():return self.json({'error':'只允许本机访问'},403)
        try:
            p=unquote(urlsplit(self.path).path)
            if p=='/api/catalog':return self.json(lib.catalog())
            if p=='/api/jobs':return self.json(jobs())
            if p=='/api/status':return self.json(dict(training_active=training_active(),training=lib.read(lib.ROOT/'runs/mojia/report.json',{}).get('status'),gpu_serial=True,cpu_workers=2))
            if p.startswith('/api/jobs/'):
                j=next((x for x in jobs() if x['id']==p.split('/')[-1]),None)
                return self.json(j or {'error':'任务不存在'},200 if j else 404)
            if p.startswith('/media/'):
                parts=p.split('/');base=JOBS/parts[2];path=base/('worker.log' if parts[-1]=='worker.log' else 'output/'+parts[-1]);allowed=JOBS.resolve()
            else:
                allowed=(lib.ROOT/'web/studio').resolve();path=allowed/('index.html' if p=='/' else p.lstrip('/'))
            path=path.resolve()
            if not path.is_relative_to(allowed) or not path.is_file():return self.json({'error':'文件不存在'},404)
            import mimetypes,re
            size=path.stat().st_size;start=0;end=size-1;partial=False
            if self.headers.get('Range'):
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',self.headers['Range'])
                if not match:return self.json({'error':'无效 Range'},416)
                a,b=match.groups()
                if a:start=int(a);end=min(int(b) if b else end,end)
                elif b:start=max(0,size-int(b))
                if start>end or start>=size:return self.json({'error':'超出文件范围'},416)
                partial=True
            self.send_response(206 if partial else 200);self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1));self.send_header('X-Content-Type-Options','nosniff')
            if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            with path.open('rb') as f:
                f.seek(start);remaining=end-start+1
                while remaining:
                    block=f.read(min(1024*1024,remaining))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:self.json({'error':str(e)},500)

def serve(port=8765):
    JOBS.mkdir(parents=True,exist_ok=True)
    if not lib.registry()['models']:
        lib.import_candidates()
    for j in jobs():
        if j['status'] in ('running','queued','waiting_training'):update(j,status='interrupted',error='服务曾中断，请重新提交')
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'机器人模型工作台 http://127.0.0.1:{port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        STOP.set();server.server_close()
        for proc in list(PROCESSES.values()):
            if proc.poll() is None:proc.terminate()
        POOL.shutdown(wait=False,cancel_futures=True);GPU_POOL.shutdown(wait=False,cancel_futures=True)
