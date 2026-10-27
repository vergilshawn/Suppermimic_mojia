"""Immutable local policy snapshots, evaluation provenance and explicit user approval."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'checkpoints/library'
LOCK=threading.RLock()

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def registry(): return read(BASE/'registry.json',dict(version=1,models=[]))

def save(value): atomic(BASE/'registry.json',value)

def get_model(model_id):
    model=next((m for m in registry()['models'] if m['id']==model_id),None)
    if not model:raise ValueError('模型不存在')
    return model

def policy_path(model):
    path=(BASE/model['file']).resolve()
    if not path.is_relative_to(BASE.resolve()) or not path.is_file():raise ValueError('模型文件缺失')
    return path

def summary(record):
    e=record.get('evaluation',record);rows=e.get('motions',[])
    full=bool(e.get('evaluation_is_full',e.get('full_library',False)))
    if not rows:return None
    import statistics
    root=[x.get('root_position_rmse_m',x.get('root_rms_distance_m',x.get('root_rmse_m'))) for x in rows]
    return dict(full=full,clips=len(rows),completed=sum(bool(x.get('completed')) for x in rows),completion_rate=e.get('completion_rate'),normalized_return=e.get('normalized_return'),mean_joint_rmse_rad=statistics.mean(x['joint_rmse_rad'] for x in rows),median_root_rmse_m=statistics.median(x for x in root if x is not None),motions=rows)

def runtime(model):
    if model['robot']=='mojia':return ROOT
    path=BASE/'runtimes'/('g1-'+model['runtime_commit'][:12])
    if not (path/'robot_control/runtime/train.py').is_file():raise ValueError('G1 隔离运行环境缺失；重新扫描模型库')
    return path

def ensure_g1_runtime():
    sha=subprocess.check_output(['git','rev-parse','g1'],cwd=ROOT,text=True).strip()
    target=BASE/'runtimes'/('g1-'+sha[:12])
    if not (target/'robot_control/runtime/train.py').is_file():
        target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryFile() as f:
            subprocess.run(['git','archive',sha,'robot_control'],cwd=ROOT,stdout=f,check=True);f.seek(0)
            with tarfile.open(fileobj=f) as archive:archive.extractall(target,filter='data')
    return sha

def import_candidates():
    """Freeze only learner files, never optimizer/replay state or arbitrary paths."""
    with LOCK:
        BASE.mkdir(parents=True,exist_ok=True);value=registry();commit=ensure_g1_runtime()
        sources=[('g1',ROOT/'checkpoints/g1/best.pt'),('mojia',ROOT/'runs/mojia/best_learner.pt'),('mojia',ROOT/'runs/mojia/learner.pt')]
        sources += [('mojia',p) for p in sorted((ROOT/'runs/mojia/evaluation').glob('learner_*.pt'))]
        imported=[]
        for robot,source in sources:
            if not source.is_file():continue
            fd,name=tempfile.mkstemp(dir=BASE,suffix='.pt');os.close(fd);temp=Path(name)
            try:
                # Read one inode; atomic source replacement cannot mix two snapshots.
                with source.open('rb') as a,temp.open('wb') as b:shutil.copyfileobj(a,b,8*1024*1024)
                sha=digest(temp)
                if any(m['sha256']==sha for m in value['models']):continue
                cp=torch.load(temp,map_location='cpu',weights_only=True,mmap=True)
                if (robot=='mojia' and cp.get('schema')!='mojia_ndof_v6') or (robot=='g1' and cp.get('schema')!=5):raise ValueError('不兼容的 checkpoint 类型')
                if 'optimizers' in cp:raise ValueError('只导入 learner，不导入完整回放池')
                step=int(cp['iteration'])+1;model_id=f'{robot}-{step:06d}-{sha[:8]}'
                folder=BASE/'models'/robot/model_id;folder.mkdir(parents=True,exist_ok=True);temp.replace(folder/'policy.pt')
                item=dict(id=model_id,robot=robot,iteration=step,label=f'{step:,} 轮',file=str((folder/'policy.pt').relative_to(BASE)),sha256=sha,bytes=(folder/'policy.pt').stat().st_size,imported_at=now(),source=str(source.relative_to(ROOT)),approval='pending',note='',evaluations={},runtime_commit=commit if robot=='g1' else None)
                record=None
                if robot=='g1':record=read(ROOT/'checkpoints/g1/evaluation.json')
                else:
                    p=ROOT/f'runs/mojia/evaluation/job_{step:08d}.result.json'
                    record=read(p)
                    if record and record.get('status')!='complete':record=None
                if record:
                    normalized=summary(record)
                    if normalized and normalized['full']:
                        item['evaluations']['cuda:0']=dict(source='historical_full_evaluation',tested_at=now(),snapshot_sha256=sha,summary=normalized)
                        atomic(folder/'evaluation_cuda.json',record)
                value['models'].append(item);imported.append(model_id)
            finally:temp.unlink(missing_ok=True)
        save(value);return imported

def record_evaluation(model_id,record,device,scope):
    with LOCK:
        value=registry();m=next(m for m in value['models'] if m['id']==model_id)
        result=summary(record)
        if not result:raise ValueError('评估没有动作结果')
        if not result['full']:raise ValueError('完整模型推荐不能使用限步评估')
        entry=dict(source=scope,tested_at=now(),snapshot_sha256=m['sha256'],summary=result)
        m['evaluations'][device]=entry
        atomic(policy_path(m).parent/('evaluation_'+device.replace(':','_')+'.json'),record)
        save(value)
        if m['approval']=='approved':_approval_files(m)

def recommendation(model):
    entry=model['evaluations'].get('cuda:0');s=entry and entry['summary']
    return (s['completion_rate'],s['normalized_return'],-s['median_root_rmse_m']) if s and s['full'] else (-1,-1,-1)

def catalog():
    value=registry()
    for robot in ('g1','mojia'):
        group=sorted((m for m in value['models'] if m['robot']==robot),key=recommendation,reverse=True)
        for i,m in enumerate(group):
            m['recommended']=i==0 and recommendation(m)[0]>=0
            m['recommendation_reason']='按同机器人 GPU 完整执行率优先、回报其次推荐' if m['recommended'] else ''
    value['robots']=[dict(id='mojia',name='Mojia v3.8',dof=23,description='无支撑杆 · 肘部零位已校准'),dict(id='g1',name='Unitree G1',dof=29,description='保留的 G1 独立运行版本')]
    value['motions']={}
    for robot in ('g1','mojia'):
        manifest=read(ROOT/'data'/robot/'manifest.json',{})
        value['motions'][robot]=[dict(id=Path(c['file']).stem,frames=c['frames'],seconds=round((c['frames']-1)/50,2),category=_category(Path(c['file']).stem)) for c in manifest.get('clips',[])]
    return value

def _category(name):
    if name.startswith('dance'):return '舞蹈'
    if name.startswith('fight'):return '格斗'
    if name.startswith('fall'):return '倒地起身'
    if name.startswith('jumps'):return '跳跃'
    if name.startswith(('run','sprint')):return '跑步'
    return '行走'

def _approval_files(model):
    folder=BASE/'approved'/model['robot']/model['id'];folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'policy.pt'
    if not dest.exists():os.link(policy_path(model),dest)
    atomic(folder/'model.json',model)

def approve(model_id,approved,note=''):
    with LOCK:
        value=registry();m=next((m for m in value['models'] if m['id']==model_id),None)
        if not m:raise ValueError('模型不存在')
        if approved and recommendation(m)[0]<0:raise ValueError('请先完成 GPU 全动作评估，再确认保留')
        if digest(policy_path(m))!=m['sha256']:raise ValueError('模型校验失败')
        m.update(approval='approved' if approved else 'pending',approved_at=now() if approved else None,note=str(note)[:500]);save(value)
        if approved:_approval_files(m)
        else:
            folder=BASE/'approved'/m['robot']/m['id']
            if folder.exists():shutil.rmtree(folder)
        return m
