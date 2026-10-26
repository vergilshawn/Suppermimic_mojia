import os
os.environ['MUJOCO_GL']='egl'
os.environ['PYOPENGL_PLATFORM']='egl'
import sys,json,copy,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np, torch,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from OpenGL import GL
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from robot_control.runtime.evaluation_worker import load
from robot_control.runtime.kernel import proprio
out=BASE/'web';torch.set_num_threads(2)
p=BASE/'checkpoints/g1/best.pt';cp,env,actor=load(p)
selected=['dance1_subject2','fight1_subject2','fallAndGetUp1_subject4','jumps1_subject5']
best=json.loads((p.parent/'evaluation.json').read_text())
results=[]
for name in selected:
 full=next(x for x in best['evaluation']['motions'] if x['clip']==name);assert full['completed']
 ctx=torch.zeros(1,33);ctx[:,0]=1;clip=env.clip_names.index(name);env.reset(clip,0,ctx[0]);state=env.state(np.zeros(29))[None];history=proprio(state,env.task.nominal,ctx,False)[:,None].expand(-1,10,-1).clone();ss=[];rr=[]
 with torch.inference_mode():
  for f in range(600):
   ref=env.references[clip,f];ss.append(state[0].numpy().copy());rr.append(ref.numpy())
   action=actor(history,env.references[clip,torch.arange(f+1,f+11)][None],state)[0];target=env.nominal+env.scale*action.numpy()
   for _ in range(4):
    env.data.ctrl[:]=np.clip(env.kp*(target-env.data.qpos[7:])-env.kd*env.data.qvel[6:],-env.effort,env.effort);mujoco.mj_step(env.model,env.data)
   nxt=env.state(action.numpy())[None]
   if env.task.termination(nxt,env.references[clip,f+1][None],False).item():raise RuntimeError(name+' failed preview')
   state=nxt;history=torch.cat((history[:,1:],proprio(state,env.task.nominal,ctx,False)[:,None]),1)
 np.savez_compressed(out/(name+'.npz'),states=np.array(ss),reference=np.array(rr))
 results.append(dict(clip=name,seconds=12,full_gpu_evaluation=full,preview_cpu_survived=True));print('recorded',name,flush=True)
xml=out/'matched_scene.xml';mujoco.mj_saveLastXML(str(xml),env.model);tree=ET.parse(xml);root=tree.getroot();world=root.find('worldbody');ghost=copy.deepcopy(world.find('body'))
for el in ghost.iter():
 if 'name' in el.attrib:el.set('name','ghost_'+el.get('name'))
 if el.tag=='geom':
  el.attrib.pop('material',None);el.set('rgba','.12 .65 1 .28');el.set('contype','0');el.set('conaffinity','0')
world.append(ghost)
for e in root.findall('keyframe'):root.remove(e)
ET.SubElement(world,'light',pos='-3 -4 6',dir='.3 .4 -1',directional='true',diffuse='.8 .8 .8',castshadow='true')
tree.write(xml);model=mujoco.MjModel.from_xml_path(str(xml));model.vis.global_.offwidth=1920;model.vis.global_.offheight=1080;model.vis.quality.offsamples=4;model.vis.quality.shadowsize=4096;model.vis.headlight.ambient[:]=.35
renderer=mujoco.Renderer(model,1080,1920);backend=GL.glGetString(GL.GL_RENDERER).decode();print('renderer',backend,flush=True)
data=mujoco.MjData(model);camera=mujoco.MjvCamera();camera.distance=3.3;camera.azimuth=135;camera.elevation=-18
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
for result in results:
 name=result['clip'];traj=np.load(out/(name+'.npz'));smooth=None
 with imageio.get_writer(out/(name+'.mp4'),fps=50,codec='libx264',macro_block_size=2,ffmpeg_params=['-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart']) as writer:
  for i,(a,r) in enumerate(zip(traj['states'],traj['reference'])):
   data.qpos[:36]=np.r_[a[:7],a[13:42]];data.qpos[36:]=np.r_[r[:7],r[13:42]];mujoco.mj_forward(model,data)
   target=np.array([a[0],a[1],.65]);smooth=target if smooth is None else .92*smooth+.08*target;camera.lookat[:]=smooth
   renderer.update_scene(data,camera=camera);frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame);draw.rectangle((0,0,1920,95),fill='#132033')
   draw.text((24,12),f'LOCAL v5 / 30,000  |  {name}  |  {i*.02:05.2f}s',font=font,fill='white');draw.text((24,52),'Solid: policy   Cyan: reference   World-space error preserved   1080p / 50 fps',font=font,fill='#b9dfff')
   writer.append_data(np.asarray(frame))
   if i==250:frame.save(out/(name+'_poster.jpg'))
 print('rendered',name,flush=True)
renderer.close();(out/'matched_report.json').write_text(json.dumps(dict(checkpoint=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),backend=backend,width=1920,height=1080,fps=50,results=results),indent=2))
