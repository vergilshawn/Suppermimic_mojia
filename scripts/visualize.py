
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Render policy/reference comparisons or serve the retained comparison website."""
import argparse
from pathlib import Path

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('render', help='Run deterministic CPU physics and render actual policy tracking.')
    r.add_argument('--checkpoint', required=True, type=Path)
    r.add_argument('--motion', required=True, nargs='+', help='Exact dataset clip names')
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--seconds', type=float, default=12)
    r.add_argument('--device', default='cpu', help='Policy physics: cpu or cuda:0; use cuda:0 to match training')
    r.add_argument('--width', type=int, default=1920)
    r.add_argument('--height', type=int, default=1080)
    ref = sub.add_parser('reference', help='Play prepared references kinematically using the robot model; no policy needed.')
    ref.add_argument('--config', type=Path, default=Path('configs/train.json'))
    ref.add_argument('--motion', nargs='+', required=True)
    ref.add_argument('--out', type=Path, required=True)
    ref.add_argument('--seconds', type=float, default=20, help='0 renders the entire clip')
    ref.add_argument('--start', type=float, default=0)
    ref.add_argument('--width', type=int, default=1920)
    ref.add_argument('--height', type=int, default=1080)
    s = sub.add_parser('serve', help='Serve existing comparison files locally.')
    s.add_argument('--directory', type=Path, default=Path('web'))
    s.add_argument('--port', type=int, default=8000)
    a = p.parse_args(argv)
    if a.mode == 'serve':
        from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
        from functools import partial
        if not a.directory.is_dir(): p.error('Website directory does not exist')
        server = ThreadingHTTPServer(('127.0.0.1', a.port), partial(SimpleHTTPRequestHandler, directory=str(a.directory.resolve())))
        print(f'http://127.0.0.1:{a.port}', flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
        finally: server.server_close()
        return
    if a.mode == 'reference':
        if a.seconds < 0 or a.start < 0 or min(a.width,a.height)<64 or a.width%2 or a.height%2:
            p.error('Use nonnegative times and even image dimensions >=64')
        return render_reference(a)
    if a.seconds <= 0 or min(a.width, a.height) < 64 or a.width % 2 or a.height % 2:
        p.error('Use positive duration and even image dimensions >=64')
    render(a)

"""Deterministic policy rollout with an unaligned translucent reference robot."""
def render(a):
    import os
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import copy, json, tempfile, hashlib
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import numpy as np
    import torch, mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from robot_control.runtime.kernel import proprio
    torch.set_num_threads(2)
    cp = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    mojia = cp.get('schema') == 'mojia_ndof_v6'
    if mojia:
        from robot_control.runtime.train import make_environment, make_networks
        from robot_control.runtime.physics import CPUVectorPhysics
        env = make_environment(cp['config'])
        if cp['contract'] != env.contract: raise ValueError('Robot/data contract mismatch')
        actor, _, _, tasks = make_networks(env, cp['config'], a.device)
        actor.load_state_dict(cp['actor']); actor.eval()
        if a.device.startswith('cuda'):
            from robot_control.runtime.physics import WarpPhysics
            physics = WarpPhysics(env, 1, device=a.device)
        else:
            physics = CPUVectorPhysics(env, 1, device='cpu')
        n, h, future_count, context_count = env.layout.dof, env.layout.history, env.layout.future, env.layout.context
        qids = env.qids
    else:
        from robot_control.runtime.evaluation_worker import load
        cp, env, actor = load(a.checkpoint)
        tasks = env.task
        n, h, future_count, context_count = 29, 10, 10, 33
        qids = np.arange(7, 36)
    unknown = set(a.motion) - set(env.clip_names)
    if unknown: raise ValueError(f'Unknown clips: {sorted(unknown)}')
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.motion:
        if (a.out / (name + '.mp4')).exists(): raise FileExistsError(a.out / (name + '.mp4'))
    dt = env.control['control_dt']
    # Copy the robot tree for a reference ghost; preserve physical world coordinates.
    with tempfile.TemporaryDirectory(prefix='motion_scene_') as tmp:
        xml = Path(tmp) / 'scene.xml'; mujoco.mj_saveLastXML(str(xml), env.model)
        root = ET.parse(xml).getroot(); world = root.find('worldbody')
        ghost = copy.deepcopy(world.find('body'))
        for el in ghost.iter():
            if 'name' in el.attrib: el.set('name', 'reference_' + el.get('name'))
            if el.tag == 'geom':
                el.attrib.pop('material', None); el.set('rgba', '.12 .65 1 .28')
                el.set('contype', '0'); el.set('conaffinity', '0')
        world.append(ghost)
        for el in root.findall('keyframe'): root.remove(el)
        ET.ElementTree(root).write(xml)
        model = mujoco.MjModel.from_xml_path(str(xml))
    model.vis.global_.offwidth = a.width; model.vis.global_.offheight = a.height
    model.vis.quality.offsamples = 4; model.vis.quality.shadowsize = 4096
    data = mujoco.MjData(model); camera = mujoco.MjvCamera()
    camera.distance = 3.3; camera.azimuth = 135; camera.elevation = -18
    try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', max(12, a.width//70))
    except OSError: font = ImageFont.load_default()
    results = []
    with mujoco.Renderer(model, height=a.height, width=a.width) as renderer, torch.inference_mode():
        for name in a.motion:
            clip = env.clip_names.index(name); ctx = torch.zeros(1, context_count, device=a.device if mojia else 'cpu'); ctx[:, 0] = 1
            state = env.references[clip, 0][None].clone().to(a.device if mojia else 'cpu'); state[:, 13+2*n:] = 0
            if mojia: physics.reset(torch.tensor([0],device=state.device), state, ctx)
            else: env.reset(clip, 0, ctx[0]); state = env.state(np.zeros(n))[None]
            def observation(s):
                return proprio(s, tasks.nominal, ctx) if mojia else proprio(s, tasks.nominal, ctx, False)
            history = observation(state)[:, None].expand(-1, h, -1).clone()
            budget = min(max(1, round(a.seconds/dt)), int(env.lengths[clip])-1)
            fallen = False; smooth = None
            with imageio.get_writer(a.out / (name+'.mp4'), fps=1/dt, codec='libx264', macro_block_size=2,
                                    ffmpeg_params=['-crf','18','-preset','fast','-threads','2','-movflags','+faststart']) as writer:
                for frame in range(budget):
                    ref = env.references[clip, frame]
                    current=state[0].detach().cpu().numpy()
                    data.qpos[:7] = current[:7]; data.qpos[qids] = current[13:13+n]
                    offset = env.model.nq
                    data.qpos[offset:offset+7] = ref[:7]; data.qpos[offset+qids] = ref[13:13+n]
                    mujoco.mj_forward(model, data)
                    target = np.array([float(state[0,0]), float(state[0,1]), .65])
                    smooth = target if smooth is None else .92*smooth+.08*target; camera.lookat[:] = smooth
                    renderer.update_scene(data, camera=camera)
                    image = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(image)
                    draw.rectangle((0,0,a.width,max(45,a.height//12)),fill='#132033')
                    draw.text((12,8),f'{name} | iteration {cp["iteration"]+1} | {frame*dt:.2f}s\nSolid: policy   Cyan: reference (world coordinates)',font=font,fill='white')
                    writer.append_data(np.asarray(image))
                    if frame == 0: image.save(a.out / (name+'_poster.jpg'))
                    first = frame+1
                    future = env.references[clip, torch.arange(first, first+future_count)][None]
                    if mojia: future=future.to(a.device)
                    action = actor(history, future, state).clamp(-20,20)
                    if mojia: nxt, _ = physics.step(action)
                    else:
                        target_q = env.nominal + env.scale*action[0].numpy()
                        for _ in range(round(dt/env.control['physics_dt'])):
                            env.data.ctrl[:] = np.clip(env.kp*(target_q-env.data.qpos[7:])-env.kd*env.data.qvel[6:],-env.effort,env.effort)
                            mujoco.mj_step(env.model,env.data)
                        nxt = env.state(action[0].numpy())[None]
                    fallen = bool(tasks.termination(nxt, env.references[clip,frame+1][None].to(nxt.device), False))
                    state = nxt; history = torch.cat((history[:,1:], observation(state)[:,None]),1)
                    if fallen: break
            result = dict(clip=name, rendered_frames=frame+1, seconds=(frame+1)*dt, fall=fallen,
                          completed=not fallen and frame+1==int(env.lengths[clip])-1,
                          budget_limited=not fallen and frame+1<int(env.lengths[clip])-1)
            results.append(result); print(json.dumps(result), flush=True)
    digest = hashlib.sha256()
    with a.checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): digest.update(block)
    (a.out/'render_report.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),physics_device=a.device,sha256=digest.hexdigest(),width=a.width,height=a.height,fps=1/dt,results=results),indent=2))


def render_reference(a):
    import os
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import json, html
    import numpy as np
    import mujoco, torch
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from robot_control.runtime.train import make_environment
    torch.set_num_threads(2)
    env = make_environment(json.loads(a.config.read_text()))
    unknown = set(a.motion)-set(env.clip_names)
    if unknown: raise ValueError(f'Unknown clips: {sorted(unknown)}')
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.motion:
        if (a.out/(name+'.mp4')).exists(): raise FileExistsError(a.out/(name+'.mp4'))
    model = env.model
    model.vis.global_.offwidth=a.width; model.vis.global_.offheight=a.height
    model.vis.quality.offsamples=4
    model.vis.headlight.ambient[:]=.5
    model.vis.headlight.diffuse[:]=.9
    camera=mujoco.MjvCamera(); camera.distance=3.3; camera.azimuth=135; camera.elevation=-18
    dt=env.control['control_dt']; n=len(env.control['joint_names'])
    qids=env.qids if hasattr(env,'qids') else np.arange(7,7+n)
    results=[]
    with mujoco.Renderer(model,height=a.height,width=a.width) as renderer:
        for name in a.motion:
            clip=env.clip_names.index(name); length=int(env.lengths[clip]); start=round(a.start/dt)
            if start>=length: raise ValueError(f'Start exceeds clip duration: {name}')
            end=length if a.seconds==0 else min(length,start+round(a.seconds/dt))
            smooth=None
            with imageio.get_writer(a.out/(name+'.mp4'),fps=1/(2*dt),codec='libx264',macro_block_size=2,
                                   ffmpeg_params=['-crf','18','-preset','fast','-threads','2','-movflags','+faststart']) as writer:
                for frame in range(start,end,2):
                    state=env.references[clip,frame].numpy()
                    env.data.qpos[:7]=state[:7]; env.data.qpos[qids]=state[13:13+n]
                    mujoco.mj_forward(model,env.data)
                    target=np.array([state[0],state[1],.65]); smooth=target if smooth is None else .9*smooth+.1*target
                    camera.lookat[:]=smooth; renderer.update_scene(env.data,camera=camera)
                    image=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(image)
                    draw.rectangle((0,0,a.width,42),fill='#132033')
                    draw.text((12,8),f'REFERENCE ONLY | {name} | {frame*dt:.2f}s / {(length-1)*dt:.2f}s | kinematic playback',fill='white')
                    writer.append_data(np.asarray(image))
                    if frame==start: image.save(a.out/(name+'_poster.jpg'))
            result=dict(clip=name,source=str(Path(env.assets)/'motions'/(name+'.npz')) if hasattr(env,'assets') else str(a.config),start_s=a.start,end_s=end*dt,full_duration_s=(length-1)*dt,kinematic_only=True)
            results.append(result); print(json.dumps(result),flush=True)
    cards=''.join('<article><h2>'+html.escape(r['clip'])+'</h2><video controls preload="metadata" poster="'+r['clip']+'_poster.jpg" src="'+r['clip']+'.mp4"></video><p>Preview '+str(round(r['end_s']-r['start_s'],2))+' s / full '+str(round(r['full_duration_s'],2))+' s</p></article>' for r in results)
    (a.out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mojia reference motions</title><style>body{background:#101827;color:#eee;font:18px sans-serif;margin:30px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:24px}video{width:100%}article{background:#1b2940;padding:18px}</style><h1>Mojia reference motions</h1><p>Kinematic playback of prepared training references. No policy or physics rollout; this does not prove dynamic feasibility.</p><main>'+cards+'</main>')
    (a.out/'reference_report.json').write_text(json.dumps(dict(config=str(a.config),results=results),indent=2))


if __name__ == "__main__":
    main()
