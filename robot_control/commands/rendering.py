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
        actor, _, _, tasks = make_networks(env, cp['config'], 'cpu')
        actor.load_state_dict(cp['actor']); actor.eval()
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
            clip = env.clip_names.index(name); ctx = torch.zeros(1, context_count); ctx[:, 0] = 1
            state = env.references[clip, 0][None].clone(); state[:, 13+2*n:] = 0
            if mojia: physics.reset(torch.tensor([0]), state, ctx)
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
                    data.qpos[:7] = state[0, :7]; data.qpos[qids] = state[0, 13:13+n]
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
                    first = frame if mojia else frame+1
                    future = env.references[clip, torch.arange(first, first+future_count)][None]
                    action = actor(history, future, state)
                    if mojia: nxt, _ = physics.step(action)
                    else:
                        target_q = env.nominal + env.scale*action[0].numpy()
                        for _ in range(round(dt/env.control['physics_dt'])):
                            env.data.ctrl[:] = np.clip(env.kp*(target_q-env.data.qpos[7:])-env.kd*env.data.qvel[6:],-env.effort,env.effort)
                            mujoco.mj_step(env.model,env.data)
                        nxt = env.state(action[0].numpy())[None]
                    fallen = bool(tasks.termination(nxt, env.references[clip,frame+1][None], False))
                    state = nxt; history = torch.cat((history[:,1:], observation(state)[:,None]),1)
                    if fallen: break
            result = dict(clip=name, rendered_frames=frame+1, seconds=(frame+1)*dt, fall=fallen,
                          completed=not fallen and frame+1==int(env.lengths[clip])-1,
                          budget_limited=not fallen and frame+1<int(env.lengths[clip])-1)
            results.append(result); print(json.dumps(result), flush=True)
    digest = hashlib.sha256()
    with a.checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): digest.update(block)
    (a.out/'render_report.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),sha256=digest.hexdigest(),width=a.width,height=a.height,fps=1/dt,results=results),indent=2))
