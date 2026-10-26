"""Independent checks for robot coordinates, endpoint goals and tracking points."""
import unittest
import numpy as np
import torch
import mujoco
from robot_control.robot_layout import RobotLayout
from robot_control.runtime.dynamics import ConsistentDynamics
from robot_control.runtime.environment import ReferenceBank
from robot_control.runtime.interfaces import motion_future
from robot_control.runtime.train import default_config,make_environment

class RobotContractTests(unittest.TestCase):
    def test_script_training_conflict_guard(self):
        import tempfile
        from pathlib import Path
        from robot_control.runtime.train import require_no_competing_training
        with tempfile.TemporaryDirectory() as folder:
            proc=Path(folder);(proc/'123').mkdir()
            cmd=proc/'123/cmdline'
            cmd.write_bytes(b'python\0scripts/train.py\0--config\0configs/train.json\0')
            with self.assertRaises(RuntimeError):require_no_competing_training(proc,ignore_pid=999)
            cmd.write_bytes(b'python\0scripts/train.py\0--preflight\0')
            require_no_competing_training(proc,ignore_pid=999)

    def test_layout_matches_actual_world_model(self):
        for n in (23,29):
            layout=RobotLayout(n);model=ConsistentDynamics([8],layout=layout)
            self.assertEqual(layout.to_dict()['dimensions']['dynamics_output'],model.net[-1].out_features)
            state=torch.zeros(2,layout.state);state[:,3]=1
            features=model.features(state,torch.zeros(2,n),torch.zeros(2,n+4))
            self.assertEqual(features.shape[-1],model.net[0].in_features)

    def test_goal_window_excludes_current_frame_and_handles_clip_end(self):
        state=torch.zeros(12,82);state[:,0]=torch.arange(12);state[:,7:13]=1;state[:,36:59]=1
        bank=ReferenceBank([state])
        future=motion_future(bank,0,0,10)
        torch.testing.assert_close(future[:,0],torch.arange(1,11,dtype=torch.float32))
        tail=motion_future(bank,0,11,10)
        self.assertTrue((tail[:,0]==11).all());self.assertTrue((tail[:,7:13]==0).all());self.assertTrue((tail[:,36:59]==0).all())

    def test_named_joint_fk_and_hand_point_velocity(self):
        env=make_environment(default_config(True));state=env.references[0,0].clone()
        state[7:13]=torch.tensor([.1,-.2,.3,.4,-.3,.2]);state[36:59]=torch.linspace(-.5,.5,23)
        p,r,v,w=env.task.fk(state[None]);points=env.task.tracking_kinematics((p,r,v,w))
        d=mujoco.MjData(env.model);d.qpos[:7]=state[:7];d.qpos[env.qids]=state[13:36]
        d.qvel[:6]=state[7:13];d.qvel[env.vids]=state[36:59];mujoco.mj_forward(env.model,d)
        np.testing.assert_allclose(p[0].numpy(),d.xpos[1:],atol=2e-6)
        before=[]
        for point in env.tracking_points:
            b=env.model.body(point['body']).id;before.append(d.xpos[b]+d.xmat[b].reshape(3,3)@np.array(point['offset_local_m']))
        np.testing.assert_allclose(points[0][0].numpy(),before,atol=2e-6)
        mujoco.mj_integratePos(env.model,d.qpos,d.qvel,1e-6);mujoco.mj_forward(env.model,d)
        after=[]
        for point in env.tracking_points:
            b=env.model.body(point['body']).id;after.append(d.xpos[b]+d.xmat[b].reshape(3,3)@np.array(point['offset_local_m']))
        np.testing.assert_allclose(points[2][0].numpy(),(np.array(after)-before)/1e-6,atol=2e-5)

if __name__=='__main__':unittest.main()
