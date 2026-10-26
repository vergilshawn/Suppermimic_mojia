"""CPU checks of scheduling, geometry, gradients and reference-derived reward behavior."""
import json
import math
from pathlib import Path
import unittest
import torch
from robot_control.runtime.dynamics import ConsistentDynamics,rotation_increment,masked_fit_loss
from robot_control.runtime.kernel import rotation
from robot_control.runtime.normalization import RunningNorm
from robot_control.runtime.train import validate_config,make_environment,make_networks,imagine_v3,fit_critics


class Improvements(unittest.TestCase):
    @unittest.skipUnless(Path("data/g1/manifest.json").exists(), "Local LAFAN dataset required")
    def test_cpu_collection_fit_and_bptt(self):
        import copy
        import mujoco
        import numpy as np
        from robot_control.runtime.collector import VectorCollector
        from robot_control.runtime.replay import ReplayV3
        from robot_control.runtime.kernel import frozen,step
        c=json.loads(Path('configs/train.json').read_text())
        c.update(simulator_environments=2,actor_batch_size=4,randomize=False)
        for key in ('encoder_widths','decoder_widths','velocity_encoder_widths','critic_widths','dynamics_widths'):c[key]=[32,32]
        env=make_environment(c);actor,critics,model,tasks=make_networks(env,c,'cpu')
        class Physics:
            def __init__(self):
                self.model=copy.copy(env.model);self.data=[mujoco.MjData(self.model) for _ in range(2)]
            def reset(self,indices,states,contexts):
                for i,s in zip(indices.tolist(),states.numpy()):
                    d=self.data[i];mujoco.mj_resetData(self.model,d)
                    d.qpos[:7]=s[:7];d.qpos[7:]=s[13:42];d.qvel[:6]=s[7:13];d.qvel[6:]=s[42:71]
                    mujoco.mj_forward(self.model,d)
            def step(self,actions):
                states=[]
                for d,a in zip(self.data,actions.numpy()):
                    target=env.nominal+env.scale*a
                    for _ in range(4):
                        d.ctrl[:]=np.clip(env.kp*(target-d.qpos[7:])-env.kd*d.qvel[6:],-env.effort,env.effort)
                        mujoco.mj_step(self.model,d)
                    states.append(np.concatenate((d.qpos[:7],d.qvel[:6],d.qpos[7:],d.qvel[6:],a)))
                return torch.tensor(np.stack(states),dtype=torch.float32),{}
        physics=Physics();collector=VectorCollector(env,physics,actor,critics,tasks,c,'cpu');collector.dynamics=model
        replay=ReplayV3(256,2);actor_before={k:v.clone() for k,v in actor.named_parameters()}
        collector.collect(64,replay,-1)
        buffers=[(v,v.clone()) for module in (actor,critics,model) for v in module.buffers()]
        c['update_normalizers']=False
        del collector.dynamics
        collector.collect(2,replay,-1)
        self.assertTrue(all(torch.equal(v,before) for v,before in buffers))
        c['update_normalizers']=True
        collector.dynamics=model
        opt=torch.optim.Adam(model.parameters(),lr=2e-4)
        step(masked_fit_loss(model,replay.sample_windows(4,8,'cpu')),opt,model.parameters())
        self.assertTrue(all(torch.equal(v,actor_before[k]) for k,v in actor.named_parameters()))
        self.assertGreater(float(model.in_norm.count),0)
        with frozen(model,critics):
            loss,data,_=imagine_v3(model,actor,critics,tasks,collector,replay.sample_starts(4,'cpu',interface=0),c)
            step(loss,torch.optim.Adam(actor.parameters(),lr=2e-4),actor.parameters(),element_clip=True)
        critic_loss=fit_critics(critics,torch.optim.Adam(critics.parameters(),lr=2e-4),data,1,2)
        self.assertTrue(math.isfinite(critic_loss));self.assertTrue(torch.isfinite(loss))

    def test_utd(self):
        c=json.loads(Path('configs/train.json').read_text());self.assertEqual(validate_config(c),2.5)
        steps=[math.floor((i+1)/.4)-math.floor(i/.4) for i in range(100)]
        self.assertEqual(steps[:4],[2,3,2,3]);self.assertEqual(sum(steps),250)
        old=dict(c,schema=3)
        with self.assertRaises(ValueError):validate_config(old)

    def test_increment_inverse_and_gradients(self):
        torch.manual_seed(14);m=ConsistentDynamics((32,32))
        a=torch.randn(8,100)*.1;a[:,3:7]=rotation_increment(torch.randn(8,3))
        b=torch.randn(8,100)*.1;b[:,3:7]=rotation_increment(torch.randn(8,3))
        reconstructed=m.integrate(a,m.targets(a,b),b[:,71:])
        torch.testing.assert_close(reconstructed[:,:3],b[:,:3],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(rotation(reconstructed[:,3:7]),rotation(b[:,3:7]),atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(reconstructed[:,7:],b[:,7:],atol=1e-6,rtol=1e-5)
        action=torch.randn(8,29,requires_grad=True);context=torch.zeros(8,33)
        # Identity orientation must retain finite Lie-group derivatives at zero.
        a[:,3:7]=torch.tensor([1.,0.,0.,0.])
        pred=m(a,action,context);m.error(pred,b).mean().backward()
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_running_stats_and_state_restore(self):
        n=RunningNorm(3);x=torch.randn(30,3)
        n.update(x[:10]);n.update(x[10:]);torch.testing.assert_close(n.mean,x.mean(0))
        torch.testing.assert_close(n.var,x.var(0,unbiased=False))
        copy=RunningNorm(3);copy.load_state_dict(n.state_dict());torch.testing.assert_close(copy(x),n(x))

    @unittest.skipUnless(Path("data/g1/manifest.json").exists(), "Local LAFAN dataset required")
    def test_reward_alignment_and_learning_graph(self):
        c=json.loads(Path('configs/train.json').read_text());env=make_environment(c)
        s=env.references[0,0][None].clone();s[:,71:]=0
        shifted=s.clone();shifted[:,0]+=.4;action=torch.zeros(1,29)
        base=env.task.reward(s,s,action,s);offset=env.task.reward(s,shifted,action,s)
        expected=.02*.5*(1-math.exp(-.4**2/.3**2))
        self.assertAlmostEqual(float(base-offset),expected,places=6)
        for key in ('encoder_widths','decoder_widths','velocity_encoder_widths','critic_widths','dynamics_widths'):c[key]=[32,32]
        actor,critics,model,tasks=make_networks(env,c,'cpu')
        future=env.references[0,torch.arange(1,11)][None]
        from robot_control.runtime.kernel import proprio
        ctx=torch.zeros(1,33);ctx[:,0]=1
        history=proprio(s,tasks.nominal,ctx)[:,None].expand(-1,10,-1).clone()
        actor.update_normalizers(history,future,s)
        out=actor(history,future,s);pred=model(s,out,ctx)
        loss=-tasks.reward(s,pred,out,env.references[0,1][None],torch.zeros(1,dtype=torch.long),torch.zeros(1,3)).mean()
        loss.backward();self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
