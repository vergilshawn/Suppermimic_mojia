import unittest
import torch
from robot_control.runtime.regression import RegressionGuard
from robot_control.runtime.train import default_config,make_environment,make_networks,validate_config
from robot_control.runtime.collector import VectorCollector
from robot_control.runtime.replay import Replay
from robot_control.runtime.dynamics import ConsistentDynamics
from robot_control.robot_layout import RobotLayout
class GuardTests(unittest.TestCase):
    def test_sustained_regression_only(self):
        g=RegressionGuard()
        def r(x,c):return dict(status='complete',evaluation_is_full=True,normalized_return=x,completion_rate=c)
        self.assertEqual(g.observe(r(.9,1)),(True,False))
        self.assertEqual(g.observe(r(.6,.5)),(False,False))
        g.observe(r(.89,1));self.assertEqual(g.bad_count,0)
        g.observe(r(.6,.5));g.observe(r(.6,.5));self.assertEqual(g.observe(r(.6,.5)),(False,True))
        self.assertEqual(g.observe(r(.91,1)),(True,False))
    def test_incomplete_evaluation_rejected(self):
        with self.assertRaises(ValueError):RegressionGuard().observe(dict(status='complete',evaluation_is_full=False))
    def test_trial_cap(self):
        c=default_config(True);c['iterations']=50001
        with self.assertRaises(ValueError):validate_config(c)
    def test_model_round_trip_and_gradients(self):
        model=ConsistentDynamics([16],layout=RobotLayout(23));s=torch.randn(4,82)*.01;s[:,3]=1
        d=torch.randn(4,58)*.01;action=torch.randn(4,23,requires_grad=True)
        nxt=model.integrate(s,d,action)
        torch.testing.assert_close(model.targets(s,nxt),d,atol=1e-6,rtol=1e-5)
        pred=model(s,action,torch.zeros(4,27));pred.square().mean().backward()
        self.assertTrue(torch.isfinite(action.grad).all())
    def test_timeout_replay_is_truncation_not_fall(self):
        c=default_config(True);c.update(reset_state_noise=False,observation_noise=False)
        env=make_environment(c);actor,critics,model,tasks=make_networks(env,c,'cpu')
        class Physics:
            def reset(self,ids,states,contexts):
                if not hasattr(self,'states'):self.states=torch.zeros(2,82)
                self.states[ids]=states
            def step(self,action):return self.states.clone(),{}
        physics=Physics();collector=VectorCollector(env,physics,actor,critics,tasks,c,'cpu');collector.dynamics=model
        collector.frames[:]=0;collector.episode_steps[:]=499
        class Capture:
            def add_batch(self,*args,**kwargs):self.args=args;self.kwargs=kwargs
        replay=Capture();collector.collect(1,replay,0)
        # terminal/truncated/clip_end fields in replay API.
        self.assertFalse(replay.args[12].any());self.assertTrue(replay.args[13].all());self.assertFalse(replay.args[14].any())
        self.assertTrue((collector.episode_steps==0).all())
if __name__=='__main__':unittest.main()
