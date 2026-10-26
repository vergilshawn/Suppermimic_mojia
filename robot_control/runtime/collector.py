"""Vector reset curriculum and current-policy world collection for v3.

Replay stays on host. Physics, observation/history, actions and reference queries
are batched on the learner GPU. Clip end is explicit, distinct from a fall.
"""
import math
import torch
from .kernel import proprio


class VectorCurriculum:
    def __init__(self,lengths,device,ema=.1):
        bins=[(c,s,min(s+50,n-1)) for c,n in enumerate(lengths) for s in range(0,n-1,50)]
        self.bins=torch.tensor(bins,device=device,dtype=torch.long)
        self.offsets=torch.tensor([sum(math.ceil((n-1)/50) for n in lengths[:c]) for c in range(len(lengths))],device=device)
        self.values=torch.zeros(len(bins),device=device);self.past=torch.ones_like(self.values)/len(bins)
        self.visits=torch.zeros_like(self.values,dtype=torch.long);self.ema=ema
    def sample(self,count):
        hard=torch.softmax(-25*self.values/10.,0)
        weights=.5/len(hard)+.4*hard+.1*self.past/self.past.sum()
        ids=torch.multinomial(weights,count,replacement=True)
        self.past+=hard
        c,s,e=self.bins[ids].unbind(-1)
        f=s+(torch.rand(count,device=s.device)*(e-s)).long()
        return c,f
    def update(self,clips,frames,returns,eligible=None):
        idx=self.offsets[clips]+frames//50
        if eligible is not None:idx=idx[eligible];returns=returns[eligible]
        count=torch.zeros_like(self.values);total=torch.zeros_like(self.values)
        count.scatter_add_(0,idx,torch.ones_like(returns));total.scatter_add_(0,idx,returns)
        used=count>0
        self.values[used]=(1-self.ema)*self.values[used]+self.ema*total[used]/count[used]
        self.visits[used]+=count[used].long()
    def state_dict(self):return {k:getattr(self,k).cpu() for k in ('values','past','visits')}
    def load_state_dict(self,state):
        for k,v in state.items():getattr(self,k).copy_(v)


def reference_commands(env,device):
    """Precompute centered 25-frame means, edge-normalized, heading planar+world yaw."""
    from .interfaces import reference_velocity_command
    result=[]
    for clip,n in enumerate(env.lengths):
        s=env.references[clip,torch.arange(n)].to(device)
        result.append(reference_velocity_command(s,torch.arange(n,device=device)))
    from .references import ReferenceBank
    # Command table uses the same offsets, indexed explicitly (not ReferenceBank's state API).
    return torch.cat(result),torch.tensor([sum(env.lengths[:c]) for c in range(len(env.lengths))],device=device)


class VectorCollector:
    def __init__(self,env,physics,actor,critics,tasks,config,device):
        from .interfaces import VelocityCommandSampler
        self.env,self.physics,self.actor,self.critics,self.tasks=env,physics,actor,critics,tasks
        self.config=config;self.device=torch.device(device);self.n=config['simulator_environments']
        self.references=env.references.to(device);self.lengths=torch.tensor(env.lengths,device=device)
        self.command_table,self.command_offsets=reference_commands(env,device)
        self.curriculum=VectorCurriculum(env.lengths,device,config.get('curriculum_ema',.1))
        self.interfaces=torch.zeros(self.n,dtype=torch.long,device=device)
        if config['interfaces']=='motion_velocity':self.interfaces[self.n//2:]=1
        self.commandsampler=VelocityCommandSampler(self.n,device,mode=config['velocity_mode'])
        self.states=torch.zeros(self.n,100,device=device);self.contexts=torch.zeros(self.n,33,device=device)
        self.history=torch.zeros(self.n,10,93,device=device)
        self.clips=torch.zeros(self.n,dtype=torch.long,device=device);self.frames=self.clips.clone()
        self.episode_ids=self.clips.clone();self.episode_rewards=torch.zeros(self.n,device=device)
        self.episode_steps=self.clips.clone();self.episode_serial=0
        self.reset(torch.arange(self.n,device=device))
    def ref_commands(self,clips=None,frames=None):
        clips=self.clips if clips is None else clips;frames=self.frames if frames is None else frames
        f=frames.minimum(self.lengths[clips]-1).clamp_min(0)
        return self.command_table[self.command_offsets[clips]+f]
    def goal_metadata(self,clips,frames,source,elapsed):
        return torch.stack((clips.float()/max(len(self.lengths)-1,1),frames.float()*.02,
                            (self.lengths[clips]-1-frames).clamp_min(0).float()*.02,
                            source.float(),elapsed.float()/250.),-1)
    def future(self,clips,frames):
        return self.references[clips[:,None],frames[:,None]+torch.arange(1,11,device=self.device)]
    @torch.no_grad()
    def reset(self,ids):
        if ids.numel()==0:return
        clips,frames=self.curriculum.sample(len(ids));context=torch.zeros(len(ids),33,device=self.device)
        context[:,0]=1.
        if self.config.get('randomize',True):
            context[:,0]=.3+.9*torch.rand(len(ids),device=self.device)
            context[:,1:30]=.02*torch.rand(len(ids),29,device=self.device)-.01
            context[:,30:]= (2*torch.rand(len(ids),3,device=self.device)-1)*context.new_tensor([.025,.05,.05])
        states=self.references[clips,frames].clone();states[:,71:]=0
        self.physics.reset(ids,states,context)
        self.states[ids]=states;self.contexts[ids]=context;self.clips[ids]=clips;self.frames[ids]=frames
        obs=proprio(states,self.tasks.base_task.nominal,context,True)
        self.history[ids]=obs[:,None].expand(-1,10,-1)
        self.commandsampler.reset(ids,self.ref_commands(clips,frames))
        self.episode_ids[ids]=torch.arange(self.episode_serial,self.episode_serial+len(ids),device=self.device)
        self.episode_serial+=len(ids);self.episode_rewards[ids]=0;self.episode_steps[ids]=0
    @torch.no_grad()
    def collect(self,steps,replay,policy_version):
        rewards=[];terms=[];clips=[];frames=[];fall_count=end_count=0
        for _ in range(steps):
            future=self.future(self.clips,self.frames)
            command=self.commandsampler.commands.clone();source=self.commandsampler.reference_mask.clone();elapsed=self.commandsampler.elapsed.clone()
            if self.config.get('update_normalizers',True):
                self.actor.update_normalizers(self.history,future,self.states)
                fm,_=self.critics.input_features(self.history,future,self.states,self.contexts,command,
                           goal_metadata=self.goal_metadata(self.clips,self.frames,source,elapsed))
                self.critics.motion[0].update(fm)
            action=self.actor(self.history,future,self.states,self.interfaces,command)
            action=action+.1*torch.randn_like(action)
            nxt,diagnostics=self.physics.step(action)
            if not torch.isfinite(nxt).all():raise FloatingPointError('Nonfinite physical transition')
            if self.config.get('update_normalizers',True):
                self.dynamics.in_norm.update(self.dynamics.features(self.states,action,self.contexts))
                self.dynamics.out_norm.update(self.dynamics.targets(self.states,nxt))
            nextframe=self.frames+1;ref=self.references[self.clips,nextframe]
            terminal=self.tasks.termination(nxt,ref,False,self.interfaces).bool()
            clip_end=nextframe>=self.lengths[self.clips]-1
            truncated=clip_end & ~terminal
            reward=self.tasks.reward(self.states,nxt,action,ref,self.interfaces,command)
            nextobs=proprio(nxt,self.tasks.base_task.nominal,self.contexts,True)
            self.commandsampler.step(self.ref_commands(self.clips,nextframe))
            nextcommand=self.commandsampler.commands.clone()
            replay.add_batch(self.states,action,nxt,self.history[:,-1],nextobs,self.contexts,
                             self.clips,self.frames,self.interfaces,command,nextcommand,source,
                             terminal,truncated,clip_end,self.episode_ids,policy_version,
                             initial_histories=self.history,rewards=reward,command_elapsed=elapsed)
            rewards.append(reward);terms.append(terminal|clip_end);clips.append(self.clips.clone());frames.append(self.frames.clone())
            self.states=nxt;self.frames=nextframe;self.history=torch.cat((self.history[:,1:],nextobs[:,None]),1)
            self.episode_rewards+=reward;self.episode_steps+=1
            fall_count+=int(terminal.sum());end_count+=int(clip_end.sum())
            self.reset(torch.where(terminal|clip_end)[0])
        # Bootstrapped per-state return estimates over this collection segment.
        val=self.critics(self.history,self.future(self.clips,self.frames),self.states,self.contexts,
                         self.interfaces,self.commandsampler.commands,
                         goal_metadata=self.goal_metadata(self.clips,self.frames,self.commandsampler.reference_mask,self.commandsampler.elapsed))
        for t in reversed(range(steps)):
            val=rewards[t]+.99*(~terms[t])*val
            self.curriculum.update(clips[t],frames[t],val,eligible=self.interfaces==0)
        return dict(physical_transitions=steps*self.n,vector_control_steps=steps,falls=fall_count,
                    clip_ends=end_count,mean_step_reward=float(torch.stack(rewards).mean()),
                    torque_saturated_world_steps=int(diagnostics['torque_saturated'].sum()) if 'torque_saturated' in diagnostics else None)
    def state_dict(self):
        fields=('states','contexts','history','clips','frames','interfaces','episode_ids','episode_rewards','episode_steps')
        return dict(tensors={k:getattr(self,k).cpu() for k in fields},episode_serial=self.episode_serial,
                    curriculum=self.curriculum.state_dict(),commandsampler=self.commandsampler.state_dict(),physics=self.physics.state_dict())
    def load_state_dict(self,state):
        for k,v in state['tensors'].items():setattr(self,k,v.to(self.device))
        self.episode_serial=state['episode_serial'];self.curriculum.load_state_dict(state['curriculum'])
        self.commandsampler.load_state_dict(state['commandsampler']);self.physics.load_state_dict(state['physics'])
