"""Packed reference storage; future goals are gathered, never padded globally."""
import torch

from .kernel import goal_features


class ReferenceBank:
    def __init__(self, references):
        self.lengths=torch.tensor([len(r) for r in references],dtype=torch.long)
        self.offsets=torch.cat((torch.zeros(1,dtype=torch.long),self.lengths.cumsum(0)[:-1]))
        self.states=torch.cat(references)
        self._devices={}

    def to(self,device):
        device=torch.device(device)
        if device==self.states.device:
            return self
        key=str(device)
        if key not in self._devices:
            bank=object.__new__(ReferenceBank)
            bank.states=self.states.to(device);bank.lengths=self.lengths.to(device)
            bank.offsets=self.offsets.to(device);bank._devices={}
            self._devices[key]=bank
        return self._devices[key]

    def __getitem__(self,index):
        clip,frame,*extra=index;device=self.states.device
        clip=torch.as_tensor(clip,dtype=torch.long,device=device)
        if isinstance(frame,slice):
            if frame.stop is None:
                raise ValueError("Reference slices must have an explicit stop")
            frame=torch.arange(frame.start or 0,frame.stop,frame.step or 1,device=device)
        frame=torch.as_tensor(frame,dtype=torch.long,device=device)
        clip,frame=torch.broadcast_tensors(clip,frame)
        length=self.lengths[clip]
        state=self.states[self.offsets[clip]+frame.clamp_min(0).minimum(length-1)]
        tail=(frame>=length)[...,None]
        # Reference time limits hold the final pose, with zero target velocity.
        result=torch.cat((state[...,:7],torch.where(tail,0.,state[...,7:13]),state[...,13:42],
                          torch.where(tail,0.,state[...,42:71]),state[...,71:]),-1)
        return result[(...,*extra)] if extra else result


class FutureBank:
    def __init__(self,references,future=10):
        self.references=references;self.future=future

    def to(self,device):
        return FutureBank(self.references.to(device),self.future)

    def __getitem__(self,index):
        clip,frame=index;device=self.references.states.device
        clip=torch.as_tensor(clip,dtype=torch.long,device=device)
        if isinstance(frame,slice):
            frame=torch.arange(frame.start or 0,frame.stop,frame.step or 1,device=device)
        frame=torch.as_tensor(frame,dtype=torch.long,device=device)
        clip,frame=torch.broadcast_tensors(clip,frame)
        states=self.references[clip[...,None],frame[...,None]+torch.arange(self.future,device=device)]
        return goal_features(states).flatten(-2)
