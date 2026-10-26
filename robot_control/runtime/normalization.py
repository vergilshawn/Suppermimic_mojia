"""Running statistics updated explicitly from physical collection/valid replay only."""
import torch
from torch import nn


class RunningNorm(nn.Module):
    def __init__(self,dim,eps=.01):
        super().__init__();self.eps=eps
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('var',torch.ones(dim))
        self.register_buffer('count',torch.zeros(()))
    @torch.no_grad()
    def update(self,x):
        x=x.detach().reshape(-1,self.mean.numel()).float()
        if x.shape[0]==0:return
        active=self.count<2e8;n=x.shape[0];total=self.count+n
        m=x.mean(0);v=x.var(0,unbiased=False);delta=m-self.mean;weight=n/total
        variance=self.var*(1-weight)+v*weight+delta.square()*weight*(1-weight)
        self.mean.copy_(torch.where(active,self.mean+delta*weight,self.mean))
        self.var.copy_(torch.where(active,variance,self.var));self.count.copy_(torch.where(active,total,self.count))
    def std(self):return (self.var+self.eps**2).sqrt()
    def forward(self,x):return (x-self.mean)/self.std()
