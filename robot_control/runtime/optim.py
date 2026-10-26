from contextlib import contextmanager
import torch

@contextmanager
def frozen(*modules):
    """Freeze weights while retaining Jacobians with respect to model inputs."""
    parameters = [p for module in modules for p in module.parameters()]
    flags = [p.requires_grad for p in parameters]
    for p in parameters:
        p.requires_grad_(False)
        p.grad = None
    try:
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)

def lambda_returns(rewards, terminations, values, gamma=.99, lam=.95):
    """values has H+1 entries; all targets are detached once per rollout."""
    result = values[-1].detach()
    targets = []
    for t in reversed(range(len(rewards))):
        result = rewards[t].detach() + gamma*(1-terminations[t].detach()) * (
            (1-lam)*values[t+1].detach() + lam*result)
        targets.append(result)
    return torch.stack(targets[::-1])