from pathlib import Path
import numpy as np
import torch

def cosine_multiplier(iteration,decay_steps=250000,minimum=.03):
    import math
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*min(iteration,decay_steps)/decay_steps))

def rng_state():
    import random
    state=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=dict(kind=state[0],keys=state[1].tolist(),pos=state[2],gauss=state[3],cached=state[4]),
                python=random.getstate())

def restore_rng(state):
    import random
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:torch.cuda.set_rng_state_all(state["cuda"])
    n=state["numpy"];np.random.set_state((n["kind"],np.array(n["keys"],dtype=np.uint32),n["pos"],n["gauss"],n["cached"]))
    random.setstate(state["python"])

def atomic_save(payload,path):
    temporary=path.with_suffix(path.suffix+".tmp")
    torch.save(payload,temporary);temporary.replace(path)
