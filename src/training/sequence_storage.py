"""CPU episode storage and bounded device windows for recurrent PPO."""
import numpy as np
import torch

from src.policy.recurrent_observations import ACTOR_KEYS
from src.training.recurrent_policy import select_state


def move_state(state, device):
    return tuple(t.detach().to(device).clone() for t in state[0]), state[1].detach().to(device).clone()


@torch.no_grad()
def replay_cache(policy, episodes, chunks, batch_size):
    """Rebuild burn-start states each epoch, cloning only needed states to CPU."""
    device = next(policy.parameters()).device
    needed = {(e,burn) for e,_,burn,_ in chunks}
    cache = {}
    for begin in range(0,len(episodes),batch_size):
        indices = list(range(begin,min(begin+batch_size,len(episodes))))
        lengths = [len(episodes[e]["actions"]) for e in indices]
        state = policy.initial_state(len(indices))
        for t in range(max(lengths)):
            for j,e in enumerate(indices):
                if (e,t) in needed:
                    cache[e,t] = move_state(select_state(state,slice(j,j+1)),"cpu")
            obs = {k:torch.as_tensor(np.stack([episodes[e]["observations"][k][min(t,lengths[j]-1)]
                                            for j,e in enumerate(indices)]),device=device) for k in ACTOR_KEYS}
            active = torch.tensor([t < n for n in lengths],device=device)[:,None]
            _, actor = policy.actor.logits(obs,state[0],active)
            critic = policy.critic.advance(obs,state[1],active)
            state = actor,critic
    return cache


def window(episodes, chunks, device, adv_mean, adv_std):
    """Only one minibatch's burn-in + loss span is padded/transferred to device."""
    burn_steps = max(start-burn for _,start,burn,_ in chunks)
    length = max(size for _,_,_,size in chunks)
    offsets = np.arange(-burn_steps,length)
    observations = {}
    for key in episodes[0]["observations"]:
        columns = []
        for e,start,_,_ in chunks:
            indices = np.clip(start+offsets,0,len(episodes[e]["actions"])-1)
            columns.append(episodes[e]["observations"][key][indices])
        observations[key] = torch.as_tensor(np.stack(columns,1),device=device)
    targets = {}
    for key in ("actions","log_probs","advantages","returns"):
        columns = []
        for e,start,_,size in chunks:
            values = episodes[e][key][start:start+size]
            columns.append(np.pad(values,(0,length-size)))
        targets[key] = torch.as_tensor(np.stack(columns,1),device=device)
    targets["advantages"] = (targets["advantages"]-adv_mean)/adv_std
    sizes = torch.tensor([c[3] for c in chunks],device=device)
    burn_sizes = torch.tensor([c[1]-c[2] for c in chunks],device=device)
    return observations,targets,sizes,burn_sizes,burn_steps
