"""Independent recurrent critic and Actor; no privileged gradient path to Actor."""
import torch
from torch import nn
from torch.nn import functional as F

from src.policy.asymmetric_observations import NAMES
from src.policy.privileged import FEATURE_COUNT
from src.policy.recurrent_actor import RecurrentActor, mlp
from src.training.factorized_policy import FactorizedDistribution


class RecurrentCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mlp([len(NAMES) + 1 + 9 + 5 + 5 + 1 + 2 + 16, 128])
        self.gru = nn.GRUCell(128, 128)
        self.truth = mlp([FEATURE_COUNT, 256, 128])
        self.value = nn.Sequential(mlp([256, 128, 128]), nn.Linear(128, 1))

    def initial_state(self, batch):
        return self.gru.weight_hh.new_zeros((batch, 128))

    def advance(self, obs, state, active):
        dt = obs["dt_s"].float().clamp_min(0)
        clock = obs["decision_clock"].float()
        history = torch.cat((obs["features"].float(), obs["valid"].float(),
            obs["action_mask"].float(), obs["selection_memory"].float(), clock/(1+clock.abs()),
            dt/(0.1+dt), obs["episode_start"].float(), obs["target_reset"].float(),
            F.one_hot(obs["candidate_map"].long(), 4).flatten(1).float()), -1)
        new_state = self.gru(self.encoder(history), state * (~obs["episode_start"].bool()).float())
        return torch.where(active.bool(), new_state, state)

    def forward(self, obs, state, active):
        new_state = self.advance(obs, state, active)
        value = self.value(torch.cat((new_state, self.truth(obs["privileged"].float())), -1)).squeeze(-1)
        return value, new_state


class RecurrentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = RecurrentActor()
        self.critic = RecurrentCritic()

    def initial_state(self, batch):
        return self.actor.initial_state(batch), self.critic.initial_state(batch)

    def forward(self, obs, state, active=None):
        if active is None:
            active = torch.ones_like(obs["dt_s"])
        logits, actor_state = self.actor.logits(obs, state[0], active)
        value, critic_state = self.critic(obs, state[1], active)
        distribution = FactorizedDistribution().proba_distribution(logits)
        distribution.apply_masking(obs["action_mask"])
        return distribution, value, (actor_state, critic_state)


def tensor_observation(observations, device):
    """Stack environment observations without interpreting unlisted dictionaries."""
    import numpy as np
    from src.policy.recurrent_observations import ACTOR_KEYS
    return {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=device)
            for k in (*ACTOR_KEYS, "privileged")}


def detach_state(state):
    return tuple(t.detach() for t in state[0]), state[1].detach()


def select_state(state, indices):
    return tuple(t[indices] for t in state[0]), state[1][indices]


def combine_states(states):
    return (tuple(torch.cat([s[0][i] for s in states]) for i in range(4)),
            torch.cat([s[1] for s in states]))
