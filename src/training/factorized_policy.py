"""Slot-first conditional fire policy over the existing nine wire actions.

The actor trunk never sees clock, referee/request state or fire-mask bits.
A separate critic consumes the full observation. Masks act on selection and
conditional firing separately, preserving selection marginals when only fire
legality changes. Deterministic decoding is hierarchical, not joint MAP.
"""
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
from torch.nn import functional as F
from sb3_contrib.common.maskable.distributions import MaskableCategorical, MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

from src.policy.observations import JOINT_SCHEMA

NAMES = JOINT_SCHEMA['features']
FIRE_FEATURES = [i for i, name in enumerate(NAMES)
                 if name.startswith(('referee.', 'referee_age_', 'since_request_'))]
SHARED_FEATURES = [i for i in range(len(NAMES)) if i not in FIRE_FEATURES]
CANDIDATE_FEATURES = [[i for i, name in enumerate(NAMES) if name.startswith(f'candidate.{slot}.')]
                      for slot in range(4)]
SELECT_ACTIONS = [0, 1, 3, 5, 7]
FIRE_ACTIONS = [2, 4, 6, 8]
CONTRACT = dict(version=1, kind='factorized_mlp', wire_actions=9, selection_actions=5,
                shared_features=[NAMES[i] for i in SHARED_FEATURES],
                fire_only_features=[NAMES[i] for i in FIRE_FEATURES],
                fire_hidden=64, selection_uses_clock=False, selection_uses_fire_mask=False,
                deterministic='selection_argmax_then_conditional_fire_gt_half',
                hold='keep_current_without_new_fire', entropy='joint_conditional')


class FactorizedDistribution(MaskableCategoricalDistribution):
    """P(hold)=alpha[0]; P(track_i/fire_i)=alpha[i+1]*(1-beta_i/beta_i).

    Input parameter layout: five selection logits followed by four conditional
    fire logits. Stored joint probabilities serve existing diagnostic tools;
    deterministic mode deliberately does not take their global argmax.
    """

    def __init__(self):
        super().__init__(9)

    def proba_distribution(self, action_logits):
        self.raw_logits = action_logits.reshape(-1, 9)
        self.apply_masking(None)
        return self

    def apply_masking(self, masks):
        logits = self.raw_logits
        mask = (torch.ones_like(logits, dtype=torch.bool) if masks is None else
                torch.as_tensor(masks, device=logits.device, dtype=torch.bool).reshape(-1, 9))
        if mask.shape != logits.shape:
            raise ValueError('factorized mask batch does not match logits')
        selection_mask = mask[:, SELECT_ACTIONS]
        fire_mask = mask[:, FIRE_ACTIONS]
        if not selection_mask.any(dim=1).all() or (fire_mask & ~selection_mask[:, 1:]).any():
            raise ValueError('mask needs a legal selection; firing requires legal same-slot tracking')
        self.selection = MaskableCategorical(logits=logits[:, :5], masks=selection_mask)
        self.fire_mask = fire_mask
        self.fire_probs = torch.where(fire_mask, logits[:, 5:].sigmoid(), 0.0)
        # Stable log probabilities avoid log(sigmoid()) underflow in PPO ratios.
        no_fire_log = torch.where(fire_mask, F.logsigmoid(-logits[:, 5:]), 0.0)
        fire_log = torch.where(fire_mask, F.logsigmoid(logits[:, 5:]), -1e8)
        pair_log = torch.stack((no_fire_log, fire_log), dim=-1) + self.selection.logits[:, 1:, None]
        joint_log = torch.cat((self.selection.logits[:, :1], pair_log.flatten(1)), dim=1)
        self.distribution = Categorical(logits=joint_log)

    def entropy(self):
        conditional = torch.where(self.fire_mask, Bernoulli(logits=self.raw_logits[:, 5:]).entropy(), 0.0)
        return self.selection.entropy() + (self.selection.probs[:, 1:] * conditional).sum(dim=1)

    def _wire_action(self, selected, fire):
        return torch.where(selected == 0, 0, 2 * selected - 1 + fire.long())

    def sample(self):
        selected = self.selection.sample()
        beta = self.fire_probs.gather(1, (selected - 1).clamp_min(0)[:, None]).squeeze(1)
        return self._wire_action(selected, torch.bernoulli(beta))

    def mode(self):
        selected = self.selection.probs.argmax(dim=1)
        beta = self.fire_probs.gather(1, (selected - 1).clamp_min(0)[:, None]).squeeze(1)
        return self._wire_action(selected, beta > .5)


def _mlp(input_width, widths, activation):
    layers = []
    for width in widths:
        layers += [nn.Linear(input_width, width), activation()]
        input_width = width
    return nn.Sequential(*layers)


class FactorizedActorCriticPolicy(MaskableMultiInputActorCriticPolicy):
    """Shared actor trunk, slot-conditioned fire head, independent full-state critic.

    PPO evaluates the executed wire action's joint log probability and exact
    joint entropy. Both actor heads backpropagate into the common trunk; value
    loss updates only the separate critic. No recurrent state is introduced.
    """

    def _build(self, lr_schedule):
        spaces = self.observation_space.spaces
        if self.action_space.n != 9 or spaces['features'].shape[1] != len(NAMES):
            raise ValueError('factorized policy requires joint v2/v3 observations and Discrete(9)')
        history = spaces['features'].shape[0]
        memory = int(np.prod(spaces['selection_memory'].shape)) if 'selection_memory' in spaces else 0
        clock = int(np.prod(spaces['decision_clock'].shape)) if 'decision_clock' in spaces else 0
        actor_width = history * len(SHARED_FEATURES) + history + memory + 5
        fire_width = history * len(FIRE_FEATURES) + clock + 4
        self.actor_trunk = _mlp(actor_width, self.net_arch['pi'], self.activation_fn)
        latent = self.net_arch['pi'][-1]
        self.selection_head = nn.Linear(latent, 5)
        self.fire_hidden = _mlp(latent + len(CANDIDATE_FEATURES[0]) + fire_width, [64], self.activation_fn)
        self.fire_head = nn.Linear(64, 1)
        self.critic_trunk = _mlp(self.features_dim, self.net_arch['vf'], self.activation_fn)
        self.value_net = nn.Linear(self.net_arch['vf'][-1], 1)
        self.action_dist = FactorizedDistribution()
        if self.ortho_init:
            for module in (self.actor_trunk, self.fire_hidden, self.critic_trunk):
                module.apply(partial(self.init_weights, gain=np.sqrt(2)))
            for module in (self.selection_head, self.fire_head):
                module.apply(partial(self.init_weights, gain=.01))
            self.value_net.apply(partial(self.init_weights, gain=1))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def actor_logits(self, obs):
        """Return five selection and four conditional fire logits from batched tensors."""
        features = obs['features'].float()
        valid = obs['valid'].float()
        # Keep padding absent even if a caller supplies nonzero invalid rows.
        features = features * valid.unsqueeze(-1)
        shared = [features[:, :, SHARED_FEATURES].flatten(1), valid,
                  obs['action_mask'][:, SELECT_ACTIONS].float()]
        if 'selection_memory' in obs:
            shared.append(obs['selection_memory'].float())
        latent = self.actor_trunk(torch.cat(shared, dim=1))
        fire_context = [features[:, :, FIRE_FEATURES].flatten(1),
                        obs['action_mask'][:, FIRE_ACTIONS].float()]
        if 'decision_clock' in obs:
            fire_context.append(obs['decision_clock'].float())
        fire_context = torch.cat(fire_context, dim=1)
        candidate = torch.stack([features[:, -1, indices] for indices in CANDIDATE_FEATURES], dim=1)
        branch = torch.cat((latent[:, None, :].expand(-1, 4, -1), candidate,
                            fire_context[:, None, :].expand(-1, 4, -1)), dim=2)
        fire_logits = self.fire_head(self.fire_hidden(branch)).squeeze(-1)
        return torch.cat((self.selection_head(latent), fire_logits), dim=1)

    def get_distribution(self, obs, action_masks=None):
        distribution = FactorizedDistribution().proba_distribution(self.actor_logits(obs))
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        return self.value_net(self.critic_trunk(features))

    def forward(self, obs, deterministic=False, action_masks=None):
        distribution = self.get_distribution(obs, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.predict_values(obs), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions, action_masks=None):
        distribution = self.get_distribution(obs, action_masks)
        return self.predict_values(obs), distribution.log_prob(actions.long().flatten()), distribution.entropy()
