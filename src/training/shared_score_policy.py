"""Permutation-equivariant candidate scoring for the joint fire policy.

The four armor candidates share one encoder and one scoring head.  Candidate
features, previous-slot history, selection memory and tracking masks are kept
candidate-local; the global trunk contains only features that do not identify a
slot.  This makes a candidate permutation move its score to the same output
slot without requiring four independently learned slot heads.
"""
from functools import partial

import numpy as np
import torch
from torch import nn

from src.policy.observations import JOINT_SCHEMA
from src.training.factorized_policy import (
    CANDIDATE_FEATURES,
    FIRE_ACTIONS,
    FIRE_FEATURES,
    FactorizedActorCriticPolicy,
    FactorizedDistribution,
    SELECT_ACTIONS,
    SHARED_FEATURES,
    _mlp,
)

NAMES = JOINT_SCHEMA["features"]
PREVIOUS_FEATURES = [NAMES.index(f"previous_slot.{slot}") for slot in range(4)]
CANDIDATE_INDICES = {index for group in CANDIDATE_FEATURES for index in group}
PREVIOUS_INDICES = set(PREVIOUS_FEATURES)
GLOBAL_FEATURES = [index for index in SHARED_FEATURES
                   if index not in CANDIDATE_INDICES and index not in PREVIOUS_INDICES]

CONTRACT = dict(
    version=1,
    kind="shared_score_mlp",
    wire_actions=9,
    selection_actions=5,
    global_features=[NAMES[index] for index in GLOBAL_FEATURES],
    candidate_features=[NAMES[index] for index in CANDIDATE_FEATURES[0]],
    candidate_context=["previous_slot_for_candidate", "return_slot_for_candidate",
                       "candidate_tracking_mask"],
    candidate_hidden=64,
    fire_hidden=64,
    selection_uses_clock=False,
    selection_uses_fire_mask=False,
    deterministic="selection_argmax_then_conditional_fire_gt_half",
    hold="keep_current_without_new_fire",
    entropy="joint_conditional",
)


class SharedScoreActorCriticPolicy(FactorizedActorCriticPolicy):
    """Factorized policy whose selection logits use a shared candidate scorer."""

    def _build(self, lr_schedule):
        spaces = self.observation_space.spaces
        if self.action_space.n != 9 or spaces["features"].shape[1] != len(NAMES):
            raise ValueError("shared score policy requires joint v2/v3 observations and Discrete(9)")
        history = spaces["features"].shape[0]
        memory = "selection_memory" in spaces
        clock = int(np.prod(spaces["decision_clock"].shape)) if "decision_clock" in spaces else 0
        candidate_width = history * (len(CANDIDATE_FEATURES[0]) + 1) + int(memory) + 1
        global_width = history * len(GLOBAL_FEATURES) + history
        self.global_trunk = _mlp(global_width, self.net_arch["pi"], self.activation_fn)
        latent = self.net_arch["pi"][-1]
        self.candidate_encoder = _mlp(candidate_width, [64], self.activation_fn)
        self.score_hidden = _mlp(latent + 64 + 64, [64], self.activation_fn)
        self.selection_head = nn.Linear(64, 1)
        self.hold_head = nn.Linear(latent, 1)
        fire_width = history * len(FIRE_FEATURES) + clock + 4
        self.fire_hidden = _mlp(latent + len(CANDIDATE_FEATURES[0]) + fire_width,
                                [64], self.activation_fn)
        self.fire_head = nn.Linear(64, 1)
        self.critic_trunk = _mlp(self.features_dim, self.net_arch["vf"], self.activation_fn)
        self.value_net = nn.Linear(self.net_arch["vf"][-1], 1)
        self.action_dist = FactorizedDistribution()
        if self.ortho_init:
            for module in (self.global_trunk, self.candidate_encoder, self.score_hidden,
                           self.fire_hidden, self.critic_trunk):
                module.apply(partial(self.init_weights, gain=np.sqrt(2)))
            for module in (self.selection_head, self.hold_head, self.fire_head):
                module.apply(partial(self.init_weights, gain=.01))
            self.value_net.apply(partial(self.init_weights, gain=1))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _candidate_context(self, features, obs, slot, memory):
        candidate = features[:, :, CANDIDATE_FEATURES[slot]]
        previous = features[:, :, PREVIOUS_FEATURES[slot]].unsqueeze(-1)
        context = [torch.cat((candidate, previous), dim=2).flatten(1)]
        if memory:
            context.append(obs["selection_memory"][:, slot + 1:slot + 2].float())
        context.append(obs["action_mask"][:, SELECT_ACTIONS[1 + slot]:SELECT_ACTIONS[1 + slot] + 1].float())
        return torch.cat(context, dim=1)

    def actor_logits(self, obs):
        """Return shared-score selection logits and shared conditional fire logits."""
        features = obs["features"].float()
        valid = obs["valid"].float()
        features = features * valid.unsqueeze(-1)
        global_input = torch.cat((features[:, :, GLOBAL_FEATURES].flatten(1), valid), dim=1)
        latent = self.global_trunk(global_input)
        memory = "selection_memory" in obs
        candidate_embeddings = []
        for slot in range(4):
            context = self._candidate_context(features, obs, slot, memory)
            candidate_embeddings.append(self.candidate_encoder(context))
        embeddings = torch.stack(candidate_embeddings, dim=1)
        candidate_mask = obs["action_mask"][:, SELECT_ACTIONS[1:]].float()
        pooled = (embeddings * candidate_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / candidate_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        score_input = torch.cat((
            latent[:, None, :].expand(-1, 4, -1), embeddings, pooled[:, None, :].expand(-1, 4, -1)
        ), dim=2)
        scores = self.selection_head(self.score_hidden(score_input)).squeeze(-1)
        hold = self.hold_head(latent)
        selection_logits = torch.cat((hold, scores), dim=1)

        fire_context = [features[:, :, FIRE_FEATURES].flatten(1),
                        obs["action_mask"][:, FIRE_ACTIONS].float()]
        if "decision_clock" in obs:
            fire_context.append(obs["decision_clock"].float())
        fire_context = torch.cat(fire_context, dim=1)
        candidate = torch.stack([features[:, -1, indices] for indices in CANDIDATE_FEATURES], dim=1)
        branch = torch.cat((latent[:, None, :].expand(-1, 4, -1), candidate,
                            fire_context[:, None, :].expand(-1, 4, -1)), dim=2)
        fire_logits = self.fire_head(self.fire_hidden(branch)).squeeze(-1)
        return torch.cat((selection_logits, fire_logits), dim=1)


__all__ = ["CONTRACT", "GLOBAL_FEATURES", "SharedScoreActorCriticPolicy"]
