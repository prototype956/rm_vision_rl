"""Independent privileged critic and restricted shared-candidate actor (schema v4)."""
from functools import partial

import numpy as np
import torch
from torch import nn

from src.policy.asymmetric_observations import SCHEMA as JOINT_SCHEMA, FIRE_FEATURES
from src.training.factorized_policy import (
    FIRE_ACTIONS,
    FactorizedActorCriticPolicy,
    FactorizedDistribution,
    SELECT_ACTIONS,
    _mlp,
)

NAMES = JOINT_SCHEMA["features"]
CANDIDATE_FEATURES = [[i for i,n in enumerate(NAMES) if n.startswith(f"candidate.{slot}.")]
                      for slot in range(4)]
SHARED_FEATURES = [i for i in range(len(NAMES)) if i not in FIRE_FEATURES]
PREVIOUS_FEATURES = [NAMES.index(f"selection.previous.{slot}") for slot in range(4)]
CANDIDATE_INDICES = {index for group in CANDIDATE_FEATURES for index in group}
PREVIOUS_INDICES = set(PREVIOUS_FEATURES)
GLOBAL_FEATURES = [index for index in SHARED_FEATURES
                   if index not in CANDIDATE_INDICES and index not in PREVIOUS_INDICES]

CONTRACT = dict(
    version=2,
    kind="asymmetric_mlp",
    observation_version=4,
    critic="independent_current_truth_plus_actor_history",
    fire_only_features=[NAMES[index] for index in FIRE_FEATURES],
    fire_uses_candidate_history=True,
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


class AsymmetricActorCriticPolicy(FactorizedActorCriticPolicy):
    """Factorized policy whose selection logits use a shared candidate scorer."""

    def _build(self, lr_schedule):
        spaces = self.observation_space.spaces
        from src.policy.privileged import FEATURE_COUNT
        if (self.action_space.n != 9 or spaces["features"].shape[1] != len(NAMES)
                or "privileged" not in spaces or spaces["privileged"].shape != (FEATURE_COUNT,)):
            raise ValueError("asymmetric policy requires restricted v4 actor and current privileged v1 state")
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
        self.fire_hidden = _mlp(latent + 64 + len(CANDIDATE_FEATURES[0]) + fire_width,
                                [64], self.activation_fn)
        self.fire_head = nn.Linear(64, 1)
        self.critic_keys = tuple(sorted(spaces))
        critic_width = sum(int(np.prod(spaces[key].shape)) for key in self.critic_keys)
        self.critic_trunk = _mlp(critic_width, self.net_arch["vf"], self.activation_fn)
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

    def predict_values(self, obs):
        # Separate encoder: no features_extractor shared with either actor head.
        values = torch.cat([obs[key].float().flatten(1) for key in self.critic_keys], dim=1)
        return self.value_net(self.critic_trunk(values))

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
        branch = torch.cat((latent[:, None, :].expand(-1, 4, -1), embeddings, candidate,
                            fire_context[:, None, :].expand(-1, 4, -1)), dim=2)
        fire_logits = self.fire_head(self.fire_hidden(branch)).squeeze(-1)
        return torch.cat((selection_logits, fire_logits), dim=1)


class DeployedActor(nn.Module):
    """Contains only actor weights; input has no privileged tensor and no value output."""
    _candidate_context = AsymmetricActorCriticPolicy._candidate_context
    actor_logits = AsymmetricActorCriticPolicy.actor_logits

    def __init__(self, policy):
        super().__init__()
        from copy import deepcopy
        for name in ("global_trunk", "candidate_encoder", "score_hidden", "selection_head",
                     "hold_head", "fire_hidden", "fire_head"):
            setattr(self, name, deepcopy(getattr(policy, name)))

    def forward(self, obs):
        logits = self.actor_logits(obs)
        selection = logits[:, :5].masked_fill(~obs["action_mask"][:, SELECT_ACTIONS].bool(), -1e8).softmax(-1)
        fire = logits[:, 5:].sigmoid() * obs["action_mask"][:, FIRE_ACTIONS].float()
        selected = selection.argmax(-1)
        beta = fire.gather(1, (selected-1).clamp_min(0)[:, None]).squeeze(1)
        action = torch.where(selected == 0, 0, 2*selected-1+(beta > .5).long())
        return selection, fire, action


def export_actor(policy, example, path):
    """Trace the restricted actor only, with dynamic batch dimension and frozen input keys."""
    import json
    from pathlib import Path
    # Trace on CPU even when optimization runs on CUDA, so deployment has no CUDA
    # device constants or requirement for a training-server GPU.
    actor = DeployedActor(policy).eval().cpu()
    inputs = {key: value.detach().cpu() for key, value in example.items() if key != "privileged"}
    traced = torch.jit.trace(actor, (inputs,), strict=True)
    traced.save(str(path))
    Path(str(path)+".json").write_text(json.dumps({"observation": JOINT_SCHEMA,
        "policy": CONTRACT, "inputs": list(inputs), "outputs": ["selection", "fire", "action"]}, indent=2))
    return traced
