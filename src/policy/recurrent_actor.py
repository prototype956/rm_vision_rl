"""v5 persistent Actor. This module needs only PyTorch and the observation schema.

State order is target, four candidate slots, gimbal, firing resources. Mapping
entries identify the old slot to move into each new slot, never simulator IDs.
"""
from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from src.policy.asymmetric_observations import NAMES

ActorState = Tuple[Tensor, Tensor, Tensor, Tensor]
CONTRACT = dict(version=1, kind="asymmetric_gru", observation_version=5,
                target_hidden=128, candidate_hidden=64, candidates=4,
                gimbal_hidden=64, resource_hidden=64, critic_hidden=128,
                layers=1, bidirectional=False, dropout=0,
                selection_actions=5, wire_actions=9,
                deterministic="selection_argmax_then_conditional_fire_gt_half",
                state_mapping="new_slot_to_old_slot", model_switch="episode_boundary")


def mlp(sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers.extend((nn.Linear(a, b), nn.Tanh()))
    return nn.Sequential(*layers)


class RecurrentActor(nn.Module):
    def __init__(self):
        super().__init__()
        target = [i for i, n in enumerate(NAMES) if n.startswith("target.") or
                  n in ("selection.age_s", "selection.previous.-1")]
        gimbal = [i for i, n in enumerate(NAMES) if n.startswith("imu.")]
        resource = [i for i, n in enumerate(NAMES) if n.startswith("fire.")]
        candidate = [[i for i, n in enumerate(NAMES) if n.startswith(f"candidate.{s}.")]
                     for s in range(4)]
        self.register_buffer("target_indices", torch.tensor(target))
        self.register_buffer("gimbal_indices", torch.tensor(gimbal))
        self.register_buffer("resource_indices", torch.tensor(resource))
        self.register_buffer("candidate_indices", torch.tensor(candidate))
        self.register_buffer("previous_indices", torch.tensor([NAMES.index(f"selection.previous.{s}") for s in range(4)]))
        self.register_buffer("select_indices", torch.tensor([0, 1, 3, 5, 7]))
        self.register_buffer("fire_indices", torch.tensor([2, 4, 6, 8]))
        self.target_encoder = mlp([len(target) + 3, 128])
        self.target_gru = nn.GRUCell(128, 128)
        self.candidate_encoder = mlp([len(candidate[0]) + 4, 64])
        self.candidate_gru = nn.GRUCell(64, 64)
        self.gimbal_encoder = mlp([len(gimbal) + 1, 64])
        self.gimbal_gru = nn.GRUCell(64, 64)
        self.resource_encoder = mlp([len(resource) + 6, 64])
        self.resource_gru = nn.GRUCell(64, 64)
        # No resource state, fire mask or fire clock reaches these selection heads.
        self.slot_score = nn.Sequential(mlp([128 + 64 + 64 + 64, 64]), nn.Linear(64, 1))
        self.hold_score = nn.Sequential(mlp([128 + 64 + 64, 64]), nn.Linear(64, 1))
        self.fire_score = nn.Sequential(
            mlp([128 + 64 + 64 + 64 + 64 + len(candidate[0]) + len(resource) + 5, 64]),
            nn.Linear(64, 1))

    @torch.jit.export
    def initial_state(self, batch: int) -> ActorState:
        ref = self.target_gru.weight_hh
        return (ref.new_zeros((batch, 128)), ref.new_zeros((batch, 4, 64)),
                ref.new_zeros((batch, 64)), ref.new_zeros((batch, 64)))

    @torch.jit.export
    def logits(self, obs: Dict[str, Tensor], state: ActorState, active: Tensor) -> Tuple[Tensor, ActorState]:
        x = obs["features"].float()
        batch = x.shape[0]
        dt = obs["dt_s"].float().clamp_min(0)
        dt = dt / (0.1 + dt)
        episode = obs["episode_start"].to(dtype=torch.bool)
        target_reset = episode | obs["target_reset"].to(dtype=torch.bool)
        target, candidates, gimbal, resource = state
        mapping = obs["candidate_map"].long()
        candidates = candidates.gather(1, mapping.unsqueeze(-1).expand(-1, -1, 64))
        target = target * (~target_reset).to(x.dtype)
        candidates = candidates * (~target_reset).to(x.dtype).unsqueeze(-1)
        gimbal = gimbal * (~episode).to(x.dtype)
        resource = resource * (~episode).to(x.dtype)
        return_slot = obs["selection_memory"].float()
        selection_mask = obs["action_mask"].index_select(1, self.select_indices).float()
        candidate_x = x[:, self.candidate_indices]
        candidate_input = torch.cat((candidate_x, x[:, self.previous_indices].unsqueeze(-1),
                                     return_slot[:, 1:].unsqueeze(-1),
                                     selection_mask[:, 1:].unsqueeze(-1),
                                     dt.unsqueeze(1).expand(-1, 4, -1)), dim=-1)
        target_next = self.target_gru(self.target_encoder(torch.cat((
            x.index_select(1, self.target_indices), return_slot[:, :1], obs["valid"].float(), dt), -1)), target)
        candidate_next = self.candidate_gru(self.candidate_encoder(candidate_input).reshape(batch * 4, 64),
                                             candidates.reshape(batch * 4, 64)).reshape(batch, 4, 64)
        gimbal_next = self.gimbal_gru(self.gimbal_encoder(torch.cat((
            x.index_select(1, self.gimbal_indices), dt), -1)), gimbal)
        resource_x = x.index_select(1, self.resource_indices)
        clock = obs["decision_clock"].float()
        clock = clock / (1 + clock.abs())
        resource_next = self.resource_gru(self.resource_encoder(torch.cat((resource_x, clock, dt), -1)), resource)
        # Padding must neither clear, permute nor advance memory.
        a = active.to(dtype=torch.bool)
        next_state = (torch.where(a, target_next, state[0]),
                      torch.where(a.unsqueeze(-1), candidate_next, state[1]),
                      torch.where(a, gimbal_next, state[2]), torch.where(a, resource_next, state[3]))
        target_next, candidate_next, gimbal_next, resource_next = next_state
        weights = selection_mask[:, 1:].unsqueeze(-1)
        mean = (candidate_next * weights).sum(1) / weights.sum(1).clamp_min(1)
        context = torch.cat((target_next, gimbal_next, mean), -1)
        context_slots = context.unsqueeze(1).expand(-1, 4, -1)
        scores = self.slot_score(torch.cat((context_slots, candidate_next), -1)).squeeze(-1)
        selection = torch.cat((self.hold_score(context), scores), -1)
        fire = self.fire_score(torch.cat((context_slots, candidate_next,
            resource_next.unsqueeze(1).expand(-1, 4, -1), candidate_x,
            resource_x.unsqueeze(1).expand(-1, 4, -1), clock.unsqueeze(1).expand(-1, 4, -1)), -1)).squeeze(-1)
        return torch.cat((selection, fire), -1), next_state

    def forward(self, obs: Dict[str, Tensor], state: ActorState) -> Tuple[Tensor, Tensor, Tensor, ActorState]:
        logits, state = self.logits(obs, state, torch.ones_like(obs["dt_s"]))
        selection_mask = obs["action_mask"].index_select(1, self.select_indices).to(dtype=torch.bool)
        fire_mask = obs["action_mask"].index_select(1, self.fire_indices).to(dtype=torch.bool)
        selection = logits[:, :5].masked_fill(~selection_mask, -1e8).softmax(-1)
        fire = logits[:, 5:].sigmoid() * fire_mask.to(logits.dtype)
        chosen = selection.argmax(-1)
        beta = fire.gather(1, (chosen - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
        action = torch.where(chosen == 0, 0, 2 * chosen - 1 + (beta > 0.5).long())
        return selection, fire, action, state
