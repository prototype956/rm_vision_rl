"""Evaluation-only loss/recovery clocks, measured relative to policy takeover."""
from src.policy.asymmetric_observations import NAMES


class TrackingMetrics:
    def __init__(self, initial):
        self.elapsed = 0.
        self.missing_since = None if self.tracking(initial) else 0.
        self.recoveries = []
        self.resets = self.missing_steps = self.unsafe_masks = self.missing_requests = 0

    @staticmethod
    def tracking(observation):
        return observation["features"][NAMES.index("target.state.tracking")] > .5

    def observe(self, observation, requested=False):
        self.elapsed += float(observation["dt_s"][0])
        present = self.tracking(observation)
        self.resets += int(observation["target_reset"][0])
        if not present:
            self.missing_steps += 1
            self.missing_requests += int(requested)
            self.unsafe_masks += int(any(observation["action_mask"][2::2]))
            if self.missing_since is None:
                self.missing_since = self.elapsed
        elif self.missing_since is not None:
            self.recoveries.append(self.elapsed-self.missing_since)
            self.missing_since = None

    def result(self):
        return dict(recovery_times_s=self.recoveries,
                    unrecovered_loss_s=None if self.missing_since is None else self.elapsed-self.missing_since,
                    target_resets=self.resets, missing_steps=self.missing_steps,
                    unsafe_fire_mask_steps=self.unsafe_masks,
                    missing_shot_requests=self.missing_requests)
