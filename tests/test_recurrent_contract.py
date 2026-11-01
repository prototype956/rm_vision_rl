"""Long-lived regressions for memory, information boundaries and recurrent PPO."""
import copy
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.policy.asymmetric_observations import NAMES, FIRE_FEATURES, _example, adapt_observation
from src.policy.recurrent_actor import RecurrentActor, CONTRACT
from src.policy.recurrent_deployment import ActorRunner, export_actor
from src.policy.recurrent_observations import RecurrentObservationPolicy
from src.policy.privileged import FEATURE_COUNT
from src.training.config import read_config
from src.training.recurrent_policy import RecurrentPolicy, select_state
from src.training.recurrent_ppo import RecurrentPPO, pack_episodes, segments, state_cache
from src.training.recurrent_rollout import advantages, collect_episodes


def observation(batch=2, first=False):
    return dict(features=torch.rand(batch, len(NAMES)) * 2-1,
        valid=torch.ones(batch, 1), action_mask=torch.ones(batch, 9),
        selection_memory=torch.tensor([[1., 0, 0, 0, 0]]).repeat(batch, 1),
        decision_clock=torch.rand(batch, 5), dt_s=torch.full((batch, 1), 0. if first else .01),
        episode_start=torch.full((batch, 1), first), target_reset=torch.full((batch, 1), first),
        candidate_map=torch.arange(4).repeat(batch, 1), privileged=torch.rand(batch, FEATURE_COUNT)*2-1)


def numpy_observation(obs, row=0):
    return {k: v[row].detach().cpu().numpy().copy() for k, v in obs.items()}


class RecurrentContractTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_actor_information_and_fire_history_isolation(self):
        policy = RecurrentPolicy()
        states = [policy.initial_state(2), policy.initial_state(2)]
        for t in range(20):
            a = observation(first=t == 0)
            b = {k: v.clone() for k, v in a.items()}
            b["features"][:, FIRE_FEATURES] = torch.rand(2, len(FIRE_FEATURES))
            b["decision_clock"] *= 10
            b["action_mask"][:, [2,4,6,8]] = 0
            b["privileged"] *= -1
            da, _, states[0] = policy(a, states[0])
            db, value, states[1] = policy(b, states[1])
            torch.testing.assert_close(da.selection.probs, db.selection.probs, rtol=0, atol=0)
        self.assertFalse(torch.equal(states[0][0][3], states[1][0][3]))
        policy.zero_grad()
        value.sum().backward()
        self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
        # Privileged-only changes leave ALL Actor outputs and memory identical.
        b = {k: v.clone() for k, v in a.items()}
        b["privileged"].normal_()
        outputs = [policy.actor(o, states[0][0]) for o in (a, b)]
        for i in range(3):
            torch.testing.assert_close(outputs[0][i], outputs[1][i], rtol=0, atol=0)

    def test_candidate_mapping_and_reset_scope(self):
        actor = RecurrentActor()
        o = observation()
        state = tuple(torch.randn_like(x) for x in actor.initial_state(2))
        permutation = torch.tensor([2, 0, 3, 1])
        mapped = {k: v.clone() for k, v in o.items()}
        mapped["features"][:, actor.candidate_indices] = o["features"][:, actor.candidate_indices[permutation]]
        mapped["features"][:, actor.previous_indices] = o["features"][:, actor.previous_indices[permutation]]
        mapped["selection_memory"][:, 1:] = o["selection_memory"][:, 1:][:, permutation]
        for indices in (actor.select_indices[1:], actor.fire_indices):
            mapped["action_mask"][:, indices] = o["action_mask"][:, indices[permutation]]
        mapped["candidate_map"] = permutation.repeat(2, 1)
        a, f, _, sa = actor(o, state)
        b, g, _, sb = actor(mapped, state)
        torch.testing.assert_close(b[:, 1:], a[:, 1:][:, permutation])
        torch.testing.assert_close(g, f[:, permutation])
        torch.testing.assert_close(sb[1], sa[1][:, permutation])
        o["target_reset"][:] = True
        _, cleared = actor.logits(o, state, torch.ones(2, 1))
        zero_target = (torch.zeros_like(state[0]), torch.zeros_like(state[1]), state[2], state[3])
        _, expected = actor.logits(o, zero_target, torch.ones(2, 1))
        for x, y in zip(cleared, expected):
            torch.testing.assert_close(x, y)
        o["episode_start"][:] = True
        _, cleared = actor.logits(o, state, torch.ones(2, 1))
        _, expected = actor.logits(o, actor.initial_state(2), torch.ones(2, 1))
        for x, y in zip(cleared, expected):
            torch.testing.assert_close(x, y)
        # Even reset/remap flags on padding cannot touch state.
        _, padding = actor.logits(mapped, state, torch.zeros(2, 1))
        for x, y in zip(padding, state):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        # One environment is unaffected by what is stored in a different row.
        single = {k: v[:1] for k, v in mapped.items()}
        one = actor(single, tuple(s[:1] for s in state))
        torch.testing.assert_close(one[0], b[:1])
        critic = RecurrentPolicy().critic
        history = torch.randn(2, 128)
        o["episode_start"][:] = False
        remembered = critic.advance(o, history, torch.ones(2, 1))
        cold = critic.advance(o, torch.zeros_like(history), torch.ones(2, 1))
        self.assertFalse(torch.equal(remembered, cold))  # target reset retains Critic context
        o["episode_start"][:] = True
        torch.testing.assert_close(critic.advance(o, history, torch.ones(2, 1)),
                                   critic.advance(o, torch.zeros_like(history), torch.ones(2, 1)))

    def test_adapter_keeps_own_feedback_during_target_loss(self):
        adapter = RecurrentObservationPolicy()
        adapter.clock.start_episode()
        packet = dict(version=1, imu=copy.deepcopy(_example["imu"]), thermal=copy.deepcopy(_example["thermal"]))
        for step in range(5):
            adapter.begin_step()
            adapter.set_tracking_metadata(dict(track_generation=1, track_reset_count=0))
            packet["thermal"].update(heat=450+step*10, allowance_remaining=300+step*20)
            packet["imu"]["gyro_rad_s"] = [step, 0, 0]
            adapter(adapt_observation(None, packet, step*10_000_000))
            o = adapter.observation()
            self.assertEqual(o["features"].shape, (len(NAMES),))
            self.assertEqual(o["action_mask"].tolist(), [1]+[0]*8)
            self.assertAlmostEqual(float(o["dt_s"][0]), 0 if step == 0 else .01)
            self.assertAlmostEqual(o["features"][NAMES.index("fire.heat")], (450+step*10)/(850+step*10), places=6)
            self.assertAlmostEqual(o["features"][NAMES.index("fire.allowance_remaining")], (300+step*20)/(500+step*20), places=6)
        self.assertFalse(any("defense" in n for n in NAMES))

    def episodes(self, policy):
        episodes = []
        with torch.no_grad():
            for length in (11, 17):
                obs = [observation(1, first=t == 0) for t in range(length+1)]
                state = policy.initial_state(1)
                actions, logp, values = [], [], []
                for o in obs[:-1]:
                    dist, value, state = policy(o, state)
                    action = dist.sample()
                    actions.append(action.item()); logp.append(dist.log_prob(action).item()); values.append(value.item())
                ep = dict(observations={k: torch.cat([o[k] for o in obs]).numpy() for k in obs[0]},
                          actions=np.array(actions), log_probs=np.array(logp, np.float32), values=np.array(values, np.float32),
                          rewards=np.linspace(0, 1, length, dtype=np.float32))
                ep["advantages"], ep["returns"] = advantages(ep["rewards"], ep["values"], 0., .999, .99)
                episodes.append(ep)
        return episodes

    def test_replay_burn_in_update_and_checkpoint(self):
        config = read_config(Path(__file__).resolve().parents[1] / "config/training/rmul_2027_infantry_curriculum_v2.json")
        config.update(device="cpu", n_envs=1)
        config["ppo"].update(sequence_length=5, burn_in=3, batch_size=2, n_epochs=2)
        model = RecurrentPPO(config)
        episodes = self.episodes(model.policy)
        packed = pack_episodes(episodes, model.device)
        chunks = segments(episodes, 5, 3)
        cache = state_cache(model.policy, packed, episodes, chunks)
        with torch.no_grad():
            for e, start, burn, size in chunks:
                state = cache[e, burn]
                for t in range(burn, start+size):
                    obs = {k: v[t, e:e+1] for k, v in packed.items()}
                    dist, value, state = model.policy(obs, state)
                    action = torch.tensor([episodes[e]["actions"][t]])
                    self.assertAlmostEqual(dist.log_prob(action).item(), episodes[e]["log_probs"][t], places=5)
                    self.assertAlmostEqual(value.item(), episodes[e]["values"][t], places=5)
        metrics = model.update(episodes)
        self.assertGreater(metrics["minibatches"], 0)
        self.assertEqual(model.num_timesteps, 28)
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "model.zip"
            from src.training.models import _digest
            from src.policy.recurrent_observations import SCHEMA
            model.save(file, dict(format_version=1, observation=dict(version=5, schema_sha256=_digest(SCHEMA))))
            restored = RecurrentPPO.load(file, device="cpu")
            o = observation(first=True)
            a = model.policy(o, model.policy.initial_state(2))
            b = restored.policy(o, restored.policy.initial_state(2))
            torch.testing.assert_close(a[0].distribution.probs, b[0].distribution.probs, rtol=0, atol=0)
            torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
            self.assertEqual(restored.completed_updates, 1)

    def test_script_export_without_critic_and_episode_switch(self):
        actor = RecurrentActor().eval()
        with tempfile.TemporaryDirectory() as d:
            export_actor(actor, d)
            runner = ActorRunner(d)
            state = actor.initial_state(1)
            for t in range(12):
                o = observation(1, first=t == 0)
                with torch.no_grad():
                    selection, fire, action, state = actor(o, state)
                packet = numpy_observation(o)
                packet.pop("privileged")
                actual, s, f = runner.predict(packet)
                self.assertEqual(actual, action.item())
                np.testing.assert_allclose(s, selection.numpy()[0], atol=1e-6)
                np.testing.assert_allclose(f, fire.numpy()[0], atol=1e-6)
            self.assertFalse(any("critic" in k for k in runner.model.state_dict()))
            old = runner.model
            runner.request_model(d)
            runner.predict(numpy_observation(observation(1)))
            self.assertIs(runner.model, old)
            runner.predict(numpy_observation(observation(1, first=True)))
            self.assertIsNot(runner.model, old)

    def test_terminal_and_timeout_gae(self):
        terminated, _ = advantages([1., 2.], [3., 4.], 0., .9, 1.)
        truncated, _ = advantages([1., 2.], [3., 4.], 5., .9, 1.)
        np.testing.assert_allclose(terminated, [-.2, -2.], atol=1e-6)
        np.testing.assert_allclose(truncated, [3.85, 2.5], atol=1e-6)

    def test_collection_barrier_and_final_observation_bootstrap(self):
        policy = RecurrentPolicy()

        class Workers:
            connections = [None, None]

            def __init__(self):
                self.pending = {}
                self.steps = [0, 0]
                self.resets = [0, 0]
                self.observations = [[numpy_observation(observation(1, first=t == 0))
                                      for t in range(length+1)] for length in (2, 3)]

            def send(self, rank, command, value=None):
                if command == "reset":
                    self.steps[rank] = 0
                    self.resets[rank] += 1
                    self.pending[rank] = (self.observations[rank][0], {})
                else:
                    self.steps[rank] += 1
                    t = self.steps[rank]
                    done = t == len(self.observations[rank])-1
                    # Deliberately different executed action: PPO must retain its sample.
                    self.pending[rank] = (self.observations[rank][t], 1., done and rank == 0,
                                          done and rank == 1, dict(executed_action=-99))

            def receive(self, rank):
                return self.pending.pop(rank)

        workers = Workers()
        episodes = collect_episodes(policy, workers, dict(ppo=dict(n_steps=4, gamma=.999, gae_lambda=.99)))
        self.assertEqual(workers.resets, [2, 2])
        self.assertEqual(sorted(len(ep["actions"]) for ep in episodes), [2, 2, 3, 3])
        for ep in episodes:
            self.assertTrue((ep["actions"] >= 0).all())
            self.assertEqual(ep["execution"][-1]["executed_action"], -99)
            if ep["worker"] == 0:
                self.assertEqual(ep["bootstrap"], 0.)
            else:
                state = policy.critic.initial_state(1)
                with torch.no_grad():
                    for t in range(len(ep["actions"])+1):
                        o = {k: torch.as_tensor(v[t:t+1]) for k, v in ep["observations"].items()}
                        value, state = policy.critic(o, state, torch.ones(1, 1))
                self.assertAlmostEqual(ep["bootstrap"], value.item(), places=6)

    def test_empty_allowance_waits_for_pending_feed_and_projectiles(self):
        from src.environment.fire import FireEnv

        class Harness(FireEnv):
            finite_episode = True

            def _prepare(self):
                self._obs = {"time": np.array([self._response["sim_time_ns"]])}
                self._prepared = dict(kind="policy_observation")

            def _switch_reward_context(self):
                return None

            def _info(self, reward, reason):
                return dict(damage=reward, end_reason=reason)

        def response(t, pending, projectiles, damage):
            own = dict(robot_id=1, hp=400, actual_shots=1, rejected_requests=0,
                       resources=dict(allowance=dict(mode="limited", remaining=0), pending_feed=pending))
            return dict(sim_time_ns=t*10_000_000, data=dict(reward_damage=damage,
                        evaluation=dict(robots=[own, dict(robot_id=2, hp=400)], projectiles=projectiles)))

        env = Harness(action_mode="joint", selection_memory=True, observation_mode="asymmetric_gru_v1")
        responses = iter((response(1, {"ready_at_ns":20_000_000}, [], 0),
                          response(2, None, [{"id":1}], 0), response(3, None, [], 20)))
        env._client = SimpleNamespace(advance=lambda **kwargs: next(responses))
        control = dict(selected_slot=-1, slot_switched=False, shot_requested=False, shot_accepted=False)
        env._bridge = SimpleNamespace(submit=lambda action: dict(command={}, control=control),
                                      ack=lambda accepted: None, cancel=lambda: None)
        env._response = response(0, None, [], 0)
        env._status = "ready"
        env._steps = env._switches = 0
        env._damage = 0
        env._scene_info = dict(episode_index=0)
        env._action_log = io.StringIO()
        env._prepare()
        for _ in range(2):
            _, reward, terminated, truncated, _ = env.step(0)
            self.assertFalse(terminated or truncated)
            self.assertEqual(reward, 0)
        _, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual((reward, info["end_reason"]), (20, "ammo_exhausted"))
        env.close()


if __name__ == "__main__":
    unittest.main()
