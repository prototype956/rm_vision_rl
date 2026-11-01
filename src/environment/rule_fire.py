"""Full traditional fire-control baseline with the same local shot opportunity clock.

Uses the existing comparison-only bridge binary; training bridge is untouched.
"""
import copy
from src.environment.multi_robot_fire import MultiRobotFireEnv
from src.policy.asymmetric_observations import adapt_observation

class RuleEnv(MultiRobotFireEnv):
    """Reuse reset/physical rules; run the existing native full RulePolicy every tick."""
    def _start(self):
        if self._stack is not None:return
        super()._start()
        original=self._bridge.exchange
        self._rule_active=False
        def exchange(request):
            request=copy.deepcopy(request)
            op=request['op']
            if op=='reset':self._rule_active=False
            if op=='begin_training':request['policy_mode']='rule'
            if op=='step' and self._rule_active:
                request['decision_due']=bool(self._history.clock.due)
            result=original(request)
            if op=='begin_training':self._rule_active=True
            return result
        self._bridge.exchange=exchange

    def _prepare(self):
        self._history.begin_step()
        self._prepared=self._bridge.prepare(self._response,external=False)
        self._actor_semantic=adapt_observation(None,self._response['data']['actor_feedback'],
            self._response['sim_time_ns'],last_request_ns=self._last_request_ns,
            request_previous=self._last_request['shot_requested'])
        # Display only: the rule controller receives its estimates directly in C++.
        self._actor_semantic['estimate']['state']=self._prepared.get('tracker_state','lost').lower()
        self._obs=self._history.observation()

    def step(self, action=None):
        if self._status!='ready':raise RuntimeError('reset required')
        result=self._prepared
        before=self._response['sim_time_ns']
        clock_before=self._history.clock.info()
        control=result['control']
        if control.get('shot_requested') and not self._history.clock.due:
            raise RuntimeError('rule requested a new shot outside the common opportunity clock')
        if getattr(self, "collect_replay_events", False):
            self._last_replay_command = dict(result["command"])
        self._response=self._client.advance(**result['command'])
        self._bridge.ack(True)
        assert self._response['sim_time_ns']==before+10_000_000
        self._steps+=1
        reward=float(self._response['data']['reward_damage'])
        self._damage+=reward
        self._switches+=int(control.get('slot_switched',False))
        self._last_request=dict(action=None,executed_action=None,wire_action=None,
            selected_slot=control['selected_slot'],slot_switched=control.get('slot_switched',False),
            shot_requested=control.get('shot_requested',False),shot_accepted=control.get('shot_accepted',False),
            reject_reason=control.get('reject_reason_name'),decision_clock_before=clock_before)
        if self._last_request['shot_requested']:self._last_request_ns=before
        self._history.clock.complete_step()
        robots=self._robots()
        reason='controlled_dead' if robots[1]['hp']<=0 else 'target_destroyed' if robots[2]['hp']<=0 else None
        if reason is None and self._ammo_exhausted():
            if robots[1]['resources']['pending_feed'] is None and not self._response['data']['evaluation']['projectiles']:
                reason='ammo_exhausted'
        terminated=reason is not None
        truncated=not terminated and self._steps>=self.episode_steps
        if truncated:reason='time_limit'
        self._prepare()
        if terminated or truncated:
            self._bridge.cancel();self._prepared=None;self._status='complete'
        return self._observation(),reward,terminated,truncated,self._info(reward,reason)
