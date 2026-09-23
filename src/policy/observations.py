"""生成带版本的有限 float32 特征，维护从旧到新排列的 8 个控制周期历史。"""
from array import array
from collections import deque
import math
import json
from pathlib import Path

VERSION = 1
HISTORY = 8
CONTROL_DT_S = 0.01
SCHEMA = json.loads(
    (Path(__file__).parents[2] / "config/observations/v1.json").read_text()
)
FEATURE_NAMES = SCHEMA["features"]


def encode(observation):
    """按固定字段顺序编码观测，排除传输元数据和评估真值。

    Args:
        observation: C++ 火控提供的语义观测，包含估计、自身状态和四个候选槽位。

    Returns:
        特征名称、90 个缩放至 [-1, 1] 的 float32 精度数值及九动作布尔掩码。

    Raises:
        ValueError: 特征非有限、字段长度或顺序不符合 schema，或动作掩码格式错误。
    """
    o = observation
    values, names = [], []
    def add(name, value, scale=1.0):
        if type(value) not in (int, float, bool) or not math.isfinite(value):
            raise ValueError('non-finite or non-numeric feature: '+name)
        names.append(name)
        values.append(max(-1.0, min(1.0, float(value)/scale)))
    def vector(name, value, scale, count):
        if len(value) != count:
            raise ValueError('wrong feature length: '+name)
        for i, v in enumerate(value):
            add(f'{name}.{i}', v, scale)
    e, f, r = o['estimate'], o['feedback'], o['referee']
    for state in ('lost', 'detecting', 'tracking', 'temp_lost'):
        add('state.'+state, e['state'] == state)
    for label in range(9):
        add('label.'+str(label), e['label'] == label)
    for kind in range(2):
        add('armor_type.'+str(kind), e['armor_type'] == kind)
    vector('center_world_div_10m', e['center_world'], 10, 3)
    vector('velocity_world_div_10mps', e['velocity_world'], 10, 3)
    vector('radii_div_0.5m', e['radii_m'], .5, 2)
    add('height_offset_div_0.5m', e['height_offset_m'], .5)
    q = e['orientation_xyzw']
    sign = -1 if q[3] < 0 else 1
    vector('orientation_xyzw', [sign*v for v in q], 1, 4)
    add('yaw_velocity_div_10radps', e['yaw_velocity_rad_s'], 10)
    add('yaw_variance_div_pi_squared', e['yaw_variance_rad2'], math.pi**2)
    add('feedback.valid', f['valid'])
    for axis in ('yaw', 'pitch'):
        add('feedback.'+axis+'.sin', math.sin(f[axis+'_rad']))
        add('feedback.'+axis+'.cos', math.cos(f[axis+'_rad']))
        add('feedback.'+axis+'_velocity_div_10radps', f[axis+'_velocity_rad_s'], 10)
    for field, scale in (('prediction_age_s',.1),('feedback_age_s',.1),('referee_age_s',.3),('since_request_s',1)):
        # 缺失的数据年龄按不可用或过期处理，不能编码成新鲜的零年龄样本。
        # 从未请求射击时不存在请求间隔限制，按已达到归一化上限编码。
        add(field+'_normalized', scale if o[field] is None else o[field], scale)
    for slot in range(-1,4):
        add('previous_slot.'+str(slot), o['previous_slot'] == slot)
    for field in ('valid','alive','fire_permitted','unlimited'):
        add('referee.'+field,r[field])
    add('referee.allowance_div_200',r['allowance_remaining'],200)
    add('referee.heat_fraction',r['heat'],max(float(r['heat_limit']),1))
    add('referee.heat_limit_div_400',r['heat_limit'],400)
    add('referee.cooling_div_100',r['cooling_per_second'],100)
    for bit in range(8):
        add('referee.block.'+str(bit),bool(r['fire_blocks'] & (1 << bit)))
    if len(o['candidates']) != 4:
        raise ValueError('four candidate slots required')
    for i,c in enumerate(o['candidates']):
        prefix=f'candidate.{i}.'
        add(prefix+'valid',c['valid'])
        # 无效候选填零，避免保留旧槽位数据。
        valid = bool(c['valid'])
        yaw = c['yaw_rad']-f['yaw_rad'] if valid else 0
        add(prefix+'yaw_error_sin',math.sin(yaw) if valid else 0)
        add(prefix+'yaw_error_cos',math.cos(yaw) if valid else 0)
        add(prefix+'pitch_error_rad',c['pitch_rad']-f['pitch_rad'] if valid else 0)
        for key,scale in (('distance_m',10),('fly_time_s',.5),('prediction_horizon_s',.5)):
            add(prefix+key+'_normalized',c[key] if valid else 0,scale)
    mask=o['action_mask']
    if len(mask)!=9 or any(type(v) is not bool for v in mask):
        raise ValueError('nine boolean action-mask entries required')
    if names != FEATURE_NAMES:
        raise ValueError("feature order differs from frozen schema")
    return names,list(array('f',values)),list(mask)


class TensorPolicy:
    """维护策略特征历史，并将独立副本交给 actor。

    即使 LOST 状态跳过策略回调，EvaluationSession 仍调用 begin_step。
    重置时清空历史；无观测周期填零并标记 valid=false。
    本适配器保留原动作掩码和返回动作，由桥接执行动作合法性检查。
    """
    def __init__(self, actor):
        self.actor=actor
        self.feature_count=len(FEATURE_NAMES)
        self.names=None
        self.reset()

    def reset(self):
        self.rows=deque([[0.0]*self.feature_count for _ in range(HISTORY)],maxlen=HISTORY)
        self.valid=deque([False]*HISTORY,maxlen=HISTORY)
        self.pending=False
        self.filled=False
        self.generation=None

    def set_generation(self, generation):
        """目标代次变化时清空历史；代次仅用于复位，不作为 actor 的输入特征。"""
        if self.generation is not None and generation != self.generation:
            pending = self.pending
            self.reset()
            if pending:
                self.begin_step()
        self.generation = generation

    def begin_step(self):
        self.rows.append([0.0]*self.feature_count)
        self.valid.append(False)
        self.pending=True
        self.filled=False

    def __call__(self, observation):
        if not self.pending or self.filled:
            raise RuntimeError('one tensor observation per begun control tick required')
        names,values,mask=encode(observation)
        if len(values)!=self.feature_count or (self.names is not None and names!=self.names):
            raise ValueError('feature schema changed')
        self.names=names
        self.rows[-1]=values
        self.valid[-1]=True
        self.filled=True
        # 向 actor 提供独立列表，防止其修改后续周期的历史或掩码。
        return self.actor(dict(version=VERSION,features=[list(v) for v in self.rows],
                               valid=list(self.valid),action_mask=mask))
