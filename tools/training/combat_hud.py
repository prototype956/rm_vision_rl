"""Seek-safe combat HUD annotations derived from a completed replay, never policy input."""
from __future__ import annotations

import argparse
import base64
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import struct
import unicodedata

from src.scoring.window import WindowScore

IMAGES = {'/vision/camera/image': '/replay/hud/first_person',
          '/replay/render/third_person': '/replay/hud/third_person'}
HUD_TOPICS = frozenset(IMAGES.values())
METADATA = 'rmvision.combat_hud'
WHITE = dict(r=1., g=1., b=1., a=1.)
GREEN = dict(r=.35, g=1., b=.45, a=1.)
YELLOW = dict(r=1., g=.85, b=.25, a=1.)
SCHEMA = dict(type='object', properties=dict(texts=dict(type='array', items=dict(type='object', properties={
    'timestamp': dict(type='object', properties=dict(sec=dict(type='integer'), nsec=dict(type='integer'))),
    'position': dict(type='object', properties=dict(x=dict(type='number'), y=dict(type='number'))),
    'text': dict(type='string'), 'font_size': dict(type='number'),
    'text_color': dict(type='object', properties={k: dict(type='number') for k in 'rgba'}),
    'background_color': dict(type='object', properties={k: dict(type='number') for k in 'rgba'}),
    'metadata': dict(type='array', items=dict(type='object', properties=dict(key=dict(type='string'), value=dict(type='string')))),
}))))
SCHEMA['properties'].update(circles=dict(type='array',items=dict(type='object')),points=dict(type='array',items=dict(type='object')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def image_size(data):
    """Read JPEG/PNG dimensions without decoding/recompressing the original pixels."""
    if data.startswith(b'\x89PNG\r\n\x1a\n') and len(data) >= 24:
        return struct.unpack('>II', data[16:24])
    if data[:2] != b'\xff\xd8': raise ValueError('HUD requires JPEG or PNG image dimensions')
    i = 2
    while i < len(data):
        if data[i] != 255: raise ValueError('invalid JPEG marker')
        while i < len(data) and data[i] == 255: i += 1
        marker = data[i]; i += 1
        if marker in (0xd8, 0x01) or 0xd0 <= marker <= 0xd7: continue
        if marker in (0xd9, 0xda): break
        length = int.from_bytes(data[i:i+2], 'big')
        if length < 2 or i+length > len(data): raise ValueError('truncated JPEG segment')
        if marker in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
            height, width = struct.unpack('>HH', data[i+3:i+7]); return width, height
        i += length
    raise ValueError('JPEG dimensions unavailable')


class CombatTimeline:
    """Build full snapshots from authoritative event IDs and past physical samples.

    Event outcomes can arrive with a later physical packet than their own event
    time. They are admitted only at/after that event time, in original ledger ID
    order. Final scores are withheld until the final source message's log time.
    """
    def __init__(self, samples, info, final_time_ns):
        if not samples: raise ValueError('HUD requires /replay/physical samples')
        self.samples = dict(sorted(samples.items())); self.times = list(self.samples)
        rounds = {r['round_id'] for r in self.samples.values()}
        if len(rounds) != 1: raise ValueError('HUD exports one recorded round at a time')
        self.round_id = next(iter(rounds)); self.info = info
        offsets = {at-r['sim_time_ns'] for at, r in self.samples.items()}
        if len(offsets) != 1: raise ValueError('inconsistent replay physical timestamp origin')
        self.epoch = next(iter(offsets)); self.final_time = final_time_ns
        self.summary = info.get('summary') or {}
        self.expected = self.summary.get('settlement') or {}
        self.start = self.summary.get('start_ns'); self.end = self.summary.get('end_ns')
        self.has_window = type(self.start) is int and type(self.end) is int and 0 <= self.start < self.end
        unique = {}
        for row in self.samples.values():
            for event in row.get('events', []):
                key = event['event_id']
                if key in unique and unique[key] != event: raise ValueError('conflicting duplicate HUD event')
                unique[key] = event
        self.events = [unique[k] for k in sorted(unique)]
        self.baseline = self.physical(self.epoch+self.start) if self.has_window else {}
        self.initial_target_hp = self.robot(self.baseline, 2).get('hp')
        self.reset()
        # Audit the entire ledger before publishing any derived recording.
        self.verified_final = False
        if self.has_window:
            self.snapshot(final_time_ns)
            actual = self.ledger.summary()
            if self.expected:
                for key in ('eligible_shots', 'damaging_projectiles', 'eligible_damage_observed',
                            'eligible_unresolved', 'excluded_own_shots', 'excluded_own_damage',
                            'own_physical_damage', 'event_count', 'closed'):
                    if actual[key] != self.expected.get(key): raise ValueError(f'HUD/source final score mismatch: {key}')
                own = self.robot(self.physical(final_time_ns), 1)
                if own.get('actual_shots') != len([s for s in self.ledger.shots.values() if s['robot_id'] == 1]):
                    raise ValueError('HUD physical launch counter mismatch')
                if own.get('damage_dealt') != actual['own_physical_damage']:
                    raise ValueError('HUD physical damage counter mismatch')
                self.verified_final = (info.get('complete') is True and self.expected.get('status') == 'complete'
                    and self.expected.get('official_damage') == actual['eligible_damage_observed']
                    and actual['eligible_unresolved'] == 0 and actual['closed'])
                if self.expected.get('status') == 'complete' and not self.verified_final:
                    raise ValueError('HUD cannot confirm source completion')
        self.reset()

    @staticmethod
    def robot(row, ident):
        return next((r for r in row.get('evaluation', {}).get('robots', []) if r.get('robot_id') == ident), {})

    def physical(self, at):
        i = bisect_right(self.times, at)-1
        return self.samples[self.times[i]] if i >= 0 else {}

    def reset(self):
        self.ledger = WindowScore(self.round_id, self.start, self.end) if self.has_window else None
        self.cursor = 0; self.last_time = -1

    def snapshot(self, at):
        if at < self.last_time: self.reset()
        now = at-self.epoch; self.last_time = at
        if self.ledger is not None:
            events = []
            while self.cursor < len(self.events) and self.events[self.cursor]['time_ns'] <= now:
                events.append(self.events[self.cursor]); self.cursor += 1
            self.ledger.ingest(dict(ok=True, round_id=self.round_id, sim_time_ns=now, data=dict(events=events)))
            score = self.ledger.summary()
        else: score = {}
        own = self.robot(self.physical(at), 1); target = self.robot(self.physical(at), 2)
        resources = own.get('resources') or {}; baseline = self.robot(self.baseline, 1)
        base_resources = baseline.get('resources') or {}
        if not self.has_window:
            reason = str(self.summary.get('status', '') or self.info.get('error', ''))
            phase = ('warming' if at < self.final_time else
                'warmup_failed' if 'warmup' in reason or 'tracking_initialization' in reason else 'unavailable')
        elif now < self.start: phase = 'warming'
        elif at >= self.final_time:
            phase = 'complete' if getattr(self, 'verified_final', False) else 'incomplete'
        elif now >= self.end: phase = 'settling'
        else: phase = 'evaluating'
        active = phase not in ('warming', 'warmup_failed', 'unavailable')
        shots = score.get('eligible_shots') if active else None
        hits = score.get('damaging_projectiles') if active else None
        damage = score.get('eligible_damage_observed') if active else None
        target_hp = number(target.get('hp')); start_hp = number(self.initial_target_hp)
        killed = bool(active and target_hp is not None and target_hp <= 0 and start_hp is not None
                      and damage is not None and damage >= start_hp)
        times = [s['damage_time_ns'] for s in self.ledger.shots.values()
                 if s['classification'] == 'eligible' and s['damage'] > 0] if self.ledger else []
        def delta(current, initial):
            current, initial = number(current), number(initial)
            return max(0., current-initial) if active and current is not None and initial is not None else None
        return dict(timestamp_ns=at, phase=phase, shots=shots, hits=hits,
            hit_efficiency=hits/shots if shots else None, damage=damage,
            pending=score.get('eligible_unresolved') if active else None,
            own_hp=number(own.get('hp')), target_hp=target_hp, killed=killed,
            kill_time_s=(max(times)-self.start)*1e-9 if killed and times else None,
            heat=number(resources.get('heat')), heat_limit=number(resources.get('heat_limit')),
            heat_locked=resources.get('cooling_locked') is True or resources.get('round_locked') is True,
            ordinary_locks=delta(resources.get('ordinary_heat_lock_count'), base_resources.get('ordinary_heat_lock_count')),
            heat_locked_s=delta(own.get('heat_locked_s'), baseline.get('heat_locked_s')),
            elapsed_s=max(0., min(now, self.end)-self.start)*1e-9 if self.has_window else None,
            official_damage=self.expected.get('official_damage') if phase == 'complete' else None)


def annotations(snapshot, info, width, height, last_image_ns, *, epoch=0, start_ns=0):
    """Five Chinese text labels; all numeric values are complete seek-safe snapshots."""
    def value(v, decimals=0):
        return '不可用' if number(v) is None else (f'{v:.{decimals}f}' if decimals else str(int(v)))
    model = info.get('model') or {}; scene = info.get('scene') or {}
    scenario = scene.get('scenario') or {}; target = scenario.get('duel', {}).get('target', {})
    profiles = dict(standard='标准', wheel_leg='轮足', low='低矮', folding='折叠')
    motions = dict(radial='径向', oblique='斜向', lateral='横移')
    kind = '传统火控' if model.get('kind') == 'traditional' else 'GRU'
    steps = '' if kind == '传统火控' else f" {value(model.get('num_timesteps'))}步"
    stage = scene.get('stage_id') or ('C'+str(scene['course']) if 'course' in scene else '课程不可用')
    header = f"{kind}{steps} | {stage} {profiles.get(target.get('profile'), '车型不可用')}/{motions.get(scene.get('motion_type'), '运动不可用')} | 案例{scene.get('case', '不可用')}"
    phase = snapshot['phase']
    if phase == 'warming': hit = '命中效率 —（预热不计分）'
    elif snapshot['shots'] is None: hit = '命中效率 不可用 | 发射/命中 不可用'
    else:
        rate = f"{snapshot['hit_efficiency']*100:.2f}%" if snapshot['shots'] else '—（0/0）'
        hit = f"命中效率 {rate} | 有效命中/出膛 {snapshot['hits']}/{snapshot['shots']} | 在途 {snapshot['pending']}"
    damage_label = '最终伤害' if phase == 'complete' else '暂计伤害'
    battle = f"{damage_label} {value(snapshot['damage'])} | 目标HP {value(snapshot['target_hp'])} | 己方HP {value(snapshot['own_hp'])}"
    if snapshot['killed']: battle += ' | 已击毁'
    thermal = f"热量 {value(snapshot['heat'],1)}/{value(snapshot['heat_limit'])} | 普通热锁 {value(snapshot['ordinary_locks'])}次 | 锁定 {value(snapshot['heat_locked_s'],2)}s"
    labels = dict(warming='预热（不计分）', evaluating='评估中（暂计）', settling='尾部结算中（暂计）',
                  complete='最终成绩（完整结算）', incomplete='结算不完整', warmup_failed='预热失败', unavailable='成绩不可用／录制未完成')
    status = labels[phase]
    if snapshot['kill_time_s'] is not None: status += f" | 击毁 {snapshot['kill_time_s']:.2f}s"
    elif snapshot['elapsed_s'] is not None: status += f" | 评估 {snapshot['elapsed_s']:.2f}s"
    if snapshot['timestamp_ns'] > last_image_ns:
        image_s = max(0., (last_image_ns-epoch-start_ns)*1e-9)
        now_s = max(0., (snapshot['timestamp_ns']-epoch-start_ns)*1e-9)
        status += f" | 定格 {image_s:.2f}s／统计 {now_s:.2f}s"
    lines = [header, hit, battle, thermal, status]
    scale = min(width/1280, height/720); font = 22*scale
    def fit(text):
        units = 0.; result = ''
        for ch in text:
            units += 1. if unicodedata.east_asian_width(ch) in ('W','F') else .65
            if units*font > width-40*scale: return result+'…'
            result += ch
        return result
    timestamp = dict(sec=snapshot['timestamp_ns']//1_000_000_000, nsec=snapshot['timestamp_ns']%1_000_000_000)
    colors = [WHITE, WHITE, GREEN if snapshot['killed'] else WHITE,
              YELLOW if snapshot['heat_locked'] else WHITE,
              GREEN if phase == 'complete' else YELLOW if phase in ('settling','incomplete','warmup_failed','unavailable') else WHITE]
    return dict(circles=[],points=[],texts=[dict(timestamp=timestamp, position=dict(x=16*scale, y=(38+28*i)*scale), text=fit(text),
        font_size=font, text_color=colors[i], background_color=dict(r=0.,g=0.,b=0.,a=.65),
        metadata=[dict(key='full_text',value=text),dict(key='score_phase',value=phase)]) for i,text in enumerate(lines)])


def augment_mcap(source, destination):
    """Write a separate derived MCAP; replace only our own previously derived HUD."""
    from mcap.reader import make_reader
    from mcap.writer import Writer
    from mcap_protobuf.decoder import DecoderFactory
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists(): raise FileExistsError('HUD output must be a new file')
    source_hash = sha(source); samples = {}; info = None; final_time = None; images = {}; sizes = {}
    decoder = DecoderFactory()
    with source.open('rb') as stream:
        reader = make_reader(stream, validate_crcs=True)
        for schema, channel, message in reader.iter_messages():
            if channel.topic == '/replay/info': info=json.loads(message.data); final_time=message.log_time
            elif channel.topic == '/replay/physical':
                row=json.loads(message.data)
                if message.log_time in samples:
                    previous=samples[message.log_time].get('events', [])
                    row['events']=previous+row.get('events', [])
                samples[message.log_time]=row
            elif channel.topic in IMAGES:
                images.setdefault(channel.topic, []).append(message.log_time)
                if channel.topic not in sizes:
                    if channel.message_encoding == 'protobuf':
                        payload=decoder.decoder_for(channel.message_encoding,schema)(message.data).data
                    elif channel.message_encoding == 'json': payload=base64.b64decode(json.loads(message.data)['data'])
                    else: raise ValueError('unsupported replay image message encoding')
                    sizes[channel.topic]=image_size(payload)
        # File-level provenance such as the existing renderer receipt is retained.
        metadata=list(reader.iter_metadata())
    if info is None or not images: raise ValueError('HUD requires replay identity and image topics')
    timeline=CombatTimeline(samples,info,final_time)
    targets={}; hold_count=0
    for image_topic, stamps in images.items():
        times=set(stamps)
        tail_start=timeline.epoch+timeline.end if timeline.has_window else final_time
        times.update(t for t in timeline.times if t > max(stamps) and t >= tail_start)
        times.add(final_time)
        for at in sorted(times):
            if at < min(stamps): continue
            last=stamps[bisect_right(stamps,at)-1]
            targets.setdefault(at, []).append((image_topic,last))
            hold_count += int(at > last)
    schemas={}; channels={}; source_messages=0; counts={topic:0 for topic in images}; hud_last={}
    expected_hash=hashlib.sha256()
    with source.open('rb') as stream, destination.open('xb') as target:
        writer=Writer(target); writer.start(profile='foxglove',library='rmvision-combat-hud')
        hud_schema=writer.register_schema('foxglove.ImageAnnotations','jsonschema',json.dumps(SCHEMA).encode())
        hud_channels={topic:writer.register_channel(IMAGES[topic],'json',hud_schema) for topic in images}
        times=iter(sorted(targets)); upcoming=next(times,None)
        def emit(at):
            snapshot=timeline.snapshot(at)
            for topic,last in targets[at]:
                annotation=annotations(snapshot,info,*sizes[topic],last,epoch=timeline.epoch,start_ns=timeline.start or 0)
                writer.add_message(hud_channels[topic],at,json.dumps(annotation,ensure_ascii=False,separators=(',',':')).encode(),at,counts[topic])
                counts[topic]+=1; hud_last[topic]=snapshot
        for schema,channel,message in make_reader(stream,validate_crcs=True).iter_messages():
            while upcoming is not None and upcoming < message.log_time:
                emit(upcoming); upcoming=next(times,None)
            if channel.topic in HUD_TOPICS: continue
            if schema is not None and schema.id not in schemas: schemas[schema.id]=writer.register_schema(schema.name,schema.encoding,schema.data)
            if channel.id not in channels: channels[channel.id]=writer.register_channel(channel.topic,channel.message_encoding,schemas.get(channel.schema_id,0),channel.metadata)
            writer.add_message(channels[channel.id],message.log_time,message.data,message.publish_time,message.sequence)
            expected_hash.update(channel.topic.encode()+message.log_time.to_bytes(8,'little')+message.publish_time.to_bytes(8,'little')+
                message.sequence.to_bytes(8,'little')+message.data);source_messages+=1
        while upcoming is not None: emit(upcoming); upcoming=next(times,None)
        for record in metadata:
            if record.name != METADATA: writer.add_metadata(record.name,record.metadata)
        writer.add_metadata(METADATA,dict(version='1',source_sha256=source_hash,scope='single_battle',
            sync_timestamps='off',timing='capture-time statistics; explicit held-image settlement updates'))
        writer.finish()
    actual_hash=hashlib.sha256();actual_counts={};last_annotations={}
    with destination.open('rb') as stream:
        for _,channel,message in make_reader(stream,validate_crcs=True).iter_messages():
            if channel.topic in HUD_TOPICS:
                actual_counts[channel.topic]=actual_counts.get(channel.topic,0)+1
                last_annotations[channel.topic]=json.loads(message.data)
            else:actual_hash.update(channel.topic.encode()+message.log_time.to_bytes(8,'little')+message.publish_time.to_bytes(8,'little')+
                message.sequence.to_bytes(8,'little')+message.data)
    if expected_hash.digest()!=actual_hash.digest() or sha(source)!=source_hash:
        raise RuntimeError('HUD export changed source messages or the source recording')
    if actual_counts!={IMAGES[k]:v for k,v in counts.items()}: raise RuntimeError('HUD message count mismatch')
    result=dict(passed=True,source=str(source),output=str(destination),source_sha256=source_hash,output_sha256=sha(destination),
        original_messages_preserved=source_messages,crcs_valid=True,hud_messages=actual_counts,held_updates=hold_count,
        final_snapshots={IMAGES[k]:v for k,v in hud_last.items()},last_annotations=last_annotations)
    destination.with_suffix('.hud.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=augment_mcap(args.input,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('last_annotations','final_snapshots')},ensure_ascii=False))


if __name__=='__main__':main()
