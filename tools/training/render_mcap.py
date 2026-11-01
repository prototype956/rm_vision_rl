"""Reconstruct two rendered views in a NEW MCAP without rerunning policy or physics.

Requires the optional requirements-replay.txt packages and the simulator's
training_preview built with --features training. Original recordings stay read-only.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from mcap.reader import make_reader
from mcap.writer import Writer
from mcap_protobuf.decoder import DecoderFactory

EPOCH = 1_000_000_000_000_000_000
ALIGN = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
IMAGE = '/vision/camera/image'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def rotation(q):
    """ROS/Bevy xyzw quaternion to an active rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def quaternion(r):
    """Stable matrix conversion, including 180 degree camera rotations."""
    candidates = [1+r[0,0]-r[1,1]-r[2,2], 1-r[0,0]+r[1,1]-r[2,2],
                  1-r[0,0]-r[1,1]+r[2,2], 1+np.trace(r)]
    i = int(np.argmax(candidates)); s = math.sqrt(max(candidates[i], 0.)) * 2
    if i == 0: q = [s/4, (r[0,1]+r[1,0])/s, (r[0,2]+r[2,0])/s, (r[2,1]-r[1,2])/s]
    elif i == 1: q = [(r[0,1]+r[1,0])/s, s/4, (r[1,2]+r[2,1])/s, (r[0,2]-r[2,0])/s]
    elif i == 2: q = [(r[0,2]+r[2,0])/s, (r[1,2]+r[2,1])/s, s/4, (r[1,0]-r[0,1])/s]
    else: q = [(r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s, s/4]
    return (np.asarray(q)/np.linalg.norm(q)).tolist()


def lerp_quaternion(a, b, t):
    a, b = np.asarray(a), np.asarray(b)
    dot = float(a@b)
    if dot < 0: b, dot = -b, -dot
    if dot > .9995: q = a+(b-a)*t
    else:
        theta = math.acos(np.clip(dot, -1, 1))
        q = (math.sin((1-t)*theta)*a+math.sin(t*theta)*b)/math.sin(theta)
    return (q/np.linalg.norm(q)).tolist()


def bracket(values, stamp):
    keys = sorted(values)
    i = bisect_left(keys, stamp)
    if i < len(keys) and keys[i] == stamp: return values[stamp], values[stamp], 0.
    if i == 0 or i == len(keys): raise ValueError(f'no physical bracket for capture {stamp}')
    left, right = keys[i-1:i+1]
    if right-left > 10_000_000: raise ValueError('physical render bracket exceeds 10 ms')
    return values[left], values[right], (stamp-left)/(right-left)


def reconstruct_camera(plates, calibration):
    """Recover optical pose from capture-time world corners and ideal truth rays.

    Control-time TF is not a capture pose. Use a planar initializer followed by
    all-ray refinement, without using noisy detections, PnP or filter estimates.
    The renderer independently checks its actual loaded meshes against these rays.
    """
    k = list(calibration.K); fx, fy, cx, cy = k[0], k[4], k[2], k[5]
    usable=[p for p in plates if p.get('ideal_corners_px') is not None and len(p['ideal_corners_px'])==4]
    if not usable:raise ValueError('no capture-time truth rays for camera reconstruction')
    # Historical files contain control-time TF, not necessarily capture-time TF.
    # Recover optical pose from a known world plate and its noise-free projection.
    # Prefer the largest image quadrilateral to avoid a grazing plane's conditioning.
    def image_area(p):
        pixels=np.asarray(p['ideal_corners_px']);a=pixels[1]-pixels[0];b=pixels[3]-pixels[0]
        return abs(a[0]*b[1]-a[1]*b[0])
    plate=max(usable,key=image_area)
    world=np.asarray(plate['corners_world']); uv=(np.asarray(plate['ideal_corners_px'])-[cx,cy])/[fx,fy]
    x=world[1]-world[0];x=x/np.linalg.norm(x)
    y=world[3]-world[0];y=y-x*(x@y);y=y/np.linalg.norm(y)
    basis=np.column_stack((x,y,np.cross(x,y)))
    plane=(world-world[0])@basis
    lhs=[];rhs=[]
    for (u,v),(px,py,_) in zip(uv,plane):
        lhs.extend([[px,py,1,0,0,0,-u*px,-u*py],[0,0,0,px,py,1,-v*px,-v*py]])
        rhs.extend([u,v])
    h=np.append(np.linalg.solve(lhs,rhs),1.).reshape(3,3)
    scale=(np.linalg.norm(h[:,0])+np.linalg.norm(h[:,1]))*.5
    columns=np.column_stack((h[:,0]/scale,h[:,1]/scale,np.cross(h[:,0],h[:,1])/scale**2))
    u,_,v=np.linalg.svd(columns);plane_camera=u@v
    if np.linalg.det(plane_camera)<0:raise ValueError('reflected camera reconstruction')
    world_camera=(plane_camera@basis.T).T
    origin_ros=world[0]-world_camera@(h[:,2]/scale)
    points, pixels = [], []
    for plate in plates:
        ideal = plate.get('ideal_corners_px')
        if ideal is not None and len(ideal) == 4:
            points.extend(plate['corners_world']); pixels.extend(ideal)
    if len(points) < 4: raise ValueError('no capture-time truth rays for camera reconstruction')
    points, pixels = np.asarray(points), np.asarray(pixels)
    r=world_camera.T;translation=-r@origin_ros
    def exp_rotation(delta):
        angle=float(np.linalg.norm(delta))
        if angle < 1e-14:return np.eye(3)
        x,y,z=delta/angle;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        return np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew)
    def project(r,t):
        actual=points@r.T+t
        return actual[:,:2]/actual[:,2,None]*[fx,fy]+[cx,cy]
    # GLB corner coordinates are serialized as f32; refine the planar initializer
    # against all available ideal truth rays, without using any noisy detections.
    for _ in range(15):
        projected=project(r,translation);residual=(projected-pixels).reshape(-1)
        if abs(residual).max()<1e-5:break
        columns=[];epsilon=1e-6
        for axis in range(6):
            delta=np.zeros(3);delta[axis%3]=epsilon
            trial=project(exp_rotation(delta)@r,translation) if axis<3 else project(r,translation+delta)
            columns.append(((trial-projected)/epsilon).reshape(-1))
        step=np.linalg.lstsq(np.column_stack(columns),-residual,rcond=None)[0]
        accepted=False
        for factor in (1.,.5,.25,.125):
            trial_r=exp_rotation(step[:3]*factor)@r;trial_t=translation+step[3:]*factor
            if np.linalg.norm(project(trial_r,trial_t)-pixels)<np.linalg.norm(residual):
                r,translation=trial_r,trial_t;accepted=True;break
        if not accepted:break
    projected=project(r,translation)
    world_camera=r.T;origin_ros=-world_camera@translation
    error = float(np.linalg.norm(projected-pixels, axis=1).max())
    if not math.isfinite(error) or error > .02:
        raise ValueError(f'capture camera/truth projection mismatch: {error:.6f} px')
    # Bevy camera looks along -Z with +Y up; optical camera uses +Z, -Y.
    render_rotation = ALIGN.T @ world_camera @ np.diag([1., -1., -1.])
    return dict(translation=(ALIGN.T@origin_ros).tolist(),rotation_xyzw=quaternion(render_rotation)), error


def prepare(source, work, *, limit=None):
    source, work = Path(source).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    decoder = DecoderFactory(); physical={}; truth={}; controls={}; images=set(); calibration=None; info=None
    with source.open('rb') as stream:
        for schema, channel, message in make_reader(stream, validate_crcs=True).iter_messages():
            topic, stamp = channel.topic, message.log_time
            if topic == '/replay/physical':
                row=json.loads(message.data); physical[stamp]=row
                if stamp != EPOCH+row['sim_time_ns']: raise ValueError('unknown physical timestamp epoch')
                for t in row['evaluation'].get('visual_truth',[]): truth[EPOCH+t['capture_time_ns']]=t
            elif topic == '/replay/info': info=json.loads(message.data)
            elif topic == '/vision/control/state': controls[stamp]=json.loads(message.data)['angles']
            elif topic == IMAGE: images.add(stamp)
            elif topic == '/vision/camera/calibration':
                calibration=decoder.decoder_for(channel.message_encoding,schema)(message.data)
    if not info or not info['complete'] or not calibration: raise ValueError('complete replay and camera calibration required')
    if (abs(calibration.K[0]-calibration.K[4])>1e-6 or abs(calibration.K[2]-calibration.width*.5)>1e-6
        or abs(calibration.K[5]-calibration.height*.5)>1e-6 or any(abs(d)>1e-12 for d in calibration.D)):
        raise ValueError('renderer currently requires centered square-pixel, undistorted camera optics')
    if len({r['round_id'] for r in physical.values()}) != 1: raise ValueError('export one round at a time')
    stamps=sorted(images)
    if limit is not None: stamps=stamps[:limit]
    frames=[]; errors=[]
    first=physical[min(physical)]
    scenario=first['evaluation'].get('scenario')
    if not scenario: raise ValueError('resolved render scenario missing')
    for stamp in stamps:
        t=truth[stamp]
        camera,error=reconstruct_camera(t['plates'],calibration);errors.append(error)
        left,right,fraction=bracket(physical,stamp)
        data=copy.deepcopy(left)
        data['evaluation']['scenario']=scenario
        later={r['robot_id']:r for r in right['evaluation']['robots']}
        captured={r['id']:r for r in t['targets']}
        for robot in data['evaluation']['robots']:
            nxt=later[robot['robot_id']]
            root=captured[robot['robot_id']]
            robot['position_bevy_m']=(ALIGN.T@root['position_world']).tolist()
            q=lerp_quaternion(robot['rotation_xyzw'],nxt['rotation_xyzw'],fraction)
            # Capture truth supplies actual chassis heading. Retain root roll/pitch.
            r=rotation(q); yaw=math.atan2(r[0,2],r[2,2]); dy=root['yaw']-yaw
            c,s=math.cos(dy),math.sin(dy)
            robot['rotation_xyzw']=quaternion(np.array([[c,0,s],[0,1,0],[-s,0,c]])@r)
            for field in ('fold','lift_m'):
                a,b=robot['mechanism'][field],nxt['mechanism'][field]
                robot['mechanism'][field]=a+(b-a)*fraction
        own=next(r for r in data['evaluation']['robots'] if r['robot_id']==1)
        if np.linalg.norm(np.asarray(camera['translation'])-own['position_bevy_m'])>1.:
            raise ValueError('reconstructed optical origin is outside the controlled robot mount envelope')
        a,b,f=bracket(controls,stamp)
        data['feedback']={key:a['feedback_'+key]+math.atan2(math.sin(b['feedback_'+key]-a['feedback_'+key]),
            math.cos(b['feedback_'+key]-a['feedback_'+key]))*f for key in ('yaw','pitch')}
        data['feedback']={'yaw_rad':data['feedback']['yaw'],'pitch_rad':data['feedback']['pitch']}
        # Projectile interpolation is display-only, never supplied to an Actor.
        later_bullets={b['projectile_id']:b for b in right['evaluation']['projectiles']}
        for bullet in data['evaluation']['projectiles']:
            if bullet['projectile_id'] in later_bullets:
                p=np.asarray(bullet['position_bevy_m']);q=np.asarray(later_bullets[bullet['projectile_id']]['position_bevy_m'])
                bullet['position_bevy_m']=(p+(q-p)*fraction).tolist()
        frames.append(dict(sequence=t['sequence'],timestamp_ns=stamp,camera=camera,data=data,plates=t['plates']))
    manifest=dict(version=1,width=calibration.width,height=calibration.height,fy=calibration.K[4],
        output=str(work/'images'),frames=frames)
    (work/'render-manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
    receipt=dict(source=str(source),source_sha256=digest(source),frames=len(frames),
        complete_source_frames=len(images),max_camera_truth_projection_error_px=max(errors),
        simulator_config_sha256=info.get('runtime_fingerprints',{}).get('simulator_config'),
        model=info['model'],first_timestamp_ns=stamps[0],last_timestamp_ns=stamps[-1],
        reconstruction='capture-time camera and target truth; <=10ms interpolated mechanisms/projectiles; no physics or inference')
    (work/'prepare.json').write_text(json.dumps(receipt,indent=2))
    return receipt


def assemble(source, work, output, renderer, *, combat_hud=True):
    source,work,output=Path(source).resolve(),Path(work).resolve(),Path(output).resolve()
    from tools.training.combat_hud import HUD_TOPICS
    requested_output=output
    if combat_hud:
        output=output.with_name(output.stem+'.render-only'+output.suffix)
    if requested_output.exists():raise FileExistsError(requested_output)
    receipt=json.loads((work/'prepare.json').read_text())
    if digest(source)!=receipt['source_sha256']:raise ValueError('source MCAP changed')
    complete=json.loads((work/'images/complete.json').read_text())
    if not complete['complete'] or complete['frames']!=receipt['complete_source_frames']:
        raise ValueError('all source image frames must have both rendered views')
    if complete.get('max_mesh_projection_error_px',float('inf')) > .5:
        raise ValueError('rendered mesh alignment was not verified')
    manifest=json.loads((work/'render-manifest.json').read_text())
    by_time={r['timestamp_ns']:r['sequence'] for r in manifest['frames']}
    provenance={**receipt,'renderer_sha256':digest(renderer),'views':['first_person','third_person'],
        'first_person_topic':IMAGE,'third_person_topic':'/replay/render/third_person',
        'measurement_canvas_topic':'/replay/render/measurement_canvas','rendered_simulation':True,'real_camera':False}
    decoder=DecoderFactory();schemas={};channels={};other_hash=hashlib.sha256();other_copy_hash=hashlib.sha256();count=0
    with source.open('rb') as stream,output.open('xb') as target:
        writer=Writer(target);writer.start(profile='foxglove',library='rmvision-rendered-replay')
        third=None;canvas=None
        for schema,channel,message in make_reader(stream,validate_crcs=True).iter_messages(log_time_order=False):
            if channel.topic in HUD_TOPICS:continue
            if schema is not None and schema.id not in schemas:schemas[schema.id]=writer.register_schema(schema.name,schema.encoding,schema.data)
            if channel.id not in channels:
                channels[channel.id]=writer.register_channel(channel.topic,channel.message_encoding,
                    schemas.get(channel.schema_id,0),channel.metadata)
            data=message.data
            if channel.topic==IMAGE:
                if third is None:
                    third=writer.register_channel('/replay/render/third_person',channel.message_encoding,schemas[channel.schema_id])
                    canvas=writer.register_channel('/replay/render/measurement_canvas',channel.message_encoding,schemas[channel.schema_id])
                writer.add_message(canvas,message.log_time,message.data,message.publish_time,message.sequence)
                proto=decoder.decoder_for(channel.message_encoding,schema)(data)
                seq=by_time[message.log_time]
                for view,cid in (('first_person',channels[channel.id]),('third_person',third)):
                    jpeg=(work/f'images/{view}/{seq:06}.jpg').read_bytes()
                    if jpeg[:2]!=b'\xff\xd8':raise ValueError('render output is not a JPEG')
                    proto.data=jpeg;proto.format='jpeg'
                    if view=='third_person':proto.frame_id='replay_third_person'
                    writer.add_message(cid,message.log_time,proto.SerializeToString(),message.publish_time,message.sequence)
                count+=1;continue
            if channel.topic=='/replay/info':
                info=json.loads(data);info['render']=provenance
                info['availability'].update(rendered_camera=True,third_person=True,real_camera=False)
                data=json.dumps(info,separators=(',',':')).encode()
            else:
                key=channel.topic.encode()+message.log_time.to_bytes(8,'little')
                other_hash.update(key+message.data);other_copy_hash.update(key+data)
            writer.add_message(channels[channel.id],message.log_time,data,message.publish_time,message.sequence)
        writer.add_metadata('rmvision.render',{'provenance':json.dumps(provenance)})
        writer.finish()
    if count!=receipt['frames'] or other_hash.digest()!=other_copy_hash.digest():raise ValueError('diagnostic preservation failed')
    # Re-read actual output rather than trusting the write loop.
    output_hash=hashlib.sha256();render_counts={IMAGE:0,'/replay/render/third_person':0}
    with output.open('rb') as stream:
        for _,channel,message in make_reader(stream,validate_crcs=True).iter_messages(log_time_order=False):
            if channel.topic in render_counts:render_counts[channel.topic]+=1
            elif channel.topic not in ('/replay/info','/replay/render/measurement_canvas'):
                output_hash.update(channel.topic.encode()+message.log_time.to_bytes(8,'little')+message.data)
    if output_hash.digest()!=other_hash.digest() or any(n!=count for n in render_counts.values()):
        raise ValueError('output diagnostics or rendered frame counts changed')
    if digest(source)!=receipt['source_sha256']:raise ValueError('source changed during render assembly')
    result=dict(passed=True,output=str(output),output_sha256=digest(output),source_sha256=digest(source),
        frames_per_view=count,diagnostic_payloads_preserved=True,crcs_valid=True,provenance=provenance)
    if combat_hud:
        from tools.training.combat_hud import augment_mcap
        hud=augment_mcap(output,requested_output)
        result.update(render_only_output=str(output),output=str(requested_output),output_sha256=hud['output_sha256'],combat_hud=True)
    (work/'validation.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--renderer',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--assets',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--limit',type=int)
    p.add_argument('--combat-hud',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args()
    if a.limit is not None and a.limit<=0:p.error('--limit must be positive')
    receipt=prepare(a.source,a.work,limit=a.limit)
    print(json.dumps({k:v for k,v in receipt.items() if k!='model'}),flush=True)
    if a.prepare_only:return
    if not all((a.renderer,a.output,a.config,a.assets)):p.error('render requires --renderer, --output, --config and --assets')
    if digest(a.config)!=receipt['simulator_config_sha256']:
        raise ValueError('restore the original simulator config before rendering')
    subprocess.run([str(a.renderer.resolve()),'--render-manifest',str((a.work/'render-manifest.json').resolve()),
        '--config',str(a.config.resolve()),'--assets',str(a.assets.resolve())],check=True)
    print(json.dumps(assemble(a.source,a.work,a.output,a.renderer,combat_hud=a.combat_hud)))


if __name__=='__main__':main()
