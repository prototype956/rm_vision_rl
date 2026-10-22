"""File-exchanged on-policy PPO: persistent NUC collection, server-only optimization.

Transfer checkpoints/NPZ over SSH, then publish the corresponding ready JSON last.
No desktop/video stream or per-control-tick network round trip is needed. A collector
keeps its native processes, estimator and history alive while waiting for new weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3.common.logger import configure

from src.training.config import read_config
from src.training.parallel import make_training_vector
from src.training.models import (MeasuredMaskablePPO, build_model, training_metadata,
                                 read_checkpoint_metadata, save_checkpoint)
from src.training.train import SamplingGuard
from src.training.asymmetric_policy import export_actor

ARRAYS = ('actions','rewards','episode_starts','values','log_probs','advantages','returns','action_masks')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2))
    os.replace(temporary,path)


def collect(config, output, updates, timeout):
    """Run on NUC. Each policy is used for exactly one complete rollout before updating."""
    if config['policy_kind'] != 'asymmetric_mlp':
        raise ValueError('split collector requires asymmetric_mlp')
    config['device']='cpu'
    config['total_timesteps']=config['ppo']['n_steps']*config['n_envs']*updates
    env=make_training_vector(config,output)
    try:
        model=build_model(config,env)
        metadata=training_metadata(config,env)
        model.set_logger(configure(str(output/'collector-log'),['stdout','csv']))
        _,callback=model._setup_learn(config['total_timesteps'],callback=SamplingGuard(),
                                      reset_num_timesteps=True)
        callback.on_training_start({}, {})
        for index in range(updates):
            success=model.collect_rollouts(env,callback,model.rollout_buffer,
                n_rollout_steps=config['ppo']['n_steps'],use_masking=True)
            if not success: raise RuntimeError('collection interrupted')
            checkpoint=output/f'rollout-{index:04d}.zip'
            batch=output/f'rollout-{index:04d}.npz'
            save_checkpoint(model,metadata,checkpoint,index)
            b=model.rollout_buffer
            arrays={key:getattr(b,key) for key in ARRAYS}
            arrays.update({'obs.'+key:value for key,value in b.observations.items()})
            arrays['contract']=np.asarray(json.dumps(dict(version=1,update=index,
                checkpoint_sha256=digest(checkpoint),num_timesteps=model.num_timesteps,
                observation=metadata['observation'])))
            with batch.with_suffix('.npz.partial').open('wb') as file:
                np.savez_compressed(file,**arrays)
            os.replace(batch.with_suffix('.npz.partial'),batch)
            publish(output/f'rollout-{index:04d}.ready.json',dict(checkpoint=checkpoint.name,
                batch=batch.name,checkpoint_sha256=digest(checkpoint),batch_sha256=digest(batch)))
            print(f'ROLLOUT_READY {index} steps={model.num_timesteps}',flush=True)
            if index+1 == updates: break
            ready=output/f'policy-{index:04d}.ready.json'
            deadline=time.monotonic()+timeout
            while not ready.exists():
                if time.monotonic()>=deadline: raise TimeoutError(f'waiting for {ready}')
                time.sleep(.2)
            response=json.loads(ready.read_text())
            incoming=output/f'policy-{index:04d}.zip'
            if digest(incoming)!=response['checkpoint_sha256']:
                raise ValueError('incomplete policy transfer')
            saved=read_checkpoint_metadata(incoming)
            if (saved['observation']!=metadata['observation'] or
                    saved['fingerprints']!=metadata['fingerprints'] or
                    saved['num_timesteps']!=model.num_timesteps or
                    saved['completed_updates']!=index+1):
                raise ValueError('received stale or incompatible policy')
            # Only tensor parameters/optimizer state cross back into the live collector.
            # Do not deserialize server-side Gym/NumPy runtime objects or replace env state.
            model.set_parameters(incoming,exact_match=True,device='cpu')
            model._n_updates=int(response['metrics']['train/n_updates'])
            # Deliberately retain _last_obs, episode starts, random streams and env history.
        callback.on_training_end()
    finally: env.close()


def update(checkpoint, batch, output, device):
    """Run on the server; no simulator/vision process is constructed by this entry."""
    metadata=read_checkpoint_metadata(checkpoint)
    if metadata['config']['policy_kind']!='asymmetric_mlp':
        raise ValueError('split update requires asymmetric checkpoint')
    torch.set_num_threads(metadata['config']['torch_threads'])
    model=MeasuredMaskablePPO.load(checkpoint,device=device)
    before={key:v.clone() for key,v in model.policy.state_dict().items()}
    b=model.rollout_buffer
    with np.load(batch,allow_pickle=False) as data:
        contract=json.loads(str(data['contract']))
        if (contract['version']!=1 or contract['checkpoint_sha256']!=digest(checkpoint)
                or contract['observation']!=metadata['observation']
                or contract['num_timesteps']!=model.num_timesteps
                or contract['update']!=metadata['completed_updates']):
            raise ValueError('rollout belongs to a different behavior policy/checkpoint')
        expected=set(ARRAYS)|{'obs.'+key for key in b.observations}|{'contract'}
        if set(data.files)!=expected: raise ValueError('rollout whitelist mismatch')
        for key in ARRAYS:
            value=data[key]
            if value.shape!=getattr(b,key).shape or not np.isfinite(value).all():
                raise ValueError('invalid rollout '+key)
            setattr(b,key,value.copy())
        for key in b.observations:
            value=data['obs.'+key]
            if value.shape!=b.observations[key].shape or not np.isfinite(value).all():
                raise ValueError('invalid observation '+key)
            b.observations[key]=value.copy()
    example={key:torch.as_tensor(value[0],device=model.device) for key,value in b.observations.items()}
    b.full=True; b.pos=b.buffer_size; b.generator_ready=False
    model.set_logger(configure(str(output/'optimizer-log'),['stdout','csv']))
    model._update_current_progress_remaining(model.num_timesteps,model._total_timesteps)
    model.train()
    if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
        raise FloatingPointError('non-finite optimizer result')
    changed=sum(float((value-before[key]).abs().sum())
                for key,value in model.policy.state_dict().items())
    if changed<=0: raise RuntimeError('optimizer did not update parameters')
    count=metadata['completed_updates']+1
    destination=output/f'policy-{count-1:04d}.zip'
    save_checkpoint(model,metadata,destination,count)
    actor_path=output/f'actor-{count-1:04d}.pt'
    deployed=export_actor(model.policy,example,actor_path)
    pure={key:value.detach().cpu() for key,value in example.items() if key!='privileged'}
    expected=model.policy.get_distribution(example,example['action_mask']).mode()
    assert torch.equal(expected.cpu(),deployed(pure)[2])
    assert not any('critic' in key or 'value_net' in key for key in deployed.state_dict())
    restored=MeasuredMaskablePPO.load(destination,device=device)
    assert torch.equal(model.policy.actor_logits(example),restored.policy.actor_logits(example))
    metrics={key:float(value) for key,value in model.logger.name_to_value.items()
             if key.startswith('train/')}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError('non-finite optimizer metrics')
    model.logger.dump(step=model.num_timesteps)
    report=dict(checkpoint_sha256=digest(destination),completed_updates=count,
                num_timesteps=model.num_timesteps,parameter_delta_l1=changed,metrics=metrics,
                actor_only_export=True,save_load=True,device=str(model.device))
    publish(output/f'policy-{count-1:04d}.ready.json',report)
    print('UPDATE_OK '+json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['collect','update'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--batch',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=1)
    parser.add_argument('--wait-timeout',type=float,default=3600)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    if args.updates<1: raise ValueError('updates must be positive')
    if args.stage=='collect': collect(read_config(args.config),args.output,args.updates,args.wait_timeout)
    else: update(args.checkpoint,args.batch,args.output,args.device)


if __name__=='__main__': main()
