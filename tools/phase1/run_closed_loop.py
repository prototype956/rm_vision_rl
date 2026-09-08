"""Run a bounded production Talos/vision validation with recording temporarily disabled."""
import argparse
import os
import signal
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seconds', type=int, default=180)
parser.add_argument('--output', required=True)
args = parser.parse_args()
vision = Path('/home/nuc/Workspace/rm_vision_2027')
config = vision / 'src/config/tool/foxglove.yaml'
original = config.read_bytes()
updated = original.replace(b'recording:\n  enabled: true', b'recording:\n  enabled: false')
if b'recording:\n  enabled: false' not in updated:
    raise SystemExit('Unrecognized recording configuration; refusing to launch.')
output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
backup = output.with_suffix('.foxglove-backup.yaml')
backup.write_bytes(original)
child = None

def interrupted(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    config.write_bytes(updated)
    with output.open('w') as log:
        child = subprocess.Popen(['./scripts/run_simulation_vision.sh'], cwd=vision,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'launcher pid={child.pid}; log={output}', flush=True)
        try:
            code = child.wait(timeout=args.seconds)
            print(f'launcher exited: {code}', flush=True)
        except subprocess.TimeoutExpired:
            print('bounded validation interval complete', flush=True)
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=15)
    if config.read_bytes() == updated:
        config.write_bytes(original)
        print('original recording configuration restored', flush=True)
    else:
        print(f'configuration changed externally; preserved it. Original backup: {backup}', flush=True)
