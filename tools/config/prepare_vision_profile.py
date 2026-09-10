"""Materialize a reviewable training config copy; never edit the vision project's defaults."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
from tools.validation.validate import require


def prepare(vision_root, profile, output):
    vision_root, output = vision_root.resolve(), output.resolve()
    source = vision_root/'src/config/modules'
    require(output != vision_root and vision_root not in output.parents,
            'profile output must be outside the vision project')
    require(not output.exists(), 'use a new profile directory; do not overwrite a prior experiment')
    settings = json.loads(profile.read_text())
    require(set(settings) == {'version', 'name', 'source_max_iterations', 'max_iterations'} and
            settings['version'] == 1, 'unsupported profile')
    for key in ('source_max_iterations', 'max_iterations'):
        require(type(settings[key]) is int and 0 < settings[key] <= 2000, 'invalid iteration budget')
    original = (source/'gimbal_trajectory_planner.yaml').read_text()
    pattern = rf"(?m)^([ \t]*max_iterations:[ \t]*){settings['source_max_iterations']}([ \t]*(?:#.*)?)$"
    require(len(re.findall(pattern, original)) == 1,
            'baseline solver config changed; review the profile again')
    target = output/'src/config/modules'
    shutil.copytree(source, target)
    (target/'gimbal_trajectory_planner.yaml').write_text(
        re.sub(pattern, lambda m: m[1]+str(settings["max_iterations"])+m[2], original))
    def hashes(root):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob('*')) if p.is_file()}
    before, after = hashes(source), hashes(target)
    require([k for k in before if before[k] != after[k]] == ['gimbal_trajectory_planner.yaml'],
            'unexpected profile differences')
    manifest = dict(profile=settings, source=str(source), source_sha256=before, profile_sha256=after)
    (output/'profile.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    for name in ('vision-root', 'profile', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.vision_root, a.profile, a.output)
    print(a.output.resolve())


if __name__ == '__main__':
    main()
