"""Read a stable Talos v7 combat snapshot for validation only; never consume or mutate IPC."""
import argparse
import json
from pathlib import Path
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
args = parser.parse_args()
meta = Path('/tmp/talos_ipc_meta')
for _ in range(20):
    data = meta.read_bytes()
    if len(data) != 74752 or struct.unpack_from('<II', data) != (0x54414C07, 7):
        raise SystemExit('Expected Talos v7 metadata')
    slot = max(range(3), key=lambda i: struct.unpack_from('<Q', data, 128 + i * 24768)[0])
    begin = 128 + slot * 24768
    end = begin + 24768
    if data[begin:end] != meta.read_bytes()[begin:end]:
        continue
    combat = begin + 6144
    count = struct.unpack_from('<I', data, combat + 48)[0]
    if count > 16:
        raise SystemExit('Invalid robot count')
    rows = []
    names = ('actual_shots', 'rejected_requests', 'damage_dealt', 'damage_taken', 'kills',
             'armor_contacts', 'damaging_hits', 'heat_lock_count', 'heat_locked_s')
    for i in range(count):
        base = combat + 192 + i * 128
        robot_id, hp, max_hp = struct.unpack_from('<QII', data, base)
        rows.append(dict(id=robot_id, hp=hp, max_hp=max_hp,
                         heat=struct.unpack_from('<f', data, base + 16)[0],
                         **dict(zip(names, struct.unpack_from('<QQQQQQQQd', data, base + 48)))))
    result = dict(round_id=struct.unpack_from('<Q', data, combat)[0],
                  sim_time_ns=struct.unpack_from('<Q', data, combat + 8)[0], robots=rows)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    break
else:
    raise SystemExit('No stable snapshot; retry')
