"""Stream TensorBoard scalar records with bounded memory; never import TensorFlow."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import struct


def scalar(value):
    if value.HasField('simple_value'):
        return float(value.simple_value)
    tensor = value.tensor
    if tensor.float_val:
        return float(tensor.float_val[0])
    if tensor.double_val:
        return float(tensor.double_val[0])
    if tensor.dtype in (1, 2) and len(tensor.tensor_content) == (4 if tensor.dtype == 1 else 8):
        return struct.unpack('<f' if tensor.dtype == 1 else '<d', tensor.tensor_content)[0]
    return None


def main():
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto import event_pb2
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('event_file', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bin-size', type=int, default=1000)
    p.add_argument('--tags', default=r'Train/|Loss/|Policy/|Performance/|Episode_Reward/(v111|world_forward|track_lin|track_ang)|Curriculum/(terrain_levels|v111|v11_platform)|Episode_Termination/')
    a = p.parse_args()
    pattern = re.compile(a.tags)
    bins = defaultdict(lambda: [0., 0, float('inf'), float('-inf')])
    names = set()
    first = last = None
    for raw in RawEventFileLoader(str(a.event_file)).Load():
        event = event_pb2.Event.FromString(raw)
        if not event.HasField('summary'):
            continue
        first = event.step if first is None else min(first, event.step)
        last = event.step if last is None else max(last, event.step)
        for value in event.summary.value:
            names.add(value.tag)
            if not pattern.search(value.tag):
                continue
            number = scalar(value)
            if number is None:
                continue
            item = bins[(event.step//a.bin_size*a.bin_size, value.tag)]
            item[0] += number
            item[1] += 1
            item[2] = min(item[2], number)
            item[3] = max(item[3], number)
    result = dict(source=str(a.event_file), first_step=first, last_step=last,
        bin_size=a.bin_size, tags=sorted(names), bins=[dict(step=step, tag=tag, mean=v[0]/v[1],
            count=v[1], minimum=v[2], maximum=v[3]) for (step, tag), v in sorted(bins.items())])
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Scalar audit: {first}..{last}; {len(names)} tags; {len(bins)} aggregate rows')


if __name__ == '__main__':
    main()
