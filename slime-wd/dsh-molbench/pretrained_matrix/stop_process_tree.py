"""Bounded cleanup of one owned Linux process tree, including TERM-resistant children."""
import argparse
import json
import os
import signal
import time
from pathlib import Path

def snapshot():
    result = {}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = p.read_text().rsplit(')', 1)[1].split()
            result[int(p.parent.name)] = (int(fields[1]), fields[19], fields[0])
        except (OSError, ValueError, IndexError):
            pass
    return result

def stop_tree(pid, grace=10):
    tree = snapshot()
    if pid not in tree:
        return {'root': pid, 'tracked': [], 'forced': [], 'remaining': []}
    tracked = {pid: tree[pid][1]}
    def alive():
        current = snapshot()
        changed = True
        while changed:
            changed = False
            for child, (parent, start, state) in current.items():
                if parent in tracked and child not in tracked and current.get(parent, (None,None))[1] == tracked[parent]:
                    tracked[child] = start; changed = True
        return [p for p, start in tracked.items() if p in current and current[p][1] == start and current[p][2] != 'Z']
    def send(pids, sig):
        for p in pids:
            try: os.kill(p, sig)
            except ProcessLookupError: pass
    send(alive(), signal.SIGTERM)
    deadline = time.monotonic() + grace
    while alive() and time.monotonic() < deadline:
        time.sleep(.1)
    forced = alive(); send(forced, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while alive() and time.monotonic() < deadline:
        time.sleep(.1)
    return {'root': pid, 'tracked': sorted(tracked), 'forced': forced, 'remaining': alive()}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('pid', type=int); p.add_argument('--grace', type=float, default=10); a=p.parse_args()
    r=stop_tree(a.pid,a.grace);print(json.dumps(r));raise SystemExit(bool(r['remaining']))
