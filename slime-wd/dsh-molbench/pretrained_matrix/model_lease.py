"""Release a finite inference RJob when its CPU evaluation finishes or vanishes."""
import json,os,sys,time
from pathlib import Path
root,pid,token=Path(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
while True:
    os.kill(pid,0)
    try:state=json.loads((root/'cpu_control.json').read_text())
    except (FileNotFoundError,ValueError):raise SystemExit('Missing CPU evaluation lease')
    if state.get('token')!=token:raise SystemExit('CPU evaluation lease owner mismatch')
    if time.time()-state['updated']>600:raise SystemExit('CPU evaluation controller lease expired')
    if state['state']=='done':raise SystemExit(state['exit_code'])
    time.sleep(5)
