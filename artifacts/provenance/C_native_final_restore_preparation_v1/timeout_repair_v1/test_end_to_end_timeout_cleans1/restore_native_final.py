import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
signal.signal(signal.SIGTERM,signal.SIG_IGN)
code="import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path("+repr(str(root/'child_ready'))+").write_text(str(os.getpid()));time.sleep(60)"
child=subprocess.Popen([sys.executable,'-c',code])
(root/'group.json').write_text(json.dumps(dict(pgid=os.getpgrp(),child_pid=child.pid)))
while not (root/'child_ready').exists():time.sleep(.01)
time.sleep(60)
