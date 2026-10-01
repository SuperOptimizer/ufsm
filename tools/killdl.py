import os, signal, sys
needle = sys.argv[1].encode()
for pid in os.listdir('/proc'):
    if not pid.isdigit() or int(pid) == os.getpid(): continue
    try: comm=open(f'/proc/{pid}/comm').read().strip(); cmd=open(f'/proc/{pid}/cmdline','rb').read()
    except Exception: continue
    if comm in ('curl','pget.sh','sh') and needle in cmd and b'killdl' not in cmd: os.kill(int(pid), signal.SIGTERM); print('killed', pid, comm)
