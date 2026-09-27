import os, time, threading
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip
BASE='https://zenodo.org/records/3625687/files/{}?download=1'
SPK=[s for s in range(1,35) if s!=21]
jobs=[]
for s in SPK:
    with RemoteZip(BASE.format(f's{s}.zip')) as z:
        for i in z.infolist():
            if i.filename.endswith('.mpg') and not i.filename.startswith('__'):
                jobs.append((f's{s}.zip', i.filename, f'video/s{s}/{os.path.basename(i.filename)}'))
print(len(jobs),'files',flush=True)
local=threading.local(); t0=time.time(); done=[0]; nb=[0]
def zf(n):
    d=getattr(local,'z',None)
    if d is None: d=local.z={}
    if n not in d: d[n]=RemoteZip(BASE.format(n))
    return d[n]
def get(j):
    zn,m,out=j
    if os.path.exists(out): return
    os.makedirs(os.path.dirname(out),exist_ok=True)
    for k in range(8):
        try:
            b=zf(zn).read(m); open(out+'.part','wb').write(b); os.replace(out+'.part',out); nb[0]+=len(b); done[0]+=1
            if done[0]%500==0: print(f'{done[0]} {nb[0]/1e9:.1f}GB {nb[0]/1e3/(time.time()-t0):.0f}KB/s',flush=True)
            return
        except Exception: time.sleep(5+5*k); local.z={}
    print('FAIL',m,flush=True)
for rnd in range(3):
    with ThreadPoolExecutor(10) as ex: list(ex.map(get,jobs))
print('DONE',sum(os.path.exists(j[2]) for j in jobs),'/',len(jobs),flush=True)
