import random, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip
BASE='https://zenodo.org/records/3625687/files/{}?download=1'
SPK=[29,30,31,32,33,34]; N=int(sys.argv[1]) if len(sys.argv)>1 else 100
os.makedirs('audio',exist_ok=True); os.makedirs('video',exist_ok=True)
with RemoteZip(BASE.format('audio_25k.zip')) as z:
    names=[i.filename for i in z.infolist() if i.filename.endswith('.wav') and not i.filename.startswith('__')]
jobs=[]
rng=random.Random(0)
for s in SPK:
    ids=sorted(os.path.basename(n)[:-4] for n in names if n.startswith(f'audio_25k/s{s}/'))
    pick=sorted(rng.sample(ids, min(N,len(ids))))
    open(f'subset_s{s}.txt','w').write('\n'.join(pick))
    for u in pick:
        jobs.append(('audio_25k.zip', f'audio_25k/s{s}/{u}.wav', f'audio/s{s}/{u}.wav'))
        jobs.append((f's{s}.zip', f's{s}/{u}.mpg', f'video/s{s}/{u}.mpg'))
print(len(jobs),'files', flush=True)
local=threading.local(); t0=time.time(); done=[0]; nbytes=[0]
def zf(name):
    d=getattr(local,'z',None)
    if d is None: d=local.z={}
    if name not in d: d[name]=RemoteZip(BASE.format(name))
    return d[name]
def get(j):
    zname,member,out=j
    if os.path.exists(out): return
    os.makedirs(os.path.dirname(out),exist_ok=True)
    for k in range(5):
        try:
            data=zf(zname).read(member); open(out+'.part','wb').write(data); os.replace(out+'.part',out)
            nbytes[0]+=len(data); done[0]+=1
            if done[0]%50==0: print(f'{done[0]} files {nbytes[0]/1e6:.0f}MB {nbytes[0]/1e3/(time.time()-t0):.0f}KB/s',flush=True)
            return
        except Exception as e:
            time.sleep(5); local.z={}
    print('FAIL',member,flush=True)
with ThreadPoolExecutor(8) as ex: list(ex.map(get,jobs))
print('DONE',done[0],flush=True)
