import json, os, re
from pathlib import Path
H = Path.home() / 'tsync_exp'
man = json.load(open(H / 'fleurs/tr_tr/manifest.json'))
out = H / 'tr_fleurs_mfa/corpus/spk'; out.mkdir(parents=True, exist_ok=True)
for uid in sorted(man)[:20]:
    t = man[uid]['text'].replace('I', 'ı').replace('İ', 'i').lower()
    t = re.sub(r"[^\w' ]", ' ', t); t = re.sub(r'\s+', ' ', t).strip()
    (out / f'{uid}.lab').write_text(t)
    l = out / f'{uid}.wav'
    if not l.exists(): os.symlink(H / f'fleurs/tr_tr/{uid}.wav', l)
print('ok', len(list(out.glob('*.lab'))))
