import json, re, difflib, sys, glob
from pathlib import Path
import numpy as np
H = Path.home() / 'tsync_exp'; sys.path.insert(0, str(H / 'atlas/paper/tools'))
from grid_layer_validation import parse_textgrid_tier
nrm = lambda w: re.sub(r"[^\w']", '', w.replace('I', 'ı').replace('İ', 'i').lower())
for f in sorted(glob.glob(str(H / 'l1/ctc_mms_tr_*.json'))):
    st = []; m = t = 0
    for uid, hyp in json.load(open(f)).items():
        tg = H / f'tr_fleurs_mfa/out/spk/{uid}.TextGrid'
        if not tg.exists(): continue
        ref = [(nrm(w), a, b) for w, a, b in parse_textgrid_tier(tg, 'words') if nrm(w)]
        h = [(nrm(w), a, b) for w, a, b in hyp if nrm(w)]
        sm = difflib.SequenceMatcher(a=[x[0] for x in ref], b=[x[0] for x in h], autojunk=False)
        for blk in sm.get_matching_blocks():
            for i in range(blk.size): st.append((h[blk.b + i][1] - ref[blk.a + i][1]) * 1000)
        m += sum(b.size for b in sm.get_matching_blocks()); t += len(ref)
    a = np.abs(st)
    print(f"{Path(f).stem:24s} n={len(a)} match={m/t*100:.0f}% MAE={a.mean():.0f} median={np.median(a):.0f} P90={np.percentile(a,90):.0f} <100ms={np.mean(a<100)*100:.0f}% signed median={np.median(st):+.0f}")
