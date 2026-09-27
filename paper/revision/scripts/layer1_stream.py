"""Low-latency Layer-1 variants, measured with the wall-clock model of layer1_wall.py.

Policies
  la2   LocalAgreement-2 (Machacek et al. 2023): Whisper on a growing buffer; a word is
        committed once two consecutive hypotheses agree on it (beyond the committed
        prefix). Buffer trimmed at the last committed word when it exceeds --max-buf.
        Needs no transcript.
  ctc   Streaming forced alignment of a KNOWN transcript (in A.T.L.A.S the Live API text
        arrives ~349 ms before the audio): CTC emissions of a wav2vec2/MMS aligner on the
        audio since the last committed word; free-end Viterbi aligns the remaining words;
        a word is committed when the next word has started and its end lies >= guard
        behind the buffer head (or at end of stream).

Wall clock: audio arrives in real time; a step processes audio up to the current head,
takes C seconds; next head = max(head + hop, now) (skip-ahead when behind).
L = emission wall time - emitted word end.  Output: JSON line with latency stats and,
per utterance, the emitted (word, start, end) triples (for accuracy scoring).
"""
import argparse, json, re, sys, time, unicodedata
import numpy as np

SR = 16000


# ── LocalAgreement-2 ───────────────────────────────────────────────────────
def run_la2(model, audio, lang, hop, max_buf):
    dur = len(audio) / SR
    head = min(hop, dur); T = head
    committed, prev, buf0 = [], [], 0.0
    out, lat, comp = [], [], []
    while True:
        seg = audio[int(buf0 * SR):int(head * SR)]
        prompt = ' '.join(w for w, _, _ in committed[-30:])
        tic = time.perf_counter()
        segs, _ = model.transcribe(seg, word_timestamps=True, language=lang, beam_size=1,
                                   vad_filter=False, condition_on_previous_text=False,
                                   initial_prompt=prompt or None)
        hyp = [(w.word.strip(), buf0 + w.start, buf0 + w.end) for s in segs for w in (s.words or [])]
        C = time.perf_counter() - tic; comp.append(C); T += C
        # drop hypothesis words that end before the committed frontier
        front = committed[-1][2] if committed else 0.0
        hyp = [h for h in hyp if h[1] >= front - 0.05]
        last = head >= dur - 1e-9
        new = []
        if last:
            new = hyp
        else:
            for a, b in zip(hyp, prev):
                if norm(a[0]) == norm(b[0]) and norm(a[0]):
                    new.append(a)
                else:
                    break
        for w in new:
            committed.append(w); out.append(w); lat.append(max(0.0, T - w[2]))
        prev = hyp[len(new):]
        if last:
            break
        if committed and head - buf0 > max_buf:
            buf0 = committed[-1][2]; prev = []
        nxt = max(head + hop, T); head = min(dur, nxt); T = max(T, head)
    return out, lat, comp


def norm(w):
    return re.sub(r"[^\w']", '', w.lower())


# ── streaming CTC alignment of a known transcript ──────────────────────────
TR_MAP = str.maketrans({'ç': 'c', 'ş': 's', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ü': 'u', 'â': 'a', 'î': 'i', 'û': 'u'})


class Aligner:
    def __init__(self, kind, device):
        import torch, torchaudio
        from torchaudio.pipelines import MMS_FA, WAV2VEC2_ASR_BASE_960H
        self.torch = torch
        if kind == 'mms':
            b = MMS_FA; self.model = b.get_model(with_star=False).to(device).eval()
            d = b.get_dict(star=None); self.lower = True
        else:
            b = WAV2VEC2_ASR_BASE_960H; self.model = b.get_model().to(device).eval()
            labels = b.get_labels(); d = {c: i for i, c in enumerate(labels)}; self.lower = False
        self.dict = d; self.blank = 0; self.device = device
        self.fps = SR / 320.0   # wav2vec2 frame rate (20 ms)

    def tokens(self, word):
        w = word.lower().translate(TR_MAP)
        w = unicodedata.normalize('NFKD', w)
        w = ''.join(c for c in w if c.isalpha() or c == "'")
        if not self.lower:
            w = w.upper()
        return [self.dict[c] for c in w if c in self.dict]

    def emissions(self, seg):
        with self.torch.inference_mode():
            x = self.torch.from_numpy(seg).float().unsqueeze(0).to(self.device)
            em, _ = self.model(x)
            return self.torch.log_softmax(em, -1)[0].cpu().numpy()


def prefix_viterbi(lp, tok_words, blank=0):
    """Free-end CTC Viterbi of a word sequence against log-probs lp (T,V).
    Returns per aligned word (first_frame, last_frame) for the best prefix."""
    labels, wid = [], []
    for k, tw in enumerate(tok_words):
        for t in tw:
            labels.append(t); wid.append(k)
    L = len(labels)
    if L == 0 or len(lp) == 0:
        return []
    S = 2 * L + 1
    ext = np.full(S, blank); ext[1::2] = labels
    Tn = len(lp)
    NEG = -1e30
    dp = np.full(S, NEG); dp[0] = lp[0, blank]; dp[1] = lp[0, ext[1]]
    bp = np.zeros((Tn, S), dtype=np.int32)
    skip_ok = np.zeros(S, bool)
    for s in range(3, S, 2):
        skip_ok[s] = ext[s] != ext[s - 2]
    for t in range(1, Tn):
        c0 = dp
        c1 = np.concatenate(([NEG], dp[:-1]))
        c2 = np.concatenate(([NEG, NEG], dp[:-2])); c2 = np.where(skip_ok, c2, NEG)
        stack = np.stack([c0, c1, c2]); arg = stack.argmax(0)
        dp = stack.max(0) + lp[t, ext]
        bp[t] = arg
    s = int(dp.argmax())
    path = np.empty(Tn, dtype=np.int32)
    for t in range(Tn - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    spans = {}
    for t, st in enumerate(path):
        if st % 2 == 1:
            k = wid[(st - 1) // 2]
            a, b = spans.get(k, (t, t)); spans[k] = (min(a, t), max(b, t))
    last_state = int(path[-1])
    # state index of each word's last label
    word_last_state = {}
    for i, k in enumerate(wid):
        word_last_state[k] = 2 * i + 1
    return spans, last_state, word_last_state


def run_ctc(al, audio, words, hop, guard, margin=0.3, lookahead=15, need_next=True, sil=None, avail=None):
    """avail: optional per-word wall time (s, on the playback clock) at which the word's text is known."""
    dur = len(audio) / SR
    tw = [al.tokens(w) for w in words]
    keep = [i for i, t in enumerate(tw) if t]
    words = [words[i] for i in keep]; tw = [tw[i] for i in keep]
    av = [avail[i] for i in keep] if avail is not None else [0.0] * len(words)
    head = min(hop, dur); T = head
    k, anchor = 0, 0.0
    out, lat, comp = [], [], []
    while k < len(words):
        s0 = max(0.0, anchor - margin)
        seg = audio[int(s0 * SR):int(head * SR)]
        tic = time.perf_counter()
        lp = al.emissions(seg) if len(seg) > 400 else np.zeros((0, 1))
        f0 = int(round((anchor - s0) * al.fps))
        known = sum(1 for t in av[k:k + lookahead] if t <= T)   # words whose text has arrived
        res = prefix_viterbi(lp[f0:], tw[k:k + known]) if (len(lp) > f0 + 1 and known) else []
        C = time.perf_counter() - tic; comp.append(C); T += C
        last = head >= dur - 1e-9
        j = 0
        if res:
            spans, last_state, wls = res
            t0 = s0 + f0 / al.fps
            while k + j < len(words) and j in spans:
                a, b = spans[j]
                end = t0 + (b + 1) / al.fps
                started_next = (j + 1) in spans
                if (j + 1) >= known and k + j + 1 < len(words) and not last:
                    started_next = False   # next word's text not yet known: cannot use it as evidence
                # silence rule: path ends in the blank right after this word, long enough
                if sil is not None and not started_next and last_state == wls[j] + 1 and end <= head - sil:
                    started_next = True
                if not last and not ((started_next or not need_next) and end <= head - guard):
                    break
                st = t0 + a / al.fps
                out.append((words[k + j], st, end)); lat.append(max(0.0, T - end))
                j += 1
            if j:
                k += j; anchor = out[-1][2]
        if last:
            if not res or not j:
                break
            continue            # flush remaining words at end of stream
        nxt = max(head + hop, T); head = min(dur, nxt); T = max(T, head)
    return out, lat, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=['la2', 'ctc'], required=True)
    ap.add_argument('--lang', required=True)
    ap.add_argument('--model', default='tiny')          # whisper size or mms|w2v
    ap.add_argument('--hop', type=float, default=0.32)
    ap.add_argument('--guard', type=float, default=0.10)
    ap.add_argument('--max-buf', type=float, default=8.0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--manifest', default=None, help='json {uid: {"text": ...}} for ctc')
    ap.add_argument('--tag', default='')
    ap.add_argument('--dump', default=None)
    ap.add_argument('--no-need-next', action='store_true')
    ap.add_argument('--sil', type=float, default=None)
    ap.add_argument('--wavs', nargs='+', required=True)
    a = ap.parse_args()
    from faster_whisper.audio import decode_audio
    import torch; torch.set_num_threads(a.threads)
    if a.policy == 'la2':
        from faster_whisper import WhisperModel
        m = WhisperModel(a.model, device='cpu', compute_type='int8', cpu_threads=a.threads)
        m.transcribe(np.zeros(SR, dtype=np.float32), language=a.lang)
    else:
        al = Aligner(a.model, a.device); al.emissions(np.zeros(SR, dtype=np.float32))
        man = json.load(open(a.manifest))
    L, C, dump, audio_s, cover = [], [], {}, 0.0, []
    from pathlib import Path
    for p in a.wavs:
        au = decode_audio(p, sampling_rate=SR); audio_s += len(au) / SR
        uid = Path(p).stem
        if a.policy == 'la2':
            out, l, c = run_la2(m, au, a.lang, a.hop, a.max_buf)
        else:
            text = man[uid]['text'] if isinstance(man[uid], dict) else man[uid]
            out, l, c = run_ctc(al, au, text.split(), a.hop, a.guard, need_next=not a.no_need_next, sil=a.sil)
            ntok = sum(1 for w in text.split() if al.tokens(w)); cover.append((len(out), ntok))
        L += l; C += c; dump[uid] = [(w, round(s, 3), round(e, 3)) for w, s, e in out]
    L = np.array(L) * 1000; C = np.array(C) * 1000
    if a.dump:
        json.dump(dump, open(a.dump, 'w'))
    print(json.dumps({'policy': a.policy, 'model': a.model, 'lang': a.lang, 'tag': a.tag, 'hop_ms': a.hop * 1000,
                      'guard_ms': a.guard * 1000, 'device': a.device, 'threads': a.threads, 'n_utt': len(a.wavs),
                      'audio_s': round(audio_s, 1), 'n_words': len(L),
                      'L_median': float(np.median(L)), 'L_p95': float(np.percentile(L, 95)), 'L_max': float(L.max()),
                      'C_median': float(np.median(C)), 'C_p95': float(np.percentile(C, 95)),
                      'keep_up_pct': float(np.mean(C < a.hop * 1000) * 100), 'need_next': not a.no_need_next, 'sil': a.sil,
                      'coverage': (sum(x for x, _ in cover) / max(1, sum(y for _, y in cover))) if cover else None}))


if __name__ == '__main__':
    main()
