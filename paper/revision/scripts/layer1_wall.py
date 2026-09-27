"""Wall-clock Layer-1 latency benchmark (revision item C2/C3/C4).

Fixes the clock origin of paper/tools/layer1_bench.py, which starts the
wall clock at 0 when the first window (ending at stream time W = 3.2 s) is
processed, so the W seconds of audio arrival are never counted and L is
clipped at 0 (hence the 0 ms English median).

Here audio arrives in real time (stream time == wall time, t = 0 at first
sample). The recognizer processes the window ending at the current buffer
head; a window can start only when its audio has arrived and the previous
window is done. If per-window compute C < h, the next window ends h later;
otherwise it takes the latest available audio (skip-ahead). A word is
finalized when head - t_end >= g (or at end of stream). Emission latency
L = wall time at which the word is emitted - word end time t_end.

Usage: python layer1_wall.py --model tiny --hop 0.32 --lang en --wavs a.wav b.wav ...
Prints one JSON line with per-word latencies and per-window compute.
"""
import argparse, json, sys, time
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

SR = 16000
GROW = False


def run_paper_clock(model, audio, lang, W, h, g):
    """Exact reproduction of paper/tools/layer1_bench.py's clock (for comparison)."""
    dur = len(audio) / SR
    window_end = min(W, dur); cum = 0.0; fin = []; lat = []; comp = []
    while True:
        ws = max(0.0, window_end - W)
        seg = audio[int(ws * SR):int(window_end * SR)]
        tic = time.perf_counter()
        segs, _ = model.transcribe(seg, word_timestamps=True, language=lang,
                                   vad_filter=False, condition_on_previous_text=False)
        words = [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or [])]
        C = time.perf_counter() - tic; comp.append(C); cum += C
        for text, a, b in words:
            te = ws + b
            if (window_end - te) < g:
                continue
            if any(text == ft and abs(te - fte) < 0.12 for ft, fte in fin):
                continue
            fin.append((text, te)); lat.append(max(0.0, cum - te))
        if window_end >= dur:
            break
        window_end = min(dur, window_end + max(h, C))
    return lat, comp


def run(model, audio, lang, W, h, g):
    dur = len(audio) / SR
    head = min(h if GROW else W, dur)
    T = head                 # wall clock: first window available once filled
    finalized, lat, comp = [], [], []
    while True:
        ws = max(0.0, head - W)
        seg = audio[int(ws * SR):int(head * SR)]
        tic = time.perf_counter()
        segs, _ = model.transcribe(seg, word_timestamps=True, language=lang,
                                   vad_filter=False, condition_on_previous_text=False,
                                   beam_size=1)
        words = [(w.word.strip().lower().strip('.,!?'), ws + w.start, ws + w.end)
                 for s in segs for w in (s.words or [])]
        C = time.perf_counter() - tic
        comp.append(C)
        T += C
        last = head >= dur - 1e-9
        for text, ts, te in words:
            if not last and (head - te) < g:
                continue
            if any(text == ft and abs(te - fte) < 0.12 for ft, fte in finalized):
                continue
            finalized.append((text, te))
            lat.append(max(0.0, T - te))
        if last:
            break
        nxt = head + h
        if T > nxt:              # backlog: jump to latest audio
            nxt = T
        head = min(dur, nxt)
        T = max(T, head)         # cannot process audio before it arrives
    return lat, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='tiny')
    ap.add_argument('--hop', type=float, default=0.32)
    ap.add_argument('--win', type=float, default=3.2)
    ap.add_argument('--guard', type=float, default=0.40)
    ap.add_argument('--lang', required=True)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--tag', default='')
    ap.add_argument('--paper-clock', action='store_true')
    ap.add_argument('--grow', action='store_true', help='start with a growing window at t=h instead of waiting for W')
    ap.add_argument('--wavs', nargs='+', required=True)
    a = ap.parse_args()
    global GROW; GROW = a.grow
    model = WhisperModel(a.model, device='cpu', compute_type='int8', cpu_threads=a.threads)
    model.transcribe(np.zeros(SR, dtype=np.float32), language=a.lang)  # warm-up
    L, C, audio_s = [], [], 0.0
    for p in a.wavs:
        au = decode_audio(p, sampling_rate=SR)
        audio_s += len(au) / SR
        l, c = (run_paper_clock if a.paper_clock else run)(model, au, a.lang, a.win, a.hop, a.guard)
        L += l; C += c
    L = np.array(L) * 1000; C = np.array(C) * 1000
    print(json.dumps({'tag': a.tag, 'grow': a.grow, 'paper_clock': a.paper_clock, 'model': a.model, 'lang': a.lang, 'hop_ms': a.hop * 1000,
                      'threads': a.threads, 'n_utt': len(a.wavs), 'audio_s': round(audio_s, 1),
                      'n_words': len(L),
                      'L_median': float(np.median(L)), 'L_p95': float(np.percentile(L, 95)),
                      'L_max': float(L.max()),
                      'C_median': float(np.median(C)), 'C_p95': float(np.percentile(C, 95)),
                      'keep_up_pct': float(np.mean(C < a.hop * 1000) * 100)}))


if __name__ == '__main__':
    main()
