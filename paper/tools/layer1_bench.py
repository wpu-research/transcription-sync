"""
Standalone benchmark of TranscriptionSync Layer 1 (Section IV-B): streaming
word-timestamp ASR with a compact int8 Whisper model on CPU, simulating the
sliding-window / playout-buffer protocol and measuring the *word emission
lag* L (Eq. 3).

Design values under test: window W = 3.2 s, hop h = 320 ms, guard g = 400 ms
(a word is finalized once its end timestamp is more than g behind the current
buffer head). Model: faster-whisper "tiny", int8, CPU, multilingual (en/tr).

Audio: 10 short utterances (5 English, 5 Turkish; same sentences as
layer23_bench.py) synthesized offline with espeak-ng into layer1_audio/.

Because the sliding window is re-transcribed on every hop, each window's
processing time C is measured directly. If C > h the recognizer cannot keep
up with hop h in real time; the simulation then advances the buffer head by
C instead of h (the realistic behaviour of a single-threaded pipeline that
processes windows back-to-back), and this backlog is reflected in L.

Run directly:
    python3 layer1_bench.py
"""
import glob
import time

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

W = 3.2     # window length, s
H = 0.32    # nominal hop, s
G = 0.40    # guard interval, s
SR = 16000

MODEL_SIZE = "tiny"


def simulate_stream(model, audio: np.ndarray, lang: str):
    duration = len(audio) / SR
    window_end = min(W, duration)
    cumulative_wall = 0.0
    finalized = []  # (text, te_abs)
    lags_ms = []
    c_times_s = []

    while True:
        window_start = max(0.0, window_end - W)
        seg = audio[int(window_start * SR):int(window_end * SR)]
        tic = time.perf_counter()
        segments, _ = model.transcribe(seg, word_timestamps=True, language=lang,
                                         vad_filter=False, condition_on_previous_text=False)
        words = [(w.word.strip(), w.start, w.end) for s in segments for w in (s.words or [])]
        C = time.perf_counter() - tic
        c_times_s.append(C)
        cumulative_wall += C

        for text, ws, we in words:
            te_abs = window_start + we
            if (window_end - te_abs) < G:
                continue  # too close to buffer head, may still be revised
            if any(text == ft and abs(te_abs - fte) < 0.12 for ft, fte in finalized):
                continue  # already finalized in a previous window
            finalized.append((text, te_abs))
            L = max(0.0, cumulative_wall - te_abs)
            lags_ms.append(L * 1000.0)

        if window_end >= duration:
            break
        window_end = min(duration, window_end + max(H, C))

    return lags_ms, c_times_s, len(finalized)


def main():
    print(f"Loading faster-whisper '{MODEL_SIZE}' (int8, CPU)...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    all_lags, all_c = [], []
    n_words_total = 0
    for path in sorted(glob.glob("layer1_audio/*.wav")):
        lang = "en" if "/en" in path else "tr"
        audio = decode_audio(path, sampling_rate=SR)
        lags, c_times, n_fin = simulate_stream(model, audio, lang)
        all_lags.extend(lags)
        all_c.extend(c_times)
        n_words_total += n_fin
        print(f"  {path}: {n_fin} words finalized, "
              f"window times (ms) median={np.median(c_times)*1000:.1f} "
              f"max={np.max(c_times)*1000:.1f}")

    all_lags = np.array(all_lags)
    all_c_ms = np.array(all_c) * 1000.0

    print()
    print(f"Model: {MODEL_SIZE} (int8, CPU), W={W*1000:.0f} ms, h={H*1000:.0f} ms, g={G*1000:.0f} ms")
    print(f"n words finalized: {n_words_total}")
    print()
    print("| Quantity | Median | P95 |")
    print("|---|---|---|")
    print(f"| Word emission lag L (ms) | {np.median(all_lags):.1f} | {np.percentile(all_lags, 95):.1f} |")
    print(f"| Window processing time C (ms) | {np.median(all_c_ms):.1f} | {np.percentile(all_c_ms, 95):.1f} |")
    print()
    keep_up = np.mean(all_c_ms < H * 1000.0) * 100
    print(f"Windows processed within hop h={H*1000:.0f} ms: {keep_up:.1f}%")
    print(f"Implied ASR CPU load (one core), C/effective-hop: "
          f"{100.0 * np.median(all_c_ms) / max(H*1000.0, np.median(all_c_ms)):.1f}%")


if __name__ == "__main__":
    main()
