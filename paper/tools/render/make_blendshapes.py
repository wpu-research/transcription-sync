"""
wav -> TSYNC 14-channel ARKit blendshape schedule JSON (60 fps), using the
standalone Layer 1-3 modules of layer1_bench.py / layer23_bench.py.

For the Phase-0 SyncNet sanity check the goal is a *well-synchronized* mouth
stream, so Layer 1 here is plain full-utterance transcription with word
timestamps (not the streaming finalization protocol — causality is irrelevant
to the sanity check; the full V-C TSYNC condition will use the streaming
simulation).

Usage:
    python make_blendshapes.py --wav en0.wav --lang en --out en0_tsync.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # paper/tools
from layer23_bench import (ARPABET_TABLE, TURKISH_TABLE, BILABIAL_PHONES_EN,
                            BILABIAL_PHONES_TR, g2p_en, g2p_tr,
                            layer2_durations, layer3_render, CHANNELS)

SR = 16000
FPS = 60


def wav_to_schedule(wav: Path, lang: str, model_size: str = 'small') -> dict:
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(wav), sampling_rate=SR)
    model = WhisperModel(model_size, device='cpu', compute_type='int8')
    segments, _ = model.transcribe(audio, word_timestamps=True, language=lang,
                                     vad_filter=False, condition_on_previous_text=False)
    triples = [(w.word.strip().lower().strip('.,!?'), w.start, w.end)
               for s in segments for w in (s.words or [])]
    if not triples:
        raise RuntimeError(f'no words recognized in {wav}')

    table = ARPABET_TABLE if lang == 'en' else TURKISH_TABLE
    bilset = BILABIAL_PHONES_EN if lang == 'en' else BILABIAL_PHONES_TR
    g2p = g2p_en if lang == 'en' else g2p_tr

    timed_phonemes = []
    for word, t_s, t_e in triples:
        phonemes = [p for p in g2p(word) if p in table]
        if not phonemes:
            continue
        _, midpoints = layer2_durations(phonemes, t_s, t_e, table)
        for p, tau in zip(phonemes, midpoints):
            cls, V = table[p]
            timed_phonemes.append((p, tau, V, p in bilset))
    timed_phonemes.sort(key=lambda x: x[1])

    frames, _ = layer3_render(timed_phonemes, table, bilset, fps=FPS)

    # layer3_render starts at first_midpoint - 3*sigma; pad leading frames so
    # frame 0 of the schedule corresponds to t=0 of the audio.
    t0 = timed_phonemes[0][1] - 0.090
    lead = max(0, round(t0 * FPS))
    rest = {c: 0.0 for c in CHANNELS}
    rest['MouthClose'] = 0.15
    duration = len(audio) / SR
    n_total = round(duration * FPS)
    padded = [dict(rest)] * lead + frames
    padded = padded[:n_total] + [dict(rest)] * max(0, n_total - len(padded))

    return {'fps': FPS, 'wav': wav.name, 'lang': lang,
            'words': [(w, round(s, 3), round(e, 3)) for w, s, e in triples],
            'frames': padded}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True)
    ap.add_argument('--lang', required=True, choices=['en', 'tr'])
    ap.add_argument('--model', default='small')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sched = wav_to_schedule(Path(args.wav), args.lang, args.model)
    Path(args.out).write_text(json.dumps(sched))
    print(f"{args.out}: {len(sched['frames'])} frames, "
          f"{len(sched['words'])} words: {' '.join(w for w, _, _ in sched['words'])}")


if __name__ == '__main__':
    main()
