"""
Section V-C batch evaluation driver. For every utterance in the audio dir
(collect_audio.py output), runs the full pipeline:

  1. MFA alignment (reference transcript -> TextGrid), once per corpus
  2. make_conditions: 6 blendshape schedules per utterance
  3. render_video: 6 videos per utterance (parallel workers, one persistent
     browser per worker)
  4. SyncNet (run_pipeline + run_syncnet) per video
  5. collects per-video {offset, lse_d, lse_c} into results.json

Computational metrics that need no video (LVD, timing error, lag) are in
compute_metrics.py and consume the schedules + results.json.

Usage:
  python run_eval.py --audio paper/eval_audio --work paper/eval_work \
      [--workers 4] [--stage all|mfa|schedules|render|syncnet]
  (resumable: every stage skips outputs that already exist)
"""
import argparse
import base64
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SYNCNET_DIR = Path('/tmp/syncnet_python')
CONDITIONS = ['amp', 'rate', 'tsync', 'tsync-dur', 'tsync-coart', 'mfa']
FPS_OUT = 25
W, H = 640, 480


def utt_ids(audio_dir: Path):
    manifest = json.loads((audio_dir / 'manifest.json').read_text())
    return sorted(manifest.keys()), manifest


# ── stage: mfa ──────────────────────────────────────────────────────────────
def stage_mfa(audio_dir: Path, work: Path):
    corpus = work / 'mfa_corpus' / 'spk'
    out = work / 'mfa_out'
    if (out / 'spk').is_dir() and any((out / 'spk').glob('*.TextGrid')):
        existing = len(list((out / 'spk').glob('*.TextGrid')))
        print(f'[mfa] {existing} TextGrids exist — skipping alignment '
              f'(delete {out} to redo)')
        return
    corpus.mkdir(parents=True, exist_ok=True)
    ids, manifest = utt_ids(audio_dir)
    for uid in ids:
        shutil.copy(audio_dir / f'{uid}.wav', corpus / f'{uid}.wav')
        (corpus / f'{uid}.lab').write_text(manifest[uid]['transcript'])
    print(f'[mfa] aligning {len(ids)} utterances...')
    subprocess.run(['conda', 'run', '-n', 'mfa', 'mfa', 'align',
                    str(corpus.parent), 'english_us_arpa', 'english_us_arpa',
                    str(out), '--clean'], check=True)
    print(f'[mfa] done -> {out}')


# ── stage: schedules ────────────────────────────────────────────────────────
def stage_schedules(audio_dir: Path, work: Path):
    sched_dir = work / 'schedules'
    sched_dir.mkdir(parents=True, exist_ok=True)
    ids, manifest = utt_ids(audio_dir)
    for uid in ids:
        if all((sched_dir / f'{uid}_{c}.json').exists() for c in CONDITIONS):
            continue
        tg = work / 'mfa_out' / 'spk' / f'{uid}.TextGrid'
        cmd = [sys.executable, str(HERE / 'make_conditions.py'),
               '--wav', str(audio_dir / f'{uid}.wav'), '--lang', 'en',
               '--transcript', manifest[uid]['transcript'],
               '--outdir', str(sched_dir)]
        if tg.exists():
            cmd += ['--textgrid', str(tg)]
        else:
            print(f'  [warn] {uid}: no TextGrid, mfa condition will be skipped')
            cmd += ['--conditions', ','.join(c for c in CONDITIONS if c != 'mfa')]
        subprocess.run(cmd, check=True)


# ── stage: render (parallel, persistent browser per worker) ────────────────
def _render_worker(jobs, audio_dir_s, work_s, worker_id):
    sys.path.insert(0, str(HERE))
    from render_video import serve_repo, lower_first, mux
    from playwright.sync_api import sync_playwright

    audio_dir, work = Path(audio_dir_s), Path(work_s)
    video_dir = work / 'videos'
    srv, port = serve_repo()
    done = 0
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--use-angle=swiftshader',
                                               '--enable-unsafe-swiftshader'])
            page = browser.new_page(viewport={'width': W, 'height': H})
            page.goto(f'http://127.0.0.1:{port}/paper/tools/render/render_page.html'
                      f'?model=/public/human.glb&w={W}&h={H}')
            page.evaluate('window.avatarReady')
            for uid, cond in jobs:
                out_mp4 = video_dir / f'{uid}_{cond}.mp4'
                if out_mp4.exists():
                    continue
                sched = json.loads((work / 'schedules' / f'{uid}_{cond}.json').read_text())
                frames = sched['frames']
                in_fps = sched['fps']
                n_out = max(1, round(len(frames) / in_fps * FPS_OUT))
                morphs = []
                for f in range(n_out):
                    src = min(len(frames) - 1, round(f / FPS_OUT * in_fps))
                    morphs.append({lower_first(k): v for k, v in frames[src].items()})
                tmp = Path(tempfile.mkdtemp(prefix=f'rw{worker_id}_'))
                try:
                    for start in range(0, n_out, 100):
                        urls = page.evaluate('(fl) => window.renderBatch(fl)',
                                              morphs[start:start + 100])
                        for i, u in enumerate(urls):
                            (tmp / f'f{start + i:05d}.png').write_bytes(
                                base64.b64decode(u.split(',', 1)[1]))
                    mux(tmp, audio_dir / f'{uid}.wav', out_mp4, FPS_OUT)
                    done += 1
                    print(f'[w{worker_id}] {uid}_{cond} ({n_out} frames) done={done}')
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
            browser.close()
    finally:
        srv.shutdown()


def stage_render(audio_dir: Path, work: Path, workers: int):
    video_dir = work / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    ids, _ = utt_ids(audio_dir)
    jobs = []
    for uid in ids:
        for cond in CONDITIONS:
            if not (work / 'schedules' / f'{uid}_{cond}.json').exists():
                continue
            if not (video_dir / f'{uid}_{cond}.mp4').exists():
                jobs.append((uid, cond))
    print(f'[render] {len(jobs)} videos to render with {workers} workers')
    if not jobs:
        return
    chunks = [jobs[i::workers] for i in range(workers)]
    procs = []
    for wi, chunk in enumerate(chunks):
        pr = mp.Process(target=_render_worker, args=(chunk, str(audio_dir), str(work), wi))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
    failed = [j for j in jobs if not (video_dir / f'{j[0]}_{j[1]}.mp4').exists()]
    if failed:
        print(f'[render] WARNING: {len(failed)} videos missing: {failed[:5]}...')


# ── stage: syncnet ──────────────────────────────────────────────────────────
def stage_syncnet(audio_dir: Path, work: Path):
    import re
    video_dir = work / 'videos'
    sn_data = (work / 'syncnet_data').resolve()
    results_path = work / 'results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    vids = sorted(video_dir.glob('*.mp4'))
    for v in vids:
        ref = v.stem
        if ref in results:
            continue
        v_abs = str(v.resolve())
        r1 = subprocess.run([sys.executable, 'run_pipeline.py', '--videofile', v_abs,
                             '--reference', ref, '--data_dir', str(sn_data),
                             '--min_track', '40', '--overwrite'],
                            cwd=SYNCNET_DIR, capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, 'run_syncnet.py', '--videofile', v_abs,
                             '--reference', ref, '--data_dir', str(sn_data)],
                            cwd=SYNCNET_DIR, capture_output=True, text=True)
        out = r2.stdout + r2.stderr
        m_off = re.search(r'AV offset:\s*(-?\d+)', out)
        m_dist = re.search(r'Min dist:\s*([\d.]+)', out)
        m_conf = re.search(r'Confidence:\s*([\d.]+)', out)
        if m_off and m_dist and m_conf:
            results[ref] = {'av_offset_frames': int(m_off.group(1)),
                            'lse_d': float(m_dist.group(1)),
                            'lse_c': float(m_conf.group(1))}
            print(f'[syncnet] {ref}: offset={m_off.group(1)} '
                  f'lse_d={m_dist.group(1)} lse_c={m_conf.group(1)}')
        else:
            results[ref] = {'error': 'no_score', 'detail': out[-300:]}
            print(f'[syncnet] {ref}: NO SCORE')
        results_path.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results.values() if 'lse_d' in r)
    print(f'[syncnet] {n_ok}/{len(vids)} scored -> {results_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--stage', default='all',
                    choices=['all', 'mfa', 'schedules', 'render', 'syncnet'])
    args = ap.parse_args()

    audio_dir, work = Path(args.audio), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if args.stage in ('all', 'mfa'):
        stage_mfa(audio_dir, work)
    if args.stage in ('all', 'schedules'):
        stage_schedules(audio_dir, work)
    if args.stage in ('all', 'render'):
        stage_render(audio_dir, work, args.workers)
    if args.stage in ('all', 'syncnet'):
        stage_syncnet(audio_dir, work)


if __name__ == '__main__':
    main()
