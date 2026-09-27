"""
Deterministic offline avatar video renderer (Section V-C pipeline).

Takes a blendshape schedule JSON (frames of {channel: value} at a given fps)
plus the matching wav, renders each frame headlessly (playwright + chromium,
render_page.html), and muxes frames + audio into an H.264 MP4 with ffmpeg.

The schedule's channel names are the paper's 14 ARKit channels with an
upper-case initial ('JawOpen', ...); the GLB morph targets use lower-camel
('jawOpen', ...) — mapped here.

Usage:
    python render_video.py --schedule utt.json --wav utt.wav --out utt.mp4
                           [--fps 25] [--w 640 --h 480] [--shift-ms 0]

--shift-ms shifts the AUDIO track relative to the video (positive = audio
later), used for the SyncNet sanity check (±200 ms desynchronized versions).
JSON schedule format: {"fps": 60, "frames": [{"JawOpen": 0.4, ...}, ...]}
Frames are resampled to the output fps by nearest-neighbour on time.
"""
import argparse
import http.server
import json
import shutil
import subprocess
import tempfile
import threading
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../Atlas


def lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


def serve_repo(port: int = 0):
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(REPO_ROOT))
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, srv.server_address[1]


def render_frames(schedule: dict, out_dir: Path, fps: int, w: int, h: int,
                   model: str = '/public/human.glb', quiet: bool = False):
    from playwright.sync_api import sync_playwright

    in_fps = schedule['fps']
    frames = schedule['frames']
    duration = len(frames) / in_fps
    n_out = max(1, round(duration * fps))

    srv, port = serve_repo()
    page_url = (f'http://127.0.0.1:{port}/paper/tools/render/render_page.html'
                f'?model={model}&w={w}&h={h}')
    try:
        with sync_playwright() as p:
            # software WebGL for headless rendering (no GPU needed)
            browser = p.chromium.launch(args=['--use-angle=swiftshader',
                                               '--enable-unsafe-swiftshader'])
            page = browser.new_page(viewport={'width': w, 'height': h})
            page.goto(page_url)
            n_morphs = page.evaluate('window.avatarReady')
            if not quiet:
                print(f'  model loaded: {n_morphs} morph targets')
            # build the per-output-frame morph list, render in in-page batches
            morph_list = []
            for f in range(n_out):
                t = f / fps
                src = min(len(frames) - 1, round(t * in_fps))
                morph_list.append({lower_first(k): v for k, v in frames[src].items()})
            import base64
            BATCH = 100
            for start in range(0, n_out, BATCH):
                chunk = morph_list[start:start + BATCH]
                urls = page.evaluate('(fl) => window.renderBatch(fl)', chunk)
                for i, u in enumerate(urls):
                    png = base64.b64decode(u.split(',', 1)[1])
                    (out_dir / f'f{start + i:05d}.png').write_bytes(png)
                if not quiet:
                    print(f'  frames {start}-{min(n_out, start + BATCH)}/{n_out}')
            browser.close()
    finally:
        srv.shutdown()
    return n_out


def mux(frames_dir: Path, wav: Path, out: Path, fps: int, shift_ms: int = 0):
    """ffmpeg: PNG sequence + wav -> H.264 yuv420p MP4. shift_ms > 0 delays audio."""
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-framerate', str(fps), '-i', str(frames_dir / 'f%05d.png')]
    if shift_ms >= 0:
        cmd += ['-itsoffset', f'{shift_ms / 1000.0:.3f}', '-i', str(wav)]
    else:
        # negative offset on input audio: trim the start instead
        cmd += ['-ss', f'{-shift_ms / 1000.0:.3f}', '-i', str(wav)]
    cmd += ['-map', '0:v', '-map', '1:a',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
            '-c:a', 'aac', '-shortest', str(out)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schedule', required=True)
    ap.add_argument('--wav', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--w', type=int, default=640)
    ap.add_argument('--h', type=int, default=480)
    ap.add_argument('--shift-ms', type=int, default=0)
    ap.add_argument('--model', default='/public/human.glb')
    args = ap.parse_args()

    schedule = json.loads(Path(args.schedule).read_text())
    tmp = Path(tempfile.mkdtemp(prefix='render_'))
    try:
        n = render_frames(schedule, tmp, args.fps, args.w, args.h, args.model)
        mux(tmp, Path(args.wav), Path(args.out), args.fps, args.shift_ms)
        print(f'{args.out}: {n} frames @ {args.fps} fps, audio shift {args.shift_ms} ms')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
