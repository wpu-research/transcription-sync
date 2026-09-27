#!/usr/bin/env python3
"""
GRID corpus video indirici (Zenodo / Sheffield)
Bağımlılık YOK — yalnızca Python stdlib. Kaldığı yerden devam eder (resume).

Kullanım (Windows):
  python grid_download.py --out C:\\Users\\wpu-ai\\Desktop\\Atlas_Journal\\grid_video
  python grid_download.py --out ... --extract          # indirip zip'leri aç
  python grid_download.py --out ... --speakers 1 2 3   # sadece belirli konuşmacılar
  python grid_download.py --out ... --source sheffield --quality normal

Notlar:
  - s21'in videosu yoktur (korpusun bilinen eksiği) — otomatik atlanır.
  - Toplam ~15–25 GB; yarıda kesilirse aynı komut kaldığı yerden sürer.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

NO_VIDEO = {21}
CHUNK    = 1 << 20          # 1 MB
RETRIES  = 5
UA       = "Mozilla/5.0 (grid-downloader; research use)"

def url_for(spk: int, source: str, quality: str) -> str:
    if source == "zenodo":
        return f"https://zenodo.org/records/3625687/files/s{spk}.zip?download=1"
    # sheffield
    tag = "mpg_vcd" if quality == "normal" else "mpg_6000"
    return f"https://spandh.dcs.shef.ac.uk/gridcorpus/s{spk}/video/s{spk}.{tag}.zip"

def remote_size(url: str) -> int | None:
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": UA})
        with urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None

def download(url: str, dest: Path) -> bool:
    """Resume destekli indirme. Başarı → True."""
    total = remote_size(url)
    have  = dest.stat().st_size if dest.exists() else 0

    if total and have == total:
        print(f"  ✔ {dest.name} zaten tam ({total/1e6:.0f} MB)")
        return True
    if total and have > total:
        print(f"  ⚠ {dest.name} bozuk görünüyor (yerel > uzak) — baştan iniyor")
        dest.unlink(); have = 0

    for attempt in range(1, RETRIES + 1):
        try:
            headers = {"User-Agent": UA}
            mode = "wb"
            if have > 0:
                headers["Range"] = f"bytes={have}-"
                mode = "ab"
            req = Request(url, headers=headers)
            with urlopen(req, timeout=60) as r, open(dest, mode) as f:
                if have > 0 and r.status != 206:
                    # sunucu Range desteklemiyor → baştan
                    f.close(); dest.unlink(); have = 0
                    f = open(dest, "wb")
                t0, done0 = time.time(), have
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk); have += len(chunk)
                    el = time.time() - t0
                    if el > 0.5:
                        mbps = (have - done0) / 1e6 / el
                        pct  = f"{100*have/total:5.1f}%" if total else f"{have/1e6:7.0f} MB"
                        print(f"\r  ↓ {dest.name}: {pct}  ({mbps:.1f} MB/s)   ",
                              end="", flush=True)
            print()
            if total and have != total:
                raise IOError(f"eksik indirme: {have}/{total}")
            return True
        except (HTTPError, URLError, IOError, TimeoutError) as e:
            print(f"\n  ⚠ deneme {attempt}/{RETRIES} başarısız: {e}")
            have = dest.stat().st_size if dest.exists() else 0
            time.sleep(min(2 ** attempt, 30))
    return False

def extract(zip_path: Path, out_dir: Path):
    target = out_dir / zip_path.stem.split(".")[0]      # s1.zip → s1/
    if target.exists() and any(target.iterdir()):
        print(f"  ✔ {target.name}/ zaten açılmış")
        return
    print(f"  📦 açılıyor: {zip_path.name} → {target.name}/")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        # zip zaten s1/... ile başlıyorsa out_dir'e, başlamıyorsa target'a aç
        root_has_spk = all(n.split("/")[0] == target.name for n in names if "/" in n)
        z.extractall(out_dir if root_has_spk else target)

def main():
    ap = argparse.ArgumentParser(description="GRID video downloader")
    ap.add_argument("--out", default="~/grid_video")
    ap.add_argument("--speakers", type=int, nargs="*", default=None,
                    help="Varsayılan: 1–34 (s21 hariç)")
    ap.add_argument("--source", choices=["zenodo", "sheffield"], default="zenodo")
    ap.add_argument("--quality", choices=["normal", "high"], default="normal",
                    help="Yalnızca sheffield için (normal=360x288, high=720x576)")
    ap.add_argument("--extract", action="store_true", help="İndirdikten sonra zip'leri aç")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    speakers = args.speakers or [s for s in range(1, 35) if s not in NO_VIDEO]
    speakers = [s for s in speakers if s not in NO_VIDEO or
                print(f"  ⚠ s{s} atlandı — videosu yok")]

    print(f"Kaynak : {args.source}"
          + (f" ({args.quality})" if args.source == "sheffield" else ""))
    print(f"Hedef  : {out}")
    print(f"Konuşmacı: {len(speakers)} adet\n")

    failed = []
    for i, spk in enumerate(speakers, 1):
        print(f"[{i}/{len(speakers)}] s{spk}")
        url  = url_for(spk, args.source, args.quality)
        dest = out / f"s{spk}.zip"
        if download(url, dest):
            if args.extract:
                try:
                    extract(dest, out)
                except zipfile.BadZipFile:
                    print(f"  ❌ {dest.name} bozuk zip — silindi, tekrar çalıştırın")
                    dest.unlink(); failed.append(spk)
        else:
            failed.append(spk)

    print()
    if failed:
        print(f"❌ Başarısız: {['s'+str(s) for s in failed]}")
        print("   Aynı komutu tekrar çalıştırın — tamamlananlar atlanır, eksikler sürer.")
        sys.exit(1)
    print("✅ Tüm videolar indirildi." + (" Zip'ler açıldı." if args.extract else ""))
    print("   Sonraki adım: grid_video_gt.py ile MediaPipe GT çıkarımı.")

if __name__ == "__main__":
    main()
