#!/usr/bin/env python3
"""Превью мира без браузера: рисует BCC-ячейки из .vfw в PNG/MP4.

Зачем: движок headless, вьюер — в браузере, а посмотреть результат нужно и в
терминале/CI. Этот скрипт читает тот же .vfw, что и вьюер, и рисует ячейки
дисками с сортировкой по глубине: размер диска = шаг решётки уровня, так что
дробление осколка видно и на статичном кадре.

    python3 tools/preview.py demos/out/world.vfw --frames 0,8,40,299 --out demos/out
    python3 tools/preview.py demos/out/world.vfw --video demos/out/world.mp4

Зависимости: numpy, Pillow (для видео ещё ffmpeg).
"""

import argparse
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 960, 640
EYE = np.array([3.6, 2.4, 4.2])
TARGET = np.array([0.0, 0.55, 0.0])
FOV = math.radians(48.0)


def read_world(path):
    b = Path(path).read_bytes()
    if b[:4] != b"VFW1":
        sys.exit(f"{path}: не VFW1")
    nf, nshapes, total, base = struct.unpack_from("<IIIf", b, 4)
    off = 20
    shapes = []
    for _ in range(nshapes):
        level, n = struct.unpack_from("<II", b, off)
        off += 8
        offs = np.frombuffer(b, dtype="<f4", count=n * 3, offset=off).reshape(n, 3)
        off += n * 12
        shapes.append({"level": level, "offsets": offs, "edge": base / (1 << level)})
    frames = []
    for _ in range(nf):
        (m,) = struct.unpack_from("<I", b, off)
        off += 4
        row = []
        for _ in range(m):
            (sid,) = struct.unpack_from("<I", b, off)
            pos = np.frombuffer(b, dtype="<f4", count=3, offset=off + 4)
            off += 4 + 12 + 16
            row.append((sid, pos.astype(np.float64)))
        frames.append(row)
    return {"nf": nf, "base": base, "shapes": shapes, "frames": frames}


def basis():
    fwd = TARGET - EYE
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up


FWD, RIGHT, UP = basis()
FOCAL = (H * 0.5) / math.tan(FOV * 0.5)


def project(p):
    d = p - EYE
    z = d @ FWD
    if z <= 0.05:
        return None
    x = (d @ RIGHT) / z * FOCAL + W * 0.5
    y = H * 0.5 - (d @ UP) / z * FOCAL
    return x, y, z


def render(frame, size=(W, H)):
    """Ячейки — диски с сортировкой по глубине; рисуем в 2× и уменьшаем."""
    Wd, Hd = size
    ss = 2
    img = Image.new("RGB", (Wd * ss, Hd * ss), (11, 15, 20))
    dr = ImageDraw.Draw(img)

    # пол и лёгкий градиент к горизонту
    for y in range(Hd * ss // 2, Hd * ss):
        t = (y - Hd * ss / 2) / (Hd * ss / 2)
        dr.line([(0, y), (Wd * ss, y)],
                fill=(int(17 + 9 * (1 - t)), int(23 + 11 * (1 - t)), int(30 + 13 * (1 - t))))

    cells = []
    for sid, pos in frame:
        sh = shapes[sid]
        pts = pos[None, :] + sh["offsets"]
        cells.append((pts, (pts - EYE) @ FWD, sh["edge"], sh["level"]))

    order = sorted(range(len(cells)), key=lambda i: -cells[i][1].mean())

    for i in order:
        pts, depth, edge, level = cells[i]
        z = float(depth.mean())
        if z <= 0.15:
            continue
        # перспективный радиус диска
        r = edge * FOCAL / z * 0.50 * ss * (Wd / W)
        # туман по глубине + оттенок по уровню: крупные чуть теплее
        fog = max(0.0, min(0.75, (z - 3.0) * 0.06))
        tint = [
            (208, 214, 224),
            (198, 212, 230),
            (188, 210, 236),
        ][min(level, 2)]
        lit = 0.55 + 0.45 * max(0.0, min(1.0, (pts[:, 1].mean() + 0.2) / 1.8))
        col = tuple(int(c * lit * (1 - fog) + 12 * fog) for c in tint)
        for (p, zz) in zip(pts, depth):
            if zz <= 0.15:
                continue
            px = (p - EYE) @ RIGHT / zz * FOCAL * (Wd / W) + Wd * 0.5
            py = Hd * 0.5 - (p - EYE) @ UP / zz * FOCAL * (Wd / W)
            rr = max(0.9, r)
            rim = tuple(max(0, int(c * 0.55)) for c in col)
            dr.ellipse(
                [(px - rr) * ss, (py - rr) * ss, (px + rr) * ss, (py + rr) * ss],
                fill=col, outline=rim, width=max(1, int(rr * 0.18 * ss)),
            )
    return img.resize((Wd, Hd), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--frames", default="0", help="список кадров через запятую")
    ap.add_argument("--out", default="demos/out")
    ap.add_argument("--video", default=None, help="путь к mp4 (рендерит весь файл)")
    ap.add_argument("--step", type=int, default=3, help="шаг кадров для видео")
    ap.add_argument("--size", default="960x640", help="размер кадра WxH")
    args = ap.parse_args()

    global shapes
    w = read_world(args.file)
    shapes = w["shapes"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.file).stem

    size = tuple(int(v) for v in args.size.lower().split("x"))

    if args.video:
        tmp = out / f".{stem}-frames"
        tmp.mkdir(parents=True, exist_ok=True)
        idx = list(range(0, w["nf"], args.step))
        for k, f in enumerate(idx):
            img = render(w["frames"][f], size)
            img.save(tmp / f"{k:04d}.png")
            if k % 10 == 0:
                print(f"кадр {f} ({k + 1}/{len(idx)})")
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", "30", "-i", str(tmp / "%04d.png"),
             "-pix_fmt", "yuv420p", "-crf", "20", args.video],
            check=True, capture_output=True,
        )
        print(f"видео: {args.video}")
        return

    for f in [int(x) for x in args.frames.split(",")]:
        f = min(f, w["nf"] - 1)
        img = render(w["frames"][f], size)
        p = out / f"{stem}-{f:03d}.png"
        img.save(p)
        cells = sum(shapes[s]["offsets"].shape[0] for s, _ in w["frames"][f])
        print(f"{p}  (ячеек {cells})")


if __name__ == "__main__":
    main()
