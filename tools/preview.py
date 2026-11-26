#!/usr/bin/env python3
"""Превью мира без браузера: растеризует меши марчинга из .vfw в PNG/MP4.

Движок headless, вьюер — в браузере, а посмотреть результат нужно и в
терминале/CI. Скрипт читает ровно тот же `.vfw`, что и вьюер (форматы
совпадают: вершины+индексы на форму, позиция+кватернион на кадр), и рисует
треугольники с z-буфером и плоским освещением.

    python3 tools/preview.py demos/out/world.vfw --frames 0,8,60,299
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
from PIL import Image

W, H = 960, 640
EYE = np.array([3.2, 2.2, 3.8])
TARGET = np.array([0.0, 0.55, 0.0])
FOV = math.radians(46.0)


def read_world(path):
    b = Path(path).read_bytes()
    if b[:4] not in (b"VFW2", b"VFW3"):
        sys.exit(f"{path}: не VFW2/VFW3 (нужен мир движка)")
    nf, nshapes, _total, base = struct.unpack_from("<IIIf", b, 4)
    off = 20
    shapes = []
    for _ in range(nshapes):
        level, nv, ni = struct.unpack_from("<III", b, off)
        off += 12
        grid = None
        if b[:4] == b"VFW3":
            glo = struct.unpack_from("<fff", b, off)
            ghi = struct.unpack_from("<fff", b, off + 12)
            gstep = struct.unpack_from("<f", b, off + 24)[0]
            grid = (glo, ghi, gstep)
            off += 28
        pos = np.frombuffer(b, dtype="<f4", count=nv * 3, offset=off).reshape(nv, 3)
        off += nv * 12
        idx = np.frombuffer(b, dtype="<u4", count=ni, offset=off).reshape(-1, 3)
        off += ni * 4
        shapes.append({"level": level, "pos": pos.astype(np.float64), "idx": idx,
                       "edge": base / (1 << level)})
    frames = []
    for _ in range(nf):
        (m,) = struct.unpack_from("<I", b, off)
        off += 4
        row = []
        for _ in range(m):
            (sid,) = struct.unpack_from("<I", b, off)
            p = np.frombuffer(b, dtype="<f4", count=3, offset=off + 4).astype(np.float64)
            q = np.frombuffer(b, dtype="<f4", count=4, offset=off + 16).astype(np.float64)
            off += 4 + 12 + 16
            row.append((sid, p, q))
        frames.append(row)
    return {"nf": nf, "shapes": shapes, "frames": frames}


def qrot(q, v):
    """Поворот вектора кватернионом [w,x,y,z] — как в VS_WORLD."""
    w, x, y, z = q
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


def auto_camera(w):
    """Камера по габариту всех кадров: иначе осколки улетают из вида."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for row in w["frames"][::10]:
        for sid, p, _q in row:
            sh = w["shapes"][sid]
            if sh["pos"].size == 0:
                continue
            lo = np.minimum(lo, p + sh["pos"].min(0))
            hi = np.maximum(hi, p + sh["pos"].max(0))
    c = (lo + hi) / 2.0
    r = float(np.linalg.norm(hi - lo)) * 0.62 + 0.4
    return c + np.array([0.85, 0.55, 1.0]) * r, c


def basis():
    fwd = TARGET - EYE
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up


FWD, RIGHT, UP = basis()
FOCAL = (H * 0.5) / math.tan(FOV * 0.5)


def render(frame, shapes, size=(W, H)):
    Wd, Hd = size
    img = np.zeros((Hd, Wd, 3), np.float32)
    # фон: небо сверху, тёмная земля снизу
    horizon = int(Hd * 0.55)
    for y in range(Hd):
        t = y / Hd
        img[y, :] = (10, 14, 20) if y < horizon else (14 + 6 * t, 18 + 7 * t, 24 + 8 * t)
    zbuf = np.full((Hd, Wd), np.inf, np.float32)

    L = np.array([0.42, 0.86, 0.32])
    L /= np.linalg.norm(L)

    for sid, p, q in frame:
        sh = shapes[sid]
        if sh["idx"].size == 0:
            continue
        pts = sh["pos"]
        # жёсткое движение: поворот вокруг своего центра, затем перенос
        rot = np.array([qrot(q, v) for v in pts])
        world = rot + p
        d = world - EYE
        z = d @ FWD
        if (z <= 0.1).any():
            continue
        sx = (d @ RIGHT) / z * FOCAL + Wd * 0.5
        sy = Hd * 0.5 - (d @ UP) / z * FOCAL
        tri = sh["idx"]
        a = world[tri[:, 0]]
        b = world[tri[:, 1]]
        c = world[tri[:, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1)
        ln[ln < 1e-12] = 1.0
        n = n / ln[:, None]
        # нормали наружу: если смотрит от камеры — разворачиваем (двусторонний свет)
        to_eye = EYE[None, :] - (a + b + c) / 3.0
        flip = (n * to_eye).sum(1) < 0
        n[flip] *= -1
        shade = 0.18 + 0.82 * np.clip(n @ L, 0.0, 1.0)
        level_tint = np.array([0.80 - 0.05 * sh["level"], 0.78 - 0.02 * sh["level"],
                               0.72 + 0.06 * sh["level"]])

        X = np.stack([sx[tri[:, 0]], sx[tri[:, 1]], sx[tri[:, 2]]], 1)
        Y = np.stack([sy[tri[:, 0]], sy[tri[:, 1]], sy[tri[:, 2]]], 1)
        Z = np.stack([z[tri[:, 0]], z[tri[:, 1]], z[tri[:, 2]]], 1)

        x0 = np.clip(np.floor(X.min(1)), 0, Wd - 1).astype(int)
        x1 = np.clip(np.ceil(X.max(1)), 0, Wd - 1).astype(int)
        y0 = np.clip(np.floor(Y.min(1)), 0, Hd - 1).astype(int)
        y1 = np.clip(np.ceil(Y.max(1)), 0, Hd - 1).astype(int)
        for t in range(tri.shape[0]):
            if x1[t] < x0[t] or y1[t] < y0[t]:
                continue
            xs = np.arange(x0[t], x1[t] + 1) + 0.5
            ys = np.arange(y0[t], y1[t] + 1) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            ax, bx, cx = X[t]
            ay, by, cy = Y[t]
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-12:
                continue
            w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / den
            w1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / den
            w2 = 1.0 - w0 - w1
            mask = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not mask.any():
                continue
            zz = w0 * Z[t, 0] + w1 * Z[t, 1] + w2 * Z[t, 2]
            sub = zbuf[y0[t]:y1[t] + 1, x0[t]:x1[t] + 1]
            upd = mask & (zz < sub)
            if not upd.any():
                continue
            sub[upd] = zz[upd]
            col = level_tint * shade[t] * 255.0
            img[y0[t]:y1[t] + 1, x0[t]:x1[t] + 1][upd] = col
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--frames", default="0", help="список кадров через запятую")
    ap.add_argument("--out", default="demos/out")
    ap.add_argument("--video", default=None)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--size", default="960x640")
    args = ap.parse_args()

    w = read_world(args.file)
    # камера по габариту мира
    global EYE, TARGET, FWD, RIGHT, UP
    EYE, TARGET = auto_camera(w)
    FWD, RIGHT, UP = basis()
    size = tuple(int(v) for v in args.size.lower().split("x"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.file).stem

    if args.video:
        tmp = out / f".{stem}-frames"
        tmp.mkdir(parents=True, exist_ok=True)
        idx = list(range(0, w["nf"], args.step))
        for k, f in enumerate(idx):
            render(w["frames"][f], w["shapes"], size).save(tmp / f"{k:04d}.png")
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
        img = render(w["frames"][f], w["shapes"], size)
        p = out / f"{stem}-{f:03d}.png"
        img.save(p)
        tris = sum(w["shapes"][s]["idx"].shape[0] for s, _, _ in w["frames"][f])
        print(f"{p}  (треугольников {tris})")


if __name__ == "__main__":
    main()
