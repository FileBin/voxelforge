/* Чтение файлов движка и эталона.
 *
 *   .vfr  — траектории тел CPU-эталона (для сверки с GPU)
 *   .vfw  — мир движка: BCC-ячейки осколков, их формы и положения по кадрам
 *
 * Форматы: `crates/world/src/types.rs` (vfr) и `crates/world/src/vfw.rs`. */

/* ArrayBuffer нельзя индексировать: File.arrayBuffer() отдаёт именно его. */
function magicOf(buf) {
  const u8 = buf instanceof ArrayBuffer
    ? new Uint8Array(buf)
    : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  return String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
}

/** "vfr" | "vfw" | null */
export function detectFormat(buf) {
  const m = magicOf(buf);
  if (m === "VFR1" || m === "VFR2") return "vfr";
  if (m === "VFW1") return "vfw";
  return null;
}

/** Траектории: `"VFR1/2" | u32 n | u32 nf | тела | nf × n × 3 f32`. */
export function parseTrace(buf) {
  const dv = new DataView(buf);
  const magic = magicOf(buf);
  if (magic !== "VFR1" && magic !== "VFR2") throw new Error("не VFR1/VFR2: " + magic);
  const n = dv.getUint32(4, true);
  const nf = dv.getUint32(8, true);
  let off = 12;
  const radius = new Float32Array(n);
  const shapes = new Uint8Array(n);
  const half = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    off += 8;                                   // масса (f64)
    radius[i] = dv.getFloat32(off, true);
    off += 4;
    if (magic === "VFR2") {
      shapes[i] = dv.getUint8(off);
      off += 1;
      for (let k = 0; k < 3; k++) {
        half[i * 3 + k] = dv.getFloat32(off, true);
        off += 4;
      }
    } else {
      half[i * 3] = half[i * 3 + 1] = half[i * 3 + 2] = radius[i];
    }
  }
  const pos = new Float32Array(nf * n * 3);
  for (let i = 0; i < nf * n * 3; i++) {
    pos[i] = dv.getFloat32(off, true);
    off += 4;
  }
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    const r = Math.max(radius[i], half[i * 3], half[i * 3 + 1], half[i * 3 + 2]);
    for (let f = 0; f < nf; f++) {
      for (let k = 0; k < 3; k++) {
        const v = pos[(f * n + i) * 3 + k];
        lo[k] = Math.min(lo[k], v - r);
        hi[k] = Math.max(hi[k], v + r);
      }
    }
  }
  return { kind: "trace", n, nf, radius, shapes, half, pos, lo, hi };
}

/** Мир движка:
 *  `"VFW1" | u32 frames | u32 shapes | u32 cells | f32 base`
 *  `shape: u32 level | u32 n | n × (f32 x, y, z)`
 *  `frame: u32 m | m × (u32 shape | f32 pos×3 | f32 quat×4)` */
export function parseWorld(buf) {
  const dv = new DataView(buf);
  const magic = magicOf(buf);
  if (magic !== "VFW1") throw new Error("не VFW1: " + magic);
  const nf = dv.getUint32(4, true);
  const nShapes = dv.getUint32(8, true);
  const total = dv.getUint32(12, true);
  const base = dv.getFloat32(16, true);

  let off = 20;
  const shapes = [];
  for (let s = 0; s < nShapes; s++) {
    const level = dv.getUint32(off, true);
    const n = dv.getUint32(off + 4, true);
    off += 8;
    const offsets = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) {
      offsets[i] = dv.getFloat32(off, true);
      off += 4;
    }
    shapes.push({ level, n, offsets, edge: base / (1 << level) });
  }

  const frames = [];
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let f = 0; f < nf; f++) {
    const m = dv.getUint32(off, true);
    off += 4;
    const row = [];
    for (let b = 0; b < m; b++) {
      const shape = dv.getUint32(off, true);
      off += 4;
      const pos = [
        dv.getFloat32(off, true),
        dv.getFloat32(off + 4, true),
        dv.getFloat32(off + 8, true),
      ];
      off += 12;
      off += 16;                                // кватернион: тела пока не вращаются
      row.push({ shape, pos });
      const sh = shapes[shape];
      const r = sh.edge * 0.5;
      for (let c = 0; c < sh.n; c++) {
        for (let k = 0; k < 3; k++) {
          const v = pos[k] + sh.offsets[c * 3 + k];
          lo[k] = Math.min(lo[k], v - r);
          hi[k] = Math.max(hi[k], v + r);
        }
      }
    }
    frames.push(row);
  }
  return { kind: "world", nf, total, base, shapes, frames, lo, hi };
}
