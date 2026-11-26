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
  if (m === "VFW1" || m === "VFW2") return "vfw";
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

/** Мир движка (v2 — с мешами марчинга):
 *  `"VFW2" | u32 frames | u32 shapes | u32 verts | f32 base`
 *  `shape: u32 level | u32 nv | u32 ni | nv × (f32 x,y,z) | ni × u32`
 *  `frame: u32 m | m × (u32 shape | f32 pos×3 | f32 quat×4)` */
export function parseWorld(buf) {
  const dv = new DataView(buf);
  const magic = magicOf(buf);
  if (magic !== "VFW2") throw new Error("не VFW2: " + magic);
  const nf = dv.getUint32(4, true);
  const nShapes = dv.getUint32(8, true);
  const total = dv.getUint32(12, true);
  const base = dv.getFloat32(16, true);

  let off = 20;
  const shapes = [];
  for (let s = 0; s < nShapes; s++) {
    const level = dv.getUint32(off, true);
    const nv = dv.getUint32(off + 4, true);
    const ni = dv.getUint32(off + 8, true);
    off += 12;
    const positions = new Float32Array(nv * 3);
    for (let i = 0; i < nv * 3; i++) {
      positions[i] = dv.getFloat32(off, true);
      off += 4;
    }
    const indices = new Uint32Array(ni);
    for (let i = 0; i < ni; i++) {
      indices[i] = dv.getUint32(off, true);
      off += 4;
    }
    // габарит формы — по вершинам меша
    const blo = [Infinity, Infinity, Infinity];
    const bhi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < nv * 3; i += 3) {
      for (let k = 0; k < 3; k++) {
        blo[k] = Math.min(blo[k], positions[i + k]);
        bhi[k] = Math.max(bhi[k], positions[i + k]);
      }
    }
    // Плоские нормали считаются здесь: меш фасеточный, у вершины на стыке
    // граней «правильной» одной нормали нет, поэтому разворачиваем треугольники
    // в отдельные вершины с нормалью грани. Иначе в шейдере normalize(vec3(0))
    // даёт NaN и объекты рисуются чёрными.
    const ntri = indices.length / 3;
    const verts = new Float32Array(ntri * 3 * 6); // (pos3 + nrm3) × 3 вершины
    let w = 0;
    for (let t = 0; t < ntri; t++) {
      const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
      const ax = positions[i0 * 3], ay = positions[i0 * 3 + 1], az = positions[i0 * 3 + 2];
      const bx = positions[i1 * 3], by = positions[i1 * 3 + 1], bz = positions[i1 * 3 + 2];
      const cx = positions[i2 * 3], cy = positions[i2 * 3 + 1], cz = positions[i2 * 3 + 2];
      const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
      const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
      let nx = e1y * e2z - e1z * e2y;
      let ny = e1z * e2x - e1x * e2z;
      let nz = e1x * e2y - e1y * e2x;
      const len = Math.hypot(nx, ny, nz) || 1;
      nx /= len; ny /= len; nz /= len;
      for (const [px, py, pz] of [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]) {
        verts[w++] = px; verts[w++] = py; verts[w++] = pz;
        verts[w++] = nx; verts[w++] = ny; verts[w++] = nz;
      }
    }
    shapes.push({ level, vertices: verts, count: ntri * 3, edge: base / (1 << level), blo, bhi });
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
      const quat = [
        dv.getFloat32(off, true),
        dv.getFloat32(off + 4, true),
        dv.getFloat32(off + 8, true),
        dv.getFloat32(off + 12, true),
      ];
      off += 16;
      row.push({ shape, pos, quat });
      const sh = shapes[shape];
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], pos[k] + sh.blo[k]);
        hi[k] = Math.max(hi[k], pos[k] + sh.bhi[k]);
      }
    }
    frames.push(row);
  }
  return { kind: "world", nf, total, base, shapes, frames, lo, hi };
}
