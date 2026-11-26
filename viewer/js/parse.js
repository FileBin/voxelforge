/* Разбор файлов: траектории (.vfr) и меш (.vfm).
 *
 * Форматы документированы в tests/refsim/src/lib.rs (vfr) и
 * tests/refsim/src/mesher.rs (encode_vfm) — здесь только чтение. */

/* ArrayBuffer нельзя индексировать: в браузере File.arrayBuffer() отдаёт
 * именно его, и `buf[0]` там undefined. Приводим к байтам явно. */
function magicOf(buf) {
  const u8 = buf instanceof ArrayBuffer
    ? new Uint8Array(buf)
    : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  return String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
}

/** "vfr" | "vfm" | null */
export function detectFormat(buf) {
  const m = magicOf(buf);
  if (m === "VFR1" || m === "VFR2") return "vfr";
  if (m === "VFM1") return "vfm";
  if (m === "VFX1") return "vfx";
  return null;
}

/** Траектории: `"VFR1/2" | u32 n | u32 nf | тела | nf × n × 3 float32`. */
export function parseTrace(buf) {
  const dv = new DataView(buf);
  const magic = magicOf(buf);
  if (magic !== "VFR1" && magic !== "VFR2") throw new Error("не VFR1/VFR2: " + magic);
  const n = dv.getUint32(4, true);
  const nf = dv.getUint32(8, true);
  let off = 12;
  const scale = [];
  const shapes = [];
  for (let i = 0; i < n; i++) {
    off += 8;                                   // масса (f64)
    const r = dv.getFloat32(off, true);
    off += 4;
    if (magic === "VFR2") {
      const shape = dv.getUint8(off);
      off += 1;
      const hx = dv.getFloat32(off, true);
      const hy = dv.getFloat32(off + 4, true);
      const hz = dv.getFloat32(off + 8, true);
      off += 12;
      scale.push([hx, hy, hz]);
      shapes.push(shape);
    } else {
      scale.push([r, r, r]);
      shapes.push(0);
    }
  }
  const pos = new Float32Array(n * nf * 3);
  for (let i = 0; i < n * nf * 3; i++) {
    pos[i] = dv.getFloat32(off, true);
    off += 4;
  }
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let f = 0; f < nf; f++) {
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 3; k++) {
        const v = pos[(f * n + i) * 3 + k];
        if (v < lo[k]) lo[k] = v;
        if (v > hi[k]) hi[k] = v;
      }
    }
  }
  return { n, nf, pos, scale, shapes, lo, hi };
}

/** Меш: `"VFM1" | u32 verts | u32 indices | u32 line_verts | ...`. */
export function parseMesh(buf) {
  const dv = new DataView(buf);
  const magic = magicOf(buf);
  if (magic !== "VFM1") throw new Error("не VFM1: " + magic);
  const nv = dv.getUint32(4, true);
  const ni = dv.getUint32(8, true);
  const nl = dv.getUint32(12, true);
  let off = 16;

  const verts = new Float32Array(nv * 3);
  for (let i = 0; i < nv * 3; i++) {
    verts[i] = dv.getFloat32(off, true);
    off += 4;
  }
  const idx = new Uint32Array(ni);
  for (let i = 0; i < ni; i++) {
    idx[i] = dv.getUint32(off, true);
    off += 4;
  }
  const lines = new Float32Array(nl * 3);
  for (let i = 0; i < nl * 3; i++) {
    lines[i] = dv.getFloat32(off, true);
    off += 4;
  }

  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nv; i++) {
    for (let k = 0; k < 3; k++) {
      const v = verts[i * 3 + k];
      if (v < lo[k]) lo[k] = v;
      if (v > hi[k]) hi[k] = v;
    }
  }
  return { nv, ni, nl, verts, idx, lines, lo, hi };
}

/** Раскол воксельного тела:
 *  `"VFX1" | u32 shards | u32 frames | u32 total_cells | f32 cell |
 *   по осколку: u32 n, n × (f32 xyz) | кадры: shards × (f32 xyz)`. */
export function parseFracture(buf) {
  const dv = new DataView(buf);
  const magic = magicOf(buf);
  if (magic !== "VFX1") throw new Error("не VFX1: " + magic);
  const nShards = dv.getUint32(4, true);
  const nf = dv.getUint32(8, true);
  const total = dv.getUint32(12, true);
  const cell = dv.getFloat32(16, true);
  let off = 20;

  const offsets = new Float32Array(total * 3);
  const startIndex = new Uint32Array(nShards);
  let cursor = 0;
  for (let s2 = 0; s2 < nShards; s2++) {
    const n = dv.getUint32(off, true);
    off += 4;
    startIndex[s2] = cursor;
    for (let i = 0; i < n * 3; i++) {
      offsets[cursor * 3 + i] = dv.getFloat32(off, true);
      off += 4;
    }
    cursor += n;
  }

  const frames = new Float32Array(nf * nShards * 3);
  for (let i = 0; i < nf * nShards * 3; i++) {
    frames[i] = dv.getFloat32(off, true);
    off += 4;
  }

  // Габарит: мир = центр осколка на нулевом кадре + локальное смещение ячейки.
  // Считать по каждому вокселю — единственный способ не промахнуться камерой.
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let s2 = 0; s2 < nShards; s2++) {
    const from = startIndex[s2];
    const to = s2 + 1 < nShards ? startIndex[s2 + 1] : total;
    const base = s2 * 3;
    for (let c = from; c < to; c++) {
      for (let k = 0; k < 3; k++) {
        const v = frames[base + k] + offsets[c * 3 + k];
        if (v < lo[k]) lo[k] = v;
        if (v > hi[k]) hi[k] = v;
      }
    }
  }
  for (let k = 0; k < 3; k++) {
    lo[k] -= cell * 0.5;
    hi[k] += cell * 0.5;
  }
  return { nShards, nf, total, cell, offsets, startIndex, frames, lo, hi };
}
