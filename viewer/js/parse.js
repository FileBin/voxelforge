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
