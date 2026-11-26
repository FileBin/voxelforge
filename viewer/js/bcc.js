/* Геометрия BCC-ячейки — усечённый октаэдр (ячейка Вороного решётки BCC),
 * а не куб. Строится как пересечение полупространств:
 *
 *   |x| ≤ 1/2, |y| ≤ 1/2, |z| ≤ 1/2          — 6 квадратных граней
 *   |±x ± y ± z| ≤ 3/4                        — 8 шестиугольных граней
 *
 * Вершины — пересечения троек плоскостей, оставленные внутри всех остальных
 * ограничений; грани — вершины на каждой плоскости, отсортированные по углу.
 * Так решётка BCC замощает пространство без щелей: ячейка по построению
 * равна ячейке Вороного узла решётки. */

const EPS = 1e-6;

function planes() {
  const p = [];
  for (let axis = 0; axis < 3; axis++) {
    for (const s of [1, -1]) {
      const n = [0, 0, 0];
      n[axis] = s;
      p.push({ n, d: 0.5 });            // n·x ≤ d
    }
  }
  for (const sx of [1, -1]) {
    for (const sy of [1, -1]) {
      for (const sz of [1, -1]) {
        p.push({ n: [sx, sy, sz], d: 0.75 });
      }
    }
  }
  return p;
}

/** Пересечение трёх плоскостей; null, если вырождено. */
function solve(a, b, c) {
  const det =
    a.n[0] * (b.n[1] * c.n[2] - b.n[2] * c.n[1]) -
    a.n[1] * (b.n[0] * c.n[2] - b.n[2] * c.n[0]) +
    a.n[2] * (b.n[0] * c.n[1] - b.n[1] * c.n[0]);
  if (Math.abs(det) < 1e-9) return null;
  const d = [a.d, b.d, c.d];
  const out = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const m = [a.n.slice(), b.n.slice(), c.n.slice()];
    m[0][i] = d[0];
    m[1][i] = d[1];
    m[2][i] = d[2];
    out[i] =
      (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) / det;
  }
  return out;
}

export function bccCell() {
  const ps = planes();
  const verts = [];
  const onPlane = ps.map(() => []);

  for (let i = 0; i < ps.length; i++) {
    for (let j = i + 1; j < ps.length; j++) {
      for (let k = j + 1; k < ps.length; k++) {
        const v = solve(ps[i], ps[j], ps[k]);
        if (!v) continue;
        let inside = true;
        for (const p of ps) {
          if (p.n[0] * v[0] + p.n[1] * v[1] + p.n[2] * v[2] > p.d + EPS) {
            inside = false;
            break;
          }
        }
        if (!inside) continue;
        // дедупликация: вершина может прийти из нескольких троек
        if (verts.some((w) => Math.abs(w[0] - v[0]) < EPS && Math.abs(w[1] - v[1]) < EPS && Math.abs(w[2] - v[2]) < EPS)) {
          continue;
        }
        const id = verts.length;
        verts.push(v);
        for (let q = 0; q < ps.length; q++) {
          const p = ps[q];
          if (Math.abs(p.n[0] * v[0] + p.n[1] * v[1] + p.n[2] * v[2] - p.d) < EPS) {
            onPlane[q].push(id);
          }
        }
      }
    }
  }

  // грани: вершины на плоскости, отсортированные по углу вокруг нормали
  const pos = [];
  const nrm = [];
  for (let q = 0; q < ps.length; q++) {
    const ids = onPlane[q];
    if (ids.length < 3) continue;
    const n = ps[q].n;
    const len = Math.hypot(n[0], n[1], n[2]);
    const nn = [n[0] / len, n[1] / len, n[2] / len];
    // центр грани
    const c = [0, 0, 0];
    for (const id of ids) for (let a = 0; a < 3; a++) c[a] += verts[id][a] / ids.length;
    // базис в плоскости
    const up = Math.abs(nn[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const u = [
      up[1] * nn[2] - up[2] * nn[1],
      up[2] * nn[0] - up[0] * nn[2],
      up[0] * nn[1] - up[1] * nn[0],
    ];
    const ul = Math.hypot(u[0], u[1], u[2]);
    u[0] /= ul; u[1] /= ul; u[2] /= ul;
    const w = [
      nn[1] * u[2] - nn[2] * u[1],
      nn[2] * u[0] - nn[0] * u[2],
      nn[0] * u[1] - nn[1] * u[0],
    ];
    ids.sort((A, B) => {
      const dA = [verts[A][0] - c[0], verts[A][1] - c[1], verts[A][2] - c[2]];
      const dB = [verts[B][0] - c[0], verts[B][1] - c[1], verts[B][2] - c[2]];
      const aA = Math.atan2(dA[0] * w[0] + dA[1] * w[1] + dA[2] * w[2], dA[0] * u[0] + dA[1] * u[1] + dA[2] * u[2]);
      const aB = Math.atan2(dB[0] * w[0] + dB[1] * w[1] + dB[2] * w[2], dB[0] * u[0] + dB[1] * u[1] + dB[2] * u[2]);
      return aA - aB;
    });
    // веер: грани выпуклые
    for (let i = 1; i + 1 < ids.length; i++) {
      const a = verts[ids[0]], b = verts[ids[i]], d = verts[ids[i + 1]];
      // ориентация наружу
      const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const e2 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
      const cr = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
      ];
      const flip = cr[0] * nn[0] + cr[1] * nn[1] + cr[2] * nn[2] < 0;
      const tri = flip ? [a, d, b] : [a, b, d];
      for (const v of tri) {
        pos.push(v[0], v[1], v[2]);
        nrm.push(nn[0], nn[1], nn[2]);
      }
    }
  }

  return {
    pos: new Float32Array(pos),
    nrm: new Float32Array(nrm),
    nverts: verts.length,
    ntris: pos.length / 9,
  };
}

/** Тот же меш в формате gl.js: массивы троек + индексы. */
export function bccMesh() {
  const c = bccCell();
  const positions = [];
  const normals = [];
  const indices = [];
  for (let i = 0; i < c.pos.length / 3; i++) {
    positions.push([c.pos[i * 3], c.pos[i * 3 + 1], c.pos[i * 3 + 2]]);
    normals.push([c.nrm[i * 3], c.nrm[i * 3 + 1], c.nrm[i * 3 + 2]]);
    indices.push(i);
  }
  return { positions, normals, indices };
}
