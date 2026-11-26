/* Инициализация WebGL2, компиляция программ и геометрия. */

export function createGL(canvas) {
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false, depth: true });
  if (!gl) throw new Error("нет WebGL2 в этом браузере");
  gl.enable(gl.DEPTH_TEST);
  /* Отсечение задних граней не включаем: при ручной намотке куба ошибка знака
   * сделала бы объекты невидимыми, а проверить это без браузера нельзя.
   * Глубина и так даёт верный результат для замкнутых мешей. */
  return gl;
}

export function compile(gl, vsSrc, fsSrc) {
  const make = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s) || "shader");
    }
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, make(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, make(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p) || "link");
  }
  return p;
}

/** Икосфера (вершины на единичной сфере, нормали — они же). */
export function icosphere(subdiv) {
  const t = (1 + Math.sqrt(5)) / 2;
  const len = (a) => Math.hypot(a[0], a[1], a[2]) || 1;
  const nrm = (a) => [a[0] / len(a), a[1] / len(a), a[2] / len(a)];
  let verts = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ].map(nrm);
  let idx = [
    0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
    1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
    3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
    4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
  ];
  for (let s = 0; s < subdiv; s++) {
    const mid = new Map();
    const getMid = (a, b) => {
      const key = a < b ? a * 100000 + b : b * 100000 + a;
      if (mid.has(key)) return mid.get(key);
      const va = verts[a];
      const vb = verts[b];
      verts.push(nrm([(va[0] + vb[0]) / 2, (va[1] + vb[1]) / 2, (va[2] + vb[2]) / 2]));
      mid.set(key, verts.length - 1);
      return verts.length - 1;
    };
    const next = [];
    for (let i = 0; i < idx.length; i += 3) {
      const a = idx[i];
      const b = idx[i + 1];
      const c = idx[i + 2];
      const ab = getMid(a, b);
      const bc = getMid(b, c);
      const ca = getMid(c, a);
      next.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    idx = next;
  }
  return { positions: verts, normals: verts, indices: idx };
}

/** Куб с гранёными нормалями (по 4 вершины на грань). */
export function cube() {
  const faces = [
    { n: [0, 1, 0], v: [[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]] },
    { n: [0, -1, 0], v: [[-1, -1, 1], [1, -1, 1], [1, -1, -1], [-1, -1, -1]] },
    { n: [1, 0, 0], v: [[1, -1, 1], [1, 1, 1], [1, 1, -1], [1, -1, -1]] },
    { n: [-1, 0, 0], v: [[-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1]] },
    { n: [0, 0, 1], v: [[-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]] },
    { n: [0, 0, -1], v: [[1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]] },
  ];
  const positions = [];
  const normals = [];
  const indices = [];
  faces.forEach((f, fi) => {
    f.v.forEach((p) => {
      positions.push(p);
      normals.push(f.n);
    });
    const b = fi * 4;
    indices.push(b, b + 1, b + 2, b, b + 2, b + 3);
  });
  return { positions, normals, indices };
}

/** Меш с атрибутами позиции/нормали и инстансным буфером внутри VAO. */
export function makeInstancedMesh(gl, loc, mesh) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  const vb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vb);
  const data = new Float32Array(mesh.positions.length * 6);
  for (let i = 0; i < mesh.positions.length; i++) {
    data[i * 6 + 0] = mesh.positions[i][0];
    data[i * 6 + 1] = mesh.positions[i][1];
    data[i * 6 + 2] = mesh.positions[i][2];
    data[i * 6 + 3] = mesh.normals[i][0];
    data[i * 6 + 4] = mesh.normals[i][1];
    data[i * 6 + 5] = mesh.normals[i][2];
  }
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc.aPos);
  gl.vertexAttribPointer(loc.aPos, 3, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(loc.aNrm);
  gl.vertexAttribPointer(loc.aNrm, 3, gl.FLOAT, false, 24, 12);

  const ib = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(mesh.indices), gl.STATIC_DRAW);

  const inst = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, inst);
  gl.enableVertexAttribArray(loc.iPos);
  gl.vertexAttribPointer(loc.iPos, 3, gl.FLOAT, false, 24, 0);
  gl.vertexAttribDivisor(loc.iPos, 1);
  gl.enableVertexAttribArray(loc.iScale);
  gl.vertexAttribPointer(loc.iScale, 3, gl.FLOAT, false, 24, 12);
  gl.vertexAttribDivisor(loc.iScale, 1);

  gl.bindVertexArray(null);
  return { vao, count: mesh.indices.length, inst };
}
