/* Отрисовка мира движка (.vfw): каждая форма — готовый меш марчинга по
 * тетраэдрам BCC, кадр задаёт жёсткое движение (перенос + поворот).
 *
 * Кубических вокселей и сфер здесь нет: поверхность тела — изоповерхность
 * поля ячеек, разрезанная тетраэдрами. Когда осколок дробит ячейки, у него
 * появляется новая форма с более мелким тетраэдральным шагом. */

import { compile } from "./gl.js";
import { VS_WORLD, FS_WORLD, VS_GRIDLINES, FS_LINE } from "./shaders.js";

export class WorldRenderer {
  constructor(gl) {
    this.gl = gl;
    this.prog = compile(gl, VS_WORLD, FS_WORLD);
    this.loc = {
      aPos: gl.getAttribLocation(this.prog, "aPos"),
      aNrm: gl.getAttribLocation(this.prog, "aNrm"),
    };
    this.uVP = gl.getUniformLocation(this.prog, "uVP");
    this.uEye = gl.getUniformLocation(this.prog, "uEye");
    this.uTint = gl.getUniformLocation(this.prog, "uTint");
    this.uPos = gl.getUniformLocation(this.prog, "uPos");
    this.uQuat = gl.getUniformLocation(this.prog, "uQuat");

    // сетка вокселей — отдельная программа линий
    this.progLines = compile(gl, VS_GRIDLINES, FS_LINE);
    this.lineLoc = {
      aPos: gl.getAttribLocation(this.progLines, "aPos"),
      uVP: gl.getUniformLocation(this.progLines, "uVP"),
      uTint: gl.getUniformLocation(this.progLines, "uTint"),
    };
    this.showGrid = false;
    this.data = null;
    this.buffers = [];
  }

  setData(data) {
    const gl = this.gl;
    this.#dispose();
    this.data = data;
    this.buffers = data.shapes.map((sh) => {
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);

      // позиция и нормаль в одном буфере: 6 float на вершину
      const vb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, sh.vertices, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(this.loc.aPos);
      gl.vertexAttribPointer(this.loc.aPos, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(this.loc.aNrm);
      gl.vertexAttribPointer(this.loc.aNrm, 3, gl.FLOAT, false, 24, 12);

      gl.bindVertexArray(null);

      // Каркас тетраэдров: именно сами воксели, а не их меш. Когда включён
      // режим «показать тетраэдры», меш не рисуется вовсе.
      const lines = sh.edges ? Array.from(sh.edges) : [];
      // если рёбер нет (старый файл) — хотя бы контур решётки
      if (lines.length === 0) {
        const lo = sh.gridLo;
        const hi = sh.gridHi;
        const pushSeg = (a, b) => lines.push(a[0], a[1], a[2], b[0], b[1], b[2]);
        for (let axis = 0; axis < 3; axis++) {
          for (let u = 0; u <= 1; u++) {
            for (let v = 0; v <= 1; v++) {
              const p0 = [0, 0, 0];
              const p1 = [0, 0, 0];
              const a1 = (axis + 1) % 3;
              const a2 = (axis + 2) % 3;
              p0[a1] = u ? hi[a1] : lo[a1];
              p0[a2] = v ? hi[a2] : lo[a2];
              p1[a1] = p0[a1];
              p1[a2] = p0[a2];
              p0[axis] = lo[axis];
              p1[axis] = hi[axis];
              pushSeg(p0, p1);
            }
          }
        }
      }
      const gvao = gl.createVertexArray();
      gl.bindVertexArray(gvao);
      const gvb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, gvb);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(this.lineLoc.aPos);
      gl.vertexAttribPointer(this.lineLoc.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);

      return { vao, vb, count: sh.count, gvao, gvb, gcount: lines.length / 3 };
    });
  }

  draw(vp, eye, frame) {
    if (!this.data) return;
    const gl = this.gl;
    const row = this.data.frames[Math.max(0, Math.min(this.data.nf - 1, frame))];
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.uVP, false, vp);
    gl.uniform3fv(this.uEye, eye);
    // оттенок зависит от уровня: крупные ячейки теплее, мелкие холоднее
    const tint = (level) => [
      0.80 - 0.05 * level,
      0.78 - 0.02 * level,
      0.72 + 0.06 * level,
    ];
    for (const body of row) {
      if (this.showGrid) continue; // режим «тетраэдры»: меш скрыт
      const sh = this.data.shapes[body.shape];
      const buf = this.buffers[body.shape];
      if (!buf || !buf.count) continue;
      gl.uniform3fv(this.uPos, body.pos);
      gl.uniform4fv(this.uQuat, body.quat);
      gl.uniform3fv(this.uTint, tint(sh.level));
      gl.bindVertexArray(buf.vao);
      gl.drawArrays(gl.TRIANGLES, 0, buf.count);
    }
    gl.bindVertexArray(null);

    if (this.showGrid) {
      gl.useProgram(this.progLines);
      gl.uniformMatrix4fv(this.lineLoc.uVP, false, vp);
      gl.uniform3fv(this.lineLoc.uTint, [0.30, 0.55, 0.85]);
      for (const body of row) {
        const buf = this.buffers[body.shape];
        if (!buf || !buf.gcount) continue;
        // линии заданы в локальных координатах — тот же перенос и поворот
        gl.uniform3fv(this.uPos, body.pos);
        gl.uniform4fv(this.uQuat, body.quat);
        gl.bindVertexArray(buf.gvao);
        gl.drawArrays(gl.LINES, 0, buf.gcount);
      }
      gl.bindVertexArray(null);
    }
  }

  #dispose() {
    if (!this.buffers.length) return;
    const gl = this.gl;
    for (const b of this.buffers) {
      gl.deleteVertexArray(b.vao);
      gl.deleteBuffer(b.vb);
      if (b.gvao) {
        gl.deleteVertexArray(b.gvao);
        gl.deleteBuffer(b.gvb);
      }
    }
    this.buffers = [];
  }
}
