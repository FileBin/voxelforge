/* Отрисовка мира движка (.vfw): каждая форма — готовый меш марчинга по
 * тетраэдрам BCC, кадр задаёт жёсткое движение (перенос + поворот).
 *
 * Кубических вокселей и сфер здесь нет: поверхность тела — изоповерхность
 * поля ячеек, разрезанная тетраэдрами. Когда осколок дробит ячейки, у него
 * появляется новая форма с более мелким тетраэдральным шагом. */

import { compile } from "./gl.js";
import { VS_WORLD, FS_WORLD } from "./shaders.js";

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

      const vb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, sh.positions, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(this.loc.aPos);
      gl.vertexAttribPointer(this.loc.aPos, 3, gl.FLOAT, false, 0, 0);

      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sh.indices, gl.STATIC_DRAW);

      gl.bindVertexArray(null);
      return { vao, vb, ib, count: sh.indices.length };
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
      const sh = this.data.shapes[body.shape];
      const buf = this.buffers[body.shape];
      if (!buf || !buf.count) continue;
      gl.uniform3fv(this.uPos, body.pos);
      gl.uniform4fv(this.uQuat, body.quat);
      gl.uniform3fv(this.uTint, tint(sh.level));
      gl.bindVertexArray(buf.vao);
      gl.drawElements(gl.TRIANGLES, buf.count, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);
  }

  #dispose() {
    if (!this.buffers.length) return;
    const gl = this.gl;
    for (const b of this.buffers) {
      gl.deleteVertexArray(b.vao);
      gl.deleteBuffer(b.vb);
      gl.deleteBuffer(b.ib);
    }
    this.buffers = [];
  }
}
