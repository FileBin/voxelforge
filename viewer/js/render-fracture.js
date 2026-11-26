/* Раскол воксельного тела (.vfx): каждый осколок — набор вокселей, которые
 * едут вместе с ним. Рисуем инстансами кубиков: на кадр считаем позиции
 * «центр осколка + локальное смещение ячейки». */

import { compile, makeInstancedMesh, cube } from "./gl.js";
import { VS_BODY, FS_BODY } from "./shaders.js";

export class FractureRenderer {
  constructor(gl) {
    this.gl = gl;
    this.prog = compile(gl, VS_BODY, FS_BODY);
    this.loc = {
      aPos: gl.getAttribLocation(this.prog, "aPos"),
      aNrm: gl.getAttribLocation(this.prog, "aNrm"),
      iPos: gl.getAttribLocation(this.prog, "iPos"),
      iScale: gl.getAttribLocation(this.prog, "iScale"),
    };
    this.uVP = gl.getUniformLocation(this.prog, "uVP");
    this.uEye = gl.getUniformLocation(this.prog, "uEye");
    this.uTint = gl.getUniformLocation(this.prog, "uTint");
    this.mesh = makeInstancedMesh(gl, this.loc, cube());
    this.data = null;
    this.instances = null;
  }

  setData(data) {
    this.data = data;
    this.instances = new Float32Array(data.total * 6);
  }

  /** Пересобрать инстансы под кадр: позиция = центр осколка + смещение ячейки. */
  #buildFrame(frame) {
    const d = this.data;
    const out = this.instances;
    const half = d.cell * 0.5;
    let i = 0;
    for (let s = 0; s < d.nShards; s++) {
      const base = (frame * d.nShards + s) * 3;
      const sx = d.frames[base];
      const sy = d.frames[base + 1];
      const sz = d.frames[base + 2];
      // сколько ячеек у осколка — из разницы начал
      const from = d.startIndex[s];
      const to = s + 1 < d.nShards ? d.startIndex[s + 1] : d.total;
      for (let c = from; c < to; c++) {
        const o = c * 3;
        out[i * 6 + 0] = sx + d.offsets[o];
        out[i * 6 + 1] = sy + d.offsets[o + 1];
        out[i * 6 + 2] = sz + d.offsets[o + 2];
        out[i * 6 + 3] = half;
        out[i * 6 + 4] = half;
        out[i * 6 + 5] = half;
        i++;
      }
    }
    return i;
  }

  draw(vp, eye, frame) {
    if (!this.data) return;
    const gl = this.gl;
    const count = this.#buildFrame(Math.min(frame, this.data.nf - 1));
    gl.bindBuffer(gl.ARRAY_BUFFER, this.mesh.inst);
    gl.bufferData(gl.ARRAY_BUFFER, this.instances, gl.DYNAMIC_DRAW);
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.uVP, false, vp);
    gl.uniform3fv(this.uEye, eye);
    gl.uniform3fv(this.uTint, [0.86, 0.62, 0.45]);
    gl.bindVertexArray(this.mesh.vao);
    gl.drawElementsInstanced(gl.TRIANGLES, this.mesh.count, gl.UNSIGNED_INT, 0, count);
    gl.bindVertexArray(null);
  }
}
