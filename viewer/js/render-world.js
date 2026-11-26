/* Отрисовка мира движка (.vfw): каждая BCC-ячейка — усечённый октаэдр.
 *
 * Кубических вокселей нет: меш один на все ячейки (bcc.js), а инстанс задаёт
 * только позицию и размер. Размер берётся из уровня ячейки, поэтому после
 * дробления осколка его ячейки в кадре становятся мельче — это видно глазом. */

import { compile, makeInstancedMesh } from "./gl.js";
import { VS_BODY, FS_BODY } from "./shaders.js";
import { bccMesh } from "./bcc.js";

export class WorldRenderer {
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

    this.cell = bccMesh();
    this.mesh = makeInstancedMesh(gl, this.loc, this.cell);
    this.data = null;
    this.instances = null;
    this.count = 0;
  }

  setData(data) {
    this.data = data;
    this.instances = new Float32Array(data.total * 6);
  }

  /** Инстансы кадра: позиция ячейки = центр тела + смещение ячейки. */
  #build(frame) {
    const d = this.data;
    const row = d.frames[Math.max(0, Math.min(d.nf - 1, frame))];
    const out = this.instances;
    let i = 0;
    for (const body of row) {
      const sh = d.shapes[body.shape];
      const e = sh.edge;
      for (let c = 0; c < sh.n; c++) {
        out[i * 6 + 0] = body.pos[0] + sh.offsets[c * 3];
        out[i * 6 + 1] = body.pos[1] + sh.offsets[c * 3 + 1];
        out[i * 6 + 2] = body.pos[2] + sh.offsets[c * 3 + 2];
        out[i * 6 + 3] = e;
        out[i * 6 + 4] = e;
        out[i * 6 + 5] = e;
        i++;
      }
    }
    this.count = i;
  }

  draw(vp, eye, frame) {
    if (!this.data) return;
    const gl = this.gl;
    this.#build(frame);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.mesh.inst);
    gl.bufferData(gl.ARRAY_BUFFER, this.instances.subarray(0, this.count * 6), gl.DYNAMIC_DRAW);
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.uVP, false, vp);
    gl.uniform3fv(this.uEye, eye);
    gl.uniform3fv(this.uTint, [0.72, 0.79, 0.86]);
    gl.bindVertexArray(this.mesh.vao);
    gl.drawElementsInstanced(gl.TRIANGLES, this.mesh.count, gl.UNSIGNED_INT, 0, this.count);
    gl.bindVertexArray(null);
  }
}
