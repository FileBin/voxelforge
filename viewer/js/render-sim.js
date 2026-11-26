/* Отрисовка тел траектории: инстансинг сфер и коробок. */

import { compile, makeInstancedMesh, icosphere, cube } from "./gl.js";
import { VS_BODY, FS_BODY } from "./shaders.js";

export class SimRenderer {
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

    this.meshSphere = makeInstancedMesh(gl, this.loc, icosphere(2));
    this.meshBox = makeInstancedMesh(gl, this.loc, cube());
    this.spheres = [];
    this.boxes = [];
  }

  /** Подготовить массивы инстансов под конкретный кадр траектории. */
  prepare(trace, frame) {
    const base = frame * trace.n * 3;
    this.spheres.length = 0;
    this.boxes.length = 0;
    for (let i = 0; i < trace.n; i++) {
      const x = trace.pos[base + i * 3];
      const y = trace.pos[base + i * 3 + 1];
      const z = trace.pos[base + i * 3 + 2];
      const s = trace.scale[i];
      if (trace.shapes[i] === 1) this.boxes.push([x, y, z, s[0], s[1], s[2]]);
      else this.spheres.push([x, y, z, s[0], s[0], s[0]]);
    }
  }

  draw(vp, eye) {
    const gl = this.gl;
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.uVP, false, vp);
    gl.uniform3fv(this.uEye, eye);
    this.#drawSet(this.meshSphere, this.spheres, [0.55, 0.72, 0.92]);
    this.#drawSet(this.meshBox, this.boxes, [0.95, 0.72, 0.42]);
    gl.bindVertexArray(null);
  }

  #drawSet(mesh, list, tint) {
    if (!list.length) return;
    const gl = this.gl;
    const data = new Float32Array(list.length * 6);
    for (let i = 0; i < list.length; i++) {
      data[i * 6 + 0] = list[i][0];
      data[i * 6 + 1] = list[i][1];
      data[i * 6 + 2] = list[i][2];
      data[i * 6 + 3] = list[i][3];
      data[i * 6 + 4] = list[i][4];
      data[i * 6 + 5] = list[i][5];
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.inst);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.uniform3fv(this.uTint, tint);
    gl.bindVertexArray(mesh.vao);
    gl.drawElementsInstanced(gl.TRIANGLES, mesh.count, gl.UNSIGNED_INT, 0, list.length);
  }
}
