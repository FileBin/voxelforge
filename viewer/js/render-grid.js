/* Пол-сетка: одна плоскость, узор считается в фрагментном шейдере. */

import { compile } from "./gl.js";
import { VS_GRID, FS_GRID } from "./shaders.js";

export class GridRenderer {
  constructor(gl) {
    this.gl = gl;
    this.prog = compile(gl, VS_GRID, FS_GRID);
    this.uVP = gl.getUniformLocation(this.prog, "uVP");
    this.uEye = gl.getUniformLocation(this.prog, "uEye");

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-500, 0, -500, 500, 0, -500, 500, 0, 500, -500, 0, 500]),
      gl.STATIC_DRAW,
    );
    const aPos = gl.getAttribLocation(this.prog, "aPos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.bindVertexArray(null);
  }

  draw(vp, eye) {
    const gl = this.gl;
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.uVP, false, vp);
    gl.uniform3fv(this.uEye, eye);
    gl.bindVertexArray(this.vao);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_INT, 0);
  }
}
