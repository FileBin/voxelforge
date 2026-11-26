/* Отрисовка воксельного меша (.vfm): треугольники и каркас BCC-ячеек.
 *
 * Нормали не хранятся в файле — считаем производными в шейдере, для
 * воксельной поверхности это даже правильнее: видно грани. */

import { compile } from "./gl.js";
import { VS_MESH, FS_MESH, VS_LINE, FS_LINE } from "./shaders.js";

export class MeshRenderer {
  constructor(gl) {
    this.gl = gl;
    this.progMesh = compile(gl, VS_MESH, FS_MESH);
    this.progLines = compile(gl, VS_LINE, FS_LINE);
    this.meshLoc = {
      aPos: gl.getAttribLocation(this.progMesh, "aPos"),
      uVP: gl.getUniformLocation(this.progMesh, "uVP"),
      uEye: gl.getUniformLocation(this.progMesh, "uEye"),
      uTint: gl.getUniformLocation(this.progMesh, "uTint"),
    };
    this.lineLoc = {
      aPos: gl.getAttribLocation(this.progLines, "aPos"),
      uVP: gl.getUniformLocation(this.progLines, "uVP"),
      uTint: gl.getUniformLocation(this.progLines, "uTint"),
    };
    this.mesh = null;
    this.showLattice = true;
  }

  setMesh(mesh) {
    const gl = this.gl;
    this.#dispose();
    this.mesh = mesh;

    const triVao = gl.createVertexArray();
    gl.bindVertexArray(triVao);
    const vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.meshLoc.aPos);
    gl.vertexAttribPointer(this.meshLoc.aPos, 3, gl.FLOAT, false, 0, 0);
    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.idx, gl.STATIC_DRAW);

    const lineVao = gl.createVertexArray();
    gl.bindVertexArray(lineVao);
    const lb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, lb);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.lines, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(this.lineLoc.aPos);
    gl.vertexAttribPointer(this.lineLoc.aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    this.tris = { vao: triVao, buffers: [vb, ib], count: mesh.idx.length };
    this.lines = { vao: lineVao, buffers: [lb], count: mesh.lines.length / 3 };
  }

  draw(vp, eye) {
    if (!this.mesh) return;
    const gl = this.gl;

    gl.useProgram(this.progMesh);
    gl.uniformMatrix4fv(this.meshLoc.uVP, false, vp);
    gl.uniform3fv(this.meshLoc.uEye, eye);
    gl.uniform3fv(this.meshLoc.uTint, [0.62, 0.82, 0.95]);
    gl.bindVertexArray(this.tris.vao);
    gl.drawElements(gl.TRIANGLES, this.tris.count, gl.UNSIGNED_INT, 0);

    if (this.showLattice && this.lines.count) {
      gl.useProgram(this.progLines);
      gl.uniformMatrix4fv(this.lineLoc.uVP, false, vp);
      gl.uniform3fv(this.lineLoc.uTint, [0.55, 0.75, 0.95]);
      gl.bindVertexArray(this.lines.vao);
      gl.drawArrays(gl.LINES, 0, this.lines.count);
    }
    gl.bindVertexArray(null);
  }

  #dispose() {
    if (!this.mesh) return;
    const gl = this.gl;
    for (const set of [this.tris, this.lines]) {
      if (!set) continue;
      gl.deleteVertexArray(set.vao);
      for (const b of set.buffers) gl.deleteBuffer(b);
    }
    this.tris = null;
    this.lines = null;
  }
}
