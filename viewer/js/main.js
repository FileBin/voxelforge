/* Точка входа: собирает вместе камеру, рендереры, панель и файл.
 *
 * Здесь только склейка — вся логика лежит в соседних модулях:
 *   parse.js        чтение .vfr и .vfm
 *   camera.js       орбита/зум/пан
 *   gl.js           WebGL2, программы, геометрия
 *   render-sim.js   инстансинг тел
 *   render-mesh.js  воксельный меш и каркас BCC
 *   render-grid.js  пол-сетка
 *   ui.js           панель, клавиши, перетаскивание
 */

import { OrbitCamera, attachControls } from "./camera.js";
import { createGL } from "./gl.js";
import { detectFormat, parseTrace, parseMesh } from "./parse.js";
import { persp, lookAt, mul } from "./mat4.js";
import { GridRenderer } from "./render-grid.js";
import { SimRenderer } from "./render-sim.js";
import { MeshRenderer } from "./render-mesh.js";
import { UI } from "./ui.js";

const canvas = document.getElementById("cv");
const cam = new OrbitCamera();
const ui = new UI({ onFile: (f) => loadFile(f) });

let gl = null;
let grid = null;
let simRenderer = null;
let meshRenderer = null;
let trace = null;
let mesh = null;

function resize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function viewProj() {
  const eye = cam.eye();
  const aspect = canvas.width / Math.max(1, canvas.height);
  const vp = mul(persp(0.9, aspect, 0.01, 5000), lookAt(eye, cam.target, [0, 1, 0]));
  return { vp, eye };
}

function render() {
  if (!gl) return;
  resize();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.043, 0.059, 0.078, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const { vp, eye } = viewProj();
  grid.draw(vp, eye);
  if (mesh) {
    meshRenderer.draw(vp, eye);
  } else if (trace) {
    simRenderer.prepare(trace, ui.frame);
    simRenderer.draw(vp, eye);
  }
}

function loadFile(file) {
  file.arrayBuffer().then((buf) => {
    const format = detectFormat(buf);
    if (!format) throw new Error("неизвестный формат (ожидался .vfr или .vfm)");

    if (format === "vfm") {
      trace = null;
      mesh = parseMesh(buf);
      meshRenderer.setMesh(mesh);
      cam.fit(mesh.lo, mesh.hi);
      ui.setMeshMode(
        `${file.name} · меш: ${mesh.nv} вершин, ${mesh.ni / 3} треугольников, ${mesh.nl / 2} рёбер каркаса`,
      );
    } else {
      mesh = null;
      trace = parseTrace(buf);
      cam.fit(trace.lo, trace.hi);
      ui.setTrace(trace.nf);
      ui.setMeta(`${file.name} · тел ${trace.n} · кадров ${trace.nf}`);
    }
    ui.hideMessage();
    render();
  }).catch((e) => ui.showError("ошибка чтения", e.message));
}

let last = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000 || 0);
  last = now;
  ui.advance(dt);
  render();
  requestAnimationFrame(loop);
}

try {
  gl = createGL(canvas);
  grid = new GridRenderer(gl);
  simRenderer = new SimRenderer(gl);
  meshRenderer = new MeshRenderer(gl);
  attachControls(canvas, cam);

  document.getElementById("lattice").addEventListener("change", (e) => {
    meshRenderer.showLattice = e.target.checked;
  });

  const q = new URLSearchParams(location.search).get("file");
  if (q) {
    fetch(q)
      .then((r) => r.arrayBuffer())
      .then((b) => loadFile(new File([b], q.split("/").pop())))
      .catch(() => {});
  }

  requestAnimationFrame(loop);
} catch (e) {
  ui.showError("WebGL2 недоступен", e.message);
}
