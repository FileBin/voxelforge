/* Точка входа вьюера: камера, рендереры, панель, файл.
 *
 * Вьюер понимает два файла:
 *   .vfw — мир движка (BCC-ячейки, раскол и дробление)
 *   .vfr — траектории CPU-эталона (проверка физики)
 *
 * Модули: parse — форматы, bcc — геометрия ячейки, gl — WebGL2 и программы,
 * render-world — инстансинг ячеек, render-sim — сферы/блоки эталона,
 * render-grid — пол, camera — управление, ui — панель. */

import { OrbitCamera, attachControls } from "./camera.js";
import { createGL } from "./gl.js";
import { detectFormat, parseTrace, parseWorld } from "./parse.js";
import { persp, lookAt, mul } from "./mat4.js";
import { GridRenderer } from "./render-grid.js";
import { SimRenderer } from "./render-sim.js";
import { WorldRenderer } from "./render-world.js";
import { UI } from "./ui.js";

const canvas = document.getElementById("cv");
const cam = new OrbitCamera();
const ui = new UI({ onFile: (f) => loadFile(f) });

let gl = null;
let grid = null;
let simRenderer = null;
let worldRenderer = null;
let trace = null;
let world = null;

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
  if (world) {
    worldRenderer.draw(vp, eye, ui.frame);
  } else if (trace) {
    simRenderer.prepare(trace, ui.frame);
    simRenderer.draw(vp, eye);
  }
}

function describe(w) {
  const levels = new Set(w.shapes.map((s) => s.level));
  const edges = [...levels].sort().map((l) => (w.base / (1 << l)).toFixed(3)).join(" → ");
  return `ячейка ${edges} м · кадров ${w.nf}`;
}

/** Загрузить файл; экспортируется, чтобы путь проверялся тестом. */
export function loadFile(file) {
  file.arrayBuffer().then((buf) => {
    const format = detectFormat(buf);
    if (!format) {
      const head = new TextDecoder().decode(new Uint8Array(buf.slice(0, 4)));
      throw new Error(`неизвестный формат (начало: ${JSON.stringify(head)}); ожидался VFW1 или VFR2`);
    }

    if (format === "vfw") {
      trace = null;
      world = parseWorld(buf);
      worldRenderer.setData(world);
      cam.fit(world.lo, world.hi);
      ui.setTrace(world.nf);
      ui.setMeta(`${file.name} · ${world.shapes.length} форм · ${describe(world)}`);
    } else {
      world = null;
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
  worldRenderer = new WorldRenderer(gl);
  attachControls(canvas, cam);

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
