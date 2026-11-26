/* Орбитальная камера: цель, расстояние, два угла. Управление — ЛКМ (орбита),
 * колесо (зум), ПКМ или Shift+ЛКМ (пан). */

export class OrbitCamera {
  constructor() {
    this.yaw = 0.6;
    this.pitch = 0.35;
    this.dist = 8;
    this.target = [0, 1, 0];
  }

  /** Поставить камеру так, чтобы в кадр попал весь габарит. */
  fit(lo, hi) {
    this.target = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    const ext = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 0.5);
    this.dist = ext * 1.7;
    this.pitch = 0.35;
    this.yaw = 0.6;
  }

  eye() {
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    return [
      this.target[0] + this.dist * cp * Math.sin(this.yaw),
      this.target[1] + this.dist * sp,
      this.target[2] + this.dist * cp * Math.cos(this.yaw),
    ];
  }
}

/** Навесить управление на канвас. */
export function attachControls(canvas, cam) {
  let drag = null;

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointerup", () => { drag = null; });

  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    drag.x = e.clientX;
    drag.y = e.clientY;
    if (drag.pan) {
      const k = cam.dist * 0.0016;
      const right = [Math.cos(cam.yaw), 0, -Math.sin(cam.yaw)];
      cam.target[0] += -right[0] * dx * k;
      cam.target[2] += -right[2] * dx * k;
      cam.target[1] += dy * k;
    } else {
      cam.yaw -= dx * 0.006;
      cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + dy * 0.006));
    }
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    cam.dist = Math.max(0.02, Math.min(50000, cam.dist * Math.exp(e.deltaY * 0.0012)));
  }, { passive: false });
}
