/* Панель и клавиши: состояние проигрывания, покадровый шаг, перетаскивание
 * файла. Никакой графики — только DOM, чтобы это можно было читать отдельно. */

const $ = (id) => document.getElementById(id);

export class UI {
  /** @param {{onFile: (f: File) => void}} handlers */
  constructor(handlers) {
    this.onFile = handlers.onFile;
    this.play = true;
    this.frame = 0;
    this.speed = 60;
    this.nf = 0;
    this.onFrameChange = null;

    $("play").addEventListener("click", () => this.togglePlay());
    $("prev").addEventListener("click", () => this.step(-1));
    $("next").addEventListener("click", () => this.step(1));
    $("prev10").addEventListener("click", () => this.step(-10));
    $("next10").addEventListener("click", () => this.step(10));
    $("speed").addEventListener("change", (e) => { this.speed = +e.target.value; });
    $("frame").addEventListener("input", (e) => this.setFrame(+e.target.value));
    $("file").addEventListener("change", (e) => {
      if (e.target.files[0]) this.onFile(e.target.files[0]);
    });

    window.addEventListener("keydown", (e) => {
      const tag = (e.target && e.target.tagName) || "";
      if (tag === "INPUT" && e.code !== "Space") return;
      switch (e.code) {
        case "Space": e.preventDefault(); this.togglePlay(); break;
        case "ArrowLeft": e.preventDefault(); this.step(e.shiftKey ? -10 : -1); break;
        case "ArrowRight": e.preventDefault(); this.step(e.shiftKey ? 10 : 1); break;
        case "ArrowUp": e.preventDefault(); this.step(10); break;
        case "ArrowDown": e.preventDefault(); this.step(-10); break;
        case "Home": if (this.nf) { e.preventDefault(); this.setFrame(0, true); } break;
        case "End": if (this.nf) { e.preventDefault(); this.setFrame(this.nf - 1, true); } break;
        default: break;
      }
    });

    for (const ev of ["dragenter", "dragover"]) {
      window.addEventListener(ev, (e) => e.preventDefault());
    }
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f) this.onFile(f);
    });

    this.syncPlay();
  }

  /** Подготовить ползунок под новую траекторию. */
  setTrace(nf) {
    this.nf = nf;
    this.frame = 0;
    $("frame").max = String(Math.max(0, nf - 1));
    $("frame").value = "0";
    $("frame").disabled = false;
    $("pos").textContent = nf ? `1 / ${nf}` : "—";
  }

  /** Режим меша: покадровый просмотр не нужен. */
  setMeshMode(text) {
    this.nf = 0;
    $("frame").disabled = true;
    $("pos").textContent = "—";
    $("meta").textContent = text;
  }

  setMeta(text) {
    $("meta").textContent = text;
  }

  hideMessage() {
    $("msg").style.display = "none";
  }

  showError(title, detail) {
    $("msg").style.display = "grid";
    $("msg").innerHTML = `<b>${title}</b><span>${detail}</span>`;
  }

  togglePlay() {
    this.play = !this.play;
    this.syncPlay();
  }

  syncPlay() {
    $("play").textContent = this.play ? "⏸ пауза" : "▶ играть";
  }

  /** Ручной шаг всегда ставит на паузу: если смотришь кадр, автолистание мешает. */
  setFrame(f, pause = false) {
    if (!this.nf) return;
    if (pause && this.play) this.togglePlay();
    this.frame = Math.max(0, Math.min(this.nf - 1, Math.round(f)));
    $("frame").value = String(this.frame);
    $("pos").textContent = `${this.frame + 1} / ${this.nf}`;
    if (this.onFrameChange) this.onFrameChange(this.frame);
  }

  step(d) {
    this.setFrame(this.frame + d, true);
  }

  /** Продвинуть время; вернуть текущий кадр. */
  advance(dt) {
    if (!this.nf || !this.play) return this.frame;
    this.acc = (this.acc || 0) + dt * this.speed;
    const steps = Math.floor(this.acc);
    if (steps > 0) {
      this.acc -= steps;
      this.frame = (this.frame + steps) % this.nf;
      $("frame").value = String(this.frame);
      $("pos").textContent = `${this.frame + 1} / ${this.nf}`;
      if (this.onFrameChange) this.onFrameChange(this.frame);
    }
    return this.frame;
  }
}
