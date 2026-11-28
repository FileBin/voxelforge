// Живая демка: окно Vulkan, орбита мышью, стрельба, повторный раскол.
//
//   nix run .#live
//
// ЛКМ — орбита (клик без движения = выстрел), ПКМ/Shift — пан, колесо — зум,
// ПРОБЕЛ/F — выстрел в центр, T — каркас тетраэдров, R — сброс.
use std::ffi::{c_char, c_void, CString};

use voxelforge_world::live::{model_matrix, Live};
use voxelforge_world::perf::{VfPerfRow, VfSceneCounts};
use voxelforge_world::sim::find_shaders;

#[repr(C)]
struct VfStatus {
    code: i32,
    msg: *const c_char,
}

#[repr(C)]
#[derive(Default)]
struct VfInput {
    alive: i32,
    resized: i32,
    width: u32,
    height: u32,
    mx: f32,
    my: f32,
    dx: f32,
    dy: f32,
    wheel: f32,
    lmb: i32,
    rmb: i32,
    lmb_pressed: i32,
    key_space: i32,
    key_r: i32,
    key_t: i32,
    key_f: i32,
    key_f3: i32,
    key_w: i32,
    key_a: i32,
    key_s: i32,
    key_d: i32,
    key_q: i32,
    key_e: i32,
    shift: i32,
    ctrl: i32,
}

unsafe extern "C" {
    /// Окно живёт на девайсе ядра (core: *mut vf_ctx).
    fn vf_view_new(
        out: *mut *mut c_void,
        core: *mut c_void,
        w: i32,
        h: i32,
        title: *const c_char,
        shader_dir: *const c_char,
    ) -> VfStatus;
    fn vf_view_free(v: *mut c_void);
    /* Две очереди: кадр ждёт тактовый таймлайн и сигналит свой. */
    fn vf_view_timelines(v: *mut c_void, tick_sem: *mut c_void, frame_sem: *mut c_void);
    fn vf_view_tick_ready(v: *mut c_void, value: u64, bank: u32);
    fn vf_view_poll(v: *mut c_void, input: *mut VfInput) -> i32;
    fn vf_view_camera(v: *mut c_void, vp: *const f32, eye: *const f32, near_m: f32, far_m: f32);
    fn vf_view_add_mesh(
        v: *mut c_void,
        pos: *const f32,
        nrm: *const f32,
        nv: u32,
        idx: *const u32,
        ni: u32,
    ) -> i32;
    fn vf_view_add_lines(v: *mut c_void, pts: *const f32, n_lines: u32) -> i32;
    fn vf_view_begin(v: *mut c_void, r: f32, g: f32, b: f32);
    fn vf_view_set_bodies(v: *mut c_void, buffer: *mut c_void, count: u32);
    /// Поза не передаётся: body — номер тела в буфере, alpha — сглаживание.
    fn vf_view_draw(v: *mut c_void, mesh: i32, body: u32, alpha: f32, tint: *const f32);
    fn vf_view_present_mode(v: *mut c_void) -> i32;
    fn vf_view_draw_lines(v: *mut c_void, lines: i32, model: *const f32, tint: *const f32);
    fn vf_view_end(v: *mut c_void);
    /// Оверлей стадий (F3 / VF_PERF=1): строки копируются внутрь окна.
    fn vf_view_perf(
        v: *mut c_void,
        rows: *const VfPerfRow,
        n: u32,
        tick_min_ms: f32,
        tick_mean_ms: f32,
        tick_max_ms: f32,
        scene: *const VfSceneCounts,
    );
}

// ---------------------------------------------------------------- матрицы

fn persp(fovy: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fovy * 0.5).tan();
    // В Vulkan ось Y в clip space смотрит ВНИЗ, в отличие от GL: без минуса
    // картинка получается вверх ногами.
    [
        f / aspect, 0.0, 0.0, 0.0,
        0.0, -f, 0.0, 0.0,
        0.0, 0.0, far / (near - far), -1.0,
        0.0, 0.0, (near * far) / (near - far), 0.0,
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-8);
    [a[0] / l, a[1] / l, a[2] / l]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = norm(sub(center, eye));
    let s = norm(cross(f, up));
    let u = cross(s, f);
    [
        s[0], u[0], -f[0], 0.0,
        s[1], u[1], -f[1], 0.0,
        s[2], u[2], -f[2], 0.0,
        -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0,
    ]
}

fn mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut o = [0.0f32; 16];
    for c in 0..4 {
        for r in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[k * 4 + r] * b[c * 4 + k];
            }
            o[c * 4 + r] = s;
        }
    }
    o
}

// ---------------------------------------------------------------- камера

/// Свободный полёт: WASD в плоскости взгляда, Q/E вниз/вверх, мышь — обзор.
struct Fly {
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
    speed: f32,
}

impl Fly {
    fn forward(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        [cp * self.yaw.sin(), self.pitch.sin(), cp * self.yaw.cos()]
    }

    /// Направление «вправо по экрану» = cross(forward, up): при yaw = 0 это -x,
    /// потому что камера смотрит в +z и мир повёрнут к ней левым боком.
    fn right(&self) -> [f32; 3] {
        [-self.yaw.cos(), 0.0, self.yaw.sin()]
    }

    fn look_at(&mut self, target: [f32; 3]) {
        let d = sub(target, self.pos);
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-4);
        self.pitch = (d[1] / len).asin();
        self.yaw = d[0].atan2(d[2]);
    }

    fn move_by(&mut self, dirs: [f32; 3], dt: f32) {
        let f = self.forward();
        let r = self.right();
        for i in 0..3 {
            self.pos[i] += (f[i] * dirs[0] + r[i] * dirs[1] + dirs[2] * if i == 1 { 1.0 } else { 0.0 })
                * self.speed
                * dt;
        }
    }
}

/// Интерполяция кватернионов (по короткой дуге) для сглаживания кадров.
fn slerp(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let b = if dot < 0.0 { [-b[0], -b[1], -b[2], -b[3]] } else { b };
    let d = dot.abs().min(1.0);
    let (sa, sb) = if d > 0.9995 {
        (1.0 - t, t)
    } else {
        let th = d.acos();
        let s = th.sin();
        (((1.0 - t) * th).sin() / s, (t * th).sin() / s)
    };
    let q = [
        a[0] * sa + b[0] * sb,
        a[1] * sa + b[1] * sb,
        a[2] * sa + b[2] * sb,
        a[3] * sa + b[3] * sb,
    ];
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
}


// ───────────────────────── фоновая физика ─────────────────────────
//
// Симуляция живёт в своём потоке и публикует снимки, а главный поток только
// рисует: кадр больше не ждёт такт физики. Между двумя последними тактами
// рендер интерполирует, поэтому картинка остаётся гладкой, даже когда такт
// стоит десятки миллисекунд.

#[derive(Clone, Copy)]
struct BSnap {
    id: u64,
    kind: u8,
    alive: bool,
    pos: [f64; 3],
    prev: [f64; 3],
    quat: [f64; 4],
    prevq: [f64; 4],
}

struct MeshData {
    pos: Vec<f32>,
    nrm: Vec<f32>,
    idx: Vec<u32>,
    edges: Vec<f32>,
}

struct Shared {
    /// VkBuffer буфера тел, как число: сброс мира создаёт новый буфер, и окну
    /// нужно перепривязать дескриптор.
    body_buf: u64,
    /// Две очереди: какой такт опубликован в этот банк (кадр ждёт его значение).
    pub_tick: u64,
    pub_bank: u32,
    bodies: Vec<BSnap>,
    pending: Vec<(u64, MeshData)>,
    tick: f32,
    t_pub: f64,
    t_pub_at: std::time::Instant,
    /// Такт считается прямо сейчас. Кадр ждёт его окончания: пока такт в
    /// полёте, позы в буфере меняются, и alpha от опубликованного снимка
    /// уже не соответствует паре prev→cur на устройстве.
    in_tick: bool,
    /// Две очереди: кадр НЕ ждёт закрытия такта — он читает банк, который
    /// опубликован прошлым тактом.
    split: bool,
    /// Измеренный интервал между публикациями (для alpha).
    interval: f32,
    last_pub_at: Option<std::time::Instant>,
    shots: usize,
    cracks: usize,
    shards: usize,
    alive: usize,
    lines_on: bool,
    /// Диагностика стадий включена (F3 / VF_PERF=1): показывать оверлей.
    perf_on: bool,
    /// Строки последнего окна диагностики и мин/сред/макс такта.
    perf_rows: Vec<VfPerfRow>,
    perf_tick: (f32, f32, f32),
    /// Счётчики сцены: всего тел, бодрствующих, точек контакта.
    perf_scene: VfSceneCounts,
    log: String,
}

enum Cmd {
    Shoot([f64; 3], [f64; 3], f64),
    Reset,
    SetLines(bool),
    /// F3: включить/выключить диагностику стадий (таблица + оверлей).
    TogglePerf,
}

struct SnapOut {
    body_buf: u64,
    /// Две очереди: значение такта и банк, из которого читает кадр.
    pub_tick: u64,
    pub_bank: u32,
    bodies: Vec<BSnap>,
    pending: Vec<(u64, MeshData)>,
    tick: f32,
    alpha: f32,
    shots: usize,
    cracks: usize,
    shards: usize,
    alive: usize,
    lines_on: bool,
    perf_on: bool,
    perf_rows: Vec<VfPerfRow>,
    perf_tick: (f32, f32, f32),
    perf_scene: VfSceneCounts,
    log: String,
    near: f32,
    far: f32,
}

enum Sim {
    Direct(Box<Live>),
    Thread {
        shared: std::sync::Arc<std::sync::Mutex<Shared>>,
        tx: std::sync::mpsc::Sender<Cmd>,
    },
}

/// `Live` держит указатели на Vulkan-объекты C-ядра. Они принадлежат только
/// физическому потоку, поэтому передача владения потоком безопасна.
struct LiveSend(Live);
unsafe impl Send for LiveSend {}

fn kind_code(k: voxelforge_world::live::Kind) -> u8 {
    match k {
        voxelforge_world::live::Kind::Bullet => 2,
        voxelforge_world::live::Kind::Target => 0,
        _ => 1,
    }
}

fn mesh_of(live: &Live, i: usize) -> Option<MeshData> {
    let shape = live.shape_for(i)?;
    let mut pos = Vec::with_capacity(shape.mesh.positions.len() * 3);
    let mut nrm = Vec::with_capacity(shape.mesh.normals.len() * 3);
    for (p, n) in shape.mesh.positions.iter().zip(shape.mesh.normals.iter()) {
        pos.extend_from_slice(p);
        nrm.extend_from_slice(n);
    }
    let mut edges = Vec::with_capacity(shape.edges.len() * 3);
    for p in &shape.edges {
        edges.extend_from_slice(p);
    }
    Some(MeshData {
        pos,
        nrm,
        idx: shape.mesh.indices.clone(),
        edges,
    })
}

fn snap_of(live: &Live, prev: &[([f64; 3], [f64; 4])]) -> Vec<BSnap> {
    live.bodies
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let (pp, pq) = prev.get(i).copied().unwrap_or((b.pos, b.quat));
            BSnap {
                id: b.id,
                kind: kind_code(b.kind),
                alive: b.alive,
                pos: b.pos,
                prev: pp,
                quat: b.quat,
                prevq: pq,
            }
        })
        .collect()
}

impl Sim {
    fn direct(live: Live, lines_on: bool, sleep_on: bool, fracture_on: bool) -> Self {
        let mut live = live;
        live.lines_on = lines_on;
        live.set_sleep(sleep_on);
        live.set_fracture(fracture_on);
        Sim::Direct(Box::new(live))
    }

    fn threaded(
        live: Live,
        tick: f32,
        lines_on: bool,
        sleep_on: bool,
        fracture_on: bool,
        substeps: u32,
        iterations: u32,
    ) -> Result<Self, String> {
        let shared = std::sync::Arc::new(std::sync::Mutex::new(Shared {
            body_buf: 0,
            pub_tick: 0,
            pub_bank: 0,
            bodies: Vec::new(),
            pending: Vec::new(),
            tick,
            t_pub: 0.0,
            t_pub_at: std::time::Instant::now(),
            in_tick: false,
            split: live.gpu.queues().4,
            interval: 0.0,
            last_pub_at: None,
            shots: 0,
            cracks: 0,
            shards: 0,
            alive: 0,
            lines_on,
            perf_on: live.gpu.perf_on(),
            perf_rows: Vec::new(),
            perf_tick: (0.0, 0.0, 0.0),
            perf_scene: VfSceneCounts::default(),
            log: String::new(),
        }));
        let (tx, rx) = std::sync::mpsc::channel::<Cmd>();
        let sh = shared.clone();
        std::thread::spawn(move || {
            // Мир с его девайсом уже создан в главном потоке и переехал сюда
            // целиком: девайс один на физику и окно.
            let mut live = live;
            live.lines_on = lines_on;
            live.set_sleep(sleep_on);
            live.set_fracture(fracture_on);
                let start = std::time::Instant::now();
            let mut last = std::time::Instant::now();
            let mut acc = 0.0f32;
                    loop {
                while let Ok(cmd) = rx.try_recv() {
                    match cmd {
                        Cmd::Shoot(from, dir, speed) => {
                            live.shoot(from, dir, speed);
                        }
                        Cmd::Reset => live.reset(),
                        Cmd::SetLines(on) => live.lines_on = on,
                        Cmd::TogglePerf => {
                            let on = live.gpu.perf_toggle();
                            sh.lock().unwrap().perf_on = on;
                        }
                    }
                }
                let now = std::time::Instant::now();
                let dt = now.duration_since(last).as_secs_f32().clamp(0.0005, 0.25);
                last = now;
                let awake = live.awake();
                let mut fresh: Vec<(u64, MeshData)> = Vec::new();
                if awake {
                    acc += dt;
                    let mut steps = 0;
                    while acc >= tick && steps < 2 {
                        sh.lock().unwrap().in_tick = true;
                        let prev: Vec<([f64; 3], [f64; 4])> =
                            live.bodies.iter().map(|b| (b.pos, b.quat)).collect();
                        live.step(tick, substeps, iterations);
                        // геометрия новых тел — сразу, чтобы главный поток успел
                        for i in 0..live.bodies.len() {
                            if !live.bodies[i].alive || live.bodies[i].mesh >= 0 {
                                continue;
                            }
                            if let Some(m) = mesh_of(&live, i) {
                                fresh.push((live.bodies[i].id, m));
                                live.bodies[i].mesh = 0;
                            }
                        }
                        let bodies = snap_of(&live, &prev);
                        let mut g = sh.lock().unwrap();
                        /* Две очереди: кадр читает банк, опубликованный
                         * ПРОШЛЫМ тактом, и ждёт его таймлайн — текущий такт
                         * в буфере он не смотрит. */
                        if live.gpu.queues().4 {
                            let (pt, pb) = live.gpu.published_bank();
                            g.pub_tick = pt;
                            g.pub_bank = pb as u32;
                            g.body_buf = live.gpu.bank_buffer(pb) as usize as u64;
                        } else {
                            g.body_buf = live.gpu.body_buffer() as usize as u64;
                        }
                        g.bodies = bodies;
                        g.pending.append(&mut fresh);
                        g.t_pub = start.elapsed().as_secs_f64();
                        let now_pub = std::time::Instant::now();
                        if let Some(prev_pub) = g.last_pub_at {
                            let d = now_pub.duration_since(prev_pub).as_secs_f32();
                            /* Измеренный интервал, а не номинальный: такт
                             * считается дольше, чем 1/20 с, и на номинале
                             * alpha упиралась в 1 раньше нового снимка —
                             * картинка замирала, потом прыгала. */
                            g.interval = if g.interval > 0.0 {
                                0.8 * g.interval + 0.2 * d
                            } else {
                                d
                            };
                        }
                        g.last_pub_at = Some(now_pub);
                        g.t_pub_at = now_pub;
                        g.in_tick = false;
                        g.shots = live.shots;
                        g.cracks = live.cracks;
                        g.shards = live.shards_made;
                        g.alive = live.alive_count();
                        g.lines_on = live.lines_on;
                        g.perf_on = live.gpu.perf_on();
                        if g.perf_on {
                            g.perf_rows = live.gpu.perf_overlay();
                            g.perf_tick = live.gpu.perf_tick_range();
                            g.perf_scene = live.gpu.perf_scene();
                        } else {
                            g.perf_rows.clear();
                            g.perf_scene = VfSceneCounts::default();
                        }
                        if !live.last_log.is_empty() {
                            g.log = std::mem::take(&mut live.last_log);
                        }
                        acc -= tick;
                        steps += 1;
                    }
                } else {
                    acc = 0.0;
                }
                if !fresh.is_empty() {
                    let mut g = sh.lock().unwrap();
                    g.pending.append(&mut fresh);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });
        Ok(Sim::Thread { shared, tx })
    }

    fn shoot(&mut self, from: [f64; 3], dir: [f64; 3], speed: f64) {
        match self {
            Sim::Direct(l) => {
                let _ = l.shoot(from, dir, speed);
            }
            Sim::Thread { tx, .. } => {
                let _ = tx.send(Cmd::Shoot(from, dir, speed));
            }
        }
    }

    fn reset(&mut self) {
        match self {
            Sim::Direct(l) => l.reset(),
            Sim::Thread { tx, .. } => {
                let _ = tx.send(Cmd::Reset);
            }
        }
    }

    fn set_lines(&mut self, on: bool) {
        match self {
            Sim::Direct(l) => l.lines_on = on,
            Sim::Thread { tx, .. } => {
                let _ = tx.send(Cmd::SetLines(on));
            }
        }
    }

    /// F3: включить/выключить диагностику стадий (консольная таблица и оверлей).
    fn toggle_perf(&mut self) {
        match self {
            Sim::Direct(l) => {
                l.gpu.perf_toggle();
            }
            Sim::Thread { tx, .. } => {
                let _ = tx.send(Cmd::TogglePerf);
            }
        }
    }

    fn first_pos(&self) -> [f64; 3] {
        match self {
            Sim::Direct(l) => l.bodies.first().map(|b| b.pos).unwrap_or([0.0; 3]),
            Sim::Thread { shared, .. } => shared
                .lock()
                .unwrap()
                .bodies
                .first()
                .map(|b| b.pos)
                .unwrap_or([0.0; 3]),
        }
    }

    /// Снимок для рисования. В скриптовом режиме (Direct) шагает физику сам
    /// и отдаёт alpha = 1: прогон должен быть воспроизводимым.
    fn frame(&mut self, tick: f32, scripted: bool, substeps: u32, iterations: u32) -> SnapOut {
        match self {
            Sim::Direct(l) => {
                if scripted {
                    l.step(tick, substeps, iterations);
                }
                let prev: Vec<([f64; 3], [f64; 4])> = l.bodies.iter().map(|b| (b.pos, b.quat)).collect();
                let mut pending = Vec::new();
                for i in 0..l.bodies.len() {
                    if !l.bodies[i].alive || l.bodies[i].mesh >= 0 {
                        continue;
                    }
                    if let Some(m) = mesh_of(l, i) {
                        pending.push((l.bodies[i].id, m));
                        l.bodies[i].mesh = 0;
                    }
                }
                let (pub_tick, pub_bank) = l.gpu.published_bank();
                SnapOut {
                    body_buf: if l.gpu.queues().4 {
                        l.gpu.bank_buffer(pub_bank) as usize as u64
                    } else {
                        l.gpu.body_buffer() as usize as u64
                    },
                    pub_tick,
                    pub_bank: pub_bank as u32,
                    bodies: snap_of(l, &prev),
                    pending,
                    tick,
                    alpha: 1.0,
                    shots: l.shots,
                    cracks: l.cracks,
                    shards: l.shards_made,
                    alive: l.alive_count(),
                    lines_on: l.lines_on,
                    perf_on: l.gpu.perf_on(),
                    perf_rows: if l.gpu.perf_on() { l.gpu.perf_overlay() } else { Vec::new() },
                    perf_tick: l.gpu.perf_tick_range(),
                    perf_scene: l.gpu.perf_scene(),
                    log: String::new(),
                    near: l.near,
                    far: l.far,
                }
            }
            Sim::Thread { shared, .. } => {
                /* НЕ ждём такт. Ожидание закрытия такта само создавало рывок:
                 * кадр вставал на время физики, и кадры выходили парами с
                 * одинаковой позой — глаз читает это как «перепутанные кадры».
                 * Вместо этого: если такт в полёте, рисуем опубликованное
                 * состояние целиком (alpha = 1), без полуинтерполяций.
                 * Позы на GPU при этом могут быть новее опубликованных, но мы
                 * их не смешиваем — смешивать нечего при alpha = 1. */
                /* Ждём ЗАКРЫТИЯ такта: пока такт считается, позы в буфере
                 * меняются, и alpha от прошлой публикации перестаёт
                 * соответствовать паре prev→cur — объекты прыгают на такт
                 * вперёд и отпрыгивают назад, при этом камера ровная (она от
                 * поз не зависит). GPU одна очередь и так ждал бы такт, так что
                 * ожидание на CPU ничего не добавляет. */
                if !shared.lock().unwrap().split {
                    for _ in 0..4000 {
                        if !shared.lock().unwrap().in_tick {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                }
                let mut g = shared.lock().unwrap();
                let step = if g.interval > 0.0 { g.interval } else { g.tick };
                /* ВАЖНО: такт «в полёте» НЕ повод рисовать alpha = 1. При 20 Гц
                 * такте и 60 fps каждый третий кадр попадал в это состояние, и
                 * картинка шагала целым тактом (пуля на 26 м/с — 1.3 м за кадр).
                 * Это и было «дрожание». alpha = 1 оставляем только для явного
                 * флага и для настоящей перегрузки (интервал такта разошёлся с
                 * номиналом), когда интерполяция заведомо врёт. */
                let overdue = std::env::var("VF_NO_INTERP").is_ok()
                    || step > g.tick * 1.25;
                let alpha = if overdue || step <= 0.0 {
                    1.0
                } else {
                    (g.t_pub_at.elapsed().as_secs_f32() / step).clamp(0.0, 1.0)
                };
                SnapOut {
                    body_buf: g.body_buf,
                    pub_tick: g.pub_tick,
                    pub_bank: g.pub_bank,
                    bodies: g.bodies.clone(),
                    pending: std::mem::take(&mut g.pending),
                    tick: g.tick,
                    alpha,
                    shots: g.shots,
                    cracks: g.cracks,
                    shards: g.shards,
                    alive: g.alive,
                    lines_on: g.lines_on,
                    perf_on: g.perf_on,
                    perf_rows: g.perf_rows.clone(),
                    perf_tick: g.perf_tick,
                    perf_scene: g.perf_scene,
                    log: std::mem::take(&mut g.log),
                    near: 0.0,
                    far: 60.0,
                }
            }
        }
    }
}

fn main() {
    let mut explicit: Option<std::path::PathBuf> = None;
    let mut max_frames: u64 = 0; // 0 — без ограничения
    let mut tick_arg: Option<f32> = None;
    let mut sleep_on = true;
    // Раскол включён по умолчанию; --no-fracture выключает его для отладки
    // столкновений: с цельной целью видно, останавливает ли контакт снаряд.
    let mut fracture_on = true;
    // Подшаги и итерации солвера: раньше были жёстко 10 и 18, и на кучу это
    // ложилось тяжело (6 подпроходов x 18 итераций на такт).
    /* Подшаги и итерации: меньше прежних 10/18. Сетка стала тяжелее (теты
     * Сомирвилля заходят в соседние ячейки), и такт перестал укладываться в
     * кадровый бюджет — а это и давало дрожание. Флаги --substeps/--iterations
     * по-прежнему переопределяют. */
    let mut substeps_arg: u32 = 8;
    let mut iterations_arg: u32 = 12;
    let mut shots: Vec<u64> = Vec::new();
    let mut tets_at_start = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--shaders" => explicit = it.next().map(Into::into),
            "--frames" => max_frames = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--tick" => tick_arg = it.next().and_then(|v| v.parse::<f32>().ok()),
            "--no-sleep" => sleep_on = false,
            "--no-fracture" => fracture_on = false,
            "--substeps" => substeps_arg = it.next().and_then(|v| v.parse().ok()).unwrap_or(10),
            "--iterations" => iterations_arg = it.next().and_then(|v| v.parse().ok()).unwrap_or(18),
            "--tets" => tets_at_start = true,
            "--shots" => {
                shots = it
                    .next()
                    .unwrap_or_default()
                    .split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect()
            }
            other => {
                eprintln!("неизвестный аргумент {other}\n  --shaders DIR --frames N --shots 10,40 [--no-sleep] [--no-fracture] [--substeps N] [--iterations N]");
                std::process::exit(2);
            }
        }
    }
    let shaders = find_shaders(explicit.as_deref())
        .expect("нет SPIR-V: nix build .#spirv или --shaders DIR");
    let shaders_c = CString::new(shaders.display().to_string()).unwrap();
    let title = CString::new("voxelforge — WASD полёт, мышь обзор, клик/пробел выстрел, F3 стадии").unwrap();

    // Мир (и вместе с ним девайс Vulkan) создаётся ДО окна: окно рендерит на
    // том же девайсе, иначе буферы тел физически недоступны графике.
    let live = Live::new_windowed(&shaders);
    let body_buf0 = live.gpu.body_buffer();
    let mut view: *mut c_void = std::ptr::null_mut();
    unsafe {
        let st = vf_view_new(
            &mut view,
            live.ctx(),
            1280,
            800,
            title.as_ptr(),
            shaders_c.as_ptr(),
        );
        if st.code != 0 {
            eprintln!("окно не открылось: код {} {}", st.code, if st.msg.is_null() {
                "".into()
            } else {
                std::ffi::CStr::from_ptr(st.msg).to_string_lossy().into_owned()
            });
            std::process::exit(1);
        }
    }

    // Окно читает позы прямо из буфера физики: дескриптор привязывается один
    // раз, а после сброса мира (R) — заново, потому что буфер новый.
    unsafe { vf_view_set_bodies(view, body_buf0, voxelforge_world::sim::MAX_BODIES as u32) };
    /* Две очереди (VF_SPLIT=1): окно ждёт тактовый таймлайн на значении
     * опубликованного банка и сигналит кадровый — по нему такт понимает,
     * что банк прочитан и его можно писать снова. */
    if live.gpu.queues().4 {
        unsafe {
            vf_view_timelines(
                view,
                live.gpu.stream_semaphore(voxelforge_world::sim::VF_STREAM_TICK),
                live.gpu.stream_semaphore(voxelforge_world::sim::VF_STREAM_FRAME),
            )
        };
    }
    let mut seen_body_buf = body_buf0 as usize as u64;
    let mut seen_pub_tick = 0u64;

    /* Физика идёт своим тактом (по умолчанию 20 Гц) независимо от частоты
     * кадров: на 240 Гц мониторе кадров четыре на такт, и движение сглаживает
     * интерполяция по alpha. */
    let tick_hz: f32 = tick_arg.unwrap_or(20.0).clamp(5.0, 240.0);
    let tick: f32 = 1.0 / tick_hz;
    let mut lines_on = tets_at_start;

    // В интерактивном режиме физика уходит в свой поток; в скриптовом
    // (--frames) она остаётся здесь, чтобы прогон был воспроизводимым.
    let mut sim = if max_frames > 0 {
        Sim::direct(live, tets_at_start, sleep_on, fracture_on)
    } else {
        match Sim::threaded(live, tick, tets_at_start, sleep_on, fracture_on, substeps_arg, iterations_arg) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("физика не запустилась: {e}");
                std::process::exit(1);
            }
        }
    };
    let mut cam = Fly {
        pos: [3.0, 2.0, 3.6],
        yaw: 0.0,
        pitch: 0.0,
        speed: 4.0,
    };
    cam.look_at(sim.first_pos().map(|v| v as f32));
    let mut near = 0.05f32;
    let mut far = 60.0f32;
    let mut meshes: std::collections::HashMap<u64, (i32, i32)> = std::collections::HashMap::new();

    // сетка пола
    let mut grid: Vec<f32> = Vec::new();
    let ext = 8.0f32;
    let mut t = -ext;
    while t <= ext + 1e-4 {
        grid.extend_from_slice(&[-ext, 0.0, t, ext, 0.0, t]);
        grid.extend_from_slice(&[t, 0.0, -ext, t, 0.0, ext]);
        t += 1.0;
    }
    let grid_id = unsafe { vf_view_add_lines(view, grid.as_ptr(), (grid.len() / 6) as u32) };

    println!("voxelforge live · WASD полёт (Shift быстрее, Ctrl медленнее, Q/E вниз/вверх), мышь обзор, клик/пробел выстрел, T каркас, F3 стадии, R сброс");

    let mut prev_lmb = 0;
    let mut pressed = false;
    let mut press_x = 0.0f32;
    let mut press_y = 0.0f32;
    let mut frames = 0u64;
    let mut frame_no = 0u64;
    let mut t0 = std::time::Instant::now();
    let mut last_frame = std::time::Instant::now();
    // Физика идёт своим фиксированным тактом, кадры — своим: рендер
    // интерполирует между двумя последними состояниями симуляции.
    let mut acc = 0.0f32;

    /* VF_STEPS=1 — мерим то, что видит глаз: позицию тела так, как её считает
     * шейдер (mix(prev, cur, alpha)), и шаг между кадрами. Скачки назад после
     * шага вперёд = «перепутанные кадры»; их доля и есть число, которое ищем. */
    let dbg_steps = std::env::var("VF_STEPS").is_ok();
    let mut last_drawn: Option<[f32; 3]> = None;
    let mut last_step: [f32; 3] = [0.0; 3];
    let (mut moves, mut flips, mut max_step) = (0u32, 0u32, 0.0f32);
    let mut frame_t0 = std::time::Instant::now();
    /* Прогрев (по умолчанию ВЫКЛЮЧЕН: включается VF_WARMUP=мс). Если
     * стартовые лаги лечатся тем, что игрок успевает пострелять, дело в
     * подъёме частот GPU, и тогда прогрев их снимает. На дискретных
     * картах частоты поднимаются не мгновенно, и первый тяжёлый кадр (целый
     * куб) выглядит как «лагает в начале». Тот же эффект лечится тем, что
     * игрок успевает пострелять в сторону, — значит это подъём частот, а не
     * алгоритм. VF_WARMUP=мс меняет длительность, 0 — выключает. */
    let warm_ms: u64 = std::env::var("VF_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let warm_until = if max_frames == 0 && warm_ms > 0 {
        eprintln!("[live] прогрев GPU {warm_ms} мс");
        Some(std::time::Instant::now() + std::time::Duration::from_millis(warm_ms))
    } else {
        None
    };
    loop {
        // Реальная дельта кадра: камера должна идти по времени, а не по
        // фиксированному 1/60 — иначе на просевшем FPS она двигается медленнее.
        let frame_dt = {
            let now = std::time::Instant::now();
            let d = now.duration_since(last_frame).as_secs_f32();
            last_frame = now;
            d.clamp(0.0005, 0.1)
        };
        let mut input = VfInput::default();
        let alive = unsafe { vf_view_poll(view, &mut input) };
        if let Some(t) = warm_until {
            if std::time::Instant::now() < t {
                /* пока греемся — не стреляем и не двигаем камеру */
                input = VfInput::default();
            }
        }
        if alive == 0 {
            break;
        }
        // Первые кадры окно ещё мапится, и SDL успевает отдать мусорные
        // события: на первом запуске это читалось как нажатие и давало
        // выстрел прямо в плиту — она и «улетала вверх». После R всё было
        // нормально, потому что окно уже отрисовано. Гасим ввод двух первых
        // кадров целиком.
        if frames < 2 {
            input = VfInput::default();
        }

        // обзор: любая зажатая кнопка мыши
        if input.lmb != 0 || input.rmb != 0 {
            cam.yaw -= input.dx * 0.005;
            cam.pitch = (cam.pitch - input.dy * 0.005).clamp(-1.5, 1.5);
        }
        if input.wheel != 0.0 {
            cam.speed = (cam.speed * (1.0 + input.wheel * 0.15)).clamp(0.4, 60.0);
        }
        // полёт: W/S вперёд-назад, A/D в стороны, Q/E вниз-вверх
        let mut dirs = [0.0f32; 3];
        if input.key_w != 0 { dirs[0] += 1.0; }
        if input.key_s != 0 { dirs[0] -= 1.0; }
        if input.key_d != 0 { dirs[1] += 1.0; }
        if input.key_a != 0 { dirs[1] -= 1.0; }
        if input.key_e != 0 { dirs[2] += 1.0; }
        if input.key_q != 0 { dirs[2] -= 1.0; }
        let boost = if input.shift != 0 { 4.0 } else if input.ctrl != 0 { 0.25 } else { 1.0 };
        if dirs != [0.0, 0.0, 0.0] {
            let save = cam.speed;
            cam.speed *= boost;
            cam.move_by(dirs, frame_dt);
            cam.speed = save;
        }
        if input.lmb_pressed != 0 {
            press_x = input.mx;
            press_y = input.my;
            pressed = true;
        }

        let mut shoot = false;
        // Выстрел кликом — только после НАСТОЯЩЕГО нажатия: без этого флага
        // отпускениe кнопки, которую никто не нажимал, тоже давало выстрел.
        if prev_lmb != 0 && input.lmb == 0 {
            let moved = (input.mx - press_x).abs() + (input.my - press_y).abs();
            if pressed && moved < 6.0 {
                shoot = true;
            }
        }
        pressed = false;
        prev_lmb = input.lmb;
        if input.key_space != 0 || input.key_f != 0 {
            shoot = true;
        }
        if input.key_t != 0 {
            lines_on = !lines_on;
            sim.set_lines(lines_on);
        }
        if input.key_f3 != 0 {
            sim.toggle_perf();
        }
        if input.key_r != 0 {
            sim.reset();
            meshes.clear();
            cam.look_at(sim.first_pos().map(|v| v as f32));
            println!("сброс: снова один цельный объект");
        }

        if !shots.is_empty() && shots.contains(&frame_no) {
            shoot = true;
        }
        let eye = cam.pos;
        if shoot {
            let dir = cam.forward();
            let from = [
                eye[0] as f64 + dir[0] as f64 * 0.6,
                eye[1] as f64 + dir[1] as f64 * 0.6,
                eye[2] as f64 + dir[2] as f64 * 0.6,
            ];
            sim.shoot(from, [dir[0] as f64, dir[1] as f64, dir[2] as f64], 26.0);
        }

        // Кадр: физика к этому моменту уже посчитана в своём потоке
        // (или шагнула здесь, если режим скриптовый).
        let out = sim.frame(tick, max_frames > 0, substeps_arg, iterations_arg);

        if dbg_steps && matches!(sim, Sim::Thread { .. }) {
            /* Позиция, которую видит шейдер = mix(prev, cur, alpha) по
             * опубликованной паре. Считаем её на CPU из снимка — так видно
             * именно то, что попадает в кадр, без чтения буферов с устройства. */
            if let Some(b) = out.bodies.get(1) {
                let (cur, prev) = (b.pos, b.prev);
                let a = out.alpha as f64;
                let p = [
                    (prev[0] + (cur[0] - prev[0]) * a) as f32,
                    (prev[1] + (cur[1] - prev[1]) * a) as f32,
                    (prev[2] + (cur[2] - prev[2]) * a) as f32,
                ];
                if let Some(l) = last_drawn {
                    let d = [p[0] - l[0], p[1] - l[1], p[2] - l[2]];
                    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    max_step = max_step.max(len);
                    if len > 1e-4 {
                        let pl = (last_step[0] * last_step[0]
                            + last_step[1] * last_step[1]
                            + last_step[2] * last_step[2])
                            .sqrt();
                        if pl > 1e-4 {
                            moves += 1;
                            if d[0] * last_step[0] + d[1] * last_step[1] + d[2] * last_step[2]
                                < 0.0
                            {
                                flips += 1;
                            }
                        }
                        last_step = d;
                    }
                }
                last_drawn = Some(p);
            }
        }

        if out.body_buf != 0
            && (out.body_buf != seen_body_buf || out.pub_tick != seen_pub_tick)
        {
            seen_body_buf = out.body_buf;
            seen_pub_tick = out.pub_tick;
            unsafe {
                vf_view_set_bodies(
                    view,
                    seen_body_buf as usize as *mut c_void,
                    voxelforge_world::sim::MAX_BODIES as u32,
                );
                vf_view_tick_ready(view, out.pub_tick, out.pub_bank);
            };
        }

        // новые формы (появились осколки) — залить в окно
        for (id, m) in out.pending.iter() {
            let mesh_id = unsafe {
                vf_view_add_mesh(
                    view,
                    m.pos.as_ptr(),
                    m.nrm.as_ptr(),
                    (m.pos.len() / 3) as u32,
                    m.idx.as_ptr(),
                    m.idx.len() as u32,
                )
            };
            let line_id = if m.edges.is_empty() {
                -1
            } else {
                unsafe { vf_view_add_lines(view, m.edges.as_ptr(), (m.edges.len() / 6) as u32) }
            };
            meshes.insert(*id, (mesh_id, line_id));
        }
        if !out.log.is_empty() {
            println!("{}", out.log);
        }

        // кадр
        let aspect = if input.height > 0 {
            input.width as f32 / input.height as f32
        } else {
            1.6
        };
        let proj = persp(1.05, aspect, near, far);
        let target = [
            eye[0] + cam.forward()[0],
            eye[1] + cam.forward()[1],
            eye[2] + cam.forward()[2],
        ];
        let viewm = look_at(eye, target, [0.0, 1.0, 0.0]);
        let vp = mul(proj, viewm);
        near = out.near.max(0.02);
        far = out.far.max(2.0);
        unsafe {
            vf_view_camera(view, vp.as_ptr(), [eye[0], eye[1], eye[2], near].as_ptr(), near, far);
            vf_view_begin(view, 0.055, 0.07, 0.10);
            vf_view_draw_lines(
                view,
                grid_id,
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    .as_ptr(),
                [0.18, 0.22, 0.28, 0.0].as_ptr(),
            );
            for (i, b) in out.bodies.iter().enumerate() {
                if !b.alive {
                    continue;
                }
                let Some(&(mesh_id, line_id)) = meshes.get(&b.id) else {
                    continue;
                };
                let tint = match b.kind {
                    2 => [0.95, 0.62, 0.28, 1.0],
                    0 => [0.74, 0.70, 0.62, 1.0],
                    _ => {
                        // камень: светлые и тёмные осколки, без кислотных оттенков
                        let k = 0.56 + 0.05 * ((i % 4) as f32);
                        [k + 0.06, k + 0.035, k, 1.0]
                    }
                };
                /* Меш: позу и сглаживание считает шейдер из буфера тел.
                 * Пока включён каркас (T), меши не рисуем: режим переключается
                 * между сеткой и поверхностью, а не накладывает одно на другое. */
                if !lines_on {
                    vf_view_draw(view, mesh_id, i as u32, out.alpha, tint.as_ptr());
                }
                if lines_on && line_id >= 0 {
                    /* Каркас тетраэдров — отладка: ему по-прежнему нужна
                     * матрица, её считаем на CPU из того же снимка. */
                    let (rpos, rquat) = if out.alpha < 0.999 {
                        (
                            [
                                b.prev[0] + (b.pos[0] - b.prev[0]) * out.alpha as f64,
                                b.prev[1] + (b.pos[1] - b.prev[1]) * out.alpha as f64,
                                b.prev[2] + (b.pos[2] - b.prev[2]) * out.alpha as f64,
                            ],
                            slerp(b.prevq, b.quat, out.alpha as f64),
                        )
                    } else {
                        (b.pos, b.quat)
                    };
                    let m = model_matrix(rpos, rquat);
                    vf_view_draw_lines(view, line_id, m.as_ptr(), [0.35, 0.42, 0.5, 0.0].as_ptr());
                }
            }
            /* Оверлей стадий: рисуется последним, поверх сцены. Числа те же,
             * что в консольной таблице, — F3 включает и выключает оба. Пустой
             * указатель при включённой диагностике значит «такт ещё не пришёл»:
             * панель так и скажет, а не исчезнет молча. */
            let prow = &out.perf_rows;
            let ptr = if out.perf_on {
                prow.as_ptr()
            } else {
                std::ptr::null()
            };
            vf_view_perf(
                view,
                ptr,
                prow.len() as u32,
                out.perf_tick.0,
                out.perf_tick.1,
                out.perf_tick.2,
                &out.perf_scene,
            );
            vf_view_end(view);
        }

        /* Ограничитель нужен только когда показ не привязан к развёртке
         * (MAILBOX/IMMEDIATE): иначе цикл молотит сотни кадров, GPU забит и
         * такт физики ждёт всю эту очередь. С FIFO ритм задаёт композитор. */
        if max_frames == 0 && unsafe { vf_view_present_mode(view) } != 0 {
            let want = std::env::var("VF_FPS")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(60.0)
                .max(5.0);
            let target = std::time::Duration::from_secs_f32(1.0 / want);
            if let Some(rest) = target.checked_sub(frame_t0.elapsed()) {
                std::thread::sleep(rest);
            }
            frame_t0 = std::time::Instant::now();
        }

        frames += 1;
        frame_no += 1;
        if max_frames > 0 && frame_no >= max_frames {
            println!(
                "[live] конец: кадров {frame_no} · тел {} · трещин {} · осколков {}",
                out.alive,
                out.cracks,
                out.shards
            );
            break;
        }
        if frames % 120 == 0 {
            let dt = t0.elapsed().as_secs_f32();
            println!(
                "кадров {frames} · {:.0} fps · тел {} · трещин {} · осколков {}",
                frames as f32 / dt,
                out.alive,
                out.cracks,
                out.shards
            );
            t0 = std::time::Instant::now();
            frames = 0;
        }
    }

    if dbg_steps {
        println!(
            "шаг кадра: движений {moves} · разворотов {flips} ({:.0}%) · макс шаг {max_step:.4}",
            if moves > 0 { 100.0 * flips as f32 / moves as f32 } else { 0.0 }
        );
    }

    unsafe { vf_view_free(view) };
}
