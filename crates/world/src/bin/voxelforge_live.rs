// Живая демка: окно Vulkan, орбита мышью, стрельба, повторный раскол.
//
//   nix run .#live
//
// ЛКМ — орбита (клик без движения = выстрел), ПКМ/Shift — пан, колесо — зум,
// ПРОБЕЛ/F — выстрел в центр, T — каркас тетраэдров, R — сброс.
use std::ffi::{c_char, c_void, CString};

use voxelforge_world::live::{model_matrix, Live};
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
    fn vf_view_new(
        w: i32,
        h: i32,
        title: *const c_char,
        shader_dir: *const c_char,
        out: *mut *mut c_void,
    ) -> VfStatus;
    fn vf_view_free(v: *mut c_void);
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
    fn vf_view_draw(v: *mut c_void, mesh: i32, model: *const f32, tint: *const f32);
    fn vf_view_draw_lines(v: *mut c_void, lines: i32, model: *const f32, tint: *const f32);
    fn vf_view_end(v: *mut c_void);
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

fn main() {
    let mut explicit: Option<std::path::PathBuf> = None;
    let mut max_frames: u64 = 0; // 0 — без ограничения
    let mut tick_arg: Option<f32> = None;
    let mut shots: Vec<u64> = Vec::new();
    let mut tets_at_start = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--shaders" => explicit = it.next().map(Into::into),
            "--frames" => max_frames = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--tick" => tick_arg = it.next().and_then(|v| v.parse::<f32>().ok()),
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
                eprintln!("неизвестный аргумент {other}\n  --shaders DIR --frames N --shots 10,40");
                std::process::exit(2);
            }
        }
    }
    let shaders = find_shaders(explicit.as_deref())
        .expect("нет SPIR-V: nix build .#spirv или --shaders DIR");
    let shaders_c = CString::new(shaders.display().to_string()).unwrap();
    let title = CString::new("voxelforge — WASD полёт, мышь обзор, клик/пробел выстрел").unwrap();

    let mut view: *mut c_void = std::ptr::null_mut();
    unsafe {
        let st = vf_view_new(1280, 800, title.as_ptr(), shaders_c.as_ptr(), &mut view);
        if st.code != 0 {
            eprintln!("окно не открылось: код {} {}", st.code, if st.msg.is_null() {
                "".into()
            } else {
                std::ffi::CStr::from_ptr(st.msg).to_string_lossy().into_owned()
            });
            std::process::exit(1);
        }
    }

    let mut world = Live::new(&shaders);
    world.lines_on = tets_at_start;
    let mut cam = Fly {
        pos: [3.0, 2.0, 3.6],
        yaw: 0.0,
        pitch: 0.0,
        speed: 4.0,
    };
    cam.look_at(world.bodies[0].pos.map(|v| v as f32));
    let mut near = 0.05f32;
    let mut far = 60.0f32;
    let mut uploaded: Vec<(usize, i32)> = Vec::new();

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

    println!("voxelforge live · WASD полёт (Shift быстрее, Ctrl медленнее, Q/E вниз/вверх), мышь обзор, клик/пробел выстрел, T каркас, R сброс");

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
    // Такт физики: 20 Гц. Рендер интерполирует между тактами, поэтому
    // гладкость от этого не страдает, а нагрузка падает втрое.
    let tick_hz: f32 = tick_arg.unwrap_or(20.0).clamp(5.0, 240.0);
    let tick: f32 = 1.0 / tick_hz;
    let mut acc = 0.0f32;
    let mut prev_state: Vec<([f64; 3], [f64; 4])> = Vec::new();
    let mut alpha = 1.0f32;
    let mut last_upload: Vec<usize> = Vec::new();

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
            world.lines_on = !world.lines_on;
        }
        if input.key_r != 0 {
            world.reset();
            uploaded.clear();
            cam.look_at(world.bodies[0].pos.map(|v| v as f32));
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
            if world.shoot(from, [dir[0] as f64, dir[1] as f64, dir[2] as f64], 26.0) {
                println!("выстрел #{}", world.shots);
            }
        }

        // Физика: столько тактов, сколько прошло реального времени. В
        // скриптовом режиме (--frames) шаг строго один на кадр — иначе прогон
        // зависел бы от скорости машины и перестал быть воспроизводимым.
        let mut stepped = 0;
        // Всё спит — симуляцию не крутим вовсе. Иначе покоящаяся куча
        // продолжала стоить полный такт каждый кадр, хотя в ней ничего
        // не меняется (метка сна стоит в misc.y).
        let any_awake = world.mirror.iter().any(|g| g.misc[1] <= 0.5);
        if !any_awake {
            acc = 0.0;
            alpha = 1.0;
        } else if max_frames > 0 {
            prev_state.clear();
            prev_state.extend(world.bodies.iter().map(|b| (b.pos, b.quat)));
            world.step(tick, 6, 10);
            stepped = 1;
            acc = 0.0;
        } else {
            acc += frame_dt;
        }
        while max_frames == 0 && acc >= tick && stepped < 4 {
            prev_state.clear();
            prev_state.extend(world.bodies.iter().map(|b| (b.pos, b.quat)));
            world.step(tick, 6, 10);
            acc -= tick;
            stepped += 1;
        }
        // доля между последним и следующим тактом — по ней и рисуем
        alpha = if stepped > 0 { (acc / tick).clamp(0.0, 1.0) } else { 1.0 };
        if std::env::var("VF_DEBUG").is_ok() && frame_no < 40 && frame_no % 4 == 0 {
            for (i, b) in world.bodies.iter().enumerate() {
                if b.alive && i > 0 {
                    println!("  кадр {frame_no}: тело {i} {:?} v={:?}", b.pos.map(|v| (v * 100.0).round() / 100.0), b.vel.map(|v| (v * 10.0).round() / 10.0));
                }
            }
        }

        // новые формы (появились осколки) — залить в окно
        for i in 0..world.bodies.len() {
            if !world.bodies[i].alive || (world.bodies[i].mesh >= 0 && world.bodies[i].lines >= 0) {
                continue;
            }
            let Some(shape) = world.shape_for(i) else { continue };
            let mut pos = Vec::with_capacity(shape.mesh.positions.len() * 3);
            let mut nrm = Vec::with_capacity(shape.mesh.normals.len() * 3);
            for (p, n) in shape.mesh.positions.iter().zip(shape.mesh.normals.iter()) {
                pos.extend_from_slice(p);
                nrm.extend_from_slice(n);
            }
            let mesh_id = unsafe {
                vf_view_add_mesh(
                    view,
                    pos.as_ptr(),
                    nrm.as_ptr(),
                    shape.mesh.positions.len() as u32,
                    shape.mesh.indices.as_ptr(),
                    shape.mesh.indices.len() as u32,
                )
            };
            let line_id = if shape.edges.is_empty() {
                -1
            } else {
                let mut e = Vec::with_capacity(shape.edges.len() * 3);
                for p in &shape.edges {
                    e.extend_from_slice(p);
                }
                unsafe { vf_view_add_lines(view, e.as_ptr(), (shape.edges.len() / 2) as u32) }
            };
            world.bodies[i].mesh = mesh_id;
            world.bodies[i].lines = line_id;
            uploaded.push((i, mesh_id));
            last_upload.push(i);
        }
        if !world.last_log.is_empty() {
            println!("{}", world.last_log);
            world.last_log.clear();
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
        near = world.near.max(0.02);
        far = world.far.max(2.0);
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
            for i in 0..world.bodies.len() {
                let b = &world.bodies[i];
                if !b.alive || b.mesh < 0 {
                    continue;
                }
                let (rpos, rquat) = match prev_state.get(i) {
                    Some((pp, pq)) if alpha < 0.999 => (
                        [
                            pp[0] + (b.pos[0] - pp[0]) * alpha as f64,
                            pp[1] + (b.pos[1] - pp[1]) * alpha as f64,
                            pp[2] + (b.pos[2] - pp[2]) * alpha as f64,
                        ],
                        slerp(*pq, b.quat, alpha as f64),
                    ),
                    _ => (b.pos, b.quat),
                };
                let m = model_matrix(rpos, rquat);
                let tint = match b.kind {
                    voxelforge_world::live::Kind::Bullet => [0.95, 0.62, 0.28, 1.0],
                    voxelforge_world::live::Kind::Target => [0.74, 0.70, 0.62, 1.0],
                    _ => {
                        // камень: светлые и тёмные осколки, без кислотных оттенков
                        let k = 0.56 + 0.05 * ((i % 4) as f32);
                        [k + 0.06, k + 0.035, k, 1.0]
                    }
                };
                vf_view_draw(view, b.mesh, m.as_ptr(), tint.as_ptr());
                if world.lines_on && b.lines >= 0 {
                    let lt = [0.55, 0.85, 1.0, 1.0];
                    vf_view_draw_lines(view, b.lines, m.as_ptr(), lt.as_ptr());
                }
            }
            vf_view_end(view);
        }

        frames += 1;
        frame_no += 1;
        if max_frames > 0 && frame_no >= max_frames {
            println!(
                "[live] конец: кадров {frame_no} · тел {} · трещин {} · осколков {}",
                world.alive_count(),
                world.cracks,
                world.shards_made
            );
            break;
        }
        if frames % 120 == 0 {
            let dt = t0.elapsed().as_secs_f32();
            println!(
                "кадров {frames} · {:.0} fps · тел {} · трещин {} · осколков {}",
                frames as f32 / dt,
                world.alive_count(),
                world.cracks,
                world.shards_made
            );
            t0 = std::time::Instant::now();
            frames = 0;
        }
    }

    unsafe { vf_view_free(view) };
}
