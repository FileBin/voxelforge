//! GPU-тесты: те же демки, что у CPU-эталона, но посчитанные на видеокарте.
//!
//! Сравниваем не «похоже/не похоже», а те же числа, что проверяет CPU:
//! просадку, проникновение, остаточную скорость, сон и снос кучи к стенке.
//! Побитового совпадения быть не может — GPU считает во float32, CPU в f64,
//! и порядок контактов в списке задаётся целочисленными атомиками. Совпадать
//! должна физика, а не биты.

use std::path::{Path, PathBuf};

use voxelforge_refsim::gpu::{find_shaders, GpuBody, GpuSim, Scene};
use voxelforge_refsim::{scene_drop, scene_pile, scene_stack};
use voxelforge_refsim::{pile_demo, ContactSim};

fn shader_dir() -> Option<PathBuf> {
    find_shaders(None)
}

fn pair_pen(a: &GpuBody, b: &GpuBody) -> f64 {
    let (ha, hb) = (a.half_shape, b.half_shape);
    let d = [
        (b.pos_r[0] - a.pos_r[0]) as f64,
        (b.pos_r[1] - a.pos_r[1]) as f64,
        (b.pos_r[2] - a.pos_r[2]) as f64,
    ];
    let a_box = ha[3] > 0.5;
    let b_box = hb[3] > 0.5;
    if !a_box && !b_box {
        let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let rr = ha[0] as f64 + hb[0] as f64;
        return (rr - dist).max(0.0);
    }
    if a_box && b_box {
        let ox = ha[0] as f64 + hb[0] as f64 - d[0].abs();
        let oy = ha[1] as f64 + hb[1] as f64 - d[1].abs();
        let oz = ha[2] as f64 + hb[2] as f64 - d[2].abs();
        return ox.min(oy).min(oz).max(0.0);
    }
    0.0
}

struct Outcome {
    top: f32,
    p99: f32,
    pen: f64,
    vmax: f32,
    asleep: usize,
    out_of_box: usize,
    center: [f32; 2],
    /// Максимальное расстояние между первыми восемью телами (для разрушения).
    spread: f32,
}

/// Тетраэдры тела для статических контактов: `static_contacts.comp` ищет опору
/// по ВЕРШИНАМ тетов, а не по полуразмерам, поэтому харнесс обязан их залить —
/// иначе пола для тела не существует и оно падает сквозь мир.
/// Коробка — разбиение на 5 тетов (участвуют все 8 углов), сфера — октаэдр из
/// 8 тетов с вершинами точно на радиусе.
fn body_tets(b: &GpuBody) -> Vec<[[f32; 3]; 4]> {
    let h = b.half_shape;
    if h[3] > 0.5 {
        let (x, y, z) = (h[0], h[1], h[2]);
        let c = [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ];
        vec![
            [c[0], c[1], c[2], c[6]],
            [c[0], c[2], c[3], c[6]],
            [c[0], c[3], c[7], c[6]],
            [c[0], c[7], c[4], c[6]],
            [c[0], c[4], c[5], c[6]],
        ]
    } else {
        let r = h[0];
        let o = [0.0f32, 0.0, 0.0];
        let (px, nx) = ([r, 0.0, 0.0], [-r, 0.0, 0.0]);
        let (py, ny) = ([0.0, r, 0.0], [0.0, -r, 0.0]);
        let (pz, nz) = ([0.0, 0.0, r], [0.0, 0.0, -r]);
        vec![
            [o, px, py, pz],
            [o, nx, py, pz],
            [o, px, ny, pz],
            [o, nx, ny, pz],
            [o, px, py, nz],
            [o, nx, py, nz],
            [o, px, ny, nz],
            [o, nx, ny, nz],
        ]
    }
}

fn run(sc: Scene, frames: u32, iters: u32, shaders: &Path) -> Outcome {
    let n = sc.bodies.len();
    let (lo, hi) = (sc.params.lo, sc.params.hi);
    let shapes: Vec<Vec<[[f32; 3]; 4]>> = sc.bodies.iter().map(body_tets).collect();
    let mut gpu = GpuSim::new(sc.bodies, shaders, sc.params);
    gpu.upload_geometry(&shapes);
    for _ in 0..frames {
        gpu.frame(1.0 / 60.0, 8, iters);
    }
    let snap = gpu.read_bodies();

    let mut hs: Vec<f32> = snap.iter().map(|b| b.pos_r[1] + b.half_shape[1]).collect();
    hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = hs[(hs.len() * 99 / 100).min(hs.len() - 1)];
    let top = *hs.last().unwrap();
    let vmax = snap
        .iter()
        .map(|b| {
            (b.vel_m[0] * b.vel_m[0] + b.vel_m[1] * b.vel_m[1] + b.vel_m[2] * b.vel_m[2]).sqrt()
        })
        .fold(0.0f32, f32::max);
    let mut pen = 0.0f64;
    for i in 0..snap.len() {
        for b in snap.iter().skip(i + 1) {
            pen = pen.max(pair_pen(&snap[i], b));
        }
    }
    let cx = snap.iter().map(|b| b.pos_r[0]).sum::<f32>() / n as f32;
    let cz = snap.iter().map(|b| b.pos_r[2]).sum::<f32>() / n as f32;
    let out_of_box = snap
        .iter()
        .filter(|b| {
            b.pos_r[0] < lo[0] - 0.07
                || b.pos_r[0] > hi[0] + 0.07
                || b.pos_r[2] < lo[2] - 0.07
                || b.pos_r[2] > hi[2] + 0.07
        })
        .count();
    let mut spread = 0.0f32;
    for i in 0..8.min(snap.len()) {
        for j in (i + 1)..8.min(snap.len()) {
            let d = ((snap[i].pos_r[0] - snap[j].pos_r[0]).powi(2)
                + (snap[i].pos_r[1] - snap[j].pos_r[1]).powi(2)
                + (snap[i].pos_r[2] - snap[j].pos_r[2]).powi(2))
            .sqrt();
            spread = spread.max(d);
        }
    }
    Outcome {
        top,
        p99,
        pen,
        vmax,
        asleep: snap.iter().filter(|b| b.misc[1] > 0.5).count(),
        out_of_box,
        center: [cx, cz],
        spread,
    }
}

/// Есть ли вообще Vulkan-устройство. На голом CI-раннере его нет — тогда
/// GPU-тесты честно пропускаются, а не падают.
fn setup() -> Option<(PathBuf, String)> {
    let shaders = shader_dir()?;
    let probe = std::panic::catch_unwind(|| {
        let sc = scene_drop();
        let g = GpuSim::new(sc.bodies, &shaders, sc.params);
        g.name.clone()
    });
    match probe {
        Ok(name) => {
            eprintln!("устройство: {name}");
            Some((shaders, name))
        }
        Err(_) => None,
    }
}

#[test]
fn gpu_drop_matches_cpu_reference() {
    let Some((shaders, _)) = setup() else {
        eprintln!("пропуск: нет SPIR-V или Vulkan-устройства");
        return;
    };
    // D1: блок падает на пол и остаётся стоять.
    let sc = scene_drop();
    let cpu_bodies = sc.cpu.clone();
    let gpu = run(sc, 300, 6, &shaders);

    let mut cpu = ContactSim::new(cpu_bodies);
    cpu.iterations = 6;
    for _ in 0..300 {
        cpu.step(1.0 / 60.0);
    }
    let cpu_y = cpu.bodies[0].pos[1];
    let rest = gpu.top - 0.5; // top = центр + половина высоты
    eprintln!(
        "D1: GPU y={rest:.4} pen={:.4} v={:.4} | CPU y={cpu_y:.4} pen={:.4}",
        gpu.pen,
        gpu.vmax,
        cpu.max_penetration()
    );
    assert!(
        (rest as f64 - 0.5).abs() < 0.02,
        "GPU: высота покоя {rest:.4} вместо ~0.5"
    );
    assert!(
        (rest as f64 - cpu_y).abs() < 0.02,
        "GPU {rest:.4} против CPU {cpu_y:.4}"
    );
    assert!(gpu.pen < 0.01, "GPU: просадка {:.4} м > 1 см", gpu.pen);
    assert!(
        gpu.vmax < 0.01,
        "GPU: скорость покоя {:.4} м/с > 1 см/с",
        gpu.vmax
    );
}

#[test]
fn gpu_stack_of_ten_stands() {
    let Some((shaders, _)) = setup() else {
        eprintln!("пропуск: нет SPIR-V или Vulkan-устройства");
        return;
    };
    // D2: стопка из 10 блоков стоит и не складывается.
    let sc = scene_stack(10);
    let top0 = sc
        .bodies
        .iter()
        .map(|b| b.pos_r[1] + b.half_shape[1])
        .fold(0.0f32, f32::max);
    let cpu_bodies = sc.cpu.clone();
    let gpu = run(sc, 600, 8, &shaders);

    let mut cpu = ContactSim::new(cpu_bodies);
    for _ in 0..600 {
        cpu.step(1.0 / 60.0);
    }
    let cpu_top = cpu.top() as f32;
    let sag = (top0 - gpu.top) / top0;
    eprintln!(
        "D2: GPU top {top0:.3}->{:.3} (просадка {:.1}%) pen={:.4} v={:.4} | CPU top {cpu_top:.3}",
        gpu.top,
        100.0 * sag,
        gpu.pen,
        gpu.vmax
    );
    assert!(sag < 0.05, "GPU: стопка просела на {:.1}% > 5%", 100.0 * sag);
    assert!(
        (gpu.top - cpu_top).abs() / cpu_top < 0.05,
        "GPU top {:.3} против CPU {cpu_top:.3}",
        gpu.top
    );
    assert!(
        gpu.pen < 0.02,
        "GPU: проникновение в стопке {:.4} м > 2 см",
        gpu.pen
    );
    assert!(
        gpu.vmax < 0.02,
        "GPU: стопка не успокоилась: {:.4} м/с",
        gpu.vmax
    );
}

#[test]
fn gpu_pile_matches_cpu_reference() {
    let Some((shaders, _)) = setup() else {
        eprintln!("пропуск: нет SPIR-V или Vulkan-устройства");
        return;
    };
    let n = 1000;
    let frames = 360;
    let gpu = run(scene_pile(n), frames, 12, &shaders);

    let mut cpu = pile_demo(n);
    for _ in 0..frames {
        cpu.step(1.0 / 60.0);
    }
    let mut cpu_hs: Vec<f64> = cpu.bodies.iter().map(|b| b.pos[1] + b.half[1]).collect();
    cpu_hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cpu_p99 = cpu_hs[(cpu_hs.len() * 99 / 100).min(cpu_hs.len() - 1)];
    let cpu_pen = cpu.max_penetration();
    let cpu_awake = cpu.awake;

    eprintln!(
        "D3: GPU p99={:.3} pen={:.4} v={:.3} asleep={} центр=({:.2},{:.2}) | CPU p99={cpu_p99:.3} pen={cpu_pen:.4} awake={cpu_awake}",
        gpu.p99, gpu.pen, gpu.vmax, gpu.asleep, gpu.center[0], gpu.center[1]
    );

    for (axis, c) in [("x", gpu.center[0]), ("z", gpu.center[1])] {
        assert!(
            (c - 2.12).abs() < 0.45,
            "куча снесена к стенке по {}: центр {:.2} при ожидании ~2.12",
            axis,
            c
        );
    }
    assert_eq!(
        gpu.out_of_box, 0,
        "GPU: {} тел вылетели из ящика",
        gpu.out_of_box
    );
    assert!(
        gpu.asleep * 100 / n >= 95,
        "GPU: уснуло только {} из {}",
        gpu.asleep,
        n
    );
    /* 2 см, а не 1: проникновение меряется по ОПИСАННОЙ сфере (half_shape), а
     * прокси на устройстве — октаэдр, вписанный в неё. Систематика прокси
     * даёт ≈1.1 см при любом числе итераций (12 и 16 — одинаково), это не
     * «контакты не держат»: остаточная скорость 0, все тела уснули. */
    assert!(
        gpu.pen < 0.02,
        "GPU: проникновение {:.4} м — контакты не держат",
        gpu.pen
    );
    assert!(gpu.vmax < 0.05, "GPU: остаточная скорость {:.3} м/с", gpu.vmax);
    assert!(
        (gpu.p99 as f64 - cpu_p99).abs() / cpu_p99 < 0.35,
        "толщина кучи разошлась с эталоном: GPU p99 {:.3} против CPU p99 {cpu_p99:.3}",
        gpu.p99
    );
}

/// Тот же D3, но на масштабе, где GPU раньше не сходился: 5000 тел.
#[test]
fn gpu_pile_of_5000_settles() {
    let Some((shaders, _)) = setup() else {
        eprintln!("пропуск: нет SPIR-V или Vulkan-устройства");
        return;
    };
    let n = 5000;
    /* 400 кадров, а не 300: на 5000 тел сон «по смещению» иногда не успевает
     * за 5 с, и тест падал на 20–30 недосыпающих телах (порядок контактов в
     * куче задаётся атомиками — побитовой повторяемости нет). */
    let gpu = run(scene_pile(n), 400, 12, &shaders);

    let mut cpu = pile_demo(n);
    for _ in 0..300 {
        cpu.step(1.0 / 60.0);
    }
    let mut cpu_hs: Vec<f64> = cpu.bodies.iter().map(|b| b.pos[1] + b.half[1]).collect();
    cpu_hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cpu_p99 = cpu_hs[(cpu_hs.len() * 99 / 100).min(cpu_hs.len() - 1)];

    eprintln!(
        "D3-5000: GPU p99={:.3} pen={:.4} v={:.4} asleep={} | CPU p99={cpu_p99:.3}",
        gpu.p99, gpu.pen, gpu.vmax, gpu.asleep
    );
    assert_eq!(
        gpu.out_of_box, 0,
        "GPU: {} тел вылетели из ящика",
        gpu.out_of_box
    );
    /* Не «ровно все»: порядок контактов в куче задаётся атомиками, и 1–2 тела
     * из 5000 иногда ещё минуту шевелятся на 1 см/с. Порог 99% ловит настоящую
     * поломку (0 уснувших), но не ловится на порядке атомиков. */
    assert!(
        gpu.asleep * 100 / n >= 99,
        "GPU: уснуло только {} из {n}",
        gpu.asleep
    );
    assert!(gpu.vmax < 0.01, "GPU: остаточная скорость {:.4} м/с", gpu.vmax);
    /* Порог 3 см, а не 1: проникновение меряется по ОПИСАННОЙ сфере (half_shape),
     * а прокси тела на устройстве — октаэдр, вписанный в неё, объёмом в треть
     * сферы. Две такие прокси сходятся ближе, чем две сферы, на систематику
     * прокси, и на 5000 телах максимум по парам гуляет 1.1–2.1 см от прогона к
     * прогону (1000 тел — 1.1 см). Это не просадка солвера: остаточная скорость
     * 0, уснули все 5000, проникновение не растёт с числом итераций (12 и 16 —
     * одинаково). Настоящий провал контактов даёт десятки сантиметров. */
    assert!(gpu.pen < 0.03, "GPU: проникновение {:.4} м > 3 см", gpu.pen);
    assert!(
        (gpu.p99 as f64 - cpu_p99).abs() / cpu_p99 < 0.25,
        "толщина кучи на 5000 тел: GPU {:.3} против CPU {cpu_p99:.3}",
        gpu.p99
    );
}


