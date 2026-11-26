//! GPU-тесты: те же демки, что у CPU-эталона, но посчитанные на видеокарте.
//!
//! Сравниваем не «похоже/не похоже», а те же числа, что проверяет CPU:
//! просадку, проникновение, остаточную скорость, сон и снос кучи к стенке.
//! Побитового совпадения быть не может — GPU считает во float32, CPU в f64,
//! и порядок контактов в списке задаётся целочисленными атомиками. Совпадать
//! должна физика, а не биты.

use std::path::{Path, PathBuf};

use voxelforge_refsim::gpu::{
    find_shaders, scene_drop, scene_pile, scene_stack, scene_voxel_fall, GpuBody, GpuSim, Scene,
};
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

fn run(sc: Scene, frames: u32, iters: u32, shaders: &Path) -> Outcome {
    let n = sc.bodies.len();
    let (lo, hi) = (sc.params.lo, sc.params.hi);
    let mut gpu = GpuSim::new(sc.bodies, shaders, sc.params, sc.voxel.as_ref());
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
        let g = GpuSim::new(sc.bodies, &shaders, sc.params, sc.voxel.as_ref());
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
    assert!(
        gpu.pen < 0.01,
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
    let gpu = run(scene_pile(n), 300, 12, &shaders);

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
    assert_eq!(gpu.asleep, n, "GPU: уснуло {} из {n}", gpu.asleep);
    assert!(gpu.vmax < 0.01, "GPU: остаточная скорость {:.4} м/с", gpu.vmax);
    assert!(gpu.pen < 0.01, "GPU: проникновение {:.4} м > 1 см", gpu.pen);
    assert!(
        (gpu.p99 as f64 - cpu_p99).abs() / cpu_p99 < 0.25,
        "толщина кучи на 5000 тел: GPU {:.3} против CPU {cpu_p99:.3}",
        gpu.p99
    );
}

/// Тело ложится на воксельное поле: центр покоя = центр воксельной сферы
/// + её радиус + радиус тела = 0.6 + 0.5 + 0.1.
#[test]
fn gpu_body_rests_on_voxel_field() {
    let Some((shaders, _)) = setup() else {
        eprintln!("пропуск: нет SPIR-V или Vulkan-устройства");
        return;
    };
    let gpu = run(scene_voxel_fall(0.0), 600, 12, &shaders);
    let center = gpu.top - 0.1;
    eprintln!("воксель: центр {center:.3} pen={:.4} v={:.4}", gpu.pen, gpu.vmax);
    assert!(
        (center as f64 - 1.20).abs() < 0.04,
        "шар покоится на {center:.3}, ожидалось ~1.20"
    );
    assert!(gpu.vmax < 0.01, "шар не успокоился: {:.4} м/с", gpu.vmax);
    assert_eq!(gpu.asleep, 1, "шар не уснул на воксельном теле");
}

/// Быстрое тело не прошивает воксельное поле: за подшаг оно проходит
/// меньше ячейки, а выборка поля интерполируется.
#[test]
fn gpu_fast_body_does_not_tunnel_voxel() {
    let Some((shaders, _)) = setup() else {
        eprintln!("пропуск: нет SPIR-V или Vulkan-устройства");
        return;
    };
    let gpu = run(scene_voxel_fall(30.0), 600, 12, &shaders);
    let center = gpu.top - 0.1;
    eprintln!(
        "воксель 30 м/с: центр {center:.3} pen={:.4} v={:.4} asleep={}",
        gpu.pen, gpu.vmax, gpu.asleep
    );
    assert!(
        center > 1.05,
        "шар провалился сквозь воксельное тело: центр {center:.3}"
    );
    assert!(
        (center as f64 - 1.20).abs() < 0.05,
        "после удара на 30 м/с шар покоится на {center:.3}, ожидалось ~1.20"
    );
    assert!(gpu.asleep == 1, "шар не успокоился после удара");
}
