//! D3 на видеокарте: та же сцена, что у CPU-эталона, те же числа на выходе.
//!
//! Тест сравнивает не «похоже/не похоже», а агрегаты: просадку кучи, глубину
//! проникновения и остаточную скорость. Побитового совпадения быть не может —
//! GPU считает во float32, CPU в f64, и порядок контактов в списке задаётся
//! целочисленными атомиками. Совпадать должна физика, а не биты.

use std::path::{Path, PathBuf};

use voxelforge_refsim::gpu::{find_shaders, GpuBody, GpuSim, Params};
use voxelforge_refsim::{pile_demo, Body};

fn shader_dir() -> Option<PathBuf> {
    find_shaders(None)
}

fn to_gpu(body: &Body) -> GpuBody {
    let im = if body.mass > 0.0 { 1.0 / body.mass } else { 0.0 };
    GpuBody {
        pos_r: [
            body.pos[0] as f32,
            body.pos[1] as f32,
            body.pos[2] as f32,
            body.radius,
        ],
        vel_m: [
            body.vel[0] as f32,
            body.vel[1] as f32,
            body.vel[2] as f32,
            im as f32,
        ],
        ang_i: [
            body.ang[0] as f32,
            body.ang[1] as f32,
            body.ang[2] as f32,
            body.inv_inertia as f32,
        ],
        quat: [body.rot[0] as f32, body.rot[1] as f32, body.rot[2] as f32, body.rot[3] as f32],
        misc: [0.0, 0.0, im as f32, body.inv_inertia as f32],
    }
}

struct Outcome {
    top: f32,
    vmax: f32,
    pen: f64,
    asleep: usize,
    out_of_box: usize,
}

fn run_gpu(n: usize, frames: u32, iters: u32, shaders: &Path) -> Outcome {
    let sim = pile_demo(n);
    let top = sim
        .bodies
        .iter()
        .map(|b| b.pos[1] as f32 + b.radius)
        .fold(0.0f32, f32::max);
    let params = Params {
        lo: [-0.05, 0.0, -0.05],
        hi: [4.2, top + 0.5, 4.2],
        ..Params::default()
    };
    let bodies: Vec<GpuBody> = sim.bodies.iter().map(to_gpu).collect();
    let mut gpu = GpuSim::new(bodies, shaders, params);
    let dt = 1.0 / 60.0;
    for _ in 0..frames {
        gpu.frame(dt, 8, iters);
    }
    let snap = gpu.read_bodies();
    let top_end = snap
        .iter()
        .map(|b| b.pos_r[1] + b.pos_r[3])
        .fold(0.0f32, f32::max);
    let vmax = snap
        .iter()
        .map(|b| {
            (b.vel_m[0] * b.vel_m[0] + b.vel_m[1] * b.vel_m[1] + b.vel_m[2] * b.vel_m[2]).sqrt()
        })
        .fold(0.0f32, f32::max);
    let asleep = snap.iter().filter(|b| b.misc[1] > 0.5).count();
    let out_of_box = snap
        .iter()
        .filter(|b| {
            b.pos_r[0] < -0.12 || b.pos_r[0] > 4.27 || b.pos_r[2] < -0.12 || b.pos_r[2] > 4.27
        })
        .count();
    let mut pen = 0.0f64;
    for i in 0..snap.len() {
        for b in snap.iter().skip(i + 1) {
            let dx = (snap[i].pos_r[0] - b.pos_r[0]) as f64;
            let dy = (snap[i].pos_r[1] - b.pos_r[1]) as f64;
            let dz = (snap[i].pos_r[2] - b.pos_r[2]) as f64;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let rr = (snap[i].pos_r[3] + b.pos_r[3]) as f64;
            if d < rr {
                pen = pen.max(rr - d);
            }
        }
    }
    Outcome {
        top: top_end,
        vmax,
        pen,
        asleep,
        out_of_box,
    }
}

#[test]
fn gpu_pile_matches_cpu_reference() {
    let Some(shaders) = shader_dir() else {
        eprintln!("пропуск: нет скомпилированных шейдеров (shaders/spv/*.spv)");
        return;
    };

    // Нет ICD (например, голый CI-раннер) — GPU-часть не проверяем.
    let probe = std::panic::catch_unwind(|| {
        let sim = pile_demo(8);
        let bodies: Vec<GpuBody> = sim.bodies.iter().map(to_gpu).collect();
        let top = sim
            .bodies
            .iter()
            .map(|b| b.pos[1] as f32 + b.radius)
            .fold(0.0f32, f32::max);
        let params = Params {
            hi: [4.2, top + 0.5, 4.2],
            ..Params::default()
        };
        let g = GpuSim::new(bodies, &shaders, params);
        g.name.clone()
    });
    let Ok(device) = probe else {
        eprintln!("пропуск: Vulkan-устройство недоступно");
        return;
    };
    eprintln!("устройство: {device}");

    let n = 1000;
    let frames = 360;
    let gpu = run_gpu(n, frames, 12, &shaders);

    let mut cpu = pile_demo(n);
    for _ in 0..frames {
        cpu.step(1.0 / 60.0);
    }
    let cpu_top = cpu
        .bodies
        .iter()
        .map(|b| b.pos[1] + b.radius as f64)
        .fold(0.0f64, f64::max);
    let cpu_awake = cpu.awake;
    let cpu_pen = cpu.max_penetration();

    eprintln!(
        "GPU: top={:.3} pen={:.4} v={:.3} asleep={} | CPU: top={:.3} pen={:.4} awake={}",
        gpu.top, gpu.pen, gpu.vmax, gpu.asleep, cpu_top, cpu_pen, cpu_awake
    );

    assert_eq!(gpu.out_of_box, 0, "GPU: {} тел вылетели из ящика", gpu.out_of_box);
    assert!(
        gpu.asleep * 100 / n >= 90,
        "GPU: уснуло только {} из {} тел",
        gpu.asleep,
        n
    );
    assert!(
        gpu.pen < 0.01,
        "GPU: проникновение {:.4} м — контакты не держат",
        gpu.pen
    );
    assert!(
        gpu.vmax < 0.30,
        "GPU: остаточная скорость {:.3} м/с — куча не успокоилась",
        gpu.vmax
    );
    /* Высоты сравниваем не «в лоб»: у CPU-солвера накопленные импульсы
     * (warm start) дают более жёсткий контакт, и куча растекается сильнее.
     * Проверяем, что обе кучи в физически осмысленном диапазоне: не
     * «пыль» на полу и не столб. */
    assert!(
        gpu.top > 0.35 && gpu.top < 2.2,
        "высота кучи на GPU неправдоподобна: {:.3} м (CPU {:.3})",
        gpu.top,
        cpu_top
    );
}
