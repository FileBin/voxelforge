//! `vf-gpu` — та же демка D3, но считается на видеокарте.
//!
//! Сцена, параметры и метрики совпадают с CPU-эталоном (`vf-ref`), поэтому
//! числа можно сравнивать напрямую: если GPU-солвер расходится с эталоном,
//! это видно по просадке, проникновению и высоте кучи.

use std::path::PathBuf;

use voxelforge_refsim::gpu::{find_shaders, GpuBody, GpuSim, Params};
use voxelforge_refsim::pile_demo;

struct Args {
    bodies: usize,
    frames: u32,
    dt: f32,
    substeps: u32,
    iterations: u32,
    out: Option<PathBuf>,
    json: Option<PathBuf>,
    compare: bool,
    shaders: Option<PathBuf>,
    friction: Option<f32>,
    beta: Option<f32>,
    max_push: Option<f32>,
    ang_damp: Option<f32>,
}

fn parse() -> Args {
    let mut a = Args {
        bodies: 2000,
        frames: 420,
        dt: 1.0 / 60.0,
        substeps: 8,
        iterations: 8,
        out: None,
        json: None,
        compare: false,
        shaders: None,
        friction: None,
        beta: None,
        max_push: None,
        ang_damp: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().expect("нужно значение аргумента");
        match k.as_str() {
            "--bodies" => a.bodies = v().parse().unwrap(),
            "--frames" => a.frames = v().parse().unwrap(),
            "--dt" => a.dt = v().parse().unwrap(),
            "--substeps" => a.substeps = v().parse().unwrap(),
            "--iterations" => a.iterations = v().parse().unwrap(),
            "--out" => a.out = Some(v().into()),
            "--json" => a.json = Some(v().into()),
            "--shaders" => a.shaders = Some(v().into()),
            "--compare" => a.compare = true,
            "--friction" => a.friction = Some(v().parse().unwrap()),
            "--beta" => a.beta = Some(v().parse().unwrap()),
            "--max-push" => a.max_push = Some(v().parse().unwrap()),
            "--ang-damp" => a.ang_damp = Some(v().parse().unwrap()),
            other => panic!("неизвестный аргумент {other}"),
        }
    }
    a
}

fn to_gpu(body: &voxelforge_refsim::Body) -> GpuBody {
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
        quat: [
            body.rot[0] as f32,
            body.rot[1] as f32,
            body.rot[2] as f32,
            body.rot[3] as f32,
        ],
        misc: [0.0, 0.0, im as f32, body.inv_inertia as f32],
    }
}

fn max_penetration(bodies: &[GpuBody]) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..bodies.len() {
        let a = &bodies[i];
        for b in bodies.iter().skip(i + 1) {
            let dx = (a.pos_r[0] - b.pos_r[0]) as f64;
            let dy = (a.pos_r[1] - b.pos_r[1]) as f64;
            let dz = (a.pos_r[2] - b.pos_r[2]) as f64;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let rr = (a.pos_r[3] + b.pos_r[3]) as f64;
            if d < rr {
                worst = worst.max(rr - d);
            }
        }
    }
    worst
}

fn main() {
    let args = parse();
    let sim = pile_demo(args.bodies);

    // Верх ящика берём из сцены: решётка ставится выше, чем плоский ящик.
    let top = sim
        .bodies
        .iter()
        .map(|b| b.pos[1] as f32 + b.radius)
        .fold(0.0f32, f32::max);
    let mut params = Params {
        lo: [-0.05, 0.0, -0.05],
        hi: [4.2, top + 0.5, 4.2],
        cell: 0.16,
        ..Params::default()
    };
    if let Some(f) = args.friction {
        params.friction = f;
    }
    if let Some(b) = args.beta {
        params.beta = b;
    }
    if let Some(m) = args.max_push {
        params.max_push = m;
    }
    if let Some(d) = args.ang_damp {
        params.ang_damp = d;
    }

    let bodies: Vec<GpuBody> = sim.bodies.iter().map(to_gpu).collect();
    let shaders = find_shaders(args.shaders.as_deref())
        .unwrap_or_else(|| panic!("нет SPIR-V: соберите shaders/*.comp через glslc или укажите --shaders"));
    let mut gpu = GpuSim::new(bodies, &shaders, params);
    println!("устройство: {}", gpu.name);
    println!(
        "сетка: {}×{}×{} = {} ячеек, тел {}",
        gpu.dims[0], gpu.dims[1], gpu.dims[2], gpu.ncells, gpu.n
    );

    let t0 = std::time::Instant::now();
    let mut slept_at = -1.0f32;
    let mut frames_out: Vec<Vec<[f32; 4]>> = Vec::new();
    let mut top_start = 0.0f32;

    for f in 0..args.frames {
        gpu.frame(args.dt, args.substeps, args.iterations);
        // С треком читаем каждый кадр: иначе во вьюере будет дёрганое видео.
        if f % 10 == 0 || f + 1 == args.frames || args.out.is_some() {
            let snap = gpu.read_bodies();
            if f == 0 {
                top_start = snap
                    .iter()
                    .map(|b| b.pos_r[1] + b.pos_r[3])
                    .fold(0.0f32, f32::max);
            }
            let vmax = snap
                .iter()
                .map(|b| (b.vel_m[0] * b.vel_m[0] + b.vel_m[1] * b.vel_m[1] + b.vel_m[2] * b.vel_m[2]).sqrt())
                .fold(0.0f32, f32::max);
            if slept_at < 0.0 && vmax < 0.05 {
                slept_at = f as f32 * args.dt;
            }
            if args.out.is_some() {
                frames_out.push(snap.iter().map(|b| b.pos_r).collect());
            }
            if f + 1 == args.frames {
                let top_end = snap
                    .iter()
                    .map(|b| b.pos_r[1] + b.pos_r[3])
                    .fold(0.0f32, f32::max);
                let pen = max_penetration(&snap);
                // Максимальная высота — метрика обманчивая: пара шаров,
                // заклинившихся в углу, делает «кучу» вдвое выше. Берём
                // 99-й процентиль: он описывает саму кучу, а не выбросы.
                let mut hs: Vec<f32> = snap.iter().map(|b| b.pos_r[1] + b.pos_r[3]).collect();
                hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p99 = hs[(hs.len() * 99 / 100).min(hs.len() - 1)];
                let (mut xmin, mut xmax, mut zmin, mut zmax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                for b in &snap {
                    xmin = xmin.min(b.pos_r[0]);
                    xmax = xmax.max(b.pos_r[0]);
                    zmin = zmin.min(b.pos_r[2]);
                    zmax = zmax.max(b.pos_r[2]);
                }
                println!("диапазон x [{xmin:.2}..{xmax:.2}] z [{zmin:.2}..{zmax:.2}] (ящик -0.05..4.20)");
                let asleep = snap.iter().filter(|b| b.misc[1] > 0.5).count();
                println!("спят: {asleep} из {}", gpu.n);
                let counts = gpu.contact_counts();
                let filled = counts.iter().filter(|&&c| c as usize >= 24).count();
                let avg = counts.iter().map(|&c| c as f64).sum::<f64>() / counts.len() as f64;
                println!("контактов на тело: среднее {avg:.1}, упёрлись в CAP: {filled}");
                println!(
                    "summary: scenario=pile bodies={} frames={} slept_at={:.2}s pen={:.4} top={:.2}->{:.2} p99={:.2} vmax={:.3}",
                    gpu.n, args.frames, slept_at, pen, top_start, top_end, p99, vmax
                );
                if args.compare {
                    let mut cpu = pile_demo(args.bodies);
                    for _ in 0..args.frames {
                        cpu.step(args.dt as f64);
                    }
                    let cpu_top = cpu
                        .bodies
                        .iter()
                        .map(|b| b.pos[1] + b.radius as f64)
                        .fold(0.0f64, f64::max);
                    let cpu_awake = cpu.awake;
                    let cpu_pen = cpu.max_penetration();
                    println!(
                        "cpu-ref:  awake={cpu_awake} pen={cpu_pen:.4} top={cpu_top:.2}"
                    );
                }
                if let Some(p) = &args.json {
                    let json = format!(
                        "{{\"scenario\":\"pile-gpu\",\"bodies\":{},\"frames\":{},\"substeps\":{},\"iterations\":{},\"dt\":{:.6},\"top_start\":{:.6},\"top_end\":{:.6},\"slept_at\":{:.4},\"max_penetration\":{:.6}}}",
                        gpu.n,
                        args.frames,
                        args.substeps,
                        args.iterations,
                        args.dt,
                        top_start,
                        snap.iter().map(|b| b.pos_r[1] + b.pos_r[3]).fold(0.0f32, f32::max),
                        slept_at,
                        pen
                    );
                    if let Some(dir) = p.parent() {
                        let _ = std::fs::create_dir_all(dir);
                    }
                    std::fs::write(p, json).unwrap();
                }
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    println!(
        "время: {:.2} с всего, {:.2} мс на кадр",
        elapsed,
        elapsed * 1000.0 / args.frames as f64
    );

    if let Some(p) = &args.out {
        // Тот же кодировщик, что у CPU-эталона: раскладка записи тела
        // (f64 масса, f32 радиус, u8 форма, 3×f32 half = 25 байт) не должна
        // разъезжаться между двумя раннерами, иначе вьюер покажет мусор.
        let frames3: Vec<Vec<[f32; 3]>> = frames_out
            .iter()
            .map(|f| f.iter().map(|q| [q[0], q[1], q[2]]).collect())
            .collect();
        let blob = voxelforge_refsim::vfr::encode(&sim.bodies, &frames3);
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(p, &blob).unwrap();
        println!("записано: {} ({} кадров)", p.display(), frames_out.len());

        // Боковая проекция «до/после» — картинка для глаз, пока вьюер не открыт.
        let svg = p.with_extension("svg");
        let panel = |frame: &Vec<[f32; 4]>, ox: f32, title: &str| {
            let (w, h) = (520.0f32, 360.0f32);
            let mut out = format!(
                r##"<g transform="translate({ox},24)"><text x="0" y="-6" fill="#9aa" font-family="sans-serif" font-size="13">{title}</text><rect x="0" y="0" width="{w}" height="{h}" fill="#0d1117" stroke="#2a3340"/>"##
            );
            let mut order: Vec<&[f32; 4]> = frame.iter().collect();
            order.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
            for q in order {
                let sx = q[0] / 4.25 * w;
                let sy = h - q[1] / 2.6 * h;
                let r = (q[3] / 4.25 * w).max(1.2);
                let depth = (q[2] / 4.25).clamp(0.0, 1.0);
                let c = (90.0 + 140.0 * depth) as u8;
                out.push_str(&format!(
                    r##"<circle cx="{sx:.1}" cy="{sy:.1}" r="{r:.1}" fill="rgb({},{c},200)" opacity="0.75"/>"##,
                    255 - c / 2
                ));
            }
            out.push_str("</g>");
            out
        };
        let first = &frames_out[0];
        let last = &frames_out[frames_out.len() - 1];
        let doc = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="410" viewBox="0 0 1080 410"><rect width="1080" height="410" fill="#0d1117"/>{}{}</svg>"##,
            panel(first, 10.0, "t = 0"),
            panel(last, 550.0, &format!("t = {:.1} с (GPU)", frames_out.len() as f32 * args.dt))
        );
        std::fs::write(&svg, doc).unwrap();
        println!("картинка: {}", svg.display());
    }
}
