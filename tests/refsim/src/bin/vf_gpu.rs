//! `vf-gpu` — те же демки D1/D2/D3, но считаются на видеокарте.
//!
//! Сцена, параметры и метрики совпадают с CPU-эталоном, поэтому числа можно
//! сравнивать напрямую: если GPU-солвер расходится с эталоном, это видно по
//! просадке, проникновению, сну и высоте кучи.

use std::path::PathBuf;

use voxelforge_refsim::gpu::{
    find_shaders, scene_drop, scene_pile, scene_stack, GpuBody, GpuSim, Params, Scene,
};
use voxelforge_refsim::{pile_demo, ContactSim};

struct Args {
    scene: String,
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
        scene: "pile".into(),
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
            "--scene" => a.scene = v(),
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

/// Глубина проникновения по паре тел — та же логика, что в шейдере:
/// сфера-сфера по расстоянию, коробки по наименьшему перекрытию осей.
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

fn max_penetration(bodies: &[GpuBody]) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..bodies.len() {
        for b in bodies.iter().skip(i + 1) {
            worst = worst.max(pair_pen(&bodies[i], b));
        }
    }
    worst
}

fn build(scene: &str, n: usize) -> Scene {
    match scene {
        "pile" => scene_pile(n),
        "drop" => scene_drop(),
        "stack" => scene_stack(n),
        other => panic!("неизвестная сцена {other}: pile | drop | stack"),
    }
}

fn main() {
    let args = parse();
    let mut sc = build(&args.scene, args.bodies);
    let mut params: Params = sc.params;
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

    let shaders = find_shaders(args.shaders.as_deref()).unwrap_or_else(|| {
        panic!("нет SPIR-V: соберите shaders/*.comp через glslc или укажите --shaders")
    });
    let n = sc.bodies.len();
    let mut gpu = GpuSim::new(std::mem::take(&mut sc.bodies), &shaders, params);
    println!("устройство: {}", gpu.name);
    println!(
        "сцена: {} · сетка {}×{}×{} = {} ячеек · тел {} · ячейка {:.3} м",
        sc.name, gpu.dims[0], gpu.dims[1], gpu.dims[2], gpu.ncells, n, params.cell
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
            let top_of = |s: &[GpuBody]| {
                s.iter()
                    .map(|b| b.pos_r[1] + b.half_shape[1])
                    .fold(0.0f32, f32::max)
            };
            if f == 0 {
                top_start = top_of(&snap);
            }
            let vmax = snap
                .iter()
                .map(|b| {
                    (b.vel_m[0] * b.vel_m[0] + b.vel_m[1] * b.vel_m[1] + b.vel_m[2] * b.vel_m[2])
                        .sqrt()
                })
                .fold(0.0f32, f32::max);
            let asleep = snap.iter().filter(|b| b.misc[1] > 0.5).count();
            if slept_at < 0.0 && asleep == n {
                slept_at = f as f32 * args.dt;
            }
            let trace = std::env::var("VF_TRACE").is_ok();
            if trace && n <= 20 {
                let ys: Vec<String> = snap
                    .iter()
                    .map(|b| {
                        format!(
                            "{:.3}/v{:.2}/s{:.0}/h{:.2}",
                            b.pos_r[1], b.vel_m[1], b.misc[1], b.half_shape[1]
                        )
                    })
                    .collect();
                println!("кадр {f}: y/v {}", ys.join(" "));
            }
            if args.out.is_some() {
                frames_out.push(snap.iter().map(|b| b.pos_r).collect());
            }
            if f + 1 == args.frames {
                let top_end = top_of(&snap);
                let pen = max_penetration(&snap);
                let mut hs: Vec<f32> = snap.iter().map(|b| b.pos_r[1] + b.half_shape[1]).collect();
                hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p99 = hs[(hs.len() * 99 / 100).min(hs.len() - 1)];
                println!("спят: {asleep} из {n}");
                println!(
                    "summary: scenario={} bodies={} frames={} slept_at={:.2}s pen={:.4} top={:.2}->{:.2} p99={:.2} vmax={:.3}",
                    sc.name, n, args.frames, slept_at, pen, top_start, top_end, p99, vmax
                );
                if args.compare {
                    match sc.name {
                        "pile" => {
                            let mut cpu = pile_demo(args.bodies);
                            for _ in 0..args.frames {
                                cpu.step(args.dt as f64);
                            }
                            let top = cpu
                                .bodies
                                .iter()
                                .map(|b| b.pos[1] + b.half[1])
                                .fold(0.0f64, f64::max);
                            let awake = cpu.awake;
                            let pen = cpu.max_penetration();
                            println!("cpu-ref:  awake={awake} pen={pen:.4} top={top:.2}");
                        }
                        _ => {
                            let mut cpu = ContactSim::new(sc.cpu.clone());
                            for _ in 0..args.frames {
                                cpu.step(args.dt as f64);
                            }
                            let top = cpu.top();
                            let pen = cpu.max_penetration();
                            let speed = cpu.max_speed();
                            println!(
                                "cpu-ref:  top={top:.3} pen={pen:.4} speed={speed:.4}"
                            );
                        }
                    }
                }
                if let Some(p) = &args.json {
                    let json = format!(
                        "{{\"scenario\":\"{}-gpu\",\"bodies\":{},\"frames\":{},\"substeps\":{},\"iterations\":{},\"dt\":{:.6},\"top_start\":{:.6},\"top_end\":{:.6},\"p99\":{:.6},\"slept_at\":{:.4},\"asleep\":{},\"max_penetration\":{:.6},\"max_speed\":{:.6}}}",
                        sc.name,
                        n,
                        args.frames,
                        args.substeps,
                        args.iterations,
                        args.dt,
                        top_start,
                        top_end,
                        p99,
                        slept_at,
                        asleep,
                        pen,
                        vmax
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
        let blob = voxelforge_refsim::vfr::encode(&sc.cpu, &frames3);
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(p, &blob).unwrap();
        println!("записано: {} ({} кадров)", p.display(), frames_out.len());
    }
}
