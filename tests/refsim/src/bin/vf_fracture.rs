//! `vf-fracture` — воксельное тело раскалывается на осколки и разлетается.
//!
//! Считается на видеокарте тем же солвером, что и коробки; на выходе `.vfx`:
//! осколки со своими вокселями и позициями по кадрам, чтобы вьюер показал
//! именно раскол воксельного тела, а не толкотню кубов.

use std::path::PathBuf;

use voxelforge_refsim::fracture::{encode_vfx, VoxelFracture};
use voxelforge_refsim::gpu::{find_shaders, GpuBody, GpuSim, Params};
use voxelforge_refsim::Body;

struct Args {
    dim: u32,
    seeds: usize,
    frames: u32,
    substeps: u32,
    iterations: u32,
    out: Option<PathBuf>,
    shaders: Option<PathBuf>,
}

fn parse() -> Args {
    let mut a = Args {
        dim: 16,
        seeds: 24,
        frames: 420,
        substeps: 8,
        iterations: 12,
        out: None,
        shaders: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().expect("нужно значение аргумента");
        match k.as_str() {
            "--dim" => a.dim = v().parse().unwrap(),
            "--seeds" => a.seeds = v().parse().unwrap(),
            "--frames" => a.frames = v().parse().unwrap(),
            "--substeps" => a.substeps = v().parse().unwrap(),
            "--iterations" => a.iterations = v().parse().unwrap(),
            "--out" => a.out = Some(v().into()),
            "--shaders" => a.shaders = Some(v().into()),
            other => panic!("неизвестный аргумент {other}"),
        }
    }
    a
}

fn main() {
    let args = parse();
    let cell = 1.0 / args.dim as f32;
    let fr = VoxelFracture::cube(args.dim, cell, args.seeds, [0.0, 0.6, 0.0], 100.0);
    let mass: f64 = fr.shards.iter().map(|s| s.mass).sum();
    println!(
        "раскол: {} осколков, {} вокселей, масса {:.1} кг, ячейка {:.4} м",
        fr.shards.len(),
        fr.shards.iter().map(|s| s.cells.len()).sum::<usize>(),
        mass,
        cell
    );

    // Осколки стартуют спящими: тело целое, пока в него не попали.
    let mut bodies: Vec<Body> = fr.bodies();
    let mut gpu_bodies: Vec<GpuBody> = bodies.iter().map(GpuBody::from_cpu).collect();
    for b in gpu_bodies.iter_mut() {
        b.misc[1] = 1.0;
        b.vel_m[3] = 0.0;
        b.ang_i[3] = 0.0;
    }
    // Снаряд летит сбоку.
    let ball = Body::new([-2.6, 0.6, 0.0], [22.0, 0.0, 0.0], 12.0, 0.25);
    gpu_bodies.push(GpuBody::from_cpu(&ball));
    bodies.push(ball);

    let max_half = gpu_bodies
        .iter()
        .map(|b| b.half_shape[0].max(b.half_shape[1]).max(b.half_shape[2]))
        .fold(0.0f32, f32::max);
    let params = Params {
        lo: [-4.0, 0.0, -4.0],
        hi: [4.0, 6.0, 4.0],
        cell: (2.0 * max_half * 1.05).max(0.17),
        friction: 0.45,
        max_push: 0.8,
        ..Params::default()
    };

    let shaders = find_shaders(args.shaders.as_deref())
        .expect("нет SPIR-V: соберите shaders/*.comp через glslc");
    let mut gpu = GpuSim::new(gpu_bodies, &shaders, params, None);
    println!("устройство: {}", gpu.name);

    let n_shards = fr.shards.len();
    let mut frames: Vec<Vec<[f32; 3]>> = Vec::with_capacity(args.frames as usize);
    for f in 0..args.frames {
        gpu.frame(1.0 / 60.0, args.substeps, args.iterations);
        let snap = gpu.read_bodies();
        frames.push(
            snap.iter()
                .take(n_shards)
                .map(|b| [b.pos_r[0], b.pos_r[1], b.pos_r[2]])
                .collect(),
        );
        if args.out.is_some() && (f % 60 == 0) {
            let moved = snap
                .iter()
                .take(n_shards)
                .filter(|b| b.misc[1] > 0.5)
                .count();
            println!("кадр {f}: спят {moved} из {n_shards}");
        }
    }

    if let Some(p) = &args.out {
        let blob = encode_vfx(&fr, &frames);
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(p, &blob).unwrap();
        println!(
            "записано: {} ({} кадров, {:.1} КБ)",
            p.display(),
            frames.len(),
            blob.len() as f64 / 1024.0
        );
        println!("смотреть: nix run .#viewer, перетащить файл в окно");
    }
}
