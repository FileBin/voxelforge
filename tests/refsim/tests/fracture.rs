//! Раскол воксельного тела: разбиение Вороного и разлёт осколков.

use std::collections::HashSet;

use voxelforge_refsim::fracture::VoxelFracture;
use voxelforge_refsim::gpu::find_shaders;

/// Каждая ячейка тела попадает ровно в один осколок — ни дыр, ни дублей.
#[test]
fn partition_covers_every_cell_once() {
    let dim = 12u32;
    let fr = VoxelFracture::cube(dim, 1.0 / dim as f32, 16, [0.0, 0.5, 0.0], 1.0);
    let mut seen: HashSet<[i32; 3]> = HashSet::new();
    let mut total = 0;
    for s in &fr.shards {
        for c in &s.cells {
            assert!(seen.insert(*c), "ячейка {c:?} попала в два осколка");
            total += 1;
        }
    }
    assert_eq!(total as u32, dim * dim * dim, "покрыты не все ячейки");
    assert_eq!(seen.len() as u32, dim * dim * dim);
}

/// Осколки обязаны быть неровными: у настоящего раскола форма не совпадает
/// со своим AABB. Если почти все осколки — полные коробки, значит это не
/// Вороной, а нарезка сетки, и «раскола» нет.
#[test]
fn shards_are_irregular() {
    let dim = 16u32;
    let cell = 1.0 / dim as f32;
    let fr = VoxelFracture::cube(dim, cell, 24, [0.0, 0.5, 0.0], 1.0);
    let mut irregular = 0;
    for s in &fr.shards {
        let nx = (2.0 * s.half[0] / cell).round().max(1.0) as usize;
        let ny = (2.0 * s.half[1] / cell).round().max(1.0) as usize;
        let nz = (2.0 * s.half[2] / cell).round().max(1.0) as usize;
        if s.cells.len() < nx * ny * nz {
            irregular += 1;
        }
    }
    eprintln!(
        "осколки: {} всего, неровных {irregular}, размеры ячеек {}",
        fr.shards.len(),
        fr.shards.iter().map(|s| s.cells.len()).min().unwrap_or(0)
    );
    assert!(fr.shards.len() >= 12, "осколков слишком мало");
    assert!(
        irregular * 4 >= fr.shards.len() * 3,
        "неровных осколков только {irregular} из {} — это нарезка, а не раскол",
        fr.shards.len()
    );
}

/// На GPU: удар разносит осколки, масса сохраняется, всё успокаивается.
#[test]
fn voxel_body_shatters_and_settles() {
    let Some(shaders) = find_shaders(None) else {
        eprintln!("пропуск: нет SPIR-V");
        return;
    };
    let probe = std::panic::catch_unwind(|| {
        let s = voxelforge_refsim::gpu::GpuSim::new(
            vec![],
            &shaders,
            voxelforge_refsim::gpu::Params::default(),
            None,
        );
        s.name.clone()
    });
    if probe.is_err() {
        eprintln!("пропуск: Vulkan-устройство недоступно");
        return;
    }

    use voxelforge_refsim::gpu::{GpuBody, GpuSim, Params};
    use voxelforge_refsim::Body;

    let dim = 12u32;
    let cell = 1.0 / dim as f32;
    let fr = VoxelFracture::cube(dim, cell, 16, [0.0, 0.6, 0.0], 100.0);
    let n = fr.shards.len();
    let mass: f64 = fr.shards.iter().map(|s| s.mass).sum();

    let mut cpu_bodies = fr.bodies();
    let mut bodies: Vec<GpuBody> = cpu_bodies.iter().map(GpuBody::from_cpu).collect();
    for b in bodies.iter_mut() {
        b.misc[1] = 1.0;
        b.vel_m[3] = 0.0;
        b.ang_i[3] = 0.0;
    }
    let ball = Body::new([-2.6, 0.6, 0.0], [22.0, 0.0, 0.0], 12.0, 0.25);
    bodies.push(GpuBody::from_cpu(&ball));
    cpu_bodies.push(ball);

    let max_half = bodies
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

    let spread = |b: &[GpuBody]| -> f32 {
        let mut w = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = ((b[i].pos_r[0] - b[j].pos_r[0]).powi(2)
                    + (b[i].pos_r[1] - b[j].pos_r[1]).powi(2)
                    + (b[i].pos_r[2] - b[j].pos_r[2]).powi(2))
                .sqrt();
                w = w.max(d);
            }
        }
        w
    };

    let mut gpu = GpuSim::new(bodies, &shaders, params, None);
    let first = gpu.read_bodies();
    let s0 = spread(&first);
    for _ in 0..300 {
        gpu.frame(1.0 / 60.0, 8, 12);
    }
    let snap = gpu.read_bodies();
    let s1 = spread(&snap);
    let asleep = snap.iter().take(n).filter(|b| b.misc[1] > 0.5).count();
    let below = snap.iter().take(n).filter(|b| b.pos_r[1] < -0.05).count();

    eprintln!(
        "раскол: {n} осколков, разброс {s0:.2} -> {s1:.2} м, спят {asleep}, под полом {below}, масса {mass:.0}"
    );
    assert!(s1 > s0 * 1.6, "осколки не разлетелись: {s0:.2} -> {s1:.2}");
    assert_eq!(below, 0, "осколки провалились под пол");
    assert!(asleep * 2 >= n, "успокоилось только {asleep} из {n}");
    assert!((mass - 100.0).abs() < 1.0, "масса тела {mass} вместо 100");
}
