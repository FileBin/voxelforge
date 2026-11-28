// Цена раскола: запекание поля осколков на CPU против устройства.
//
//   probe_bake_cost                  — 8/64/256/1000 затравок
//   VF_SEEDS=1000 probe_bake_cost    — только одно число
//
// CPU-путь — `TetBody::shatter_fields` (так было до переноса): поле каждого
// осколка считается на хосте. GPU-путь — `shatter_bakes` (только параметры) плюс
// два диспатча `shard_field.comp`. Слотов SDF всего 64, поэтому для 1000
// осколков слоты переиспользуются по кругу: время от этого не зависит, данные —
// да (нам нужны только числа времени).
use voxelforge_world::sim::{find_shaders, GpuBody, GpuSim, Params};
use voxelforge_world::tet::TetBody;

fn main() {
    let sh = find_shaders(None).unwrap();
    let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
    let parent = plate.sdf().clone();
    let params = Params {
        lo: [-8.0, 0.0, -8.0],
        hi: [8.0, 10.0, 8.0],
        ..Params::default()
    };
    let mut gb = GpuBody::default();
    gb.pos_r = [0.0, 0.6, 0.0, 0.5];
    gb.quat = [1.0, 0.0, 0.0, 0.0];
    gb.misc = [0.0, 0.0, 1.0 / 590.0, 1.0 / 60.0];
    gb.vel_m = [0.0, 0.0, 0.0, 1.0 / 590.0];
    gb.ang_i = [0.0, 0.0, 0.0, 1.0 / 60.0];
    gb.half_shape = [0.56, 0.56, 0.56, 1.0];
    let mut gpu = GpuSim::new(vec![gb], &sh, params);
    println!("устройство: {}", gpu.name);
    gpu.upload_sdf(&[Some((&parent, [0.0; 3]))]);

    let seeds: Vec<usize> = std::env::var("VF_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .map(|n| vec![n])
        .unwrap_or_else(|| vec![8, 64, 256, 1000]);
    println!("затравок · CPU поле осколков, мс · GPU поле осколков, мс · узлов всего");
    for &n in &seeds {
        let pts = plate.shatter_pts(n, 0x51ED_2A17, [0.0, 0.0, 0.0]);

        let t0 = std::time::Instant::now();
        let cpu: Vec<TetBody> = plate.shatter_fields(&pts);
        let ms_cpu = t0.elapsed().as_secs_f64() * 1e3;
        let cpu_nodes: usize = cpu.iter().map(|b| b.sdf().val.len()).sum();

        /* GPU: только параметры + два диспатча (скан бокса и запекание) */
        let t1 = std::time::Instant::now();
        let bakes = plate.shatter_bakes(&pts);
        let jobs: Vec<(usize, usize, &voxelforge_world::tet::ShardBake)> = bakes
            .iter()
            .enumerate()
            .map(|(k, b)| (0usize, 1 + (k % 63), b))
            .collect();
        let _ = gpu.bake_shard_fields(&jobs, false);
        gpu.wait_idle();
        let ms_gpu = t1.elapsed().as_secs_f64() * 1e3;
        let gpu_nodes: usize = bakes.iter().map(|b| b.dims[0] * b.dims[1] * b.dims[2]).sum();

        println!(
            "{n:7} · {ms_cpu:9.1} · {ms_gpu:9.1} · CPU {cpu_nodes} / GPU {gpu_nodes} узлов · осколков {}",
            cpu.len()
        );
    }
}
