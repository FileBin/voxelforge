// Меш на устройстве против CPU-мешера: объём, число треугольников и время.
//
//   probe_mesh_gpu            — плита (куб 8³ из тетов) и шар-снаряд
//   VF_LEVEL=6 probe_mesh_gpu — другой уровень плотности
use voxelforge_world::sim::{find_shaders, GpuBody, GpuSim, Params};
use voxelforge_world::tet::TetBody;

fn main() {
    let sh = find_shaders(None).unwrap();
    let level: u32 = std::env::var("VF_LEVEL").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let mut gb = GpuBody::default();
    gb.pos_r = [0.0, 0.6, 0.0, 0.5];
    gb.quat = [1.0, 0.0, 0.0, 0.0];
    gb.misc = [0.0, 0.0, 1.0 / 590.0, 1.0 / 60.0];
    gb.vel_m = [0.0, 0.0, 0.0, 1.0 / 590.0];
    gb.ang_i = [0.0, 0.0, 0.0, 1.0 / 60.0];
    gb.half_shape = [0.56, 0.56, 0.56, 1.0];
    let params = Params {
        lo: [-8.0, 0.0, -8.0],
        hi: [8.0, 10.0, 8.0],
        ..Params::default()
    };
    let mut gpu = GpuSim::new(vec![gb], &sh, params);
    println!("устройство: {}", gpu.name);

    for (name, body) in [
        ("плита", TetBody::cube([0.0, 0.0, 0.0], level as i32, 1)),
        ("шар", TetBody::ball([0.0, 0.0, 0.0], 0.14, 0)),
    ] {
        let f = body.sdf().clone();
        gpu.upload_sdf(&[Some((&f, [0.0; 3]))]);

        let t0 = std::time::Instant::now();
        gpu.build_meshes();
        gpu.wait_idle();
        let ms_gpu = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = std::time::Instant::now();
        let cpu = body.surface();
        let ms_cpu = t1.elapsed().as_secs_f64() * 1e3;

        let idx = gpu.mesh_index_count(0);
        println!("  {name}: GPU {} тр. · CPU {} тр.", idx / 3, cpu.triangles());
        let (verts, _) = gpu.read_mesh_debug(0, (idx as usize).min(98303));
        let vol_gpu = vol_of(&verts, idx as usize);
        let vol_cpu = vol_cpu(&cpu);
        let vol_tets = body.volume() as f32;
        println!(
            "{name:6}: GPU {ms_gpu:6.2} мс · {idx:6} индексов ({:5} тр.) · объём {vol_gpu:.4} · CPU {ms_cpu:6.2} мс · {} тр. · объём {vol_cpu:.4} · теты {vol_tets:.4} ({:+.1}%)",
            idx / 3,
            cpu.triangles(),
            100.0 * (vol_gpu - vol_tets) / vol_tets
        );
    }
}

fn vol_of(v: &[[f32; 6]], n_idx: usize) -> f32 {
    let mut acc = 0.0f64;
    for t in (0..n_idx / 3).map(|k| k * 3) {
        let a = v[t];
        let b = v[t + 1];
        let c = v[t + 2];
        acc += (a[0] as f64 * (b[1] as f64 * c[2] as f64 - b[2] as f64 * c[1] as f64)
            - a[1] as f64 * (b[0] as f64 * c[2] as f64 - b[2] as f64 * c[0] as f64)
            + a[2] as f64 * (b[0] as f64 * c[1] as f64 - b[1] as f64 * c[0] as f64))
            / 6.0;
    }
    acc.abs() as f32
}

fn vol_cpu(m: &voxelforge_world::tet::Mesh) -> f32 {
    let mut acc = 0.0f64;
    for t in m.indices.chunks(3) {
        let a = m.positions[t[0] as usize];
        let b = m.positions[t[1] as usize];
        let c = m.positions[t[2] as usize];
        acc += (a[0] as f64 * (b[1] as f64 * c[2] as f64 - b[2] as f64 * c[1] as f64)
            - a[1] as f64 * (b[0] as f64 * c[2] as f64 - b[2] as f64 * c[0] as f64)
            + a[2] as f64 * (b[0] as f64 * c[1] as f64 - b[1] as f64 * c[0] as f64))
            / 6.0;
    }
    acc.abs() as f32
}
