// Пары тетов на GPU против CPU: сравнение наборов.
use std::collections::HashSet;
use voxelforge_world::collide;
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    for _ in 0..20 { w.step(1.0 / 20.0, 10, 18); }

    let local: Vec<Vec<[[f32; 3]; 4]>> = w.bodies.iter().map(|b| b.local_tets().clone()).collect();
    let rotated: Vec<Vec<[[f32; 3]; 4]>> = w.bodies.iter().map(|b| b.rotated_tets()).collect();
    let pos: Vec<[f32; 3]> = w.bodies.iter().map(|b| [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32]).collect();
    let walls = (w.params.lo, w.params.hi);
    let cell = w.params.cell; // как в шейдере (prm1.w)

    w.gpu.upload_geometry(&local);
    let n_gpu = w.gpu.run_tet_pairs();
    let gpu: Vec<u8> = Vec::new();
    let _ = gpu;

    let cpu_pairs = collide::build_pairs_swept(&rotated, &pos, walls, cell, &vec![[0.0f32; 3]; w.bodies.len()]);
    let cpu: HashSet<(u32, u32)> = cpu_pairs.iter().map(|(b1, t1, b2, t2)| ((b1 << 20) | t1, (b2 << 20) | t2)).collect();
    let gpu_buf: Vec<[u32; 2]> = w.gpu.read_pairs();
    let g: HashSet<(u32, u32)> = gpu_buf.iter().take(n_gpu).map(|p| (p[0], p[1])).collect();
    let only_gpu = g.difference(&cpu).count();
    let only_cpu = cpu.difference(&g).count();
    println!("пар: GPU {n_gpu} (уникальных {}), CPU {} → только на GPU {only_gpu}, только на CPU {only_cpu}", g.len(), cpu.len());
}
