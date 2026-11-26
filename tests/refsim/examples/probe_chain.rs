// Обновляется ли буфер между подпроходами: два такта подряд, позиция по x.
use voxelforge_world::live::{Kind, Live};
use voxelforge_world::sim::find_shaders;
fn main() {
    let iters: u32 = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(18);
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 3.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    let slot = w.bodies.iter().position(|b| b.alive && b.kind == Kind::Bullet).unwrap();
    let local: Vec<Vec<[[f32; 3]; 4]>> = w.bodies.iter().map(|b| b.local_tets().clone()).collect();
    w.gpu.upload_geometry(&local);
    w.gpu.set_pairs(&[]);
    for k in 0..3 {
        w.gpu.frame_gpu_contacts(0.05, 6, iters);
        let gb = w.gpu.read_bodies();
        println!("такт {k} (iters={iters}): x={:.3} velx={:.2}", gb[slot].pos_r[0], gb[slot].vel_m[0]);
    }
}
