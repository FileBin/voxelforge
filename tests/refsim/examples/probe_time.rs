// Сколько стоит такт: пары на CPU против пар на устройстве.
use std::time::Instant;
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    for (label, gpu_collide, gpu_pairs) in [
        ("пары CPU + коллизия CPU", false, false),
        ("пары CPU + коллизия GPU", true, false),
        ("пары GPU + коллизия GPU", true, true),
    ] {
        let mut w = Live::new(&sh);
        w.gpu_collide = gpu_collide;
        w.gpu_pairs = gpu_pairs;
        let p = w.bodies[0].pos;
        w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
        for _ in 0..10 { w.step(1.0 / 20.0, 10, 18); }
        let n = 60;
        let t = Instant::now();
        for _ in 0..n { w.step(1.0 / 20.0, 10, 18); }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("{label}: {ms:6.2} мс/такт, тел {}", w.bodies.iter().filter(|b| b.alive).count());
    }
}
