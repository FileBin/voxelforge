// Быстрый снаряд: проходит сквозь плиту или ломает её?
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    for speed in [26.0f64, 60.0, 120.0, 240.0] {
        let mut w = Live::new(&sh);
        let p = w.bodies[0].pos;
        w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], speed);
        for _ in 0..40 { w.step(1.0 / 20.0, 10, 18); }
        // в такте позы на CPU не читаются — проба просит их явно
        w.refresh_poses();
        let far = w.bodies.iter().filter(|b| b.alive).map(|b| b.pos[0]).fold(f64::MIN, f64::max);
        println!("скорость {speed:5.0}: тел {:2}, трещин {}, дальше всех x = {far:.2}", w.bodies.iter().filter(|b| b.alive).count(), w.cracks);
        if speed >= 120.0 {
            for b in w.bodies.iter().filter(|b| b.alive) {
                println!("   тело x={:6.2} vx={:7.2} вид={:?}", b.pos[0], b.vel[0], b.kind);
            }
        }
    }
}
