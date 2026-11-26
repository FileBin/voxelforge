// Летит ли снаряд на GPU-пути и виноват ли проход сна.
use voxelforge_world::live::{Kind, Live};
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    if std::env::var("NO_SLEEP").is_ok() { w.set_sleep(false); }
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 3.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    for f in 0..6 {
        w.step(1.0 / 20.0, 10, 18);
        if let Some(b) = w.bodies.iter().find(|b| b.alive && b.kind == Kind::Bullet) {
            let v = (b.vel[0].powi(2) + b.vel[1].powi(2) + b.vel[2].powi(2)).sqrt();
            println!("кадр {f}: x={:.2} y={:.2} |v|={:.1}", b.pos[0], b.pos[1], v);
        }
    }
}
