// Что происходит при попадании снаряда в плиту: пролёт насквозь или застревание.
use voxelforge_world::live::{Kind, Live};
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    w.set_sleep(false);
    if std::env::var("NO_FRAC").is_ok() { w.set_fracture(false); }
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 2.5, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    for f in 0..14 {
        w.step(1.0 / 20.0, 10, 18);
        if let Some(b) = w.bodies.iter().find(|b| b.alive && b.kind == Kind::Bullet) {
            let v = (b.vel[0].powi(2) + b.vel[1].powi(2) + b.vel[2].powi(2)).sqrt();
            println!("кадр {f:2}: x={:+.3} y={:.3} |v|={:5.1} тел {}", b.pos[0], b.pos[1], v, w.alive_count());
        } else { println!("кадр {f:2}: снаряда нет, тел {}", w.alive_count()); }
    }
}
