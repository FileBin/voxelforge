// Что видит impact.comp: энергия удара на тело.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 120.0);
    for f in 0..12 {
        w.step(1.0 / 20.0, 10, 18);
        let imp = w.gpu.read_impact();
        let mx = imp.iter().take(w.bodies.len()).map(|r| r[0]).fold(0.0f32, f32::max);
        let bx = w.bodies.iter().filter(|b| b.alive).map(|b| b.pos[0]).fold(f64::MIN, f64::max);
        let minx = w.bodies.iter().filter(|b| b.alive).map(|b| b.pos[0]).fold(f64::MAX, f64::min);
        println!("кадр {f}: пар {:5}, энергия max {mx:9.1}, x в [{minx:.2}, {bx:.2}]", w.gpu.n_pairs);
    }
    println!("трещин {}", w.cracks);
}
