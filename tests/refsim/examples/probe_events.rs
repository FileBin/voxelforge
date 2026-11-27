// Что видит CPU от удара: энергия, события раскола и число трещин по тактам.
// Нужен, чтобы отличать «контакт не нашёлся» от «событие не доехало».
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn main() {
    let sh = find_shaders(None).unwrap();
    for speed in [26.0f64, 120.0, 240.0] {
        let mut w = Live::new(&sh);
        let p = w.bodies[0].pos;
        w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], speed);
        for f in 0..12 {
            w.step(1.0 / 20.0, 10, 18);
            /* такт отправляется асинхронно: чтобы увидеть его результат, ждём */
            w.gpu.wait_idle();
            let imp = w.gpu.read_impact();
            let e = imp.iter().map(|r| r[0]).fold(0.0f32, f32::max);
            let alive = w.bodies.iter().filter(|b| b.alive).count();
            println!(
                "скорость {speed:5.0} такт {f:2}: макс. энергия {:8.0} Дж · vmax {:.0} · тел {alive:2} · трещин {}",
                e, w.last_vmax, w.cracks
            );
        }
    }
}
