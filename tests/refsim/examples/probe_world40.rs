// «40 кадров живого мира» без окна: тот же такт, что в окне (Live::step),
// с включённым расколом и выстрелами. Печатает, развалился ли мир.
//
//   probe_world40            — путь по умолчанию (SDF-поля)
//   VF_COLLIDE=gpu probe_world40 — тет-путь
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn main() {
    let mode = std::env::var("VF_COLLIDE").unwrap_or_else(|_| "по умолчанию (sdf)".into());
    let frames: usize = std::env::var("VF_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    w.set_fracture(true);
    w.set_sleep(true);
    let mut nan_frames = 0usize;
    let mut max_speed = 0.0f64;
    for f in 0..frames {
        if f == 10 || f == 25 {
            w.shoot([0.0, 3.8, 0.0], [0.0, -1.0, 0.0], 26.0);
        }
        w.step(0.05, 8, 12);
        w.gpu.wait_idle();
        let snap = w.gpu.read_bodies();
        let bad = snap.iter().filter(|b| !b.pos_r[0].is_finite()).count();
        if bad > 0 {
            nan_frames += 1;
        }
        let sp = w.max_speed();
        if sp.is_finite() {
            max_speed = max_speed.max(sp);
        }
    }
    let snap = w.gpu.read_bodies();
    let alive = w.alive_count();
    let finite = snap.iter().filter(|b| b.pos_r.iter().all(|v| v.is_finite())).count();
    let lowest = snap
        .iter()
        .filter(|b| b.misc[1] <= 0.5)
        .map(|b| b.pos_r[1])
        .fold(f64::MAX as f32, f32::min);
    println!(
        "ИТОГ world{frames} [{mode}]: тел живых {alive} · слот-состояний с конечной позой {finite} · трещин {} · осколков {} · кадров с NaN {nan_frames} · |v| max {max_speed:.1} м/с · самая низкая y {lowest:.3}",
        w.cracks, w.shards_made
    );
}
