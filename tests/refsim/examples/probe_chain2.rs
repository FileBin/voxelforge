// Непрерывность пары prev→cur: prev_pos на такте N обязан совпадать с pos на
// такте N-1. Если нет — кадр интерполирует между несвязанными позами, и любое
// тело «дрожит» при верной траектории.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    for _ in 0..14 {
        w.step(1.0 / 20.0, 10, 18);
    }
    w.gpu.wait_idle();

    let mut prev_snap = w.gpu.read_bodies();
    let mut worst = 0.0f32;
    let mut worst_body = 0usize;
    for t in 0..20 {
        w.step(1.0 / 20.0, 10, 18);
        w.gpu.wait_idle();
        let now = w.gpu.read_bodies();
        for i in 0..now.len().min(12) {
            let cur = now[i].pos_r;
            if cur[1] < -100.0 {
                continue; // припаркованное тело
            }
            let pv = now[i].prev_pos_r;
            let was = prev_snap[i].pos_r;
            let d = ((pv[0] - was[0]).powi(2) + (pv[1] - was[1]).powi(2) + (pv[2] - was[2]).powi(2))
                .sqrt();
            if d > worst {
                worst = d;
                worst_body = i;
            }
            if d > 1e-3 {
                println!("такт {t}: тело {i}: prev=({:.4},{:.4},{:.4}) а было pos=({:.4},{:.4},{:.4}) — расхождение {d:.4}",
                    pv[0], pv[1], pv[2], was[0], was[1], was[2]);
            }
        }
        prev_snap = now;
    }
    println!("максимум расхождения prev↔прошлый pos: {worst:.5} (тело {worst_body})");
}
