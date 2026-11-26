// Момент инерции по телам: у мелких осколков он не должен быть почти нулевым.
use voxelforge_world::live::{Kind, Live};
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    w.set_sleep(false); // иначе тела спят, и обратная инерция в зеркале нулевая
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    for _ in 0..40 { w.step(1.0 / 20.0, 10, 18); }
    let mut rows: Vec<(usize, f64, f32)> = Vec::new();
    for i in 0..w.bodies.len() {
        let b = &w.bodies[i];
        if !b.alive || b.kind == Kind::Bullet { continue; }
        let ii = w.mirror[i].ang_i[3];
        let inertia = if ii > 0.0 { 1.0 / ii as f64 } else { f64::INFINITY };
        rows.push((b.tets.len(), b.mass, inertia as f32));
    }
    rows.sort_by_key(|r| r.0);
    println!("тетов | масса, кг | момент инерции, кг·м²");
    for (t, m, i) in rows.iter().take(8) { println!("{t:5} | {m:8.2} | {i:.5}"); }
    if let Some((t, m, i)) = rows.last() { println!("самый крупный: {t} тетов, {m:.0} кг, момент {i:.2}"); }
}
