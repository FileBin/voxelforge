// Два кубика летят навстречу друг другу: после контакта они расходятся или
// слипаются? Печатаем зазор и относительную скорость вдоль линии сближения.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    w.set_fracture(false);
    w.set_sleep(false);
    // мимо плиты: стреляем горизонтально выше неё
    w.shoot([-3.0, 1.2, 0.0], [1.0, 0.0, 0.0], 20.0);
    w.shoot([3.0, 1.2, 0.0], [-1.0, 0.0, 0.0], 20.0);
    w.refresh_poses();

    let bullets: Vec<usize> = (0..w.bodies.len())
        .filter(|&i| w.bodies[i].alive && w.bodies[i].kind == voxelforge_world::live::Kind::Bullet)
        .collect();
    if bullets.len() < 2 {
        println!("не нашлось двух снарядов: {:?}", bullets);
        return;
    }
    let (a, b) = (bullets[0], bullets[1]);
    println!("снаряды: {a} и {b}");

    let mut prev_gap = f64::MAX;
    for k in 0..40 {
        w.step(1.0 / 20.0, 10, 18);
        w.refresh_poses();
        let (pa, pb) = (w.bodies[a].pos, w.bodies[b].pos);
        let gap = ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt();
        // положительное — расходятся, отрицательное — сближаются
        let closing = (pb[0] - pa[0]).signum() * (w.bodies[b].vel[0] - w.bodies[a].vel[0]);
        let grew = gap > prev_gap;
        println!(
            "такт {k:2}: зазор {gap:6.3} {:>8} · vx a={:7.2} b={:7.2} · скорость расхождения {closing:7.2}",
            if grew { "растёт" } else { "падает" },
            w.bodies[a].vel[0],
            w.bodies[b].vel[0]
        );
        prev_gap = gap;
    }
}
