// Сверка контактов SDF↔SDF с тет-версией на сценарии «плита + снаряд 26 м/с».
//
//   VF_COLLIDE=gpu  probe_sdf_collide      — тет-версия (sat.comp), эталон
//   VF_COLLIDE=sdf  probe_sdf_collide      — поля (sdf_contact.comp)
//
// Печатает по кадрам число контактов, максимальную глубину проникновения и
// скорости плиты и снаряда — чтобы сравнивать числами, а не на глаз.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn main() {
    let mode = std::env::var("VF_COLLIDE").unwrap_or_else(|_| "gpu".into());
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    /* Раскол выключен: сцена должна остаться той же, иначе сравнивать нечего
     * (осколки меняют и тела, и контакты). */
    w.set_fracture(false);

    /* даём плите встать и уснуть, как в живом мире */
    for _ in 0..6 {
        w.step(0.05, 8, 12);
    }
    w.gpu.wait_idle();

    let plate_top = {
        let b = &w.bodies[0];
        let mut hi = f64::MIN;
        for t in b.rotated_tets() {
            for v in t {
                hi = hi.max((b.pos[1] + v[1] as f64) as f64);
            }
        }
        hi
    };
    let r = 0.14f64;
    /* снаряд целим сверху, в 2.5 м над плитой — удар на 26 м/с */
    let from = [0.0, plate_top + r + 2.5, 0.0];
    w.shoot(from, [0.0, -1.0, 0.0], 26.0);

    println!(
        "режим {mode} · плита сверху {plate_top:.3} м · снаряд из y={:.3} на 26 м/с",
        from[1]
    );
    let mut peak_contacts = 0u64;
    let mut peak_pen = 0f32;
    let mut peak_plate_v = 0f32;
    let mut peak_ball_v = 0f32;
    for f in 0..24 {
        let t0 = std::time::Instant::now();
        w.step(0.05, 8, 12);
        w.gpu.wait_idle();
        let dt_ms = t0.elapsed().as_secs_f64() * 1e3;
        let cs = w.gpu.read_contacts();
        let snap = w.gpu.read_bodies();
        let total: u64 = cs.iter().map(|v| v.len() as u64).sum();
        let pen = cs.iter().flatten().map(|c| c.pen).fold(0.0f32, f32::max);
        let plate_v = snap[0].vel_m;
        let plate_speed = (plate_v[0] * plate_v[0] + plate_v[1] * plate_v[1] + plate_v[2] * plate_v[2])
            .sqrt() as f32;
        /* снаряд — последний живой слот пули */
        let ball = w
            .bullets
            .iter()
            .rev()
            .find(|&&s| s < w.bodies.len() && w.bodies[s].alive)
            .filter(|&&s| s < snap.len())
            .map(|&s| snap[s].vel_m)
            .unwrap_or([0.0; 4]);
        let ball_speed = (ball[0] * ball[0] + ball[1] * ball[1] + ball[2] * ball[2]).sqrt() as f32;
        let slp = |g: &voxelforge_world::sim::GpuBody| if g.misc[1] > 0.5 { "спит" } else { "бдит" };
        let plate_state = slp(&snap[0]);
        let ball_slot = w
            .bullets
            .iter()
            .rev()
            .find(|&&s| s < snap.len())
            .copied()
            .unwrap_or(usize::MAX);
        let ball_state = if ball_slot != usize::MAX { slp(&snap[ball_slot]) } else { "—" };
        peak_contacts = peak_contacts.max(total);
        peak_pen = peak_pen.max(pen);
        peak_plate_v = peak_plate_v.max(plate_speed);
        peak_ball_v = peak_ball_v.max(ball_speed);
        println!(
            "кадр {f:2}: контактов {total:4} ({plate_state}/{ball_state}) · глубина max {pen:.4} · плита {plate_speed:6.3} м/с · снаряд {ball_speed:6.2} м/с · такт {dt_ms:6.1} мс · ccd {} ({:.2} мс)",
            w.last_ccd, w.last_ccd_ms
        );
    }
    println!(
        "ИТОГ {mode}: контактов max {peak_contacts} · глубина max {peak_pen:.4} м · плита max {peak_plate_v:.3} м/с · снаряд max {peak_ball_v:.2} м/с · трещин {}",
        w.cracks
    );
}
