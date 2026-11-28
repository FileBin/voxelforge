// Туннелирование быстрых тел: пускаем снаряд в плиту на разных скоростях и
// печатаем ЧИСЛА — столкнулся/пролетел, глубину, контакты, скорость после.
//
//   VF_COLLIDE=sdf probe_ccd    — контакты по SDF (sdf_contact.comp)
//   VF_COLLIDE=gpu probe_ccd    — тет-версия (sat.comp), для сравнения
//
// VF_SPEEDS=5,20,60,120,240   список скоростей
// VF_SUBSTEPS=8 VF_ITERS=12   нарезка такта (дефолты живого мира)
// VF_TICKS=12                 сколько тактов гонять (1 такт = 1/20 с)
// VF_DBG=1                    печатать каждый такт полёта
//
// Гравитация выключена: мерим чистый полёт по прямой, а не падение.
// Расстояние до плиты — ОТНОСИТЕЛЬНОЕ (снаряд минус плита): иначе «пролетел»
// путается с тем, что плита сама уехала от удара.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn env_u32(name: &str, def: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn main() {
    let mode = std::env::var("VF_COLLIDE").unwrap_or_else(|_| "sdf".into());
    let sh = find_shaders(None).unwrap();
    let speeds: Vec<f64> = std::env::var("VF_SPEEDS")
        .unwrap_or_else(|_| "5,20,60,120,240".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let substeps = env_u32("VF_SUBSTEPS", 8);
    let iters = env_u32("VF_ITERS", 12);
    let ticks = env_u32("VF_TICKS", 12) as usize;
    let dbg = std::env::var("VF_DBG").is_ok();
    let dt = 1.0 / 20.0;

    println!(
        "режим {mode} · подшагов {substeps} · итераций {iters} · такт {:.0} мс · гравитация ВЫКЛ",
        dt * 1e3
    );
    println!("скорость · шаг/подшаг · исход · контактов max · глубина max · x_отн · vx · Δv плиты · ccd");

    for &speed in &speeds {
        let mut w = Live::new(&sh);
        w.set_fracture(false);
        w.params.gravity = 0.0;
        w.gpu.params.gravity = 0.0;
        /* плите дать встать/уснуть и узнать свою геометрию */
        for _ in 0..6 {
            w.step(dt, substeps, iters);
        }
        w.gpu.wait_idle();

        let (plate_half, field_step) = {
            let b = &w.bodies[0];
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for t in b.rotated_tets() {
                for v in t {
                    for k in 0..3 {
                        lo[k] = lo[k].min(v[k]);
                        hi[k] = hi[k].max(v[k]);
                    }
                }
            }
            (
                [hi[0] - b.pos[0] as f32, hi[1] - b.pos[1] as f32, hi[2] - b.pos[2] as f32],
                b.sdf.as_ref().map(|f| f.step).unwrap_or(0.0),
            )
        };
        let plate_y = w.bodies[0].pos[1];

        let from = [-3.0, plate_y, 0.0];
        w.shoot(from, [1.0, 0.0, 0.0], speed);
        let slot = *w.bullets.back().unwrap();
        w.gpu.wait_idle();

        let per_sub = speed * dt as f64 / substeps as f64;
        let mut max_contacts = 0usize;
        let mut max_pen = 0.0f32;
        let mut hit_ticks = Vec::new();
        let mut max_plate_dv = 0.0f32;
        let plate_v0 = w.gpu.read_bodies()[0].vel_m;
        let mut x_rel = 0.0f32;
        let mut vx_end = 0.0f32;
        let mut ccd_hits = 0u32;
        let mut ccd_ms_max = 0.0f32;
        let mut spin_max = 0.0f32;

        for t in 0..ticks {
            let t0 = std::time::Instant::now();
            w.step(dt, substeps, iters);
            w.gpu.wait_idle();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let snap = w.gpu.read_bodies();
            let cs = if w.bullets.contains(&slot) { w.gpu.read_contacts() } else { Vec::new() };
            let n_c = cs.get(slot).map(|v| v.len()).unwrap_or(0);
            let pen = cs
                .get(slot)
                .map(|v| v.iter().map(|c| c.pen).fold(0.0f32, f32::max))
                .unwrap_or(0.0);
            let b = &snap[slot];
            x_rel = b.pos_r[0] - snap[0].pos_r[0];
            vx_end = b.vel_m[0] - snap[0].vel_m[0];
            spin_max = spin_max.max(
                (b.ang_i[0] * b.ang_i[0] + b.ang_i[1] * b.ang_i[1] + b.ang_i[2] * b.ang_i[2]).sqrt(),
            );
            let plate_dv = {
                let v = snap[0].vel_m;
                ((v[0] - plate_v0[0]).powi(2) + (v[1] - plate_v0[1]).powi(2) + (v[2] - plate_v0[2]).powi(2)).sqrt()
            };
            max_plate_dv = max_plate_dv.max(plate_dv);
            ccd_hits += w.last_ccd;
            ccd_ms_max = ccd_ms_max.max(w.last_ccd_ms);
            if n_c > 0 {
                hit_ticks.push(t);
                max_contacts = max_contacts.max(n_c);
                max_pen = max_pen.max(pen);
            }
            if dbg {
                println!(
                    "  t={t:2} x_отн={x_rel:7.3} vx_отн={vx_end:7.2} контактов {n_c:2} глубина {pen:.4} такт {ms:5.1} мс · vmax {:.1} · ccd {}",
                    w.last_vmax, w.last_ccd
                );
            }
        }

        /* «пролетел» — снаряд оказался ЗА плитой (по её собственной оси). */
        let outcome = if !w.bodies[slot].alive {
            "исчез"
        } else if x_rel > plate_half[0] {
            "ПРОЛЕТЕЛ"
        } else if x_rel > 0.0 {
            "в плите"
        } else if vx_end < -0.5 {
            "отбит"
        } else {
            "столкнулся"
        };
        println!(
            "{speed:6.0} · {per_sub:5.2} м · {outcome:11} · {max_contacts:3} · {max_pen:.4} · x_отн={x_rel:7.3} · vx_отн={vx_end:7.2} · Δv_плиты {max_plate_dv:.3} · ccd {ccd_hits} ({ccd_ms_max:.2} мс) · ω_max {spin_max:.2} · контакты {:?} · поле {field_step:.4} м",
            &hit_ticks[..hit_ticks.len().min(6)]
        );
    }
}
