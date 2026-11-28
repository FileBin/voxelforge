// CCD по SDF на устройстве: сверка с CPU-эталоном и таблица туннелирования.
//
//   probe_ccd_gpu                 — скорости 5/20/60/120/240, печатает строку на скорость
//   VF_SPEEDS=240 probe_ccd_gpu   — только одна скорость
//   VF_DBG=1 probe_ccd_gpu        — печатать каждый такт: позу, скорость, события
//   VF_CCD=0 probe_ccd_gpu        — тот же прогон без CCD (контроль: должно туннелировать)
//
// Гравитация выключена: мерим чистый полёт по прямой. Раскол выключен: нужен
// один снаряд и одна плита, а не туча осколков.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn env_u32(name: &str, def: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn main() {
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
    let dt = 1.0f32 / 20.0;

    println!(
        "устройство: {} · CCD {} · подшагов {substeps} · такт {:.0} мс",
        {
            let w = Live::new(&sh);
            let n = w.gpu.name.clone();
            drop(w);
            n
        },
        if matches!(std::env::var("VF_CCD").as_deref(), Ok("0")) { "ВЫКЛ" } else { "на GPU" },
        dt * 1e3
    );
    println!("скорость · исход · x_отн · vx_отн · событий CCD · t · глубина м · такт max мс");

    for &speed in &speeds {
        let mut w = Live::new(&sh);
        w.set_fracture(false);
        w.params.gravity = 0.0;
        w.gpu.params.gravity = 0.0;
        for _ in 0..6 {
            w.step(dt, substeps, iters);
        }
        w.gpu.wait_idle();
        let plate_half = {
            let b = &w.bodies[0];
            let mut hi = f32::MIN;
            for t in b.rotated_tets() {
                for v in t {
                    hi = hi.max(v[0]);
                }
            }
            hi - b.pos[0] as f32
        };
        let plate_y = w.bodies[0].pos[1];
        w.shoot([-3.0, plate_y, 0.0], [1.0, 0.0, 0.0], speed);
        let slot = *w.bullets.back().unwrap();
        w.gpu.wait_idle();

        let (mut x_rel, mut vx_rel) = (0.0f32, 0.0f32);
        let mut n_ev = 0usize;
        let mut first_t = 0.0f32;
        let mut depth = 0.0f32;
        let mut tick_ms_max = 0.0f32;
        let mut nan = false;
        for t in 0..ticks {
            let t0 = std::time::Instant::now();
            w.step(dt, substeps, iters);
            w.gpu.wait_idle();
            tick_ms_max = tick_ms_max.max(t0.elapsed().as_secs_f64() as f32 * 1e3);
            // события такта: у CCD в ev0.w лежит доля шага t
            let ev = w.gpu.read_impact_events();
            for e in &ev {
                if e[3] > 0.0 {
                    n_ev += 1;
                    if first_t == 0.0 {
                        first_t = e[3];
                        depth = e[7];
                    }
                }
            }
            let snap = w.gpu.read_bodies();
            let b = &snap[slot];
            x_rel = b.pos_r[0] - snap[0].pos_r[0];
            vx_rel = b.vel_m[0] - snap[0].vel_m[0];
            if !x_rel.is_finite() || !vx_rel.is_finite() {
                nan = true;
            }
            if dbg {
                println!(
                    "  t={t:2} x_отн={x_rel:7.3} vx_отн={vx_rel:7.2} v_шар={:7.2} событий {}",
                    b.vel_m[0],
                    ev.len()
                );
            }
        }
        let outcome = if nan {
            "NaN"
        } else if x_rel > plate_half {
            "ПРОЛЕТЕЛ"
        } else if x_rel > 0.0 {
            "в плите"
        } else if vx_rel < -0.5 {
            "отбит"
        } else {
            "столкнулся"
        };
        println!(
            "{speed:6.0} · {outcome:11} · x_отн={x_rel:7.3} · vx_отн={vx_rel:7.2} · событий {n_ev:2} · t={first_t:.3} · глубина {depth:.4} · такт {tick_ms_max:5.1} мс"
        );
    }
}
