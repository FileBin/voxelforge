// ЦЕНА CCD В МОМЕНТ КАСАНИЯ: сколько стоит такт, когда включается continuous
// и сколько из этого — сам CCD по фазам.
//
//   VF_SPEED=240 VF_TICKS=40 VF_COLLIDE=sdf probe_ccd_cost
//
// Печатает: такт (мин/сред/макс/p99), самый дорогой такт и его фазы CCD
// (пробные точки, интервал, бисекция), число тел/трещин, попадания CCD.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn env_f64(name: &str, def: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn main() {
    let sh = find_shaders(None).unwrap();
    let speed = env_f64("VF_SPEED", 240.0);
    let ticks = env_f64("VF_TICKS", 40.0) as usize;
    let dt = 1.0 / 20.0f32;
    println!(
        "скорость {speed} м/с · тактов {ticks} · режим {} · шейдеры {:?}",
        std::env::var("VF_COLLIDE").unwrap_or_else(|_| "по умолчанию".into()),
        sh.file_name()
    );

    let mut w = Live::new(&sh);
    /* Раскол ВКЛЮЧЁН: всплеск такта приходится именно на кадр раскола —
     * появляются десятки новых тел, и CCD собирает для них точки. */
    w.set_fracture(true);
    for _ in 0..6 {
        w.step(dt, 8, 12);
    }
    w.gpu.wait_idle();
    let _ = w.shoot([-3.0, 0.8, 0.0], [1.0, 0.0, 0.0], speed);
    w.gpu.wait_idle();

    let mut tick_ms: Vec<f32> = Vec::new();
    let mut ccd_ms: Vec<f32> = Vec::new();
    let mut ccd_hits = 0u32;
    let mut worst = (0.0f32, 0usize);
    let mut worst_ph = voxelforge_world::live::CcdPhases::default();
    println!("\nтакт · стена мс · CCD мс · попаданий · тел · трещин · шейдеров");
    for t in 0..ticks {
        let t0 = std::time::Instant::now();
        w.step(dt, 8, 12);
        w.gpu.wait_idle();
        let ms = t0.elapsed().as_secs_f32() * 1e3;
        tick_ms.push(ms);
        ccd_ms.push(w.last_ccd_ms);
        ccd_hits += w.last_ccd;
        if ms > worst.0 {
            worst = (ms, t);
            worst_ph = w.ccd_ph;
        }
        println!(
            "{t:3} · {ms:8.2} · {:8.2} · {:9} · {:4} · {:6} · {}",
            w.last_ccd_ms,
            w.last_ccd,
            w.alive_count(),
            w.cracks,
            w.shards_made
        );
    }

    let mut v = tick_ms.clone();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let stat = |s: &[f32]| -> (f32, f32, f32, f32) {
        let n = s.len();
        let mean = s.iter().sum::<f32>() / n as f32;
        let p99 = s[((n as f32 * 0.99) as usize).min(n - 1)];
        (s[0], mean, s[n - 1], p99)
    };
    let (mn, me, mx, p99) = stat(&v);
    let mut c = ccd_ms.clone();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (cmn, cme, cmx, cp99) = stat(&c);
    println!(
        "\nТАКТ (стена): мин {mn:.2} · средн {me:.2} · макс {mx:.2} · p99 {p99:.2} мс"
    );
    println!("CCD ВНУТРИ ТАКТА: мин {cmn:.2} · средн {cme:.2} · макс {cmx:.2} · p99 {cp99:.2} мс");
    println!(
        "самый дорогой такт: {} · {:.2} мс; его CCD {:.2} мс: точки {:.2} мс ({} шт) · интервал {:.2} · бисекция {:.2} ({} итер) · пар {}",
        worst.1,
        worst.0,
        worst_ph.total_us as f32 * 1e-3,
        worst_ph.probes_us as f32 * 1e-3,
        worst_ph.probes_n,
        worst_ph.trace_us as f32 * 1e-3,
        worst_ph.bisect_us as f32 * 1e-3,
        worst_ph.bisect_iters,
        worst_ph.pairs
    );
    println!(
        "итог: попаданий CCD {ccd_hits} · трещин {} · осколков {} · тел {}",
        w.cracks,
        w.shards_made,
        w.alive_count()
    );
}
