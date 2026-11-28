// Ёмкость тел и рисайклинг слотов.
//
//  1) 200 размеренных выстрелов, 2) 200 подряд без тактов, 3) 600 по одному
//     с тактом — сколько тел создано, сколько спавнов ОТКАЗАНО, сколько
//     слотов освобождено деспавном;
//  4) набиваем мир осколками до 128 и 256 тел и меряем такт;
//  5) печатаем пик RSS процесса.
use std::io::Read;

use voxelforge_world::live::{Kind, Live, MAX_BODIES, MIN_TETS};
use voxelforge_world::sim::{find_shaders, MAX_BODIES as SIM_BODIES, MAX_POOL_TETS};

fn pool_stats(w: &Live) -> (usize, usize) {
    let mut sum = 0usize;
    let mut max = 0usize;
    for b in w.bodies.iter().filter(|b| b.alive) {
        sum += b.tets.len();
        max = max.max(b.tets.len());
    }
    (sum, max)
}

fn rss_mb() -> f64 {
    let mut s = String::new();
    if let Ok(mut f) = std::fs::File::open("/proc/self/status") {
        let _ = f.read_to_string(&mut s);
    }
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            if let Some(kb) = v.trim().split_whitespace().next() {
                return kb.parse::<f64>().unwrap_or(0.0) / 1024.0;
            }
        }
    }
    0.0
}

fn report(tag: &str, w: &Live, peak_alive: usize, peak_slots: usize, refused_shots: usize) {
    let alive = w.bodies.iter().filter(|b| b.alive).count();
    let (tets, tmax) = pool_stats(w);
    println!(
        "[{tag}] создано {} · отказов {} (shoot {}) · деспавнов {} (границы {}, TTL {}, сон {}, вытеснено {}) · живых {} (пик {}) · слотов {} (пик {}) · снарядов {} · трещин {} · осколков {}",
        w.created,
        w.refused,
        refused_shots,
        w.despawned(),
        w.despawn_bounds,
        w.despawn_ttl,
        w.despawn_sleep,
        w.despawn_evicted,
        alive,
        peak_alive,
        w.bodies.len(),
        peak_slots,
        w.bullets.len(),
        w.cracks,
        w.shards_made,
    );
    println!(
        "[{tag}] пул тетов: занято {tets} из {MAX_POOL_TETS} ({:.1}%), максимум на тело {tmax}",
        100.0 * tets as f64 / MAX_POOL_TETS as f64
    );
}

/// Дробить самое крупное тело, пока в мире не станет `target` тел.
fn fill(w: &mut Live, target: usize) {
    let mut guard = 0;
    while w.alive_count() < target && guard < 4000 {
        guard += 1;
        let mut best: Option<usize> = None;
        let mut best_t = 0usize;
        for (i, b) in w.bodies.iter().enumerate() {
            if b.alive && b.kind != Kind::Bullet && b.tets.len() > best_t {
                best_t = b.tets.len();
                best = Some(i);
            }
        }
        let Some(i) = best else { break };
        if best_t < 2 * MIN_TETS {
            break;
        }
        let p = w.bodies[i].pos;
        let before = w.alive_count();
        w.split(i, [p[0] as f32, p[1] as f32, p[2] as f32], 4000.0, 8);
        if w.alive_count() == before {
            break; // не делится дальше
        }
        /* пара тактов, чтобы осколки разъехались: иначе все 400 тел лежат
         * в одной точке, и это уже не «много тел», а одна куча-патология */
        if guard % 4 == 0 {
            w.step(1.0 / 20.0, 10, 18);
        }
    }
}

/// Среднее и максимум такта, мс.
fn time_ticks(w: &mut Live, n: usize) -> (f32, f32) {
    let mut avg = 0.0f32;
    let mut max = 0.0f32;
    for _ in 0..n {
        let t = std::time::Instant::now();
        w.step(1.0 / 20.0, 10, 18);
        let ms = t.elapsed().as_secs_f32() * 1e3;
        avg += ms;
        max = max.max(ms);
    }
    (avg / n as f32, max)
}

fn main() {
    let sh = find_shaders(None).unwrap();
    /* PROBE_ONLY=d — только фаза D (набитый мир), чтобы не гонять минуты
     * выстрелов ради проверки ёмкости. */
    let only = std::env::var("PROBE_ONLY").unwrap_or_default();
    let skip_abc = only == "d";
    let skip_a = only == "a" || only == "b";
    let skip_c = only == "c" || only == "b";
    println!(
        "ёмкость: live::MAX_BODIES={MAX_BODIES}, sim::MAX_BODIES={SIM_BODIES}, MAX_POOL_TETS={MAX_POOL_TETS}, старт RSS {:.0} МБ",
        rss_mb()
    );

    // --- A: 200 выстрелов по одному, с тактами между ними ---
    let mut w = Live::new(&sh);
    let p0 = w.bodies[0].pos;
    let from = [p0[0] - 4.0, p0[1], p0[2]];
    let (mut peak_alive, mut peak_slots) = (0, 0);
    let mut fails = 0;
    for i in 0..if skip_abc || skip_a { 0 } else { 200 } {
        if !w.shoot(from, [1.0, 0.0, 0.0], 26.0) {
            fails += 1;
            if fails <= 3 {
                println!(
                    "  отказ на выстреле {i}: живых {}, слотов {}",
                    w.alive_count(),
                    w.bodies.len()
                );
            }
        }
        for _ in 0..5 {
            w.step(1.0 / 20.0, 10, 18);
        }
        peak_alive = peak_alive.max(w.alive_count());
        peak_slots = peak_slots.max(w.bodies.len());
    }
    report("A: 200 размеренных", &w, peak_alive, peak_slots, fails);

    // --- B: 200 выстрелов за короткое время (без тактов) ---
    let mut w = Live::new(&sh);
    let p0 = w.bodies[0].pos;
    let from = [p0[0] - 4.0, p0[1], p0[2]];
    let (mut peak_alive, mut peak_slots) = (0, 0);
    let mut fails = 0;
    if !skip_abc {
        for i in 0..200 {
        if !w.shoot(from, [1.0, 0.0, 0.0], 26.0) {
            fails += 1;
            if fails <= 3 {
                println!(
                    "  отказ на выстреле {i}: живых {}, слотов {}",
                    w.alive_count(),
                    w.bodies.len()
                );
            }
        }
        peak_alive = peak_alive.max(w.alive_count());
        peak_slots = peak_slots.max(w.bodies.len());
    }
    }
    println!("  быстрых выстрелов до такта: отказов {fails}");
    for _ in 0..40 {
        w.step(1.0 / 20.0, 10, 18);
        peak_alive = peak_alive.max(w.alive_count());
        peak_slots = peak_slots.max(w.bodies.len());
    }
    report("B: 200 быстрых", &w, peak_alive, peak_slots, fails);

    // --- C: держим мир в пределе и смотрим, возвращаются ли слоты ---
    let mut w = Live::new(&sh);
    let p0 = w.bodies[0].pos;
    let from = [p0[0] - 4.0, p0[1], p0[2]];
    let mut fails = 0;
    let (mut peak_alive, mut peak_slots) = (0, 0);
    if !skip_c {
    for i in 0..600 {
        if !w.shoot(from, [1.0, 0.0, 0.0], 26.0) {
            fails += 1;
        }
        w.step(1.0 / 20.0, 10, 18);
        if i % 100 == 99 {
            println!(
                "  такт {i}: живых {}, слотов {}, свободных {}, отказов {}",
                w.alive_count(),
                w.bodies.len(),
                w.free_slots(),
                fails
            );
        }
        peak_alive = peak_alive.max(w.alive_count());
        peak_slots = peak_slots.max(w.bodies.len());
    }
    report("C: 600 по одному с тактом", &w, peak_alive, peak_slots, fails);
    }

    // --- D: сколько тел реально принимает мир и как дорог такт ---
    let mut w = Live::new(&sh);
    let targets: Vec<usize> = std::env::var("PROBE_TARGETS")
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|_| vec![64, 128, 256, 400]);
    for target in targets {
        fill(&mut w, target);
        // даём телам осесть
        let (avg, max) = time_ticks(&mut w, 20);
        let alive = w.alive_count();
        let (tets, tmax) = pool_stats(&w);
        println!(
            "  D: цель {target} → живых {alive}, слотов {}, отказов {}, такт сред {avg:.1} мс, макс {max:.1} мс, пул {tets} тетов (макс на тело {tmax})",
            w.bodies.len(),
            w.refused
        );
    }
    report("D: набитый мир", &w, w.alive_count(), w.bodies.len(), 0);
    // Сколько тел создаётся поверх ёмкости и что при этом говорит мир
    let mut fails = 0;
    for _ in 0..100 {
        if !w.shoot([0.0, 6.0, 0.0], [0.0, -1.0, 0.0], 20.0) {
            fails += 1;
        }
    }
    println!("  D: 100 выстрелов в набитом мире → отказов {fails}");
    println!("пик RSS процесса: {:.0} МБ", rss_mb());
}
