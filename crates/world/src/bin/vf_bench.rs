//! Замер цепочки пар тетов на живой сцене — без окна.
//!
//! Гоняет `Live` (та же сцена, что в окне: плита, выстрелы, раскол) и печатает
//! то, чего в таблице стадий не видно: время такта по стенным часам, число
//! записей «тет x ячейка», заполнение самой большой ячейки, сколько итераций
//! делает проход пар, сколько пар переживает префильтры и сколько их влезает в
//! буфер.
//!
//!   VF_BENCH_TICKS=160 VF_BENCH_SHOTS=4,20,36 VF_BENCH_WINDOW=40:120 \
//!     ./target/release/vf-bench
//!
//! VF_BENCH_GRID=1 — разобрать сетку в конце, VF_BENCH_DUMP_AT=t1,t2 — по
//! тактам, VF_BENCH_LOG=1 — печатать каждый такт, VF_PAIR_STATS=1 — счётчики
//! из шейдера (без него проход пар работает без единой лишней атомарной
//! операции сверх записи самой пары, и время честное).
use std::collections::HashMap;
use std::time::Instant;

use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn main() {
    let ticks: u32 = std::env::var("VF_BENCH_TICKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(160);
    let dt: f32 = std::env::var("VF_BENCH_DT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.05);
    let shots: Vec<u32> = std::env::var("VF_BENCH_SHOTS")
        .unwrap_or_else(|_| "4,20,36,52,68".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let (win_lo, win_hi) = std::env::var("VF_BENCH_WINDOW")
        .unwrap_or_else(|_| "0:0".into())
        .split_once(':')
        .map(|(a, b)| {
            (
                a.trim().parse().unwrap_or(0),
                b.trim().parse().unwrap_or(0),
            )
        })
        .unwrap_or((0, 0));
    let verbose = std::env::var("VF_BENCH_LOG").is_ok();
    let dumps: Vec<u32> = std::env::var("VF_BENCH_DUMP_AT")
        .unwrap_or_default()
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();

    let shaders = find_shaders(None).expect("нет SPIR-V: соберите shaders/spv");
    let mut live = Live::new(&shaders);
    println!("[bench] девайс {}", live.gpu.name);
    println!(
        "[bench] тактов {ticks}, dt {dt}, выстрелы на {shots:?}, статистика пар {}",
        if live.gpu.pair_stats { "вкл" } else { "выкл" }
    );

    let mut last_alive = 0usize;
    let mut times: Vec<f32> = Vec::with_capacity(ticks as usize);
    let mut pair_times: Vec<f32> = Vec::with_capacity(ticks as usize);
    for t in 0..ticks {
        if shots.contains(&t) {
            let from = [0.0f64, 1.6, 4.5];
            let aim = [0.0f64, 0.55, 0.0];
            let d = [aim[0] - from[0], aim[1] - from[1], aim[2] - from[2]];
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            live.shoot(from, [d[0] / l, d[1] / l, d[2] / l], 26.0);
        }
        let t0 = Instant::now();
        live.step(dt, 8, 12);
        /* Такт отправляется асинхронно: время шага — это ещё и ожидание
         * предыдущего submit'а, то есть честная пропускная способность. */
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        times.push(ms);
        pair_times.push(live.gpu.last_pairs_ms);
        let alive = live.alive_count();
        if alive != last_alive || verbose {
            println!(
                "[bench] такт {t}: тел {alive} · бодр {} · пар {} · пары {:.1} мс · такт {:.1} мс",
                live.last_awake,
                live.gpu.read_pair_count(),
                live.gpu.last_pairs_ms,
                ms
            );
            last_alive = alive;
        }
        if dumps.contains(&t) {
            println!("[bench] --- снимок сетки на такте {t} ---");
            dump_grid(&live);
        }
    }

    let n_bodies = live.bodies.iter().filter(|b| b.alive).count();
    let tets: usize = live.bodies.iter().filter(|b| b.alive).map(|b| b.local_tets().len()).sum();
    println!(
        "[bench] конец: тел {n_bodies}, тетов {tets}, бодр {}, трещин {}, осколков {}, контактов {}",
        live.last_awake,
        live.cracks,
        live.shards_made,
        live.gpu.contact_count()
    );
    /* Пары встроены в такт, а такт отправляется асинхронно: счётчики прохода
     * читаются только после fence. Без этого ожидания `pcnt[0]` — это ноль,
     * который поставил CPU перед отправкой, то есть «пар 0» на любой сцене. */
    live.gpu.wait_idle();
    let d = live.gpu.read_pair_debug();
    println!(
        "[bench] проход пар: пар {} · итераций {} · канон {} · не влезло {} · макс. ячейка {} · по записям {}",
        d[0], d[6], d[5], d[7], d[2], d[1]
    );
    print_times(&times, win_lo, win_hi, "такт");
    print_times(&pair_times, win_lo, win_hi, "пары (сетка+испускание)");

    if std::env::var("VF_BENCH_GRID").is_ok() {
        dump_grid(&live);
    }
}

fn print_times(times: &[f32], lo: u32, hi: u32, what: &str) {
    let mut v: Vec<f32> = times
        .iter()
        .enumerate()
        .filter(|(i, _)| *i as u32 >= lo && (hi == 0 || *i as u32 <= hi))
        .map(|(_, v)| *v)
        .collect();
    if v.is_empty() {
        v = times.to_vec();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    println!(
        "[bench] {what} (стенные часы, {} шт., окно {lo}..{hi}): мин {:.1} · среднее {:.1} · медиана {:.1} · макс {:.1} мс",
        v.len(),
        v[0],
        mean,
        v[v.len() / 2],
        v[v.len() - 1]
    );
}

/// Разбор сетки тетов: заполнение ячеек и сколько итераций из него выходит.
fn dump_grid(live: &Live) {
    let (starts, entries) = live.gpu.read_tet_grid();
    let ncells = starts.len() - 1;
    let total = starts[ncells] as usize;

    /* AABB тел — по мировым тетам: тот же префильтр, что делает шейдер. */
    let mut boxes: Vec<[f32; 6]> = Vec::with_capacity(live.bodies.len());
    for b in &live.bodies {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        if b.alive {
            for t in b.rotated_tets() {
                for v in t {
                    for k in 0..3 {
                        lo[k] = lo[k].min(v[k]);
                        hi[k] = hi[k].max(v[k]);
                    }
                }
            }
        }
        boxes.push([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]);
    }
    let overlap = |a: usize, b: usize| -> bool {
        let (x, y) = (&boxes[a], &boxes[b]);
        !(x[0] > y[3] || y[0] > x[3] || x[1] > y[4] || y[1] > x[4] || x[2] > y[5] || y[2] > x[5])
    };

    let mut nonempty = 0u64;
    let mut max_fill = 0u32;
    let mut max_cell = 0usize;
    let mut iter_all = 0u64; // пары записей внутри ячейки
    let mut iter_xbody = 0u64; // из них — разные тела
    let mut iter_broad = 0u64; // из них — тела с пересечением AABB

    for c in 0..ncells {
        let s = starts[c] as usize;
        let e = starts[c + 1] as usize;
        let k = e - s;
        if k == 0 {
            continue;
        }
        nonempty += 1;
        let k = k as u32;
        if k > max_fill {
            max_fill = k;
            max_cell = c;
        }
        iter_all += (k as u64) * (k as u64 - 1) / 2;
        let mut per_body: HashMap<u32, u32> = HashMap::new();
        for i in s..e {
            *per_body.entry(entries[i][0]).or_insert(0) += 1;
        }
        for (&ba, &ka) in &per_body {
            for (&bb, &kb) in &per_body {
                if ba >= bb {
                    continue;
                }
                let n = ka as u64 * kb as u64;
                iter_xbody += n;
                if overlap(ba as usize, bb as usize) {
                    iter_broad += n;
                }
            }
        }
    }
    let tets: usize = live
        .bodies
        .iter()
        .filter(|b| b.alive)
        .map(|b| b.local_tets().len())
        .sum();

    let mean = total as f64 / nonempty.max(1) as f64;
    println!("[grid] ячеек {ncells}, занятых {nonempty}, записей {total}");
    println!(
        "[grid] самая большая ячейка {max_fill} записей (ячейка {max_cell}), средняя {mean:.2}, записей на тет {:.2}",
        total as f64 / tets.max(1) as f64
    );
    println!(
        "[grid] пар записей в ячейках: всех {iter_all}, разных тел {iter_xbody} ({:.1}%), после префильтра AABB тел {iter_broad} ({:.1}%)",
        100.0 * iter_xbody as f64 / iter_all.max(1) as f64,
        100.0 * iter_broad as f64 / iter_all.max(1) as f64
    );
}
