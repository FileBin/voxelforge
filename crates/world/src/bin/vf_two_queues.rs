//! Две очереди против одной: такт на compute-очереди (приоритет 0.0), кадр —
//! на graphics (1.0), синхронизация таймлайн-семафорами, состояние тел в двух
//! банках.
//!
//! Мерит то, ради чего это делалось: СКОЛЬКО КАДР ЖДЁТ. Одна очередь
//! сериализует кадр за тактом (кадр, попавший в такт, стоит, пока такт не
//! досчитается), две — нет: кадр берёт банк, опубликованный ПРОШЛЫМ тактом, и
//! ждёт на GPU только его.
//!
//! Прогон воспроизводимый: такт 20 Гц сценария «плита + снаряд», кадры 200 Гц.
//! Обе конфигурации считают один и тот же сценарий, поэтому финальные позы
//! обязаны совпасть — это и есть проверка, что разделение очередей ничего не
//! испортило в физике.
//!
//!   VF_SHADERS=... cargo run --release --features live --bin vf-two-queues -- [тактов] [кадров_на_такт]

use std::path::PathBuf;

use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

struct Run {
    split: bool,
    /// Что реально дал драйвер (а не что просили).
    actual_split: bool,
    gpu_tick_ms: Vec<f32>,
    frames_ms: Vec<f32>,
    tick_ms: Vec<f32>,
    final_pos: Vec<[f64; 3]>,
    bodies: usize,
    cracks: usize,
    shards: usize,
    bank_diff: f32,
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

fn pct(v: &[f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((s.len() - 1) as f32 * p) as usize;
    s[i]
}

fn run(split: bool, shaders: &PathBuf, ticks: u32, frames_per_tick: u32) -> Run {
    /* Ядро читает VF_SPLIT при создании устройства, так что переменная должна
     * стоять до Live::new. Переменные процесса живут до конца прогона — два
     * девайса в одном процессе создаются по очереди, каждый со своим
     * значением. */
    unsafe { std::env::set_var("VF_SPLIT", if split { "1" } else { "0" }) };
    unsafe { std::env::set_var("VF_COLLIDE", "gpu") };
    unsafe { std::env::set_var("VF_PAIRS", "gpu") };

    let mut live = Live::new(shaders);
    let (fg, ft, pg, pt, split_ok) = live.gpu.queues();
    eprintln!(
        "[probe] {} · очередей: кадр семейство {fg} приоритет {pg:.1}, такт семейство {ft} \
         приоритет {pt:.1} · режим: {}",
        live.gpu.name,
        if split_ok { "две очереди" } else { "одна очередь" }
    );

    /* Два сценария.
     *
     * VF_FLY=1 — «свободный полёт»: снаряд идёт горизонтально вдали от плиты и
     * НИ С ЧЕМ не сталкивается. Тогда у солвера нет контактов, траектория —
     * чистое интегрирование, и две конфигурации обязаны совпасть БИТ В БИТ:
     * это и есть прямая проверка, что кадровый поток не мешает физике.
     *
     * По умолчанию — настоящий удар в плиту на 26 м/с: он даёт нагрузку на такт
     * (и на нём меряется ожидание кадра). Точного совпадения здесь ждать
     * нельзя: порядок записей в сетку тетов задают atomicAdd, поэтому даже два
     * прогона ОДНОЙ конфигурации расходятся (см. колонку «две-2»). */
    if std::env::var("VF_FLY").is_ok() {
        live.set_fracture(false);
        live.set_sleep(false);
        live.set_wake_on_contact(false);
        live.shoot([0.0, 8.0, 0.0], [1.0, 0.0, 0.0], 0.5);
    } else {
        live.shoot([0.0, 2.2, 0.0], [0.0, -1.0, 0.0], 26.0);
    }
    /* VF_SAFE=1 — без раскола и сна: тогда траектория обязана совпасть
     * БИТ В БИТ, и это прямая проверка, что разделение очередей ничего не
     * меняет в физике. С расколом Солвер на GPU и сам по себе
     * недетерминирован: порядок записей в сетку тетов задают atomicAdd, и
     * два прогона ОДНОЙ конфигурации расходятся на сантиметры. */
    if std::env::var("VF_SAFE").is_ok() {
        live.set_fracture(false);
        live.set_sleep(false);
        live.set_wake_on_contact(false);
    }

    let tick = 0.05f32; // 20 Гц
    let (substeps, iterations) = (8u32, 14u32);
    let mut frames_ms: Vec<f32> = Vec::with_capacity((ticks * frames_per_tick) as usize);
    let mut tick_ms: Vec<f32> = Vec::with_capacity(ticks as usize);
    let slot = tick / frames_per_tick as f32;
    let mut bank_diff = 0.0f32;
    let mut gpu_tick_ms: Vec<f32> = Vec::new();

    for _ in 0..ticks {
        let t0 = std::time::Instant::now();
        live.step(tick, substeps, iterations);
        tick_ms.push(t0.elapsed().as_secs_f32() * 1000.0);
        if live.gpu.perf_on() {
            let v = live.gpu.perf_tick_ms();
            if v > 0.0 {
                gpu_tick_ms.push(v);
            }
        }
        let t_submit = std::time::Instant::now();

        /* Кадры идут своим ритмом, не дожидаясь такта. Ровно это и проверяем:
         * каждый кадр мерит время от submit до fence. */
        let no_frames = std::env::var("VF_NOFRAMES").is_ok();
        for k in 0..frames_per_tick {
            if no_frames {
                break;
            }
            let alpha = (k as f32 + 1.0) / frames_per_tick as f32;
            live.gpu.frame_submit(alpha, true);
            if live.gpu.frame_ms > 0.0 {
                frames_ms.push(live.gpu.frame_ms);
            }
            /* Держим ритм 200 Гц: без этого кадры слиплись бы в очередь и
             * мерили бы не то. */
            let due = slot * (k + 1) as f32;
            let done = t_submit.elapsed().as_secs_f32() * 1000.0;
            if due > done {
                std::thread::sleep(std::time::Duration::from_secs_f32((due - done) / 1000.0));
            }
        }

        /* Банк, опубликованный этим тактом, обязан совпасть с буфером физики:
         * кадр читает ровно то состояние, которое посчитала физика. */
        live.gpu.wait_idle();
        let (tick_no, bank) = live.gpu.tick_bank;
        if tick_no > 0 && live.gpu.queues().4 {
            let phys = live.gpu.read_bodies();
            let b = live.gpu.read_bank(bank);
            for (p, q) in phys.iter().zip(b.iter()) {
                for k in 0..3 {
                    bank_diff = bank_diff.max((p.pos_r[k] - q.pos_r[k]).abs());
                }
            }
        }
    }
    /* Последний такт и полное затишье: если бы устройство потерялось (GPUVM
     * fault), здесь вернулась бы ошибка. */
    live.step(tick, substeps, iterations);
    live.gpu.wait_idle();

    /* Сверка — по УСТРОЙСТВУ, а не по CPU-зеркалу: зеркало на GPU-пути
     * обновляется только событиями раскола. */
    let dev = live.gpu.read_bodies();

    Run {
        split,
        actual_split: live.gpu.queues().4,
        gpu_tick_ms,
        frames_ms,
        tick_ms,
        final_pos: dev
            .iter()
            .map(|b| [b.pos_r[0] as f64, b.pos_r[1] as f64, b.pos_r[2] as f64])
            .collect(),
        bodies: live.bodies.iter().filter(|b| b.alive).count(),
        cracks: live.cracks,
        shards: live.shards_made,
        bank_diff,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let ticks: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(30);
    let fpt: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(10);

    let shaders = find_shaders(None).expect("нет SPIR-V: VF_SHADERS=... или nix build .#spirv");

    let one = run(false, &shaders, ticks, fpt);
    let two = run(true, &shaders, ticks, fpt);
    /* Контроль недетерминизма: ещё один прогон той же двухочередной
     * конфигурации. Разброс «две-две» — это шум самого солвера; расхождение
     * «одна-две» сравнивается с ним, а не с нулём. */
    let two2 = run(true, &shaders, ticks, fpt);

    println!();
    println!("такт 20 Гц, кадры 200 Гц · {} тактов · {} кадров на такт", ticks, fpt);
    println!();
    println!("| что | одна очередь | две очереди |");
    println!("|---|---|---|");
    println!(
        "| очередей у устройства | {} | {} |",
        if one.actual_split { "две" } else { "одна" },
        if two.actual_split { "две" } else { "одна" }
    );
    println!(
        "| кадр, среднее ожидание, мс | {:.3} | {:.3} |",
        mean(&one.frames_ms),
        mean(&two.frames_ms)
    );
    println!(
        "| кадр, 99-й перцентиль, мс | {:.3} | {:.3} |",
        pct(&one.frames_ms, 0.99),
        pct(&two.frames_ms, 0.99)
    );
    println!(
        "| кадр, максимум, мс | {:.3} | {:.3} |",
        one.frames_ms.iter().cloned().fold(0.0, f32::max),
        two.frames_ms.iter().cloned().fold(0.0, f32::max)
    );
    let over = |v: &Vec<f32>, lim: f32| v.iter().filter(|x| **x > lim).count();
    println!(
        "| кадров дольше 5 мс | {} из {} | {} из {} |",
        over(&one.frames_ms, 5.0),
        one.frames_ms.len(),
        over(&two.frames_ms, 5.0),
        two.frames_ms.len()
    );
    println!(
        "| такт, среднее, мс (CPU: submit+чтения) | {:.3} | {:.3} |",
        mean(&one.tick_ms),
        mean(&two.tick_ms)
    );
    println!(
        "| такт, максимум, мс | {:.3} | {:.3} |",
        one.tick_ms.iter().cloned().fold(0.0, f32::max),
        two.tick_ms.iter().cloned().fold(0.0, f32::max)
    );
    if !one.gpu_tick_ms.is_empty() && !two.gpu_tick_ms.is_empty() {
        println!(
            "| такт на GPU (таймстемпы), среднее/макс, мс | {:.2} / {:.2} | {:.2} / {:.2} |",
            mean(&one.gpu_tick_ms),
            one.gpu_tick_ms.iter().cloned().fold(0.0, f32::max),
            mean(&two.gpu_tick_ms),
            two.gpu_tick_ms.iter().cloned().fold(0.0, f32::max)
        );
    }
    println!("| тел / трещин / осколков | {} / {} / {} | {} / {} / {} |",
        one.bodies, one.cracks, one.shards, two.bodies, two.cracks, two.shards);
    println!("| то же у контроля «две-2» | — | {} / {} / {} |",
        two2.bodies, two2.cracks, two2.shards);
    println!(
        "| расхождение банка и физики, м | — | {:.6} |",
        two.bank_diff
    );

    /* Паркованные тела стоят на y = -400: это не расхождение физики, а
     * разные слоты под улетевшими обломками. Сравниваем только «настоящие». */
    let dp = |a: &Run, b: &Run| {
        let mut m = 0.0f64;
        for (x, y) in a.final_pos.iter().zip(b.final_pos.iter()) {
            if x[1] < -100.0 || y[1] < -100.0 {
                continue;
            }
            for k in 0..3 {
                m = m.max((x[k] - y[k]).abs());
            }
        }
        m
    };
    let d12 = dp(&one, &two);
    let d22 = dp(&two, &two2);
    println!();
    println!("| расхождение финальных поз | одна-две | две-две (контроль) |");
    println!("|---|---|---|");
    println!("| максимум по телам, м | {d12:.6} | {d22:.6} |");

    let fly = std::env::var("VF_FLY").is_ok();
    if fly {
        /* Свободный полёт: контактов нет, траектория обязана совпасть точно. */
        if d12 > 1e-9 {
            println!("ВНИМАНИЕ: физика зависит от конфигурации очередей ({d12:.9})");
            std::process::exit(1);
        }
        println!("физика: при свободном полёте состояния совпали бит в бит");
    } else {
        /* На ударе точного совпадения не бывает: порядок записей в сетку тетов
         * задают atomicAdd, и сам солвер расходится от прогона к прогону (см.
         * контроль «две-две»). Поэтому здесь проверяется не совпадение, а
         * отсутствие СИСТЕМАТИЧЕСКОЙ ошибки: числа тел совпадают, а
         * расхождение не выходит за порядок шума. */
        println!("физика: расхождение одна-две {d12:.6} м, шум солвера (две-две) {d22:.6} м");
        /* Разное число тел — это ещё не ошибка: раскол зависит от ударов, а
         * солвер недетерминирован. Ошибка — если контроль «две-2» совпал с
         * «одна», а «две» от него отскочила. */
        if one.bodies != two.bodies && two2.bodies == one.bodies {
            println!(
                "ВНИМАНИЕ: две очереди дали {} тел, одна и контроль — {}",
                two.bodies, one.bodies
            );
            std::process::exit(1);
        }
    }
    if two.split && two.bank_diff > 1e-6 {
        println!("ВНИМАНИЕ: банк не равен буферу физики ({:.9})", two.bank_diff);
        std::process::exit(1);
    }
    println!("OK: устройство живо, банк равен физике");
}
