// ЧИСЛА ПО ТУННЕЛИРОВАНИЮ В МЕЛКИЕ ОСКОЛКИ.
//
// Плита дробится несколько раз подряд (1-е, 2-е, 3-е поколение осколков), затем
// в выбранный осколок пускается снаряд на 5/20/60/120/240 м/с. Печатается
// таблица: столкнулся/пролетел, зазор до коробки осколка, глубина, размеры
// осколка и его поля (узлы, шаг, толщина занятой области в вокселях), число
// пробных точек, время CCD.
//
// Отдельно: пол (снаряд и осколки вниз) и мелкий осколок в мелкий осколок.
//
//   VF_COLLIDE=sdf probe_ccd_small       контакты по SDF (как в живом мире)
//   VF_SPEEDS=5,20,60,120,240            список скоростей
//   VF_TICKS=8                           минимум тактов на выстрел
//   VF_DBG=1                             печатать каждый такт
use voxelforge_world::live::{Kind, Live, LiveBody};
use voxelforge_world::sim::find_shaders;

fn env_u32(name: &str, def: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
}

/// Статистика поля осколка.
struct FieldInfo {
    dims: [usize; 3],
    step: f32,
    nodes: usize,
    neg: usize,
    /// Толщина ЗАНЯТОЙ области в узлах — по самому тонкому направлению.
    thick_vox: usize,
}

fn field_info(b: &LiveBody) -> Option<FieldInfo> {
    let f = b.sdf.as_ref()?;
    let (nx, ny, nz) = (f.dims[0], f.dims[1], f.dims[2]);
    let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
    let mut neg = 0usize;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if f.val[(z * ny + y) * nx + x] < 0.0 {
                    neg += 1;
                    let p = [f.lo[0] + x as i32, f.lo[1] + y as i32, f.lo[2] + z as i32];
                    for k in 0..3 {
                        lo[k] = lo[k].min(p[k]);
                        hi[k] = hi[k].max(p[k]);
                    }
                }
            }
        }
    }
    let thick_vox = if lo[0] > hi[0] {
        0
    } else {
        (0..3).map(|k| (hi[k] - lo[k] + 1) as usize).min().unwrap_or(0)
    };
    Some(FieldInfo { dims: [nx, ny, nz], step: f.step, nodes: nx * ny * nz, neg, thick_vox })
}

/// Коробка поля тела в мире: (центр, полуразмеры). Зазор меряется по ней —
/// радиус описанной сферы у вытянутого осколка бесполезен.
fn fbox(b: &LiveBody) -> Option<([f32; 3], [f32; 3])> {
    let f = b.sdf.as_ref()?;
    let half = [
        (f.dims[0] as f32 - 1.0) * f.step * 0.5,
        (f.dims[1] as f32 - 1.0) * f.step * 0.5,
        (f.dims[2] as f32 - 1.0) * f.step * 0.5,
    ];
    let c = [
        b.pos[0] as f32 + b.sdf_shift[0] + half[0],
        b.pos[1] as f32 + b.sdf_shift[1] + half[1],
        b.pos[2] as f32 + b.sdf_shift[2] + half[2],
    ];
    Some((c, half))
}

/// Расстояние от точки до коробки (0 внутри).
fn dist_box(p: [f32; 3], c: [f32; 3], h: [f32; 3]) -> f32 {
    let mut d = [0.0f32; 3];
    let mut any = false;
    for k in 0..3 {
        let v = (p[k] - c[k]).abs() - h[k];
        if v > 0.0 {
            d[k] = v;
            any = true;
        }
    }
    if any {
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    } else {
        0.0
    }
}

/// Годный для пробы осколок: не вырожденный (у тела в два тета обратная масса
/// уходит в бесконечность, и NaN приходит из солвера, а не из CCD).
fn valid(b: &LiveBody) -> bool {
    b.kind == Kind::Shard && b.tets.len() >= 60 && b.radius.is_finite() && b.radius > 1e-3
}

/// Поставить тело в позу напрямую (CPU-зеркало + буфер устройства).
fn place(w: &mut Live, i: usize, pos: [f32; 3], vel: [f32; 3]) {
    w.bodies[i].pos = [pos[0] as f64, pos[1] as f64, pos[2] as f64];
    w.bodies[i].vel = [vel[0] as f64, vel[1] as f64, vel[2] as f64];
    let m = &mut w.mirror[i];
    m.pos_r = [pos[0], pos[1], pos[2], m.pos_r[3]];
    m.prev_pos_r = [pos[0], pos[1], pos[2], m.prev_pos_r[3]];
    m.vel_m = [vel[0], vel[1], vel[2], m.vel_m[3]];
    m.misc[1] = 0.0;
    m.anchor = [pos[0], pos[1], pos[2], 0.0];
    w.gpu.write_bodies(&w.mirror);
}

/// Убрать тело из мира (как `Live::despawn`): далеко вниз и спящим.
fn hide(w: &mut Live, i: usize) {
    w.bodies[i].alive = false;
    let m = &mut w.mirror[i];
    m.pos_r = [0.0, -400.0, 0.0, m.pos_r[3]];
    m.prev_pos_r = m.pos_r;
    m.vel_m = [0.0; 4];
    m.ang_i = [0.0; 4];
    m.misc[1] = 1.0;
    m.anchor = [0.0, -400.0, 0.0, 0.0];
}

fn sync_mirror(w: &mut Live) {
    let snap = w.gpu.read_bodies();
    for (i, g) in snap.iter().enumerate() {
        if i < w.mirror.len() {
            w.mirror[i] = *g;
        }
    }
}

/// Сообщить такту максимальную скорость: проба ставит скорости напрямую, а
/// порог CCD берёт vmax из статистики устройства (прошлый такт). Публичный путь
/// один — выстрел, который пересчитывает vmax; снаряд тут же убираем.
fn poke_vmax(w: &mut Live) {
    if w.shoot([7.0, 9.0, -7.0], [1.0, 0.0, 0.0], 0.01) {
        if let Some(slot) = w.bullets.back().copied() {
            hide(w, slot);
            w.bullets.pop_back();
        }
    }
    w.gpu.write_bodies(&w.mirror);
}

/// Дробим плиту три поколения подряд. Ни одного такта между дроблениями:
/// позы осколков считаются только из позы родителя, и сцена воспроизводится
/// от запуска к запуску (иначе атомики GPU дают разные позы, а за ними и
/// разные осколки — таблицы до/после перестают быть про одно и то же).
fn build_shards(w: &mut Live) -> Vec<usize> {
    sync_mirror(w);
    let p = w.bodies[0].pos;
    let _ = w.split(0, [p[0] as f32 + 0.2, p[1] as f32 + 0.3, p[2] as f32], 9000.0, 6);
    sync_mirror(w);
    /* Ещё три поколения: каждый шаг — три самых крупных куска на 4 части.
     * Так доходим до осколков в 15–25 см с полем в 10–20 узлов по оси. */
    for _ in 0..3 {
        let mut big: Vec<usize> = (0..w.bodies.len())
            .filter(|&i| w.bodies[i].alive && w.bodies[i].kind == Kind::Shard)
            .collect();
        big.sort_by_key(|&i| std::cmp::Reverse(w.bodies[i].tets.len()));
        for &bi in big.iter().take(3) {
            let p = w.bodies[bi].pos;
            let _ = w.split(bi, [p[0] as f32 + 0.05, p[1] as f32 + 0.05, p[2] as f32], 4000.0, 4);
            sync_mirror(w);
        }
    }
    (0..w.bodies.len()).filter(|&i| w.bodies[i].alive).collect()
}

fn print_table(w: &Live, slots: &[usize]) {
    println!(
        "{:>4} {:>5} {:>10} {:>8} {:>8} {:>7} {:>5} {:>9} {:>9} {:>9}",
        "слот", "тетов", "поле", "шаг,мм", "узлов", "внутри", "тонк", "толщ,вокс", "обвод,м", "поле,м"
    );
    for &i in slots {
        let b = &w.bodies[i];
        let Some(fi) = field_info(b) else {
            println!("{i:>4} {:>5}   поля нет", b.tets.len());
            continue;
        };
        println!(
            "{:>4} {:>5} {:>4}×{:>2}×{:>2} {:>8.2} {:>8} {:>7} {:>5} {:>9} {:>9.3} {:>9.3}",
            i,
            b.tets.len(),
            fi.dims[0],
            fi.dims[1],
            fi.dims[2],
            fi.step * 1e3,
            fi.nodes,
            fi.neg,
            if fi.thick_vox < 3 { "да" } else { "нет" },
            fi.thick_vox,
            2.0 * b.radius,
            (fi.dims[0] as f32 - 1.0) * fi.step * 0.5,
        );
    }
}

fn main() {
    let sh = find_shaders(None).unwrap();
    let speeds: Vec<f64> = std::env::var("VF_SPEEDS")
        .unwrap_or_else(|_| "5,20,60,120,240".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let ticks = env_u32("VF_TICKS", 8) as usize;
    let dbg = std::env::var("VF_DBG").is_ok();
    let dt = 1.0 / 20.0;
    println!(
        "режим {} · гравитация ВЫКЛ · такт {:.0} мс · тактов на выстрел ≥ {ticks}",
        std::env::var("VF_COLLIDE").unwrap_or_else(|_| "по умолчанию".into()),
        dt * 1e3
    );

    let mut w = Live::new(&sh);
    w.set_fracture(false);
    w.params.gravity = 0.0;
    w.gpu.params.gravity = 0.0;
    let slots = build_shards(&mut w);
    println!("\n== осколки после трёх дроблений ({} живых) ==", slots.len());
    print_table(&w, &slots);

    /* ---------- снаряд в МЕЛКИЙ осколок ---------- */
    let target = slots
        .iter()
        .copied()
        .filter(|&i| valid(&w.bodies[i]))
        .min_by(|&a, &b| w.bodies[a].radius.partial_cmp(&w.bodies[b].radius).unwrap())
        .expect("годные осколки");
    for &i in &slots {
        if i != target {
            hide(&mut w, i);
        }
    }
    w.gpu.write_bodies(&w.mirror);
    let ti = field_info(&w.bodies[target]);
    println!(
        "\n== цель: слот {target} · тетов {} · обвод {:.3} м · поле {}×{}×{} шаг {:.2} мм · узлов {} · внутри {} · толщина {} вокс ==",
        w.bodies[target].tets.len(),
        2.0 * w.bodies[target].radius,
        ti.as_ref().map(|f| f.dims[0]).unwrap_or(0),
        ti.as_ref().map(|f| f.dims[1]).unwrap_or(0),
        ti.as_ref().map(|f| f.dims[2]).unwrap_or(0),
        ti.as_ref().map(|f| f.step * 1e3).unwrap_or(0.0),
        ti.as_ref().map(|f| f.nodes).unwrap_or(0),
        ti.as_ref().map(|f| f.neg).unwrap_or(0),
        ti.as_ref().map(|f| f.thick_vox).unwrap_or(0),
    );
    println!("\nскорость · шаг/подшаг · исход · зазор до коробки · глубина · x_отн · vx_отн · Δv цели(max) · ccd · ccd мс макс · ω цели(max) · контакты");

    for &speed in &speeds {
        place(&mut w, target, [0.0, 5.0, 0.0], [0.0; 3]);
        sync_mirror(&mut w);
        poke_vmax(&mut w);
        let ok = w.shoot([-3.0, 5.0, 0.0], [1.0, 0.0, 0.0], speed);
        assert!(ok, "снаряд не создался");
        let slot = *w.bullets.back().unwrap();
        w.gpu.wait_idle();
        sync_mirror(&mut w);

        let bb = fbox(&w.bodies[target]);
        let (mut min_gap, mut max_pen) = (f32::MAX, 0.0f32);
        let (mut ccd_hits, mut ccd_ms_max, mut hit_contacts) = (0u32, 0.0f32, 0usize);
        let target_v0 = w.gpu.read_bodies()[target].vel_m;
        let (mut dv_max, mut spin, mut crossed) = (0.0f32, 0.0f32, false);
        /* Медленному снаряду нужно больше тактов, чтобы вообще долететь. */
        let n_ticks = (((3.5 / (speed * dt as f64)) as usize) + 2).max(ticks);
        for t in 0..n_ticks {
            w.step(dt, 8, 12);
            w.gpu.wait_idle();
            let snap = w.gpu.read_bodies();
            let b = snap[slot];
            let tar = snap[target];
            if let Some((c, h)) = bb {
                let p = [b.pos_r[0], b.pos_r[1], b.pos_r[2]];
                min_gap = min_gap.min(dist_box(p, c, h) - 0.14);
                if (0..3).all(|k| (p[k] - c[k]).abs() <= h[k]) {
                    crossed = true;
                }
            }
            let cs = w.gpu.read_contacts();
            let n_c = cs.get(slot).map(|v| v.len()).unwrap_or(0);
            if let Some(v) = cs.get(slot) {
                if v.iter().any(|c| c.other == target as i32) {
                    hit_contacts += 1;
                }
                max_pen = max_pen.max(v.iter().map(|c| c.pen).fold(0.0f32, f32::max));
            }
            let dv = {
                let v = tar.vel_m;
                ((v[0] - target_v0[0]).powi(2)
                    + (v[1] - target_v0[1]).powi(2)
                    + (v[2] - target_v0[2]).powi(2))
                .sqrt()
            };
            dv_max = dv_max.max(dv);
            ccd_hits += w.last_ccd;
            ccd_ms_max = ccd_ms_max.max(w.last_ccd_ms);
            spin = spin.max(
                (tar.ang_i[0] * tar.ang_i[0] + tar.ang_i[1] * tar.ang_i[1] + tar.ang_i[2] * tar.ang_i[2])
                    .sqrt(),
            );
            if dbg {
                println!(
                    "  t={t} x_rel={:.3} vx_rel={:.2} зазор={min_gap:.3} контактов {n_c} ccd {} · {:.2} мс",
                    b.pos_r[0] - tar.pos_r[0],
                    b.vel_m[0] - tar.vel_m[0],
                    w.last_ccd,
                    w.last_ccd_ms
                );
            }
        }
        let snap = w.gpu.read_bodies();
        let x_rel = snap[slot].pos_r[0] - snap[target].pos_r[0];
        let vx_rel = snap[slot].vel_m[0] - snap[target].vel_m[0];
        let outcome = if !w.bodies[slot].alive {
            "исчез"
        } else if !x_rel.is_finite() {
            "NaN!"
        } else if dv_max > 0.5 || hit_contacts > 0 {
            "столкнулся"
        } else if crossed {
            "ПРОЛЕТЕЛ"
        } else {
            "не дошёл"
        };
        println!(
            "{speed:6.0} · {:5.2} м · {outcome:11} · {min_gap:8.3} · {max_pen:.4} · x_отн={x_rel:7.3} · vx_отн={vx_rel:8.2} · Δv {dv_max:6.2} · ccd {ccd_hits:2} ({ccd_ms_max:5.2} мс) · ω {spin:6.2} · конт.цели {hit_contacts}",
            speed * dt as f64 / 8.0
        );
        /* Снаряд убираем из мира: иначе на следующей скорости он остаётся
         * висеть у цели и ловит новый снаряд (NaN и мусор в таблице). */
        hide(&mut w, slot);
        w.gpu.write_bodies(&w.mirror);
        w.bullets.pop_back();
    }

    /* ---------- ПОЛ ---------- */
    println!("\n== пол: тело летит вниз (пол y=0, старт 4 м) ==");
    println!("тело · скорость · y после · пролетел пол?");
    for &speed in &speeds {
        for kind in ["шар", "мелкий", "крупный"] {
            w.reset();
            w.set_fracture(false);
            w.params.gravity = 0.0;
            w.gpu.params.gravity = 0.0;
            let slots = build_shards(&mut w);
            if kind == "шар" {
                for &i in &slots {
                    hide(&mut w, i);
                }
                w.gpu.write_bodies(&w.mirror);
                assert!(w.shoot([0.0, 4.0, 0.0], [0.0, -1.0, 0.0], speed));
                let slot = *w.bullets.back().unwrap();
                sync_mirror(&mut w);
                poke_vmax(&mut w);
                for _ in 0..ticks {
                    w.step(dt, 8, 12);
                    w.gpu.wait_idle();
                }
                let y = w.gpu.read_bodies()[slot].pos_r[1];
                println!("{kind:7} · {speed:6.0} · y={y:8.3} · {}", if y < -0.05 { "ДА" } else { "нет" });
                continue;
            }
            let mut cand: Vec<usize> = slots.iter().copied().filter(|&i| valid(&w.bodies[i])).collect();
            if cand.is_empty() {
                continue;
            }
            cand.sort_by(|&a, &b| w.bodies[a].radius.partial_cmp(&w.bodies[b].radius).unwrap());
            let ti = if kind == "мелкий" { cand[0] } else { cand[cand.len() - 1] };
            for &i in &slots {
                if i != ti {
                    hide(&mut w, i);
                }
            }
            place(&mut w, ti, [0.0, 4.0, 0.0], [0.0, -speed as f32, 0.0]);
            sync_mirror(&mut w);
            poke_vmax(&mut w);
            w.last_awake = 8;
            for _ in 0..ticks {
                w.step(dt, 8, 12);
                w.gpu.wait_idle();
            }
            let y = w.gpu.read_bodies()[ti].pos_r[1];
            println!(
                "{kind:7} · {speed:6.0} · y={y:8.3} · {} · обвод {:.2} м",
                if y < -0.05 { "ДА" } else { "нет" },
                2.0 * w.bodies[ti].radius
            );
        }
    }

    /* ---------- мелкий осколок в мелкий осколок ---------- */
    println!("\n== мелкий осколок в мелкий осколок (центры в 1 м) ==");
    println!("скорость · исход · зазор до коробки · x_отн · Δv цели(max) · ccd · ccd мс макс · обводы");
    for &speed in &speeds {
        w.reset();
        w.set_fracture(false);
        w.params.gravity = 0.0;
        w.gpu.params.gravity = 0.0;
        let slots = build_shards(&mut w);
        let mut small: Vec<usize> = slots.iter().copied().filter(|&i| valid(&w.bodies[i])).collect();
        small.sort_by(|&a, &b| w.bodies[a].radius.partial_cmp(&w.bodies[b].radius).unwrap());
        if small.len() < 2 {
            println!("{speed:6.0} · осколков меньше двух");
            continue;
        }
        let (a, b) = (small[0], small[1]);
        for &i in &slots {
            if i != a && i != b {
                hide(&mut w, i);
            }
        }
        place(&mut w, a, [-0.5, 5.0, 0.0], [speed as f32, 0.0, 0.0]);
        place(&mut w, b, [0.5, 5.0, 0.0], [0.0; 3]);
        sync_mirror(&mut w);
        poke_vmax(&mut w);
        w.last_awake = 8;
        let rs = w.gpu.read_bodies();
        let rb0 = rs[b].vel_m;
        let bb = fbox(&w.bodies[b]);
        let (mut min_gap, mut ccd, mut ccd_ms) = (f32::MAX, 0u32, 0.0f32);
        let (mut dv_max, mut crossed) = (0.0f32, false);
        for _ in 0..ticks {
            w.step(dt, 8, 12);
            w.gpu.wait_idle();
            let snap = w.gpu.read_bodies();
            if let Some((c, h)) = bb {
                let p = [snap[a].pos_r[0], snap[a].pos_r[1], snap[a].pos_r[2]];
                min_gap = min_gap.min(dist_box(p, c, h));
                if (0..3).all(|k| (p[k] - c[k]).abs() <= h[k]) {
                    crossed = true;
                }
            }
            let v = snap[b].vel_m;
            dv_max = dv_max.max(
                ((v[0] - rb0[0]).powi(2) + (v[1] - rb0[1]).powi(2) + (v[2] - rb0[2]).powi(2)).sqrt(),
            );
            ccd += w.last_ccd;
            ccd_ms = ccd_ms.max(w.last_ccd_ms);
        }
        let snap = w.gpu.read_bodies();
        let outcome = if !snap[a].pos_r[0].is_finite() {
            "NaN!"
        } else if dv_max > 0.5 {
            "столкнулся"
        } else if crossed {
            "ПРОЛЕТЕЛ"
        } else {
            "не дошёл"
        };
        println!(
            "{speed:6.0} · {outcome:11} · {min_gap:8.3} · x_отн={:7.3} · Δv {dv_max:6.2} · ccd {ccd:2} ({ccd_ms:5.2} мс) · {:.2}+{:.2} м",
            snap[a].pos_r[0] - snap[b].pos_r[0],
            2.0 * w.bodies[a].radius,
            2.0 * w.bodies[b].radius
        );
    }
}
