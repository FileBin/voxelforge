// ЖЁСТКОСТЬ КОНТАКТОВ: таблица по тактам для тет-пути и SDF-пути.
//
//   VF_COLLIDE=gpu probe_hardness   — SAT по тетам (прежний рабочий путь)
//   VF_COLLIDE=sdf probe_hardness   — контакты по запечённым полям
//
// Сцены (все на одной и той же плите 1.12 м из `scene_plate`):
//   drop    — шар r=0.14 падает с 2 м на плиту (удар ~6.3 м/с);
//   cube    — куб 0.28 м падает с 2 м на плиту (куб на куб);
//   ballball— шар на 20 м/с влетает в лежащий на плите шар (шар в шар);
//   wall45  — шар на 12 м/с под 45° в стену мира (x = +8).
//
// По такту печатается: центр по Y, скорость, число контактов тела, число пар
// широкого прохода, максимальная глубина из контактов и ПРОСАДКА (насколько
// низ тела ушёл ниже НУЛЯ ПОЛЯ плиты — геометрическая мера, не зависящая от
// того, что движок сам про себя думает). Плюс строка «ИТОГ» по сцене.
//
// Просадка считается по полю, а не по вершинам тетов: поверхность физики —
// изоповерхность запечённого SDF (меш тоже строится по ней).
use voxelforge_world::live::{Kind, Live, LiveBody};
use voxelforge_world::sim::{find_shaders, GpuBody};
use voxelforge_world::tet::TetBody;

const DT: f32 = 0.05;
const SUB: u32 = 8;
const IT: u32 = 12;

/// Значение поля тела в мировой точке (МЕТРЫ, минус внутри).
fn field_m(b: &LiveBody, p: [f64; 3]) -> Option<f32> {
    let f = b.sdf.as_ref()?;
    /* обратный поворот: q = (w, x, y, z), v' = v - 2w(u×v) + 2u×(u×v) */
    let q = b.quat;
    let d = [p[0] - b.pos[0], p[1] - b.pos[1], p[2] - b.pos[2]];
    let u = [q[1], q[2], q[3]];
    let w = q[0];
    let c = [
        u[1] * d[2] - u[2] * d[1],
        u[2] * d[0] - u[0] * d[2],
        u[0] * d[1] - u[1] * d[0],
    ];
    let cc = [
        u[1] * c[2] - u[2] * c[1],
        u[2] * c[0] - u[0] * c[2],
        u[0] * c[1] - u[1] * c[0],
    ];
    let local = [
        d[0] - 2.0 * w * c[0] + 2.0 * cc[0] - b.sdf_shift[0] as f64,
        d[1] - 2.0 * w * c[1] + 2.0 * cc[1] - b.sdf_shift[1] as f64,
        d[2] - 2.0 * w * c[2] + 2.0 * cc[2] - b.sdf_shift[2] as f64,
    ];
    let step = f.step as f64;
    Some(f.sample_m([
        (local[0] / step) as f32,
        (local[1] / step) as f32,
        (local[2] / step) as f32,
    ]))
}

/// Верх поверхности плиты (ноль поля) под точкой (x, z): идём СВЕРХУ вниз и
/// берём первое пересечение нуля в отрицательную сторону (снизу вверх было бы
/// первое попадание в блок поля, а не в тело).
fn surface_y(b: &LiveBody, x: f64, z: f64) -> Option<f64> {
    let mut y = b.pos[1] + 0.9;
    let y0 = b.pos[1] - 0.9;
    let mut prev = field_m(b, [x, y, z])?;
    let h = 0.002;
    while y > y0 {
        y -= h;
        let cur = field_m(b, [x, y, z])?;
        if prev >= 0.0 && cur < 0.0 {
            let t = prev / (prev - cur);
            return Some(y + h - h * t as f64);
        }
        prev = cur;
    }
    None
}

fn speed(g: &GpuBody) -> f64 {
    let v = g.vel_m;
    ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as f64).sqrt()
}

struct Sum {
    max_sink: f64,
    sink_in_contact: Vec<f64>,
    max_pen: f32,
    zero_contact_ticks: usize,
    first_contact: Option<usize>,
    stopped: Option<usize>,
    max_speed_after_impact: f64,
    through: bool,
}

fn report(name: &str, mode: &str, s: &Sum, extra: &str) {
    let mean_sink = if s.sink_in_contact.is_empty() {
        0.0
    } else {
        s.sink_in_contact.iter().sum::<f64>() / s.sink_in_contact.len() as f64
    };
    println!(
        "[{name}] ИТОГ {mode}: просадка max {:+.4} м · просадка сред {:+.4} м · глубина контактов max {:.4} м · тактов без контактов {} · первый контакт {:?} · остановка {:?} · |v| после удара {:.2} м/с · СКВОЗЬ {} {extra}",
        s.max_sink,
        mean_sink,
        s.max_pen,
        s.zero_contact_ticks,
        s.first_contact,
        s.stopped,
        s.max_speed_after_impact,
        if s.through { "ДА" } else { "нет" },
    );
}

fn main() {
    let mode = std::env::var("VF_COLLIDE").unwrap_or_else(|_| "gpu".into());
    let sh = find_shaders(None).unwrap();

    /* ── drop и cube: падение с 2 м ───────────────────────────────────── */
    for scene in ["drop", "cube"] {
        let mut w = Live::new(&sh);
        w.set_fracture(false);
        w.set_sleep(false);
        for _ in 0..4 {
            w.step(DT, SUB, IT);
        }
        w.gpu.wait_idle();
        let top = surface_y(&w.bodies[0], 0.0, 0.0).expect("поверхность плиты не найдена");
        let half = if scene == "drop" { 0.14 } else { 0.14 };
        let slot = if scene == "drop" {
            /* шар r=0.14 — тот же, что снаряд: путь через shoot, чтобы сцена
             * была ровно та же, что в живом мире */
            w.shoot([0.0, top + 0.14 + 2.0, 0.0], [0.0, 0.0, 0.0], 0.0);
            *w.bullets.back().unwrap()
        } else {
            let cube = TetBody::cube([0.0, 0.0, 0.0], 2, 1);
            w.spawn_tet(&cube, [0.0, top + half + 2.0, 0.0], [0.0; 3], Kind::Shard)
                .expect("нет свободного слота")
        };
        let mut s = Sum {
            max_sink: f64::MIN,
            sink_in_contact: Vec::new(),
            max_pen: 0.0,
            zero_contact_ticks: 0,
            first_contact: None,
            stopped: None,
            max_speed_after_impact: 0.0,
            through: false,
        };
        println!("── сцена {scene} · режим {mode} · поверхность плиты {top:.4} м ──");
        println!("такт |      y |     vy |   |v| | конт | пар |   pen  | просадка");
        let mut last_y = 0.0;
        let mut prev_contacts = false;
        let mut speeds_after = Vec::new();
        for t in 0..30 {
            w.step(DT, SUB, IT);
            w.gpu.wait_idle();
            let snap = w.gpu.read_bodies();
            let cs = w.gpu.read_contacts();
            let pairs = w.gpu.read_sdf_pairs().len();
            let mine: Vec<_> = cs[slot].iter().filter(|c| c.other == 0).collect();
            let pen = mine.iter().map(|c| c.pen).fold(0.0f32, f32::max);
            let n = mine.len();
            let g = &snap[slot];
            let y = g.pos_r[1] as f64;
            let vy = g.vel_m[1] as f64;
            let sp = speed(g);
            let sy = surface_y(&w.bodies[0], 0.0, 0.0).unwrap_or(top);
            let sink = sy - (y - half);
            if !mine.is_empty() {
                s.first_contact.get_or_insert(t);
                s.sink_in_contact.push(sink);
            } else if prev_contacts {
                s.zero_contact_ticks += 1;
            }
            if s.first_contact.is_some() && t >= s.first_contact.unwrap() {
                speeds_after.push(sp);
                if sp > s.max_speed_after_impact {
                    s.max_speed_after_impact = sp;
                }
            }
            s.max_sink = s.max_sink.max(sink);
            s.max_pen = s.max_pen.max(pen);
            last_y = y;
            prev_contacts = !mine.is_empty();
            println!(
                "{t:>4} | {y:>6.3} | {vy:>6.2} | {sp:>4.1} | {n:>4} | {pairs:>3} | {pen:>6.4} | {sink:>+8.4}"
            );
        }
        s.through = last_y - half < top - 0.1;
        /* остановка: первый такт, с которого |v| < 0.1 до конца */
        let mut st = None;
        let tail: Vec<f64> = speeds_after.clone();
        for i in 0..tail.len() {
            if tail[i..].iter().all(|&v| v < 0.1) {
                st = Some(s.first_contact.unwrap_or(0) + i);
                break;
            }
        }
        s.stopped = st;
        report(scene, &mode, &s, &format!("· y в конце {last_y:.3}"));
    }

    /* ── ballball: шар 20 м/с в лежащий шар ───────────────────────────── */
    {
        let mut w = Live::new(&sh);
        w.set_fracture(false);
        w.set_sleep(false);
        for _ in 0..4 {
            w.step(DT, SUB, IT);
        }
        w.gpu.wait_idle();
        let top = surface_y(&w.bodies[0], 0.0, 0.0).expect("поверхность плиты не найдена");
        let target = w
            .spawn_tet(
                &TetBody::ball([0.0, 0.0, 0.0], 0.14, 0),
                [0.0, top + 0.14, 0.0],
                [0.0; 3],
                Kind::Shard,
            )
            .unwrap();
        for _ in 0..4 {
            w.step(DT, SUB, IT);
        }
        w.gpu.wait_idle();
        let tgt = w.gpu.read_bodies()[target];
        println!(
            "  цель: поза {:?} · скорость {:?} · спит {}",
            tgt.pos_r, tgt.vel_m, tgt.misc[1] > 0.5
        );
        w.shoot([-0.6, top + 0.14, 0.0], [1.0, 0.0, 0.0], 20.0);
        let striker = *w.bullets.back().unwrap();
        let mut s = Sum {
            max_sink: f64::MIN,
            sink_in_contact: Vec::new(),
            max_pen: 0.0,
            zero_contact_ticks: 0,
            first_contact: None,
            stopped: None,
            max_speed_after_impact: 0.0,
            through: false,
        };
        println!("── сцена ballball · режим {mode} · лежащий шар в слоте {target} ──");
        println!("такт |      x |    |v| | конт | конт цели | пар |   pen  | Δv цели");
        let mut target_speed_before = 0.0;
        let mut target_dv = 0.0f64;
        let mut max_x = f64::MIN;
        for t in 0..16 {
            w.step(DT, SUB, IT);
            w.gpu.wait_idle();
            let snap = w.gpu.read_bodies();
            let cs = w.gpu.read_contacts();
            let pairs = w.gpu.read_sdf_pairs().len();
            let mine: Vec<_> = cs[striker].iter().filter(|c| c.other == target as i32).collect();
            let theirs: Vec<_> = cs[target].iter().filter(|c| c.other == striker as i32).collect();
            let pen = mine.iter().map(|c| c.pen).fold(0.0f32, f32::max);
            let g = &snap[striker];
            let x = g.pos_r[0] as f64;
            let sp = speed(g);
            let tsp = speed(&snap[target]);
            if !theirs.is_empty() && s.first_contact.is_none() {
                s.first_contact = Some(t);
            }
            if s.first_contact.is_some() {
                target_dv = target_dv.max(tsp - target_speed_before);
                s.max_speed_after_impact = s.max_speed_after_impact.max(sp);
            }
            target_speed_before = tsp;
            if !mine.is_empty() {
                s.sink_in_contact.push(pen as f64);
            }
            s.max_pen = s.max_pen.max(pen);
            max_x = max_x.max(x);
            println!(
                "{t:>4} | {x:>6.3} | {sp:>4.1} | {n:>4} | {tn:>9} | {pairs:>3} | {pen:>6.4} | {target_dv:>6.2}",
                n = mine.len(),
                tn = theirs.len()
            );
        }
        let _ = max_x;
        println!(
            "[ballball] ИТОГ {mode}: первый контакт {:?} · глубина max {:.4} м · Δv цели {:.2} м/с (аналитика для равных масс 10) · тактов без контактов {}",
            s.first_contact,
            s.max_pen,
            target_dv,
            s.zero_contact_ticks
        );
    }

    /* ── wall45: шар под 45° в стену ──────────────────────────────────── */
    {
        let mut w = Live::new(&sh);
        w.set_fracture(false);
        w.set_sleep(false);
        for _ in 0..4 {
            w.step(DT, SUB, IT);
        }
        w.gpu.wait_idle();
        let d = 1.0 / 2.0f64.sqrt();
        w.shoot([6.0, 3.0, 0.0], [d, -d, 0.0], 12.0);
        let slot = *w.bullets.back().unwrap();
        let wall = 8.0f64;
        println!("── сцена wall45 · режим {mode} · стена x = {wall} ──");
        println!("такт |      x |     vy | конт | пар |   pen  | просадка стены");
        let mut max_pen = 0.0f32;
        let mut max_sink = f64::MIN;
        let mut contacts = 0usize;
        let mut worst_x = f64::MIN;
        for t in 0..24 {
            w.step(DT, SUB, IT);
            w.gpu.wait_idle();
            let snap = w.gpu.read_bodies();
            let cs = w.gpu.read_contacts();
            let pairs = w.gpu.read_sdf_pairs().len();
            let mine: Vec<_> = cs[slot].iter().filter(|c| c.other == -1).collect();
            let pen = mine.iter().map(|c| c.pen).fold(0.0f32, f32::max);
            let g = &snap[slot];
            let x = g.pos_r[0] as f64;
            let vy = g.vel_m[1] as f64;
            if !mine.is_empty() {
                contacts += 1;
            }
            max_pen = max_pen.max(pen);
            worst_x = worst_x.max(x);
            let sink = x - (wall - 0.14);
            max_sink = max_sink.max(sink);
            println!(
                "{t:>4} | {x:>6.3} | {vy:>6.2} | {n:>4} | {pairs:>3} | {pen:>6.4} | {sink:>+8.4}",
                n = mine.len()
            );
        }
        println!(
            "[wall45] ИТОГ {mode}: x max {worst_x:.3} (стена {wall}) · просадка стены max {max_sink:+.4} м · глубина контактов max {max_pen:.4} м · тактов с контактом {contacts}"
        );
    }
}
