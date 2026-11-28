//! `voxelforge` — движок. Один запуск, один мир, только BCC-ячейки.
//!
//! Мир: воксельная плита на полу и воксельный снаряд, который в неё влетает.
//! От удара плита раскалывается на неровные осколки (Вороной по центрам
//! ячеек), а потом каждый осколок **дробит свои ячейки**: крупные BCC-ячейки
//! (`base`) становятся вдвое мельче, затем ещё вдвое — прямо в полёте.
//! Объём и масса при дроблении сохраняются точно.
//!
//! Считается на видеокарте (Vulkan compute через C-ядро), CPU только собирает
//! сцену и пишет `.vfw`. Смотреть: `nix run .#viewer`.

use std::path::PathBuf;

use voxelforge_world::bcc::BASE;
use voxelforge_world::collide;
use voxelforge_world::tet::{split_surfaces, Lcg, TetBody};
use voxelforge_world::sim::{find_shaders, GpuBody, GpuSim, Params};
use voxelforge_world::types::Body;
use voxelforge_world::vfw::{self, BodyFrame, Shape};

struct Args {
    frames: u32,
    substeps: u32,
    iterations: u32,
    out: Option<PathBuf>,
    shaders: Option<PathBuf>,
    quiet: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        frames: 420,
        substeps: 8,
        iterations: 14,
        out: None,
        shaders: None,
        quiet: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().expect("нужно значение аргумента");
        match k.as_str() {
            "--frames" => a.frames = v().parse().unwrap(),
            "--substeps" => a.substeps = v().parse().unwrap(),
            "--iterations" => a.iterations = v().parse().unwrap(),
            "--out" => a.out = Some(v().into()),
            "--shaders" => a.shaders = Some(v().into()),
            "--quiet" => a.quiet = true,
            "--help" | "-h" => {
                println!("voxelforge — мир BCC-вокселей");
                println!("  --frames N --substeps N --iterations N --out FILE --shaders DIR --quiet");
                std::process::exit(0);
            }
            other => panic!("неизвестный аргумент {other}"),
        }
    }
    a
}

/// Плотность материала, кг/м³ (лёгкий камень).
const DENSITY: f64 = 420.0;
/// Сколько слотов заранее выделено под осколки: тела появляются при ударе,
/// но буфер на GPU фиксированного размера, поэтому слоты резервируются.
const MAX_FRAGS: usize = 12;

/// Тело физики по воксельному куску.
fn body_of(frag: &TetBody) -> Body {
    let c = frag.world_centroid();
    let (lo, hi) = frag.bounds();
    let half = [
        0.5 * (hi[0] - lo[0]) as f64,
        0.5 * (hi[1] - lo[1]) as f64,
        0.5 * (hi[2] - lo[2]) as f64,
    ];
    let mut b = Body::block([c[0] as f64, c[1] as f64, c[2] as f64], half);
    b.mass = DENSITY * frag.volume();
    b
}

/// Форма для вьюера: меш марчинга + каркас тетраэдров + решётка.
fn shape_of(frag: &TetBody) -> Shape {
    vfw::shape_of(frag)
}

fn main() {
    let args = parse_args();

    // ---- мир ----------------------------------------------------------
    // Плита ЦЕЛАЯ: 4×4×4 кубические ячейки, 768 тетраэдров. Раскола нет,
    // пока в неё не попали — это и есть настоящая физика разрушения.
    let plate = TetBody::cube([0.0, 1.0, 0.0], 4, 0);
    let plate_vol = plate.volume();
    let ball = TetBody::ball([-3.2, 1.0, 0.0], 0.30, 1);
    let ball_vol = ball.volume();

    let ball_speed: f64 = std::env::var("VF_SPEED").ok().and_then(|v| v.parse().ok()).unwrap_or(34.0);
    let ball_density: f64 = std::env::var("VF_BDENS").ok().and_then(|v| v.parse().ok()).unwrap_or(1200.0);

    // формы: 0 — целая плита, 1 — снаряд; формы осколков добавятся при ударе
    let mut shapes: Vec<Shape> = vec![shape_of(&plate), shape_of(&ball)];
    let mut shape_id: Vec<u32> = vec![0; MAX_FRAGS];
    let ball_slot = MAX_FRAGS;
    let ball_shape = 1u32;

    // ---- тела ---------------------------------------------------------
    let mut bodies: Vec<GpuBody> = Vec::with_capacity(MAX_FRAGS + 1);
    let mut tet_shapes: Vec<Vec<[[f32; 3]; 4]>> = Vec::with_capacity(MAX_FRAGS + 1);
    let mut masses: Vec<f64> = Vec::with_capacity(MAX_FRAGS + 1);

    let whole = body_of(&plate);
    masses.push(whole.mass);
    bodies.push(GpuBody::from_cpu(&whole));
    tet_shapes.push(plate.surface_tets());

    // зарезервированные слоты осколков: припаркованы далеко и спят
    for _ in 1..MAX_FRAGS {
        let parked = Body::new([0.0, -300.0, 0.0], [0.0, 0.0, 0.0], 1.0, 0.05);
        let mut g = GpuBody::from_cpu(&parked);
        g.misc[1] = 1.0;
        bodies.push(g);
        tet_shapes.push(Vec::new());
        masses.push(1.0);
    }

    let mut proj = Body::new([-3.2, 1.0, 0.0], [0.0, 0.0, 0.0], ball_density * ball_vol, 0.30);
    proj.vel = [ball_speed, 0.0, 0.0];
    masses.push(proj.mass);
    bodies.push(GpuBody::from_cpu(&proj));
    tet_shapes.push(ball.surface_tets());

    let max_half = bodies
        .iter()
        .map(|b| b.half_shape[0].max(b.half_shape[1]).max(b.half_shape[2]))
        .fold(0.0f32, f32::max);
    let params = Params {
        lo: [-5.0, 0.0, -5.0],
        hi: [5.0, 8.0, 5.0],
        cell: (2.0 * max_half * 1.05).max(0.18),
        friction: 0.32,
        max_push: 0.35,
        ..Params::default()
    };

    let shaders = find_shaders(args.shaders.as_deref()).expect(
        "нет SPIR-V: соберите shaders/*.comp (nix build .#spirv) или укажите --shaders",
    );
    let mut gpu = GpuSim::new(bodies, &shaders, params);

    if !args.quiet {
        println!("voxelforge · {} · шаг решётки {:.3} м", gpu.name, BASE);
        println!(
            "мир: плита {:.3} м³ ({} тетраэдров, ЦЕЛАЯ), снаряд {:.3} м³",
            plate_vol,
            plate.tets.len(),
            ball_vol
        );
        println!(
            "масса плиты {:.1} кг, снаряда {:.1} кг",
            DENSITY * plate_vol,
            ball_density * ball_vol
        );
    }

    // ---- прогон --------------------------------------------------------
    let mut frames_out: Vec<Vec<BodyFrame>> = Vec::with_capacity(args.frames as usize);
    let mut log: Vec<String> = Vec::new();
    let mut broke = false;
    let mut active = 1usize; // живых слотов осколков (сейчас — одна плита)
    let mut snap = gpu.read_bodies();

    for f in 0..args.frames {
        let mut hit_now = false;
        // снимок на начало кадра: к моменту раскола солвер уже погасил
        // скорость снаряда, а энергия удара — это скорость ДО столкновения
        let snap0 = snap.clone();
        // контакты по тетраэдрам живых тел
        let mut positions: Vec<[f32; 3]> = snap
            .iter()
            .map(|b| [b.pos_r[0], b.pos_r[1], b.pos_r[2]])
            .collect();
        let mut contacts = collide::build(
            &tet_shapes,
            &positions,
            0.0,
            (params.lo, params.hi),
            BASE * 0.5,
        );
        // припаркованные слоты в контактах не участвуют
        for i in active..MAX_FRAGS {
            contacts[i].clear();
        }

        let chunks = 4u32;
        let per = (args.substeps / chunks).max(1);
        let mut _worst_pen = 0.0f32;
        for _ in 0..chunks {
            let positions: Vec<[f32; 3]> = snap
                .iter()
                .map(|b| [b.pos_r[0], b.pos_r[1], b.pos_r[2]])
                .collect();
            let mut cs = collide::build(&tet_shapes, &positions, 0.0, (params.lo, params.hi), BASE * 0.5);
            for i in active..MAX_FRAGS {
                cs[i].clear();
            }
            // Удар фиксируем ЗДЕСЬ: к началу следующего кадра солвер уже
            // растащит тела, и контакт снаряд–плита исчезнет из списка.
            if std::env::var("VF_DEBUG").is_ok() && f >= 3 && f <= 5 {
                println!(
                    "    кадр {f}: снаряд x={:.2} контактов {} ({:?})",
                    snap[ball_slot].pos_r[0],
                    cs[ball_slot].len(),
                    cs[ball_slot].iter().map(|c| (c.other, c.pen)).collect::<Vec<_>>()
                );
            }
            if !broke && cs[ball_slot].iter().any(|c| c.other == 0) {
                hit_now = true;
            }
            for list in &cs {
                for c in list {
                    _worst_pen = _worst_pen.max(c.pen);
                }
            }
            gpu.write_contacts(&cs);
            gpu.frame_with_contacts(1.0 / 60.0 / chunks as f32, per, args.iterations);
            snap = gpu.read_bodies();

            // Позиционная развязка: солвер работает со скоростями, и при
            // глубоком входе тела остаются вложенными. Расталкиваем по
            // нормали контакта пропорционально обратным массам.
            let mut fixed = snap.clone();
            let mut moved = false;
            for i in 0..=ball_slot {
                for c in &cs[i] {
                    if c.pen < 0.004 {
                        continue;
                    }
                    // пол и стены: партнёра нет, двигаем только тело
                    if c.other < 0 {
                        for k in 0..3 {
                            fixed[i].pos_r[k] += c.n[k] * c.pen * 0.8;
                        }
                        moved = true;
                        continue;
                    }
                    let (mi, mo) = (masses[i], masses[c.other as usize]);
                    if mi <= 0.0 || mo <= 0.0 {
                        continue;
                    }
                    let wi = 1.0 / mi;
                    let wo = 1.0 / mo;
                    let share = wi / (wi + wo);
                    // нормаль смотрит ОТ тела К партнёру, значит расталкивать
                    // надо в обратную сторону (для статики выше — наоборот,
                    // там нормаль это направление выталкивания)
                    let push = 0.5 * c.pen * share as f32 * 0.8;
                    for k in 0..3 {
                        fixed[i].pos_r[k] -= c.n[k] * push;
                        fixed[c.other as usize].pos_r[k] += c.n[k] * push;
                    }
                    moved = true;
                }
            }
            if moved {
                gpu.write_bodies(&fixed);
                snap = fixed;
            }
        }
        if std::env::var("VF_DEBUG").is_ok() && f < 8 {
            println!("  кадр {f}: снаряд x={:.2} vx={:.1} asleep={}", snap[ball_slot].pos_r[0], snap[ball_slot].vel_m[0], snap[ball_slot].misc[1]);
        }
        // Метрика после всех подшагов и развязки: сколько тела ещё вложены
        if std::env::var("VF_TIME").is_ok() && f % 30 == 0 {
            let positions: Vec<[f32; 3]> = snap
                .iter()
                .map(|b| [b.pos_r[0], b.pos_r[1], b.pos_r[2]])
                .collect();
            let mut cs = collide::build(&tet_shapes, &positions, 0.0, (params.lo, params.hi), BASE * 0.5);
            for i in active..MAX_FRAGS {
                cs[i].clear();
            }
            // Удар фиксируем ЗДЕСЬ: к началу следующего кадра солвер уже
            // растащит тела, и контакт снаряд–плита исчезнет из списка.
            if std::env::var("VF_DEBUG").is_ok() && f >= 3 && f <= 5 {
                println!(
                    "    кадр {f}: снаряд x={:.2} контактов {} ({:?})",
                    snap[ball_slot].pos_r[0],
                    cs[ball_slot].len(),
                    cs[ball_slot].iter().map(|c| (c.other, c.pen)).collect::<Vec<_>>()
                );
            }
            if !broke && cs[ball_slot].iter().any(|c| c.other == 0) {
                hit_now = true;
            }
            let mut body_pen = 0.0f32;
            let mut floor_pen = 0.0f32;
            for list in cs.iter() {
                for c in list {
                    if c.other < 0 {
                        floor_pen = floor_pen.max(c.pen);
                    } else {
                        body_pen = body_pen.max(c.pen);
                    }
                }
            }
            let mut who = (0usize, 0i32, 0.0f32);
            for (i, list) in cs.iter().enumerate() {
                for c in list {
                    if c.other >= 0 && c.pen > who.2 {
                        who = (i, c.other, c.pen);
                    }
                }
            }
            println!(
                "  кадр {f}: тело-тело {body_pen:.3} м, тело-пол {floor_pen:.3} м (глубже всех: {} -> {} = {:.3})",
                who.0, who.1, who.2
            );
        }

        // Удар: снаряд коснулся плиты — только теперь она раскалывается.
        if !broke && hit_now {
            broke = true;

            // Энергия удара: скорость снаряда ДО кадра (в кадре он уже успел
            // передать часть импульса), масса известна.
            let vpre = [snap0[ball_slot].vel_m[0], snap0[ball_slot].vel_m[1], snap0[ball_slot].vel_m[2]];
            let v2 = vpre[0] * vpre[0] + vpre[1] * vpre[1] + vpre[2] * vpre[2];
            let e_kin = 0.5 * masses[ball_slot] * v2 as f64;

            // Жёсткость удара задаёт число осколков: слабый — крупные куски,
            // сильный — мелкие отколы у места попадания.
            let n_want = (6.0 + (e_kin / 15000.0)).clamp(6.0, MAX_FRAGS as f64) as usize;
            let mut frags = plate.shatter_biased(
                n_want,
                0xC0FF_EE00_1234,
                [
                    snap0[ball_slot].pos_r[0],
                    snap0[ball_slot].pos_r[1],
                    snap0[ball_slot].pos_r[2],
                ],
            );
            frags.truncate(MAX_FRAGS);
            let n = frags.len();
            log.push(format!(
                "кадр {f}: удар {:.0} кДж — раскалывается на {n} осколков",
                e_kin / 1000.0
            ));

            // формы осколков: МЕШ ПО ПОЛЮ осколка (тот же марчинг, что у целых
            // тел). Раньше здесь стоял `split_surfaces` — сборка меша из граней
            // тетов, и осколки выходили «старой» грубой формы, пока целые тела
            // были гладкими.
            let base = shapes.len() as u32;
            for (i, fr) in frags.iter().enumerate() {
                shapes.push(shape_of(fr));
                shape_id[i] = base + i as u32;
            }

            // Скорости: часть кинетической энергии снаряда уходит в разлёт.
            // Направление — от точки удара наружу, поэтому отколы у места
            // попадания летят сильнее всего.
            let hit = [
                snap0[ball_slot].pos_r[0],
                snap0[ball_slot].pos_r[1],
                snap0[ball_slot].pos_r[2],
            ];
            let mass_total: f64 = frags.iter().map(|fr| DENSITY * fr.volume()).sum();
            let share = 0.35;
            let v_rms = (2.0 * share * e_kin / mass_total.max(1e-6)).sqrt();
            let mut rng = Lcg::new(0x1BAD_B002_5EED);
            let mut p_lost = [0.0f64; 3];
            let mut bodies_now = snap.clone();
            for (i, fr) in frags.iter().enumerate() {
                let b = body_of(fr);
                let c = fr.world_centroid();
                let d = [c[0] - hit[0], c[1] - hit[1], c[2] - hit[2]];
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3);
                let speed = (v_rms * (0.6 + 0.8 * rng.unit() as f64)) as f32;
                let mut g = GpuBody::from_cpu(&b);
                for k in 0..3 {
                    g.vel_m[k] = d[k] / len * speed;
                    g.pos_r[k] += d[k] / len * 0.004; // не стартовать вложенными
                    p_lost[k] += b.mass * (d[k] / len) as f64 * speed as f64;
                }
                g.misc[1] = 0.0;
                g.misc[0] = 0.0;
                g.anchor = [g.pos_r[0], g.pos_r[1], g.pos_r[2], 0.0];
                bodies_now[i] = g;
                tet_shapes[i] = fr.surface_tets();
                masses[i] = b.mass;
            }
            for k in 0..3 {
                bodies_now[ball_slot].vel_m[k] -= (p_lost[k] / masses[ball_slot]) as f32;
            }
            active = n;
            gpu.write_bodies(&bodies_now);
            snap = bodies_now;
        }

        // запись кадра: живые осколки + снаряд
        let mut row = Vec::with_capacity(active + 1);
        for i in 0..active {
            row.push(BodyFrame {
                shape: shape_id[i],
                pos: [snap[i].pos_r[0], snap[i].pos_r[1], snap[i].pos_r[2]],
                quat: [snap[i].quat[0], snap[i].quat[1], snap[i].quat[2], snap[i].quat[3]],
            });
        }
        row.push(BodyFrame {
            shape: ball_shape,
            pos: [
                snap[ball_slot].pos_r[0],
                snap[ball_slot].pos_r[1],
                snap[ball_slot].pos_r[2],
            ],
            quat: [1.0, 0.0, 0.0, 0.0],
        });
        frames_out.push(row);
    }

    if !args.quiet {
        let tris: usize = shapes.iter().map(|s| s.mesh.triangles()).sum();
        println!("кадров {}, форм {}, треугольников {}", frames_out.len(), shapes.len(), tris);
        for line in &log {
            println!("  {line}");
        }
    }

    if let Some(p) = &args.out {
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let blob = vfw::encode(&frames_out, &shapes);
        std::fs::write(p, &blob).unwrap();
        if !args.quiet {
            println!(
                "записано: {} ({:.1} КБ)",
                p.display(),
                blob.len() as f64 / 1024.0
            );
            println!("смотреть: nix run .#viewer");
        }
    }
}
