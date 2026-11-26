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

fn main() {
    let args = parse_args();

    // ---- мир ----------------------------------------------------------
    // Плита: 3 кубические ячейки по ребру = 0.84 м, ячейки уровня 0.
    let plate = TetBody::cube([0.0, 1.0, 0.0], 4, 0);
    let plate_vol = plate.volume() as f64;

    // Снаряд — воксельный шар, летит сбоку.
    let ball = TetBody::ball([-3.0, 1.0, 0.0], 0.30, 1);
    let ball_vol = ball.volume() as f64;

    // Раскол плиты по Вороному: 9 затравок → неровные осколки.
    let mut shards: Vec<TetBody> = plate.shatter(14, 0xC0FF_EE00_1234);
    // Меши всех осколков считаются ОДНИМ проходом по общему списку тетраэдров:
    // грани на стыке двух осколков попадают в оба меша, поэтому у каждого
    // осколка поверхность замкнута.
    fn rebuild(shards: &[TetBody], origin: [f32; 3]) -> Vec<Shape> {
        let mut tets = Vec::new();
        let mut owner = Vec::new();
        for (i, s) in shards.iter().enumerate() {
            tets.extend_from_slice(&s.tets);
            owner.extend(std::iter::repeat(i).take(s.tets.len()));
        }
        let mut meshes = split_surfaces(origin, &tets, &owner, shards.len());
        // Вершины локальные: кадр прибавит позицию тела, поэтому вычитаем
        // центроид осколка — тот же, что уходит в физику.
        for (k, m) in meshes.iter_mut().enumerate() {
            let c = shards[k].centroid();
            for p in m.positions.iter_mut() {
                p[0] -= c[0];
                p[1] -= c[1];
                p[2] -= c[2];
            }
        }
        meshes.into_iter().map(|m| Shape { level: m.level, mesh: m }).collect()
    }

    let mut shapes: Vec<Shape> = rebuild(&shards, plate.origin);
    let mut shape_id: Vec<u32> = (0..shards.len() as u32).collect();
    let ball_shape = shapes.len() as u32;
    shapes.push(vfw::shape_of(&ball));

    // ---- тела для солвера ---------------------------------------------
    let mut bodies: Vec<GpuBody> = Vec::new();
    let mut cpu: Vec<Body> = Vec::new();
    for s in &shards {
        // центроид тетраэдров локальный — возвращаем его в мир через origin
        let c0 = s.centroid();
        let c = [c0[0] + plate.origin[0], c0[1] + plate.origin[1], c0[2] + plate.origin[2]];
        let (lo, hi) = s.bounds();
        // Коллизионная коробка осколка чуть меньше его габарита: у осколка
        // рваные края, и по полному AABB соседние осколки «склеены» — они бы
        // не разлетались, а толкали друг друга монолитом.
        let half = [
            0.80 * (hi[0] - lo[0]) as f64,
            0.80 * (hi[1] - lo[1]) as f64,
            0.80 * (hi[2] - lo[2]) as f64,
        ];
        let mut b = Body::block([c[0] as f64, c[1] as f64, c[2] as f64], half);
        b.mass = DENSITY * s.volume() as f64;
        cpu.push(b);
        bodies.push(GpuBody::from_cpu(&b));
    }
    // Осколки спят, пока в плиту не попали. Массу не трогаем: нулевая
    // обратная масса сделала бы их статикой, и удар ничего бы не сдвинул.
    for b in bodies.iter_mut() {
        b.misc[1] = 1.0;
    }
    // Параметры снаряда вынесены в env — так их можно подбирать, не пересобирая.
    let ball_speed: f64 = std::env::var("VF_SPEED").ok().and_then(|v| v.parse().ok()).unwrap_or(34.0);
    let ball_density: f64 = std::env::var("VF_BDENS").ok().and_then(|v| v.parse().ok()).unwrap_or(1200.0);
    let mut proj = Body::new([-3.2, 1.0, 0.0], [0.0, 0.0, 0.0], ball_density * ball_vol, 0.30);
    proj.vel = [ball_speed, 0.0, 0.0];
    cpu.push(proj);
    bodies.push(GpuBody::from_cpu(&proj));

    let max_half = bodies
        .iter()
        .map(|b| b.half_shape[0].max(b.half_shape[1]).max(b.half_shape[2]))
        .fold(0.0f32, f32::max);
    let params = Params {
        // Ящик шире плиты, чтобы осколкам было куда разлетаться.
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
        let tets: usize = shards.iter().map(|s| s.tets.len()).sum();
        println!("voxelforge · {} · шаг решётки {:.3} м", gpu.name, BASE);
        println!(
            "мир: плита {:.3} м³ на {tets} тетраэдров, {} осколков, снаряд {:.3} м³",
            plate_vol,
            shards.len(),
            ball_vol
        );
        println!("масса плиты {:.1} кг, снаряда {:.1} кг", DENSITY * plate_vol, DENSITY * ball_vol);
    }

    // ---- прогон --------------------------------------------------------
    let n_shards = shards.len();
    let mut frames: Vec<Vec<BodyFrame>> = Vec::with_capacity(args.frames as usize);
    let mut cooldown = vec![0u32; n_shards];
    let mut global_cd = 0u32;
    let mut subdiv_log: Vec<String> = Vec::new();
    let mut broke = false;

    for f in 0..args.frames {
        gpu.frame(1.0 / 60.0, args.substeps, args.iterations);
        let snap = gpu.read_bodies();

        // Удар: снаряд коснулся плиты. Мгновенный раскол — это не украшение,
        // а модель: упругая энергия, запасённая до разрушения, уходит в разлёт.
        // Импульс не берётся из воздуха — он вычитается из импульса снаряда.
        if !broke && (0..n_shards).any(|i| snap[i].misc[1] < 0.5) {
            broke = true;
            // Источник разлёта — центр плиты, а не точка касания: снаряд
            // останавливается на поверхности, и радиальные лучи от неё дают
            // всем осколкам одно направление. Трещина идёт изнутри наружу.
            let hit = plate.centroid();
            let mut bodies_now = snap.clone();
            let mut rng = Lcg::new(0x1BAD_B002_5EED);
            let mut p_lost = [0.0f64; 3];
            for i in 0..n_shards {
                let c = [snap[i].pos_r[0], snap[i].pos_r[1], snap[i].pos_r[2]];
                let d = [c[0] - hit[0], c[1] - hit[1], c[2] - hit[2]];
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3);
                let boost = 2.0 + 3.0 * rng.unit();
                let m = cpu[i].mass;
                for k in 0..3 {
                    bodies_now[i].vel_m[k] += d[k] / len * boost;
                    p_lost[k] += m * (d[k] / len) as f64 * boost as f64;
                }
                // Сон и обратная масса обязаны быть согласованы с новой
                // скоростью: спящее тело с ненулевой скоростью ломает солвер.
                bodies_now[i].misc[1] = 0.0;
                bodies_now[i].vel_m[3] = bodies_now[i].misc[2];
                bodies_now[i].ang_i[3] = bodies_now[i].misc[3];
                bodies_now[i].misc[0] = 0.0;
            }
            let m_ball = cpu[n_shards].mass;
            for k in 0..3 {
                bodies_now[n_shards].vel_m[k] -= (p_lost[k] / m_ball) as f32;
            }
            gpu.write_bodies(&bodies_now);
            if !args.quiet {
                let mut t = 0.0;
                for k in 0..3 {
                    t += p_lost[k] * p_lost[k];
                }
                println!("кадр {f}: удар — импульс раскола {:.1} кг·м/с", t.sqrt());
            }
        }

        global_cd = global_cd.saturating_sub(1);
        for i in 0..n_shards {
            cooldown[i] = cooldown[i].saturating_sub(1);
            if cooldown[i] > 0 || snap[i].misc[1] > 0.5 {
                continue;
            }
            let speed = (snap[i].vel_m[0].powi(2) + snap[i].vel_m[1].powi(2) + snap[i].vel_m[2].powi(2)).sqrt();
            if speed < 1.5 {
                continue;
            }
            // Детализация дробится у ВСЕХ осколков разом: если уточнить только
            // один, на стыке с соседом получится T-стык и щель в поверхности.
            let lvl = shards[0].tets[0].level;
            if lvl >= shards[0].level_cap || global_cd > 0 {
                continue;
            }
            let mut refined = false;
            for s in shards.iter_mut() {
                refined |= s.subdivide();
            }
            if refined {
                let base_len = shapes.len();
                let new_meshes = rebuild(&shards, plate.origin);
                shapes.extend(new_meshes);
                for (k, id) in shape_id.iter_mut().enumerate() {
                    *id = (base_len + k) as u32;
                }
                global_cd = 14;
                let step = BASE / (1u32 << shards[0].tets[0].level) as f32;
                subdiv_log.push(format!(
                    "кадр {f}: детализация → {} тетраэдров на осколок, шаг {:.3} м",
                    shards[0].tets.len(),
                    step
                ));
            }
            let _ = i;
        }

        if std::env::var("VF_DEBUG").is_ok() && f < 6 {
            let vmax = snap.iter().map(|b| (b.vel_m[0].powi(2) + b.vel_m[1].powi(2) + b.vel_m[2].powi(2)).sqrt()).fold(0.0f32, f32::max);
            let pmax = snap.iter().map(|b| b.pos_r[0].abs().max(b.pos_r[1].abs()).max(b.pos_r[2].abs())).fold(0.0f32, f32::max);
            println!("  отладка кадр {f}: vmax {vmax:.1} м/с, |p|max {pmax:.2} м");
            if std::env::var("VF_BODIES").is_ok() {
                for (i, b) in snap.iter().enumerate() {
                    let sp = (b.vel_m[0].powi(2) + b.vel_m[1].powi(2) + b.vel_m[2].powi(2)).sqrt();
                    if sp > 200.0 || !sp.is_finite() {
                        println!("    тело {i}: p={:?} v={:?} half={:?} misc={:?} quat={:?}", b.pos_r, b.vel_m, b.half_shape, b.misc, b.quat);
                    }
                }
            }
        }

        let mut row = Vec::with_capacity(n_shards + 1);
        for i in 0..n_shards {
            row.push(BodyFrame {
                shape: shape_id[i],
                pos: [snap[i].pos_r[0], snap[i].pos_r[1], snap[i].pos_r[2]],
                quat: [snap[i].quat[0], snap[i].quat[1], snap[i].quat[2], snap[i].quat[3]],
            });
        }
        row.push(BodyFrame {
            shape: ball_shape,
            pos: [
                snap[n_shards].pos_r[0],
                snap[n_shards].pos_r[1],
                snap[n_shards].pos_r[2],
            ],
            quat: [1.0, 0.0, 0.0, 0.0],
        });
        frames.push(row);
    }

    let total_tris: usize = shapes.iter().map(|s| s.mesh.triangles()).sum();
    if !args.quiet {
        println!(
            "кадров {}, форм {}, треугольников в мешах {}",
            frames.len(),
            shapes.len(),
            total_tris
        );
        for line in subdiv_log.iter().take(6) {
            println!("  {line}");
        }
        if subdiv_log.len() > 6 {
            println!("  … всего дроблений: {}", subdiv_log.len());
        }
    }

    if let Some(p) = &args.out {
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let blob = vfw::encode(&frames, &shapes);
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
