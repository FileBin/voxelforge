//! CLI: прогон сценария на CPU-эталоне и запись траектории.
//!
//! ```text
//! vf-ref --scenario orbit    --frames 1500 --periods 2 --substeps 4 --out orbit.vfr --svg orbit.svg
//! vf-ref --scenario collapse --bodies 400 --frames 900 --seed 7 --out collapse.vfr
//! ```
//!
//! Всегда headless: окно не открывается, вывод — файлы и строки метрик.
//! Сценарий `orbit` интегрируется ровно целое число периодов, поэтому `closure`
//! (возврат планеты в стартовую точку) имеет смысл как метрика точности.

use std::process::ExitCode;

use voxelforge_refsim::{svg, vfr, Body, Rng, Sim};

struct Args {
    scenario: String,
    frames: usize,
    dt: f64,
    bodies: usize,
    seed: u64,
    periods: f64,
    substeps: usize,
    out: Option<String>,
    json: Option<String>,
    svg: Option<String>,
    size: u32,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            scenario: "orbit".into(),
            frames: 1500,
            dt: 0.01,
            bodies: 300,
            seed: 1,
            periods: 2.0,
            substeps: 4,
            out: None,
            json: None,
            svg: None,
            size: 900,
        };
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let key = argv[i].as_str();
            let val = argv.get(i + 1).cloned().unwrap_or_default();
            match key {
                "--scenario" => a.scenario = val,
                "--frames" => a.frames = val.parse().unwrap_or(a.frames),
                "--dt" => a.dt = val.parse().unwrap_or(a.dt),
                "--bodies" => a.bodies = val.parse().unwrap_or(a.bodies),
                "--seed" => a.seed = val.parse().unwrap_or(a.seed),
                "--periods" => a.periods = val.parse().unwrap_or(a.periods),
                "--substeps" => a.substeps = val.parse().unwrap_or(a.substeps),
                "--out" => a.out = Some(val),
                "--json" => a.json = Some(val),
                "--svg" => a.svg = Some(val),
                "--size" => a.size = val.parse().unwrap_or(a.size),
                other => {
                    eprintln!("unknown argument: {other}");
                    std::process::exit(2);
                }
            }
            i += 2;
        }
        a
    }
}

/// Звезда + три планеты на круговых орбитах. Возвращает сим и период средней.
fn orbit_scenario() -> (Sim, f64) {
    let m: f64 = 1000.0;
    let g: f64 = 1.0;
    let mut bodies = vec![Body::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], m, 1.6)];
    for (i, (r, mass)) in [(6.0f64, 1e-4f64), (10.0, 2e-4), (16.0, 4e-4)]
        .iter()
        .enumerate()
    {
        let (r, mass) = (*r, *mass);
        let v = (g * m / r).sqrt();
        let angle = i as f64 * 2.1;
        let pos = [r * angle.cos(), r * angle.sin(), 0.0];
        let vel = [-v * angle.sin(), v * angle.cos(), 0.0];
        bodies.push(Body::new(pos, vel, mass, 0.35));
    }
    let mut sim = Sim::new(bodies);
    sim.g = g;
    sim.softening = 1e-4;
    let period = 2.0 * std::f64::consts::PI * 10.0f64 / (g * m / 10.0f64).sqrt();
    (sim, period)
}

fn collapse_scenario(n: usize, seed: u64) -> Sim {
    let mut rng = Rng::new(seed);
    let mut bodies = Vec::with_capacity(n);
    for _ in 0..n {
        let pos = [rng.signed() * 10.0, rng.signed() * 10.0, rng.signed() * 10.0];
        let vel = [rng.signed() * 0.4, rng.signed() * 0.4, rng.signed() * 0.4];
        bodies.push(Body::new(pos, vel, 1.0 + rng.signed().abs(), 0.25));
    }
    let mut sim = Sim::new(bodies);
    sim.g = 1.0;
    sim.softening = 0.35;
    sim
}

fn main() -> ExitCode {
    let args = Args::parse();

    let (mut sim, dt) = match args.scenario.as_str() {
        "orbit" => {
            let (sim, period) = orbit_scenario();
            let steps = (args.frames * args.substeps).max(1) as f64;
            (sim, period * args.periods / steps)
        }
        "collapse" => (collapse_scenario(args.bodies, args.seed), args.dt),
        other => {
            eprintln!("unknown scenario: {other} (orbit|collapse)");
            return ExitCode::from(2);
        }
    };

    // Замыкание орбиты меряем по средней планете (r = 10), для неё же посчитан период.
    let tracked = 2usize.min(sim.bodies.len().saturating_sub(1));
    let start = sim.bodies[tracked].pos;
    let e0 = sim.energy();
    let p0 = sim.momentum();

    let mut frames: Vec<Vec<[f32; 3]>> = Vec::with_capacity(args.frames);
    for _ in 0..args.frames {
        for _ in 0..args.substeps {
            sim.step(dt);
        }
        frames.push(
            sim.bodies
                .iter()
                .map(|b| [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32])
                .collect(),
        );
    }

    let e1 = sim.energy();
    let drift = if e0 != 0.0 { ((e1 - e0) / e0).abs() } else { 0.0 };
    let p1 = sim.momentum();
    let dp = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let dp_norm = (dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]).sqrt();

    let end = sim.bodies[tracked].pos;
    let start_r = (start[0] * start[0] + start[1] * start[1] + start[2] * start[2]).sqrt();
    let closure = if args.scenario == "orbit" {
        ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2) + (end[2] - start[2]).powi(2))
            .sqrt()
    } else {
        0.0
    };
    let closure_rel = if start_r > 0.0 { closure / start_r } else { 0.0 };

    if let Some(path) = &args.out {
        let bytes = vfr::encode(&sim.bodies, &frames);
        if let Err(e) = std::fs::write(path, &bytes) {
            eprintln!("write {path}: {e}");
            return ExitCode::FAILURE;
        }
        println!("wrote {path} ({} KiB)", bytes.len() / 1024);
    }

    if let Some(path) = &args.svg {
        let text = svg(&sim.bodies, &frames, args.size);
        if let Err(e) = std::fs::write(path, text.as_bytes()) {
            eprintln!("write {path}: {e}");
            return ExitCode::FAILURE;
        }
        println!("wrote {path}");
    }

    let json = format!(
        concat!(
            "{{",
            "\"scenario\":\"{}\",\"bodies\":{},\"frames\":{},\"substeps\":{},\"dt\":{:.9},\"seed\":{},",
            "\"energy_start\":{:.12e},\"energy_end\":{:.12e},\"energy_drift\":{:.6e},",
            "\"momentum_drift\":{:.6e},\"closure_rel\":{:.6e}",
            "}}"
        ),
        args.scenario,
        sim.bodies.len(),
        args.frames,
        args.substeps,
        dt,
        args.seed,
        e0,
        e1,
        drift,
        dp_norm,
        closure_rel
    );

    if let Some(path) = &args.json {
        if let Err(e) = std::fs::write(path, &json) {
            eprintln!("write {path}: {e}");
            return ExitCode::FAILURE;
        }
        println!("wrote {path}");
    }

    println!("{json}");
    println!(
        "summary: scenario={} bodies={} frames={} substeps={} dt={:.6} energy_drift={:.3e} closure_rel={:.3e}",
        args.scenario,
        sim.bodies.len(),
        args.frames,
        args.substeps,
        dt,
        drift,
        closure_rel
    );

    // Гейты для CI. Демка падает, а не «печатает и едет дальше».
    if drift > 0.01 {
        eprintln!("FAIL: energy drift {:.4}% > 1%", drift * 100.0);
        return ExitCode::FAILURE;
    }
    if args.scenario == "orbit" && closure_rel > 0.01 {
        eprintln!("FAIL: orbit closure {:.4}% > 1%", closure_rel * 100.0);
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
