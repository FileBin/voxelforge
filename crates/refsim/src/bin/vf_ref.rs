//! CLI: прогон сценария на CPU-эталоне и запись траектории.
//!
//! ```text
//! vf-ref --scenario orbit    --frames 1500 --periods 2 --substeps 4 --out orbit.vfr
//! vf-ref --scenario collapse --bodies 400 --frames 900 --seed 7
//! vf-ref --scenario drop     --frames 300 --out drop.vfr --svg drop.svg
//! vf-ref --scenario stack    --bodies 10 --frames 600 --out stack.vfr
//! ```
//!
//! Всегда headless: окно не открывается, вывод — файлы и строки метрик.
//! `orbit` интегрируется ровно целое число периодов, поэтому замыкание орбиты
//! имеет смысл как метрика точности. Контактные сценарии (`drop`, `stack`)
//! падают, если просадка, проникновение или остаточная скорость выходят за
//! пороги демок D1 и D2 — это и есть приёмочный тест.

use std::process::ExitCode;

use voxelforge_refsim::{svg, vfr, Body, ContactSim, Rng, Sim};

struct Args {
    scenario: String,
    frames: usize,
    dt: f64,
    bodies: usize,
    seed: u64,
    periods: f64,
    substeps: usize,
    iterations: usize,
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
            iterations: 6,
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
                "--iterations" => a.iterations = val.parse().unwrap_or(a.iterations),
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

/// D1: блок падает на пол с высоты 3 м.
fn drop_scenario(substeps: usize, iterations: usize) -> ContactSim {
    let mut sim = ContactSim::new(vec![Body::block([0.0, 3.0, 0.0], [0.5, 0.5, 0.5])]);
    sim.gravity = 9.81;
    sim.substeps = substeps;
    sim.iterations = iterations;
    sim
}

/// D2: стопка из `count` блоков, поставленных с зазором 2 мм.
fn stack_scenario(count: usize, substeps: usize, iterations: usize) -> ContactSim {
    let half = [0.5, 0.2, 0.5];
    let mut bodies = Vec::new();
    for i in 0..count.max(1) {
        let y = half[1] + i as f64 * (2.0 * half[1] + 0.002);
        bodies.push(Body::block([0.0, y, 0.0], half));
    }
    let mut sim = ContactSim::new(bodies);
    sim.gravity = 9.81;
    sim.substeps = substeps;
    sim.iterations = iterations;
    sim
}

fn write_outputs(args: &Args, bodies: &[Body], frames: &[Vec<[f32; 3]>]) -> Result<(), String> {
    if let Some(path) = &args.out {
        let bytes = vfr::encode(bodies, frames);
        std::fs::write(path, &bytes).map_err(|e| format!("write {path}: {e}"))?;
        println!("wrote {path} ({} KiB)", bytes.len() / 1024);
    }
    if let Some(path) = &args.svg {
        let text = svg(bodies, frames, args.size);
        std::fs::write(path, text.as_bytes()).map_err(|e| format!("write {path}: {e}"))?;
        println!("wrote {path}");
    }
    Ok(())
}

fn write_json(args: &Args, json: &str) -> Result<(), String> {
    if let Some(path) = &args.json {
        std::fs::write(path, json).map_err(|e| format!("write {path}: {e}"))?;
        println!("wrote {path}");
    }
    Ok(())
}

fn run_gravity(args: &Args) -> ExitCode {
    let (mut sim, dt) = match args.scenario.as_str() {
        "orbit" => {
            let (sim, period) = orbit_scenario();
            let steps = (args.frames * args.substeps).max(1) as f64;
            (sim, period * args.periods / steps)
        }
        _ => (collapse_scenario(args.bodies, args.seed), args.dt),
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

    if let Err(e) = write_outputs(args, &sim.bodies, &frames) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
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
    if let Err(e) = write_json(args, &json) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
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

fn run_contact(args: &Args) -> ExitCode {
    let mut sim = match args.scenario.as_str() {
        "drop" => drop_scenario(args.substeps, args.iterations),
        _ => stack_scenario(args.bodies, args.substeps, args.iterations),
    };

    let top0 = sim.top();
    let dt = if args.dt > 0.0 { args.dt } else { 1.0 / 60.0 };

    let mut frames: Vec<Vec<[f32; 3]>> = Vec::with_capacity(args.frames);
    for _ in 0..args.frames {
        sim.step(dt);
        frames.push(
            sim.bodies
                .iter()
                .map(|b| [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32])
                .collect(),
        );
    }

    let top1 = sim.top();
    let sag = if top0 > 0.0 { (top0 - top1) / top0 } else { 0.0 };
    let pen = sim.max_penetration();
    let speed = sim.max_speed();
    let contacts = sim.contacts.len();

    if let Err(e) = write_outputs(args, &sim.bodies, &frames) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }

    let json = format!(
        concat!(
            "{{",
            "\"scenario\":\"{}\",\"bodies\":{},\"frames\":{},\"substeps\":{},\"iterations\":{},\"dt\":{:.9},",
            "\"contacts\":{},\"top_start\":{:.6},\"top_end\":{:.6},\"sag_rel\":{:.6e},",
            "\"max_penetration\":{:.6e},\"max_speed\":{:.6e}",
            "}}"
        ),
        args.scenario,
        sim.bodies.len(),
        args.frames,
        args.substeps,
        args.iterations,
        dt,
        contacts,
        top0,
        top1,
        sag,
        pen,
        speed
    );
    if let Err(e) = write_json(args, &json) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }

    println!("{json}");
    println!(
        "summary: scenario={} bodies={} frames={} substeps={} iterations={} sag_rel={:.3e} max_pen={:.4} max_speed={:.4}",
        args.scenario,
        sim.bodies.len(),
        args.frames,
        args.substeps,
        args.iterations,
        sag,
        pen,
        speed
    );

    // Гейты D1/D2.
    let mut failed = false;
    if pen > 0.02 {
        eprintln!("FAIL: проникновение {:.4} м > 2 см", pen);
        failed = true;
    }
    if speed > 0.02 {
        eprintln!("FAIL: остаточная скорость {:.4} м/с > 2 см/с", speed);
        failed = true;
    }
    if args.scenario == "stack" && sag > 0.05 {
        eprintln!("FAIL: просадка стопки {:.2}% > 5%", sag * 100.0);
        failed = true;
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    match args.scenario.as_str() {
        "orbit" | "collapse" => run_gravity(&args),
        "drop" | "stack" => run_contact(&args),
        other => {
            eprintln!("unknown scenario: {other} (orbit|collapse|drop|stack)");
            ExitCode::from(2)
        }
    }
}
