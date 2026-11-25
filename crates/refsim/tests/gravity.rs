//! Интеграционные тесты гравитации (демка D0).
//!
//! Здесь только публичное API крейта: так тесты проверяют то, чем реально
//! пользуется игра, а не внутренности, которые можно переделать свободно.

use voxelforge_refsim::{vfr, Body, Sim};

/// Центральное тело массы M и спутник на круговой орбите радиуса r.
fn circular_orbit() -> (Sim, f64) {
    let m: f64 = 1000.0;
    let r: f64 = 10.0;
    let g: f64 = 1.0;
    let v: f64 = (g * m / r).sqrt();
    let star = Body::new([0.0, 0.0, 0.0], [0.0; 3], m, 1.0);
    let planet = Body::new([r, 0.0, 0.0], [0.0, 0.0, v], 1e-6, 0.2);
    let mut sim = Sim::new(vec![star, planet]);
    sim.g = g;
    (sim, 2.0 * std::f64::consts::PI * r / v)
}

#[test]
fn orbit_closes_after_ten_periods() {
    let (mut sim, period) = circular_orbit();
    let dt = period / 4096.0;
    let start = sim.bodies[1].pos;
    for _ in 0..(4096 * 10) {
        sim.step(dt);
    }
    let end = sim.bodies[1].pos;
    let err = ((end[0] - start[0]).powi(2)
        + (end[1] - start[1]).powi(2)
        + (end[2] - start[2]).powi(2))
    .sqrt();
    let r = (start[0].powi(2) + start[1].powi(2) + start[2].powi(2)).sqrt();
    assert!(err / r < 0.01, "closure error {:.4}%", 100.0 * err / r);
}

#[test]
fn energy_drift_bounded() {
    let (mut sim, period) = circular_orbit();
    sim.softening = 1e-6;
    let dt = period / 4096.0;
    let e0 = sim.energy();
    for _ in 0..(4096 * 10) {
        sim.step(dt);
    }
    let e1 = sim.energy();
    let drift = ((e1 - e0) / e0).abs();
    assert!(drift < 0.01, "energy drift {:.4}%", 100.0 * drift);
}

#[test]
fn momentum_conserved() {
    let mut sim = Sim::new(vec![
        Body::new([1.0, 0.0, 0.0], [0.0, 0.5, 0.0], 1.0, 0.2),
        Body::new([-1.0, 0.0, 0.0], [0.0, -0.5, 0.0], 3.0, 0.3),
        Body::new([0.0, 2.0, 0.0], [0.3, 0.0, 0.0], 2.0, 0.25),
    ]);
    let p0 = sim.momentum();
    for _ in 0..2000 {
        sim.step(1e-3);
    }
    let p1 = sim.momentum();
    for k in 0..3 {
        assert!((p1[k] - p0[k]).abs() < 1e-9, "momentum drift on axis {k}");
    }
}

/// Одинаковый вход обязан давать побайтово одинаковую запись: иначе
/// демки нельзя сравнивать с GPU-версией, а отчёты — воспроизводить.
#[test]
fn recording_is_byte_identical() {
    let run = || {
        let mut sim = Sim::new(vec![
            Body::new([1.0, 0.0, 0.0], [0.0, 0.7, 0.0], 5.0, 0.2),
            Body::new([-2.0, 0.5, 0.0], [0.0, -0.3, 0.1], 4.0, 0.2),
        ]);
        let mut frames = Vec::new();
        for _ in 0..64 {
            sim.step(1e-3);
            frames.push(
                sim.bodies
                    .iter()
                    .map(|b| [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32])
                    .collect::<Vec<_>>(),
            );
        }
        vfr::encode(&sim.bodies, &frames)
    };
    assert_eq!(run(), run(), "recording must be byte-identical");
}
