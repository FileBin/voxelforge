//! Интеграционные тесты контактов: демки D1 (падение) и D2 (стопка).

use voxelforge_refsim::{Body, ContactSim};

fn drop_scene() -> ContactSim {
    let mut sim = ContactSim::new(vec![Body::block([0.0, 3.0, 0.0], [0.5, 0.5, 0.5])]);
    sim.gravity = 9.81;
    sim.substeps = 8;
    sim.iterations = 6;
    sim
}

fn stack_scene(count: usize) -> ContactSim {
    let half = [0.5, 0.2, 0.5];
    let mut bodies = Vec::new();
    for i in 0..count {
        let y = half[1] + i as f64 * (2.0 * half[1] + 0.002);
        bodies.push(Body::block([0.0, y, 0.0], half));
    }
    let mut sim = ContactSim::new(bodies);
    sim.gravity = 9.81;
    sim.substeps = 8;
    sim.iterations = 8;
    sim
}

/// D1: коробка падает на пол и остаётся стоять — без просадки и дрожания.
#[test]
fn box_lands_and_rests() {
    let mut sim = drop_scene();
    let dt = 1.0 / 60.0;
    for _ in 0..300 {
        sim.step(dt);
    }
    let rest_y = sim.bodies[0].pos[1];
    let pen = sim.max_penetration();
    let speed = sim.max_speed();
    assert!(
        (rest_y - 0.5).abs() < 0.02,
        "rest height {rest_y:.4}, ожидалось ~0.5"
    );
    assert!(pen < 0.01, "просадка {:.4} м > 1 см", pen);
    assert!(speed < 0.01, "скорость покоя {speed:.4} м/с > 1 см/с");
}

/// D2: стопка из 10 блоков стоит 10 секунд и не складывается.
#[test]
fn stack_of_ten_stands() {
    let mut sim = stack_scene(10);
    let top0 = sim.top();
    let dt = 1.0 / 60.0;
    for _ in 0..600 {
        sim.step(dt);
    }
    let top1 = sim.top();
    let sag = (top0 - top1) / top0;
    let pen = sim.max_penetration();
    let speed = sim.max_speed();
    assert!(sag < 0.05, "просадка стопки {:.2}% > 5%", 100.0 * sag);
    assert!(pen < 0.02, "проникновение в стопке {:.4} м > 2 см", pen);
    assert!(speed < 0.02, "стопка не успокоилась: {speed:.4} м/с");
}

/// Два прогона контактной сцены должны совпасть побитово.
#[test]
fn contacts_are_deterministic() {
    let run = || {
        let mut sim = stack_scene(6);
        let dt = 1.0 / 60.0;
        for _ in 0..90 {
            sim.step(dt);
        }
        sim.bodies
            .iter()
            .flat_map(|b| b.pos.iter().map(|v| v.to_bits()))
            .collect::<Vec<u64>>()
    };
    assert_eq!(run(), run(), "два прогона дали разные позиции");
}
