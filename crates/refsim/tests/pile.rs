//! Интеграционные тесты плотной кучи: правило больших масс (D4),
//! детерминизм и сцена D3 (5000 тел ложатся и засыпают).

use voxelforge_refsim::{pile_demo, Body, PileSim};

/// Сцена D3 — та же, что в демке: решётка с детерминированным дрожанием.
fn lattice_scene(count: usize) -> PileSim {
    let mut sim = pile_demo(count);
    sim.friction = 0.7;
    sim
}

/// Правило больших масс: пара с κ > 1000 решается односторонне —
/// лёгкое тело не проваливается в тяжёлое и не вылетает.
///
/// Горошина стоит ровно в полюсе валуна: сфера на сфере — неустойчивое
/// равновесие, при любом смещении она обязана скатиться (это физика, а не
/// баг), поэтому смещение по x/z здесь строго нулевое.
#[test]
fn heavy_body_does_not_swallow_light_one() {
    let mut bodies = vec![
        Body::new([0.0, 1.0, 0.0], [0.0; 3], 10_000.0, 1.0),
        Body::new([0.0, 3.0, 0.0], [0.0; 3], 1.0, 0.08),
    ];
    bodies[0].vel = [0.0; 3];
    let mut sim = PileSim::new(bodies);
    sim.mass_ratio_limit = 1000.0;
    let dt = 1.0 / 60.0;
    for _ in 0..240 {
        sim.step(dt);
    }
    let pebble_y = sim.bodies[1].pos[1];
    assert!(
        pebble_y > 1.5,
        "горошина провалилась в валун: y = {pebble_y:.3}"
    );
    assert!(pebble_y < 2.5, "горошину выбросило: y = {pebble_y:.3}");
}

#[test]
fn pile_is_deterministic() {
    let run = || {
        let mut sim = lattice_scene(400);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            sim.step(dt);
        }
        sim.bodies
            .iter()
            .flat_map(|b| b.pos.iter().map(|v| v.to_bits()))
            .collect::<Vec<u64>>()
    };
    assert_eq!(run(), run(), "куча разошлась между прогонами");
}

/// D3: куча из 5k тел должна улечься и уснуть, не проваливаясь в пол.
///
/// Проходит только на **скоростном** контактном слое. Прежняя позиционная
/// версия (`v = Δx/h`) разлеталась до 25 м/с: коррекции проникновения не
/// успевали разрешиться и превращались в скорость.
#[test]
fn pile_of_5000_settles_and_sleeps() {
    let mut sim = lattice_scene(5000);
    let dt = 1.0 / 60.0;
    let mut seconds = 0.0;
    while sim.awake > 0 && seconds < 12.0 {
        sim.step(dt);
        seconds += dt;
    }
    let pen = sim.max_penetration();
    assert!(
        sim.awake == 0,
        "не уснули: {} тел за {:.1} с",
        sim.awake,
        seconds
    );
    assert!(seconds < 8.0, "уснули только за {:.1} с", seconds);
    assert!(pen < 0.02, "проникновение {:.4} м > 2 см", pen);
}
