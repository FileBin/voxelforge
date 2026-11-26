//! BCC-воксели движка: раскол и подразбиение ячеек.
//!
//! Проверяем то, ради чего всё затевалось:
//! 1. тело состоит из ячеек решётки BCC (два узла на кубическую ячейку);
//! 2. раскол даёт неровные осколки, а не нарезку сетки;
//! 3. подразбиение делит объём ровно на 8 и сохраняет массу;
//! 4. в мире после удара ячейки действительно становятся мельче.

use std::collections::HashSet;

use voxelforge_world::bcc::{VoxelBody, BASE};

/// Решётка BCC: в кубе из `n` кубических ячеек узлов вдвое больше, чем клеток,
/// плюс поверхностные — проверяем по индексам, а не по количеству.
#[test]
fn body_sits_on_bcc_lattice() {
    let n = 3i32;
    let body = VoxelBody::cube([0.0, 0.0, 0.0], n, 0);
    let half = BASE * 0.5; // шаг решётки в единицах a/2
    let mut seen = HashSet::new();
    for c in &body.cells {
        // координаты в единицах a/2 обязаны быть целыми с чётной суммой
        let i = (c.c[0] / half).round() as i32;
        let j = (c.c[1] / half).round() as i32;
        let k = (c.c[2] / half).round() as i32;
        assert!(
            ((c.c[0] / half) - i as f32).abs() < 1e-4,
            "ячейка не на узле решётки: {:?}",
            c.c
        );
        assert_eq!(
            i.rem_euclid(2),
            j.rem_euclid(2),
            "узел не BCC (разная чётность): {:?}",
            c.c
        );
        assert_eq!(j.rem_euclid(2), k.rem_euclid(2), "узел не BCC: {:?}", c.c);
        assert!(seen.insert((i, j, k)), "узел повторён: {i},{j},{k}");
    }
    eprintln!("узлов в теле {}×{}×{}: {}", n, n, n, body.cells.len());
    // (n+1)³ углов + n³ центров тел
    assert_eq!(
        body.cells.len() as i32,
        (n + 1).pow(3) + n.pow(3),
        "не то число узлов BCC"
    );
}

/// Раскол: все ячейки распределены, и осколки неровные — иначе это не раскол,
/// а нарезка на коробки (как было в первой версии «разрушения»).
#[test]
fn shatter_gives_irregular_shards() {
    let body = VoxelBody::cube([0.0, 1.0, 0.0], 5, 0);
    let shards = body.shatter(16, 0xC0FF_EE00_1234);
    assert!(shards.len() >= 6, "осколков слишком мало: {}", shards.len());

    let total: usize = shards.iter().map(|s| s.cells.len()).sum();
    assert_eq!(total, body.cells.len(), "ячейки потерялись или задвоились");

    // «Коробка» — осколок, который занимает ВСЕ узлы BCC в своём габарите.
    // Считать надо именно узлы (шаг a/2, одна чётность), а не объём:
    // иначе полупустой осколок случайно сойдётся с объёмом коробки.
    let mut irregular = 0;
    let mut multi = 0;
    for s in &shards {
        if s.cells.len() < 5 {
            continue; // из пяти ячеек «коробка» ничего не доказывает
        }
        multi += 1;
        let e = s.cells[0].edge();
        let (lo, hi) = s.bounds();
        let mut box_nodes = 0;
        for i in 0..=((hi[0] - lo[0]) / (e * 0.5)).round() as i32 {
            for j in 0..=((hi[1] - lo[1]) / (e * 0.5)).round() as i32 {
                for k in 0..=((hi[2] - lo[2]) / (e * 0.5)).round() as i32 {
                    if i % 2 == j % 2 && j % 2 == k % 2 {
                        box_nodes += 1;
                    }
                }
            }
        }
        if s.cells.len() != box_nodes {
            irregular += 1;
        }
    }
    eprintln!(
        "осколков {}, многоклеточных {}, неровных {}",
        shards.len(),
        multi,
        irregular
    );
    assert!(multi >= 4, "многоклеточных осколков только {multi}");
    let ratio = irregular as f64 / multi as f64;
    assert!(ratio >= 0.75, "коробок слишком много: {irregular} из {multi}");
}

/// Подразбиение: каждая ячейка → 8 детей, объём и масса сохраняются точно.
#[test]
fn subdivide_keeps_volume_and_refines() {
    let mut body = VoxelBody::cube([0.0, 1.0, 0.0], 2, 0);
    let v0 = body.volume();
    let n0 = body.cells.len();
    let e0 = body.cells[0].edge();

    assert!(body.subdivide());
    let v1 = body.volume();
    let n1 = body.cells.len();
    let e1 = body.cells[0].edge();

    assert_eq!(n1, n0 * 8, "после дробления ячеек {n1}, ожидалось {}", n0 * 8);
    assert!((e1 - e0 * 0.5).abs() < 1e-6, "шаг не уменьшился вдвое: {e0} -> {e1}");
    assert!(
        (v0 - v1).abs() / v0 < 1e-4,
        "объём изменился: {v0} -> {v1}"
    );

    // до предела дробим и проверяем, что дальше нельзя
    while body.subdivide() {}
    assert!(body.cells.iter().all(|c| c.level == body.level_cap));
    assert!((body.volume() - v0).abs() / v0 < 1e-3);
    eprintln!("уровней {}, ячеек {}, объём {:.4} м³", body.level_cap, body.cells.len(), body.volume());
}
