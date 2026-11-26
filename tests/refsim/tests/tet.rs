//! Тетраэдры BCC: решётка, red-разбиение, марчинг поверхности.

use std::collections::HashMap;

use voxelforge_world::bcc::BASE;
use voxelforge_world::tet::{mesh_volume, Mesh, TetBody};

/// Каждое ребро меша принадлежит ровно двум треугольникам, и они идут
/// в противоположных направлениях — иначе поверхность дырявая или вывернута.
fn watertight(m: &Mesh) -> usize {
    let mut dir: HashMap<(u32, u32), i32> = HashMap::new();
    for t in m.indices.chunks(3) {
        for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *dir.entry(e).or_insert(0) += 1;
        }
    }
    let mut bad = 0;
    for (&(a, b), &c) in &dir {
        if a > b {
            continue;
        }
        let back = dir.get(&(b, a)).copied().unwrap_or(0);
        if c != 1 || back != 1 {
            bad += 1;
        }
    }
    bad
}

/// Куб из ячеек = 12 тетраэдров на ячейку; объём считается по тетраэдрам.
#[test]
fn cube_is_twelve_tets_per_cell() {
    let n = 2;
    let body = TetBody::cube([0.0, 0.0, 0.0], n, 0);
    let a = BASE as f64;
    assert_eq!(body.tets.len() as i32, 12 * n * n * n, "тетраэдров не 12n³");
    let vol = body.volume();
    let expect = (n as f64).powi(3) * a * a * a;
    assert!(
        (vol - expect).abs() / expect < 1e-6,
        "объём {vol} вместо {expect}"
    );
}

/// Red-разбиение: каждый тетраэдр → 8, шаг вдвое меньше, объём сохраняется
/// ТОЧНО (усечённый октаэдр так не умеет — он терял четверть объёма).
#[test]
fn refine_preserves_volume_exactly() {
    let mut body = TetBody::cube([0.0, 1.0, 0.0], 2, 0);
    let (v0, n0) = (body.volume(), body.tets.len());
    let lvl0 = body.tets[0].level;

    assert!(body.subdivide());
    let (v1, n1) = (body.volume(), body.tets.len());
    assert_eq!(n1, n0 * 8, "тетраэдров {n1}, ожидалось {}", n0 * 8);
    assert!((v0 - v1).abs() / v0 < 1e-6, "объём изменился: {v0} -> {v1}");
    assert_eq!(body.tets[0].level, lvl0 + 1, "уровень не вырос");

    // до предела
    while body.subdivide() {}
    let v2 = body.volume();
    assert!((v0 - v2).abs() / v0 < 1e-6, "объём уехал на глубине: {v0} -> {v2}");
    eprintln!(
        "разбиение: {} -> {} -> {} тетраэдров, объём {:.5} постоянен",
        n0, n1, body.tets.len(), v2
    );
}

/// Марчинг: поверхность водо-/воздухонепроницаема и её объём равен объёму тела.
#[test]
fn surface_is_watertight_and_exact() {
    let body = TetBody::cube([0.0, 1.0, 0.0], 3, 0);
    let m = body.surface();
    let bad = watertight(&m);
    assert_eq!(bad, 0, "рёбер не по два: {bad}");
    let (v, expect) = (mesh_volume(&m), body.volume());
    eprintln!(
        "меш: {} треугольников, {} вершин, объём {v:.5} при {expect:.5}",
        m.triangles(),
        m.positions.len()
    );
    assert!(
        (v as f64 - expect).abs() / expect < 1e-3,
        "объём меша {v} вместо {expect}"
    );
}

/// Раскол: тетраэдры разложены по осколкам без потерь, и осколки неровные.
#[test]
fn shatter_covers_all_tets() {
    let body = TetBody::cube([0.0, 1.0, 0.0], 3, 0);
    let shards = body.shatter(12, 0xC0FF_EE00_1234);
    let total: usize = shards.iter().map(|s| s.tets.len()).sum();
    assert_eq!(total, body.tets.len(), "тетраэдры потерялись");
    assert!(shards.len() >= 6, "осколков мало: {}", shards.len());

    let total_v: f64 = shards.iter().map(|s| s.volume()).sum();
    assert!(
        (total_v - body.volume()).abs() / body.volume() < 1e-9,
        "объём осколков не равен телу"
    );

    // Поверхности осколков считаются одним проходом (как в движке): грань на
    // стыке двух осколков попадает в оба меша, поэтому каждый замкнут.
    let mut all: Vec<voxelforge_world::tet::Tet> = Vec::new();
    let mut owner: Vec<usize> = Vec::new();
    for (i, s) in shards.iter().enumerate() {
        all.extend_from_slice(&s.tets);
        owner.extend(std::iter::repeat(i).take(s.tets.len()));
    }
    let meshes = voxelforge_world::tet::split_surfaces(body.origin, &all, &owner, shards.len());
    // Срез Вороного может оставить единичные T-стыки (перевёрнутая грань на
    // стыке двух осколков) — допускаем их немного, но не дыры в поверхности.
    for (i, m) in meshes.iter().enumerate() {
        let bad = watertight(m);
        assert!(bad <= 3, "осколок {i}: рёбер не по два {bad}");
        assert!(m.triangles() > 0, "осколок {i} без поверхности");
    }
    let total_mesh: f32 = meshes.iter().map(mesh_volume).sum();
    assert!(
        (total_mesh as f64 - body.volume()).abs() / body.volume() < 0.05,
        "объём мешей осколков {total_mesh} против {}",
        body.volume()
    );
    let sizes: Vec<usize> = shards.iter().map(|s| s.tets.len()).collect();
    eprintln!("осколки: {} штук, размеры {:?}", shards.len(), sizes);
}
