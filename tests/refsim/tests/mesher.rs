//! BCC-мешер: поверхность нулевого уровня должна быть замкнутой, а уточнение
//! поля — уменьшать отклонение меша от истинной поверхности.

use std::collections::HashMap;

use voxelforge_refsim::mesher::mesh_isosurface;
use voxelforge_refsim::VoxelBody;

/// Каждое ребро замкнутой поверхности принадлежит ровно двум треугольникам.
fn is_closed(mesh: &voxelforge_refsim::mesher::Mesh) -> (usize, usize) {
    let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
    for t in mesh.indices.chunks(3) {
        for k in 0..3 {
            let a = t[k];
            let b = t[(k + 1) % 3];
            let key = (a.min(b), a.max(b));
            *edges.entry(key).or_insert(0) += 1;
        }
    }
    let open = edges.values().filter(|c| **c != 2).count();
    (edges.len(), open)
}

#[test]
fn sphere_mesh_is_closed() {
    let body = VoxelBody::sphere([0.0, 0.0, 0.0], 0.5, 24, 0.05);
    let mesh = mesh_isosurface(&body, 0.05);
    let (edges, open) = is_closed(&mesh);
    eprintln!(
        "меш: {} вершин, {} треугольников, {} рёбер, незакрытых {}",
        mesh.vertices.len(),
        mesh.indices.len() / 3,
        edges,
        open
    );
    assert!(mesh.indices.len() >= 3 * 50, "треугольников подозрительно мало");
    assert_eq!(open, 0, "поверхность не замкнута: {open} рёбер с неверным числом граней");
}

/// Уточнённое поле обязано давать более точный меш: вершины должны лежать
/// ближе к истинной сфере.
#[test]
fn refinement_makes_mesh_more_accurate() {
    let center = [0.0f32, 0.0, 0.0];
    let radius = 0.5f32;
    let coarse = VoxelBody::sphere(center, radius, 20, 0.1);
    let fine = VoxelBody::sphere(center, radius, 40, 0.05);

    let err = |m: &voxelforge_refsim::mesher::Mesh| -> f64 {
        let mut worst = 0.0f64;
        for v in &m.vertices {
            let d = ((v[0] - center[0]).powi(2) + (v[1] - center[1]).powi(2) + (v[2] - center[2]).powi(2))
                .sqrt() as f64;
            worst = worst.max((d - radius as f64).abs());
        }
        worst
    };
    let mc = mesh_isosurface(&coarse, 0.1);
    let mf = mesh_isosurface(&fine, 0.05);
    let (ec, ef) = (err(&mc), err(&mf));
    eprintln!(
        "точность меша: крупная сетка {ec:.4} ({} треуг.), мелкая {ef:.4} ({} треуг.)",
        mc.indices.len() / 3,
        mf.indices.len() / 3
    );
    assert!(ef < ec * 0.75, "уточнение не улучшило меш: {ec:.4} -> {ef:.4}");
}
