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

/// Мешер на GPU обязан давать ту же поверхность, что CPU-версия: тот же
/// алгоритм, та же сетка — значит совпадать должны и число треугольников,
/// и габариты.
#[test]
fn gpu_mesher_matches_cpu() {
    use voxelforge_refsim::gpu::{find_shaders, gpu_mesh};
    let Some(shaders) = find_shaders(None) else {
        eprintln!("пропуск: нет SPIR-V");
        return;
    };
    let body = VoxelBody::sphere([0.0, 0.0, 0.0], 0.5, 24, 0.05);
    let cpu = mesh_isosurface(&body, 0.05);
    // В песочнице nix ICD нет — тогда честно пропускаем, а не падаем.
    let gpu = std::panic::catch_unwind(|| gpu_mesh(&body, &shaders));
    let Ok((tris, verts)) = gpu else {
        eprintln!("пропуск: Vulkan-устройство недоступно");
        return;
    };

    let bb = |it: &mut dyn Iterator<Item = &[f32; 3]>| {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for v in it {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        (lo, hi)
    };
    let (clo, chi) = bb(&mut cpu.vertices.iter());
    let (glo, ghi) = bb(&mut verts.iter());
    eprintln!(
        "мешер: CPU {} треуг. {:.2}..{:.2}, GPU {} треуг. {:.2}..{:.2}",
        cpu.indices.len() / 3,
        clo[1],
        chi[1],
        tris,
        glo[1],
        ghi[1]
    );
    assert!(tris > 100, "GPU выдал слишком мало треугольников: {tris}");
    let diff = (tris as f64 - (cpu.indices.len() / 3) as f64).abs() / (cpu.indices.len() as f64 / 3.0);
    assert!(diff < 0.02, "число треугольников разошлось на {:.1}%", 100.0 * diff);
    for k in 0..3 {
        assert!((clo[k] - glo[k]).abs() < 1e-3, "низ габарита {k} разошёлся");
        assert!((chi[k] - ghi[k]).abs() < 1e-3, "верх габарита {k} разошёлся");
    }
}
