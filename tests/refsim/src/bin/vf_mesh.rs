//! `vf-mesh` — собрать BCC-меш из поля и записать `.vfm` для вьюера.
//!
//! Это тот же мешер, что используют тесты, но с сохранением на диск: вьюер в
//! браузере открывает `.vfm` и показывает поверхность плюс каркас BCC-ячеек,
//! чтобы решётку было видно глазами, а не только в отчёте теста.

use std::path::PathBuf;

use voxelforge_refsim::mesher::{encode_vfm, lattice_wireframe, mesh_isosurface};
use voxelforge_refsim::VoxelBody;

fn main() {
    let mut shape = "sphere".to_string();
    let mut dim = 24u32;
    let mut cell = 0.05f32;
    let mut out = PathBuf::from("demos/out/voxel.vfm");
    let mut lines = 4000usize;

    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().expect("нужно значение");
        match k.as_str() {
            "--shape" => shape = v(),
            "--dim" => dim = v().parse().unwrap(),
            "--cell" => cell = v().parse().unwrap(),
            "--out" => out = v().into(),
            "--lines" => lines = v().parse().unwrap(),
            other => panic!("неизвестный аргумент {other}"),
        }
    }

    let center = [0.0f32, 1.0, 0.0];
    let body = match shape.as_str() {
        "sphere" => VoxelBody::sphere(center, 0.6, dim, cell),
        "cube" => VoxelBody::cube(center, [0.5, 0.5, 0.5], dim, cell),
        other => panic!("неизвестная форма {other}: sphere | cube"),
    };

    let mesh = mesh_isosurface(&body, cell);
    let (lv, li) = lattice_wireframe(&body, lines);
    let mut line_pairs = Vec::with_capacity(li.len() / 2);
    for pair in li.chunks(2) {
        line_pairs.push((lv[pair[0] as usize], lv[pair[1] as usize]));
    }
    let blob = encode_vfm(&mesh, &line_pairs);

    if let Some(dir) = out.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, &blob).unwrap();
    println!(
        "меш: {} вершин, {} треугольников, {} рёбер каркаса → {} ({:.1} КБ)",
        mesh.vertices.len(),
        mesh.indices.len() / 3,
        line_pairs.len(),
        out.display(),
        blob.len() as f64 / 1024.0
    );
    println!("смотреть: открыть viewer/index.html и перетащить файл");
}
