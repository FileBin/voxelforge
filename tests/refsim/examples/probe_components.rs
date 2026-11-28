// GPU-разметка компонент связности против CPU-оракула.
//
//   VF_SHADERS=<spv dir> cargo run --release --example probe_components
//
// Печатает: число компонент (должно совпасть один-в-один), совпадение
// разбиения узлов, число итераций распространения и цену CPU/GPU при 64³ и
// 128³ узлах.
use std::time::Instant;

use voxelforge_world::field::NodeField;
use voxelforge_world::live::Live;
use voxelforge_world::parts;
use voxelforge_world::sim::find_shaders;

fn field_from<F: Fn(f32, f32, f32) -> f32>(dim: usize, f: F) -> NodeField {
    let mut val = vec![0.0f32; dim * dim * dim];
    let c = dim as f32 * 0.5;
    for z in 0..dim {
        for y in 0..dim {
            for x in 0..dim {
                val[(z * dim + y) * dim + x] = f(x as f32 - c, y as f32 - c, z as f32 - c);
            }
        }
    }
    NodeField { lo: [0; 3], dims: [dim, dim, dim], val, step: 1.0 }
}

fn sphere(p: [f32; 3], c: [f32; 3], r: f32) -> f32 {
    ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt() - r
}

fn two_balls(dim: usize) -> NodeField {
    field_from(dim, |x, y, z| {
        sphere([x, y, z], [-10.0, 0.0, 0.0], 9.0).min(sphere([x, y, z], [10.0, 0.0, 0.0], 9.0))
    })
}

/// Два блока, касающиеся ТОЛЬКО углом: узлы (24,24,24) и (25,25,24) — диагональ.
fn corner_boxes(dim: usize) -> NodeField {
    let b = |x: f32, y: f32, z: f32, c: [f32; 3], h: f32| -> f32 {
        let q = [x - c[0], y - c[1], z - c[2]];
        let dx = q[0].abs() - h;
        let dy = q[1].abs() - h;
        let dz = q[2].abs() - h;
        (dx.max(0.0).powi(2) + dy.max(0.0).powi(2) + dz.max(0.0).powi(2)).sqrt()
            + dx.max(dy.max(dz)).min(0.0)
    };
    let c = dim as f32 * 0.5;
    field_from(dim, |x, y, z| {
        b(x, y, z, [c - 8.0, c - 8.0, c], 8.0).min(b(x, y, z, [c + 9.0, c + 9.0, c], 8.0))
    })
}

/// Сравнение разбиения: узлы в одной компоненте CPU обязаны быть в одной и той
/// же компоненте GPU, и наоборот (метки разные — сравниваем классы).
fn same_partition(cpu: &parts::Labels, gpu: &parts::Labels) -> bool {
    if cpu.count() != gpu.count() || cpu.label.len() != gpu.label.len() {
        return false;
    }
    let mut map = vec![u32::MAX; cpu.count()];
    let mut back = vec![u32::MAX; gpu.count()];
    let mut bad = 0usize;
    for i in 0..cpu.label.len() {
        let (a, b) = (cpu.label[i], gpu.label[i]);
        if (a == u32::MAX) != (b == u32::MAX) {
            if bad < 5 {
                println!("  расхождение занятости: узел {i}: cpu {a}, gpu {b}");
            }
            bad += 1;
            continue;
        }
        if a == u32::MAX {
            continue;
        }
        if map[a as usize] == u32::MAX {
            map[a as usize] = b;
        } else if map[a as usize] != b {
            if bad < 5 {
                println!("  cpu-компонента {a} разошлась в gpu {} и {b} (узел {i})", map[a as usize]);
            }
            bad += 1;
        }
        if back[b as usize] == u32::MAX {
            back[b as usize] = a;
        } else if back[b as usize] != a {
            if bad < 5 {
                println!("  gpu-компонента {b} собрала cpu {} и {a} (узел {i})", back[b as usize]);
            }
            bad += 1;
        }
    }
    if bad > 0 {
        println!("  расхождений: {bad}");
    }
    bad == 0
}

fn main() {
    let sh = find_shaders(None).expect("нет SPIR-V: nix build .#spirv");
    let mut w = Live::new(&sh);
    println!("устройство: {}", w.gpu.name);
    /* Прогрев: первая GPU-разметка греет буферы и кэш драйвера (пайплайн
     * `components` создаётся общим циклом вместе с остальными) — её время в
     * замеры не идёт. */
    let warm = two_balls(16);
    let _ = w.gpu.run_components(&warm, 1);
    /* Пустое поле 16³ — мерим ЧИСТУЮ цену проходов/синхронизаций: работы
     * почти нет, и вся цена — накладные расходы на submit. */
    let g = w.gpu.run_components(&two_balls(16), 1).expect("16³");
    println!("16³ (почти пусто, цена накладных): {:.2} мс, проходов {}", g.ms, g.iterations);
    for (name, dim, f) in [
        ("два шара", 64usize, two_balls(64)),
        ("два шара", 128, two_balls(128)),
        ("угловой контакт", 64, corner_boxes(64)),
    ] {
        let mut cpu_ms = f32::MAX;
        let mut cpu = parts::label_field(&f, 1);
        for _ in 0..3 {
            let t = Instant::now();
            cpu = parts::label_field(&f, 1);
            cpu_ms = cpu_ms.min(t.elapsed().as_secs_f32() * 1e3);
        }
        let mut gpu = w.gpu.run_components(&f, 1).expect("GPU-разметка не запустилась");
        for _ in 0..2 {
            let g = w.gpu.run_components(&f, 1).expect("GPU-разметка не запустилась");
            if g.ms < gpu.ms {
                gpu = g;
            }
        }
        let nodes: usize = cpu.counts.iter().sum();
        let gpu_nodes: usize = gpu.labels.counts.iter().sum();
        let gpu_ms = gpu.ms;
        let it = gpu.iters_stage;
        println!(
            "{name} {dim}³: CPU {} компонент за {cpu_ms:.2} мс · GPU {} компонент за {gpu_ms:.2} мс · проходов {} (ядро {}, кожа {}, остаток {}) · узлов CPU {nodes} / GPU {gpu_nodes} · разбиение {}",
            cpu.count(),
            gpu.labels.count(),
            gpu.iterations,
            it[0],
            it[1],
            it[2],
            if same_partition(&cpu, &gpu.labels) { "СОВПАЛО" } else { "РАСХОЖДЕНИЕ" },
        );
        assert_eq!(cpu.count(), gpu.labels.count(), "{name} {dim}³: число компонент");
        assert!(same_partition(&cpu, &gpu.labels), "{name} {dim}³: разбиение узлов");
        assert_eq!(nodes, gpu_nodes, "{name} {dim}³: число занятых узлов");
    }
    println!("все сверки сошлись");
}
