//! Вода: частицы без ограничения объёма.
//!
//! Проверяем не «красиво ли», а три свойства, которые и означают «жидкость
//! считается»: частицы не разлетаются в бесконечность, не проваливаются под
//! пол, и вода растекается за пределы начального пятна — то есть домена у неё
//! действительно нет.

use std::path::PathBuf;

use voxelforge_refsim::gpu::find_shaders;
use voxelforge_refsim::water::{WaterParams, WaterSim};

/// Кубик частиц, поставленный над полом.
fn blob(side: usize, spacing: f32, center: [f32; 3]) -> Vec<[f32; 3]> {
    let mut v = Vec::with_capacity(side * side * side);
    let half = 0.5 * (side as f32 - 1.0) * spacing;
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                v.push([
                    center[0] - half + x as f32 * spacing,
                    center[1] - half + y as f32 * spacing,
                    center[2] - half + z as f32 * spacing,
                ]);
            }
        }
    }
    v
}

fn setup() -> Option<PathBuf> {
    let shaders = find_shaders(None)?;
    let probe = std::panic::catch_unwind(|| {
        let s = WaterSim::new(blob(4, 0.06, [0.0, 1.0, 0.0]), &shaders, WaterParams::default());
        s.n
    });
    probe.ok().map(|_| shaders)
}

fn spread(points: &[[f32; 4]]) -> f32 {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for p in points {
        lo = lo.min(p[0]);
        hi = hi.max(p[0]);
    }
    hi - lo
}

#[test]
fn water_spreads_without_a_domain() {
    let Some(shaders) = setup() else {
        eprintln!("пропуск: нет SPIR-V или Vulkan-устройства");
        return;
    };
    let start = blob(16, 0.06, [0.0, 1.4, 0.0]);
    let n = start.len();
    let mut sim = WaterSim::new(start, &shaders, WaterParams::default());
    let spread0 = spread(&sim.read());

    for _ in 0..150 {
        sim.frame(1.0 / 60.0, 4);
    }
    let mid = spread(&sim.read());
    for _ in 0..150 {
        sim.frame(1.0 / 60.0, 4);
    }
    let p = sim.read();

    let mut bad = 0;
    let mut below = 0;
    let mut vmax = 0.0f32;
    let mut dens_err = 0.0f64;
    for q in &p {
        if !q[0].is_finite() || !q[1].is_finite() || !q[2].is_finite() {
            bad += 1;
        }
        if q[1] < -0.02 {
            below += 1;
        }
        vmax = vmax.max(q[1].abs());
        dens_err += ((q[3] - sim.params.rho0).abs() / sim.params.rho0) as f64;
    }
    let spread1 = spread(&p);
    let mean_dens_err = dens_err / n as f64;

    eprintln!(
        "вода: {n} частиц, разброс {spread0:.2} -> {mid:.2} -> {spread1:.2} м, NaN {bad}, \
         под полом {below}, ошибка плотности {:.0}%, y max {vmax:.2}",
        100.0 * mean_dens_err
    );

    assert_eq!(bad, 0, "{bad} частиц с NaN");
    assert_eq!(below, 0, "{below} частиц провалились под пол");
    assert!(
        spread1 > spread0 * 1.25,
        "вода не растеклась: {spread0:.2} -> {spread1:.2} м"
    );
    // Жидкость обязана затухать: если разброс продолжает расти темпом
    // первой секунды — это уже газ, а не вода.
    let grow1 = mid / spread0;
    let grow2 = spread1 / mid;
    assert!(
        grow2 < grow1,
        "растекание не замедляется: {grow1:.2}× за первую половину, {grow2:.2}× за вторую"
    );
    assert!(
        mean_dens_err < 0.6,
        "средняя ошибка плотности {:.0}% — жидкость не держит объём",
        100.0 * mean_dens_err
    );
}
