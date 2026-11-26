//! Тесты воксельного поля: уточнение и выборка.
//!
//! Здесь проверяется именно то, ради чего воксели делаются разного размера:
//! уточнённый патч описывает ту же поверхность точнее. Форма не должна
//! «поехать» — только уменьшиться ошибка.

use voxelforge_refsim::VoxelBody;

/// Уточнение обязано уменьшить ошибку описания поверхности.
#[test]
fn refinement_improves_surface_accuracy() {
    let center = [0.0f32, 0.0, 0.0];
    let radius = 0.5f32;
    let mut body = VoxelBody::sphere(center, radius, 24, 0.1);
    let lo = [-0.3f32, 0.35, -0.3];
    let hi = [0.3f32, 0.75, 0.3];

    // Сравниваем на той же области, которую уточняем: ошибка в остальной
    // сфере осталась прежней и «съела» бы разницу.
    let coarse_err = body.surface_error_in(center, radius, lo, hi, 48);
    body.refine_region(center, radius, lo, hi);
    let fine_err = body.surface_error_in(center, radius, lo, hi, 48);

    eprintln!("ошибка: крупная сетка {coarse_err:.4}, после уточнения {fine_err:.4}");
    assert!(
        coarse_err > 0.0,
        "крупная сетка должна иметь заметную ошибку, иначе тест ничего не проверяет"
    );
    assert!(
        fine_err < coarse_err * 0.75,
        "уточнение почти не улучшило точность: {coarse_err:.4} → {fine_err:.4}"
    );
}

/// Знак поля: внутри отрицательно, снаружи положительно, и поверхность
/// проходит там, где поле равно нулю.
#[test]
fn field_sign_and_zero_level() {
    let center = [0.0f32, 0.0, 0.0];
    let radius = 0.5f32;
    let body = VoxelBody::sphere(center, radius, 32, 0.05);
    assert!(body.sample([0.0, 0.0, 0.0]) < 0.0, "центр должен быть внутри");
    assert!(body.sample([0.0, 0.0, 0.0]).abs() > 0.3, "в центре глубина ~радиус");
    assert!(body.sample([0.0, 1.0, 0.0]) > 0.0, "снаружи расстояние положительное");
    let on_surface = body.sample([radius, 0.0, 0.0]);
    assert!(
        on_surface.abs() < 0.03,
        "на поверхности поле должно быть около нуля, получено {on_surface}"
    );
}

/// Мелкая ячейка не «проваливается» сквозь крупную: уточнённый патч
/// перекрывает базовую сетку в своей области, а не заменяет её дырой.
#[test]
fn refinement_has_no_holes() {
    let center = [0.0f32, 0.0, 0.0];
    let radius = 0.5f32;
    let mut body = VoxelBody::sphere(center, radius, 20, 0.12);
    body.refine_region(center, radius, [-0.25, 0.25, -0.25], [0.25, 0.75, 0.25]);

    // Плотно прощупываем сферу: значения не должны быть «далеко снаружи»
    // (f32::MAX) — это означало бы дыру в поле.
    let mut holes = 0;
    for i in 0..48 {
        for j in 0..48 {
            let theta = std::f32::consts::PI * (i as f32 + 0.5) / 48.0;
            let phi = 2.0 * std::f32::consts::PI * (j as f32 + 0.5) / 48.0;
            let p = [
                center[0] + radius * theta.sin() * phi.cos(),
                center[1] + radius * theta.cos(),
                center[2] + radius * theta.sin() * phi.sin(),
            ];
            let s = body.sample(p);
            if !s.is_finite() || s.abs() > 1.0 {
                holes += 1;
            }
        }
    }
    assert_eq!(holes, 0, "в поле {holes} дыр на поверхности");
}

/// Пре-фрактура обязана точно замостить исходное тело: объём и масса
/// сохраняются, осколки не перекрываются и не оставляют щелей.
#[test]
fn fracture_octants_tile_the_body() {
    let center = [0.0f32, 1.0, 0.0];
    let half = [0.5f32, 0.5, 0.5];
    let body = VoxelBody::cube(center, half, 24, 0.05);
    let chunks = body.fracture_octants(center, half, 1.0);

    assert_eq!(chunks.len(), 8, "октантов должно быть восемь");

    let full_volume = 8.0 * (half[0] as f64) * (half[1] as f64) * (half[2] as f64);
    let sum_mass: f64 = chunks.iter().map(|(_, _, m)| *m).sum();
    assert!(
        (sum_mass - full_volume).abs() < 1e-9,
        "масса осколков {sum_mass} != масса тела {full_volume}"
    );

    // Каждая точка тела покрыта ровно одним осколком.
    for i in 0..8 {
        for j in (i + 1)..8 {
            let (a, b) = (&chunks[i], &chunks[j]);
            let overlap = (0..3).all(|k| {
                (a.0[k] - b.0[k]).abs() < a.1[k] + b.1[k] - 1e-6
            });
            assert!(!overlap, "осколки {i} и {j} пересекаются");
        }
    }
    // Габариты объединения совпадают с исходными.
    let lo: Vec<f32> = (0..3).map(|k| chunks.iter().map(|c| c.0[k] - c.1[k]).fold(f32::MAX, f32::min)).collect();
    let hi: Vec<f32> = (0..3).map(|k| chunks.iter().map(|c| c.0[k] + c.1[k]).fold(f32::MIN, f32::max)).collect();
    for k in 0..3 {
        assert!((lo[k] - (center[k] - half[k])).abs() < 1e-5, "низ {k} разошёлся");
        assert!((hi[k] - (center[k] + half[k])).abs() < 1e-5, "верх {k} разошёлся");
    }
}
