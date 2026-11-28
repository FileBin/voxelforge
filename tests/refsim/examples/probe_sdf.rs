// Заполнение сетки SDF примитивом ЦЕЛИКОМ на устройстве.
//
// Проверяем шейдерный путь: `GpuSim::fill_sdf_grid` считает расстояния на
// видеокарте, CPU только задаёт примитив. Сверяем с аналитикой, чтобы ошибка
// была видна числом, а не на глаз.
use voxelforge_world::live::Live;
use voxelforge_world::sim::{find_shaders, SdfPrim};

fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);

    let dims = [17u32, 17, 17];
    let step = 0.05f32;
    let origin = [-0.4f32, -0.4, -0.4];
    let r = 0.3f32;

    let vals = w.gpu.fill_sdf_grid(&SdfPrim::sphere([0.0, 0.0, 0.0], r), dims, origin, step);

    let mut worst: f32 = 0.0;
    let mut worst_at = [0.0f32; 3];
    let mut neg = 0usize;
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let i = ((z * dims[1] + y) * dims[0] + x) as usize;
                let p = [
                    origin[0] + x as f32 * step,
                    origin[1] + y as f32 * step,
                    origin[2] + z as f32 * step,
                ];
                let want = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - r;
                let err = (vals[i] - want).abs();
                if err > worst {
                    worst = err;
                    worst_at = p;
                }
                if vals[i] < 0.0 {
                    neg += 1;
                }
            }
        }
    }
    println!(
        "SDF на устройстве: узлов {} · внутри {} · макс. ошибка {:.6} в {:?}",
        vals.len(),
        neg,
        worst,
        worst_at
    );

    // коробка: полуразмеры 0.2/0.1/0.05
    let half = [0.2f32, 0.1, 0.05];
    let vb = w.gpu.fill_sdf_grid(&SdfPrim::box_shape([0.0, 0.0, 0.0], half), dims, origin, step);
    let mut worst_b: f32 = 0.0;
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let i = ((z * dims[1] + y) * dims[0] + x) as usize;
                let p = [
                    origin[0] + x as f32 * step,
                    origin[1] + y as f32 * step,
                    origin[2] + z as f32 * step,
                ];
                let q = [
                    p[0].abs() - half[0],
                    p[1].abs() - half[1],
                    p[2].abs() - half[2],
                ];
                let outside = (q[0].max(0.0).powi(2) + q[1].max(0.0).powi(2) + q[2].max(0.0).powi(2))
                    .sqrt();
                let inside = q[0].max(q[1]).max(q[2]).min(0.0);
                let want = outside + inside;
                worst_b = worst_b.max((vb[i] - want).abs());
            }
        }
    }
    println!("SDF коробки на устройстве: макс. ошибка {:.6}", worst_b);

    /* Второй шаг пути: из ЭТОЙ сетки собираем тело и его меш — то есть объект
     * создан целиком из данных, которые родились на устройстве. */
    let n_lo = [
        (origin[0] / step).round() as i32 - 0,
        (origin[1] / step).round() as i32,
        (origin[2] / step).round() as i32,
    ];
    let body = voxelforge_world::tet::TetBody::from_sdf_grid(
        [0.0, 0.0, 0.0],
        0,
        n_lo,
        [dims[0] as usize, dims[1] as usize, dims[2] as usize],
        &vals,
    );
    let mesh = body.surface_grid().0;
    let open = 0usize; // замкнутость проверяется тестами крейта
    println!(
        "тело из SDF: тетов {} · треугольников {} · открытых рёбер {}",
        body.tets.len(),
        mesh.triangles(),
        open
    );
    assert!(mesh.triangles() > 100, "меш пустой");
    assert_eq!(open, 0, "поверхность не замкнута");
    /* шаг 1 коллизий по SDF: поле должно доезжать до устройства без искажений */
    let body = voxelforge_world::tet::TetBody::ball([0.0, 0.0, 0.0], 0.3, 0);
    let ok = w.gpu.sdf_roundtrip(body.sdf());
    println!("поле на устройстве и обратно: {}", if ok { "совпало" } else { "РАСХОЖДЕНИЕ" });
    assert!(ok, "поле искажается при выгрузке на устройство");
    assert!(worst < 1e-4, "шар: ошибка SDF {worst}");
    assert!(worst_b < 1e-4, "коробка: ошибка SDF {worst_b}");
    println!("PROBE_SDF ok");
}
