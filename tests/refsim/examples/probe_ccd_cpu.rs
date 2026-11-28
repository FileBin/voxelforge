// CPU-оракул: контакты SDF↔SDF для снаряда, утопленного в плиту на разную
// глубину. Печатает число контактов, глубину max и есть ли NaN — шейдер
// повторяет этот же алгоритм, и NaN здесь означал бы ошибку самого оракула.
use voxelforge_world::sdf_contact::contacts;
use voxelforge_world::tet::TetBody;

fn main() {
    let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
    let r = 0.14f32;
    let half = 0.56f32; /* полуразмер плиты по тетам, м */
    for depth in [0.002f32, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5] {
        /* центр шара на half - r + depth: чем больше depth, тем глубже внутрь */
        let y = half + r - depth;
        let ball = TetBody::ball([0.0, y, 0.0], r, 0);
        let cs = contacts(plate.sdf(), plate.origin, ball.sdf(), ball.origin, 16);
        let nan = cs
            .iter()
            .filter(|c| c.depth.is_nan() || c.p.iter().any(|v| v.is_nan()) || c.n.iter().any(|v| v.is_nan()))
            .count();
        let dmax = cs
            .iter()
            .map(|c| c.depth)
            .filter(|d| !d.is_nan())
            .fold(f32::MIN, f32::max);
        let p0 = cs.first().map(|c| c.p);
        println!(
            "глубина шара {depth:.3}: контактов {:2} · глубина max {dmax:.4} · NaN {nan} · первая точка {p0:?}",
            cs.len()
        );
    }
}
