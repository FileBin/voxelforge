// CPU-оракул: куб 0.28 на плите, заглублён на заданную величину. Печатает
// глубины контактов — проверка «глубина в метрах, а не в узлах».
use voxelforge_world::sdf_contact::contacts;
use voxelforge_world::tet::TetBody;

fn main() {
    let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
    let cube = TetBody::cube([0.0, 0.0, 0.0], 2, 1);
    let pf = plate.sdf();
    /* ноль поля плиты сверху: ищем смену знака вниз от верха блока */
    let at = |y: f32| pf.sample_m([(0.0 - plate.origin[0]) / pf.step, (y - plate.origin[1]) / pf.step, (0.0 - plate.origin[2]) / pf.step]);
    let mut top = plate.origin[1] + 0.66;
    while at(top) < 0.0 {
        top -= 0.001;
    }
    println!("ноль поля плиты сверху: {top:.4} (f={:+.4})", at(top));
    for want in [0.0f32, 0.01, 0.03, 0.05, 0.09] {
        let y = top + 0.14 - want;
        let c = TetBody::cube([0.0, y, 0.0], 2, 1);
        let cs = contacts(&pf, plate.origin, c.sdf(), c.origin, 24);
        println!(
            "заглубление {want:.3}: контактов {:2} · глубина max {:+.4} · первые {:?}",
            cs.len(),
            cs.iter().fold(f32::MIN, |m, k| m.max(k.depth)),
            cs.iter().take(3).map(|k| (k.depth * 1000.0).round() / 1000.0).collect::<Vec<_>>()
        );
    }
}
