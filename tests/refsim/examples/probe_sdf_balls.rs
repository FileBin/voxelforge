// Точечная проверка: ОДНА пара шаров, перекрытие задаётся, шаг крошечный.
// Печатает контакты из списка и сверяет с CPU-оракулом (глубины и точки).
use voxelforge_world::live::{Kind, Live};
use voxelforge_world::sim::find_shaders;
use voxelforge_world::tet::TetBody;

fn main() {
    let sh = find_shaders(None).unwrap();
    let mode = std::env::var("VF_COLLIDE").unwrap_or_else(|_| "sdf".into());
    let mut w = Live::new(&sh);
    w.set_fracture(false);
    w.set_sleep(false);
    for _ in 0..4 {
        w.step(0.05, 8, 12);
    }
    w.gpu.wait_idle();
    /* два шара r=0.14 в воздухе: A на высоте 3 м, B смещён по X так, что
     * перекрытие — заданная величина */
    let a = 0.14f64;
    let overlap: f64 = std::env::var("VF_OVERLAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.02);
    let ia = w
        .spawn_tet(
            &TetBody::ball([0.0, 0.0, 0.0], 0.14, 0),
            [0.0, 3.0, 0.0],
            [0.0; 3],
            Kind::Shard,
        )
        .unwrap();
    let dx = 2.0 * a - overlap;
    let ib = w
        .spawn_tet(
            &TetBody::ball([0.0, 0.0, 0.0], 0.14, 0),
            [dx, 3.0, 0.0],
            [0.0; 3],
            Kind::Shard,
        )
        .unwrap();
    w.step(0.0005, 1, 1);
    w.gpu.wait_idle();
    let cs = w.gpu.read_contacts();
    println!("режим {mode} · перекрытие {overlap:.3} м · A={ia} B={ib} · dx={dx:.3}");
    for (slot, name) in [(ia, "A"), (ib, "B")] {
        println!("  {name}: контактов {}", cs[slot].len());
        for c in cs[slot].iter().take(8) {
            println!(
                "    → {}: pen {:+.4} · n [{:+.2} {:+.2} {:+.2}] · p [{:+.3} {:+.3} {:+.3}]",
                c.other, c.pen, c.n[0], c.n[1], c.n[2], c.p[0], c.p[1], c.p[2]
            );
        }
    }
    let fa = w.bodies[ia].sdf.as_ref().unwrap();
    let fb = w.bodies[ib].sdf.as_ref().unwrap();
    let cpu = voxelforge_world::sdf_contact::contacts(
        fa,
        w.bodies[ia].origin,
        fb,
        w.bodies[ib].origin,
        16,
    );
    println!(
        "  CPU: контактов {} · глубины {:?}",
        cpu.len(),
        cpu.iter().map(|c| c.depth).collect::<Vec<_>>()
    );
    for c in cpu.iter().take(4) {
        println!("    p [{:+.3} {:+.3} {:+.3}] n [{:+.2} {:+.2} {:+.2}]", c.p[0], c.p[1], c.p[2], c.n[0], c.n[1], c.n[2]);
    }
}
