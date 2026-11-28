// Диагностика списка SDF-контактов: что именно приходит в солвер.
//
//   VF_COLLIDE=sdf probe_sdf_dump
//
// Сцена: шар r=0.14 падает на плиту. После каждого такта печатается список
// контактов шара (pen, нормаль, точка) — и то же для плиты. NaN в позе тела
// печатается отдельно: он приходит из солвера, значит контакт уже испорчен.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    w.set_fracture(false);
    w.set_sleep(false);
    for _ in 0..4 {
        w.step(0.05, 8, 12);
    }
    w.gpu.wait_idle();
    let top = {
        let b = &w.bodies[0];
        let f = b.sdf.as_ref().unwrap();
        let o = b.origin;
        let at = |y: f32| f.sample_m([(0.0 - o[0]) / f.step, (y - o[1]) / f.step, (0.0 - o[2]) / f.step]);
        let mut y = b.pos[1] as f32;
        while at(y) <= 0.0 && y < b.pos[1] as f32 + 2.0 {
            y += 0.001;
        }
        y
    };
    w.shoot([0.0, top as f64 + 0.14 + 2.0, 0.0], [0.0, 0.0, 0.0], 0.0);
    let ball = *w.bullets.back().unwrap();
    println!("шар в слоте {ball}, поверхность плиты {top:.4}");
    for t in 0..18 {
        w.step(0.05, 8, 12);
        w.gpu.wait_idle();
        let cs = w.gpu.read_contacts();
        let meta = w.gpu.read_contacts_meta();
        let snap = w.gpu.read_bodies();
        let g = &snap[ball];
        let diag: Vec<[i32; 4]> = meta[ball].iter().copied().filter(|m| m[1] >= 7).collect();
        if !diag.is_empty() {
            println!("  ДИАГНОСТИКА, {} шт:", diag.len());
            for (k, c) in cs[ball].iter().enumerate() {
                let m = &meta[ball][k];
                if m[1] < 7 { continue; }
                println!(
                    "    стадия {} · партнёр {} · fs {:+.4} узл · ft {:+.4} узл · |d| {:+.3} va {:+.3} ga.x {:+.3} vb {:+.3} · w [{:+.3} {:+.3} {:+.3}]",
                    m[1], m[0], m[2] as f32 * 1e-4, m[3] as f32 * 1e-4,
                    c.n[0], c.n[1], c.n[2], c.pen,
                    c.p[0], c.p[1], c.p[2]
                );
            }
        }
        println!(
            "такт {t}: шар y={:.3} vy={:.3} | контактов шара {} · плиты {} | y NaN: {}",
            g.pos_r[1],
            g.vel_m[1],
            cs[ball].len(),
            cs[0].len(),
            g.pos_r[1].is_nan()
        );
        for c in cs[ball].iter().take(6) {
            println!(
                "    шар → {}: pen {:+.4} · n [{:+.2} {:+.2} {:+.2}] · p [{:+.3} {:+.3} {:+.3}]",
                c.other, c.pen, c.n[0], c.n[1], c.n[2], c.p[0], c.p[1], c.p[2]
            );
        }
        for c in cs[0].iter().take(3) {
            println!(
                "    плита → {}: pen {:+.4} · n [{:+.2} {:+.2} {:+.2}]",
                c.other, c.pen, c.n[0], c.n[1], c.n[2]
            );
        }
        if g.pos_r[1].is_nan() {
            /* тот же кадр на CPU-оракуле: если он даёт осмысленные числа, а
             * движок — NaN, значит расходится шейдер, а не алгоритм */
            let a = w.bodies[0].sdf.as_ref().map(|f| (f, w.bodies[0].origin));
            let b = w.bodies[ball].sdf.as_ref().map(|f| (f, w.bodies[ball].origin));
            if let (Some((fa, oa)), Some((fb, ob))) = (a, b) {
                let cs = voxelforge_world::sdf_contact::contacts(fa, oa, fb, ob, 24);
                println!(
                    "  CPU-оракул на той же паре: контактов {} · глубины {:?}",
                    cs.len(),
                    cs.iter().map(|c| c.depth).collect::<Vec<_>>()
                );
            }
            break;
        }
    }
}
