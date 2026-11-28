// ШАГ ЗА ШАГОМ: шар входит в плиту, подшаги вручную (один подшаг за вызов).
// Здесь видно, на каком подшаге рождается NaN и что в списке контактов.
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
        let at = |y: f32| {
            f.sample_m([
                (0.0 - o[0]) / f.step,
                (y - o[1]) / f.step,
                (0.0 - o[2]) / f.step,
            ])
        };
        let mut y = b.pos[1] as f32;
        while at(y) <= 0.0 && y < b.pos[1] as f32 + 2.0 {
            y += 0.001;
        }
        y as f64
    };
    /* старт: нижняя точка шара на 2 см выше поверхности, скорость 6.4 вниз */
    let start = top + 0.14 + 0.02;
    w.shoot([0.0, start, 0.0], [0.0, -1.0, 0.0], 6.4);
    let ball = *w.bullets.back().unwrap();
    println!("поверхность {top:.4} · старт {start:.4} · шар {ball}");
    {
        /* сверка двух способов адресации поля: через origin (CPU-оракул) и
         * через pos + sdf_shift (то, что делает шейдер) */
        for (k, b) in w.bodies.iter().enumerate().take(2) {
            if let Some(f) = b.sdf.as_ref() {
                let ox = |y: f32| {
                    f.sample_m([
                        (0.0 - b.origin[0]) / f.step,
                        (y - b.origin[1]) / f.step,
                        (0.0 - b.origin[2]) / f.step,
                    ])
                };
                let px = |y: f64| {
                    let l = [0.0 - b.pos[0] - b.sdf_shift[0] as f64, y - b.pos[1] - b.sdf_shift[1] as f64, 0.0 - b.pos[2] - b.sdf_shift[2] as f64];
                    f.sample_m([(l[0] / f.step as f64) as f32, (l[1] / f.step as f64) as f32, (l[2] / f.step as f64) as f32])
                };
                let mut ya = b.pos[1] as f32 - 1.0;
                while ox(ya) <= 0.0 && ya < b.pos[1] as f32 + 1.5 { ya += 0.001; }
                let mut yb = b.pos[1] - 1.0;
                while px(yb) <= 0.0 && yb < b.pos[1] + 1.5 { yb += 0.001; }
                println!(
                    "  тело {k}: dims {:?} step {:.5} lo {:?} shift {:?} pos {:?} · верх через origin {:.4} · через shift {:.4}",
                    f.dims, f.step, f.lo, b.sdf_shift, b.pos, ya, yb
                );
            }
        }
    }
    let h = 0.05 / 8.0;
    for s in 0..24 {
        w.step(h, 1, 12);
        w.gpu.wait_idle();
        let snap = w.gpu.read_bodies();
        let cs = w.gpu.read_contacts();
        let g = &snap[ball];
        let pen = cs[ball].iter().map(|c| c.pen).fold(f32::MIN, f32::max);
        println!(
            "подшаг {s:>2}: y={:>8.4} vy={:>8.3} |v|={:>7.3} · контактов {:>2} · pen {:+.4} · плита y={:>7.3} · misc шар {:?} плита {:?} · w шар {:?}",
            g.pos_r[1],
            g.vel_m[1],
            (g.vel_m[0] * g.vel_m[0] + g.vel_m[1] * g.vel_m[1] + g.vel_m[2] * g.vel_m[2]).sqrt(),
            cs[ball].len(),
            pen,
            snap[0].pos_r[1],
            g.misc,
            snap[0].misc,
            g.ang_i
        );
        for c in cs[ball].iter().take(3) {
            /* сверка: что CPU видит в этой самой точке по ОБОИМ полям */
            let at = |b: &voxelforge_world::live::LiveBody| -> f32 {
                let f = b.sdf.as_ref().unwrap();
                let l = [
                    c.p[0] as f64 - b.pos[0] - b.sdf_shift[0] as f64,
                    c.p[1] as f64 - b.pos[1] - b.sdf_shift[1] as f64,
                    c.p[2] as f64 - b.pos[2] - b.sdf_shift[2] as f64,
                ];
                f.sample_m([(l[0] / f.step as f64) as f32, (l[1] / f.step as f64) as f32, (l[2] / f.step as f64) as f32])
            };
            let (fa, fb) = (at(&w.bodies[0]), at(&w.bodies[ball]));
            println!(
                "      → {}: pen {:+.4} · n [{:+.2} {:+.2} {:+.2}] · p [{:+.3} {:+.3} {:+.3}] · CPU: плита {:+.4} шар {:+.4} (сумма {:+.4})",
                c.other, c.pen, c.n[0], c.n[1], c.n[2], c.p[0], c.p[1], c.p[2], fa, fb, fa + fb
            );
        }
        if !g.pos_r[1].is_finite() {
            println!("  NaN на подшаге {s}");
            break;
        }
    }
}
