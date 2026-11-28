// Поштучная сверка контактов SDF↔SDF: ОДНА фиксированная пара тел.
//
//   VF_COLLIDE=sdf probe_sdf_pair   — контакты по полям (sdf_contact.comp)
//   VF_COLLIDE=gpu probe_sdf_pair   — контакты по тетам (sat.comp)
//
// Сцена: шар-снаряд r=0.14 утоплен в плиту на заданную глубину, такт
// минимальный — тела не успевают сдвинуться, поэтому числа сравнимы напрямую,
// в том числе с CPU-оракулом sdf_contact::contacts на той же паре.
//
// Глубина отсчитывается от НУЛЯ ПОЛЯ плиты (а не от вершин тетов): поверхность
// физики — это изоповерхность запечённого SDF, и она не совпадает с оболочкой
// тетов (ноль проходит между узлами).
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn main() {
    let mode = std::env::var("VF_COLLIDE").unwrap_or_else(|_| "gpu".into());
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    w.set_fracture(false);
    w.set_sleep(false);
    for _ in 0..4 {
        w.step(0.05, 8, 12);
    }
    w.gpu.wait_idle();

    /* поверхность плиты по полю: ищем смену знака вдоль оси Y над центром */
    let (field_top, plate_f) = {
        let b = &w.bodies[0];
        let f = b.sdf.as_ref().unwrap();
        let o = b.origin;
        let at = |y: f32| -> f32 {
            f.sample_m([(0.0 - o[0]) / f.step, (y - o[1]) / f.step, (0.0 - o[2]) / f.step])
        };
        /* идём от центра плиты (там поле отрицательно) вверх до смены знака */
        let mut y = b.pos[1] as f32;
        while at(y) <= 0.0 && y < b.pos[1] as f32 + 2.0 {
            y += 0.001;
        }
        (y, at(y - 0.02))
    };

    let r = 0.14f64;
    /* Заглубление отсчитываем от НУЛЯ ПОЛЯ: поверхность физики по SDF — это
     * изоповерхность, она не совпадает с оболочкой тетов (у повёрнутой
     * коробки — на ~2 шага узла выше). 0.17 — чтобы перекрытие было
     * однозначным, а не «касание». */
    let want = 0.25f64;
    let from = [0.0, field_top as f64 + r - want, 0.0];
    w.shoot(from, [0.0, 0.0, 0.0], 0.0);
    let ball_slot = *w.bullets.back().unwrap();

    /* один крошечный такт: контакты считаются, тела почти не двигаются */
    w.step(0.0005, 1, 1);
    w.gpu.wait_idle();

    let cs = w.gpu.read_contacts();
    let pair: Vec<_> = cs[ball_slot].iter().filter(|c| c.other == 0).collect();
    let pairs = w.gpu.read_sdf_pairs();
    println!("  широкий проход SDF: пар тел {}", pairs.len());
    let max_pen = pair.iter().map(|c| c.pen).fold(0.0f32, f32::max);
    println!(
        "режим {mode}: ноль поля плиты y={field_top:.3} (f={plate_f:+.3}) · шар утоплен на {want:.3} м"
    );
    println!(
        "  движок: контактов {} · глубина max {max_pen:.4} м",
        pair.len()
    );

    /* CPU-оракул на той же паре (позы — как их залили в мир) */
    let a = w.bodies[0].pos;
    let b = w.bodies[ball_slot].pos;
    let oa = [a[0] as f32, a[1] as f32, a[2] as f32];
    let ob = [b[0] as f32, b[1] as f32, b[2] as f32];
    let cpu = voxelforge_world::sdf_contact::contacts(
        w.bodies[0].sdf.as_ref().unwrap(),
        oa,
        w.bodies[ball_slot].sdf.as_ref().unwrap(),
        ob,
        16,
    );
    let cpu_pen = cpu.iter().map(|c| c.depth).fold(0.0f32, f32::max);
    println!(
        "  CPU-оракул: контактов {} · глубина max {cpu_pen:.4} м · расхождение с движком {:.4} м",
        cpu.len(),
        (max_pen - cpu_pen).abs()
    );
    let step = w.bodies[0].sdf.as_ref().unwrap().step;
    println!(
        "  шаг узла плиты {step:.4} м · ошибка глубины против аналитики {:.4} м",
        (max_pen as f64 - want).abs()
    );
}
