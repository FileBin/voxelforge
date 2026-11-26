// Живой мир без окна: проверяем, что объект раскалывается по удару и
// ломается ПОВТОРНО, когда снаряд попадает в осколок.
use voxelforge_world::live::{Kind, Live};
use voxelforge_world::sim::find_shaders;

fn world() -> Option<Live> {
    // без Vulkan-драйвера (например, в голом CI) проверять нечего
    if std::env::var("VK_ICD_FILENAMES").map(|v| v.is_empty()).unwrap_or(true)
        && !std::path::Path::new("/dev/dri").exists()
    {
        eprintln!("нет Vulkan-драйвера — проверка живого мира пропущена");
        return None;
    }
    let sh = match find_shaders(None) {
        Some(p) => p,
        None => {
            eprintln!("нет SPIR-V (VF_SHADERS) — проверка раскола пропущена");
            return None;
        }
    };
    Some(Live::new(&sh))
}

#[test]
fn first_shot_breaks_the_whole_plate() {
    let Some(mut w) = world() else { return };
    assert_eq!(w.alive_count(), 1, "в начале должен быть один цельный объект");
    assert!(w.bodies[0].tets.len() > 500, "плита — один кусок, а не набор осколков");

    let p = w.bodies[0].pos;
    assert!(w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0));
    for _ in 0..90 {
        w.step(1.0 / 60.0, 4, 10);
    }
    assert!(w.cracks >= 1, "удар не расколол плиту");
    assert!(w.alive_count() >= 3, "осколков слишком мало: {}", w.alive_count());
    assert!(w.shards_made >= 2);
    // осколки должны быть мельче исходной плиты
    let biggest = w
        .bodies
        .iter()
        .filter(|b| b.alive)
        .map(|b| b.tets.len())
        .max()
        .unwrap_or(0);
    assert!(biggest < 500, "после удара остался кусок размером с плиту: {biggest}");
}

#[test]
fn second_shot_breaks_a_shard_again() {
    let Some(mut w) = world() else { return };
    let p0 = w.bodies[0].pos;
    assert!(w.shoot([p0[0] - 4.0, p0[1], p0[2]], [1.0, 0.0, 0.0], 26.0));
    for _ in 0..80 {
        w.step(1.0 / 60.0, 4, 10);
    }
    let first = w.cracks;
    assert!(first >= 1, "первый удар не сработал");

    // целимся в самый крупный живой осколок
    let target = w
        .bodies
        .iter()
        .enumerate()
        .filter(|(_, b)| b.alive && b.kind != Kind::Bullet && b.tets.len() > 20)
        .max_by_key(|(_, b)| b.tets.len())
        .map(|(i, _)| i)
        .expect("нет живых осколков");
    let p = w.bodies[target].pos;
    let from: [f64; 3] = [p[0] - 3.0, p[1] + 0.4, p[2]];
    let d: [f64; 3] = [3.0, -0.4, 0.0];
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    assert!(w.shoot(from, [d[0] / l, d[1] / l, d[2] / l], 30.0));

    for _ in 0..120 {
        w.step(1.0 / 60.0, 4, 10);
    }
    assert!(
        w.cracks > first,
        "второй удар не дал новой трещины: трещин {} → {}",
        first,
        w.cracks
    );
    assert!(w.shards_made > 2);
}

#[test]
fn fast_bullet_does_not_pass_through() {
    let Some(mut w) = world() else { return };
    let p = w.bodies[0].pos;
    // 40 м/с: за кадр снаряд пролетает 0.67 м — больше ячейки, но кадр
    // дробится, поэтому контакт обязан найтись
    assert!(w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 40.0));
    for _ in 0..90 {
        w.step(1.0 / 60.0, 4, 10);
    }
    assert!(w.cracks >= 1, "быстрый снаряд прошёл сквозь плиту");
}

#[test]
fn fragments_settle_without_jitter() {
    let Some(mut w) = world() else { return };
    let p = w.bodies[0].pos;
    assert!(w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0));
    for _ in 0..60 {
        w.step(1.0 / 60.0, 4, 10);
    }
    assert!(w.cracks >= 1, "удар не расколол плиту");
    for _ in 0..360 {
        w.step(1.0 / 60.0, 4, 10);
    }
    assert!(
        w.max_speed() < 0.05,
        "осколки дрожат: скорость {:.3} м/с",
        w.max_speed()
    );
    assert!(
        w.max_pen() < 0.03,
        "тела провалились друг в друга: {:.3} м",
        w.max_pen()
    );
}
