// Сколько байт за ТАКТ уходит между устройством и CPU и сколько стоит такт.
// Это счётчик для плана «снять последнюю синхронизацию»: в спокойном такте
// должно остаться 4 Б на счётчик событий, а не 9 КБ.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    // прогрев: первый такт поднимает геометрию и пары
    for _ in 0..10 {
        w.step(1.0 / 20.0, 10, 18);
    }
    let (up0, down0) = (w.gpu.bytes_up, w.gpu.bytes_down);
    let n = 60;
    let t = std::time::Instant::now();
    for _ in 0..n {
        w.step(1.0 / 20.0, 10, 18);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let up = w.gpu.bytes_up - up0;
    let down = w.gpu.bytes_down - down0;
    println!(
        "такт: {ms:6.2} мс · CPU→GPU {:7} Б ({:.1} Б/такт) · GPU→CPU {:7} Б ({:.1} Б/такт)",
        up,
        up as f64 / n as f64,
        down,
        down as f64 / n as f64
    );
    println!(
        "тел {}, трещин {}, осколков {}",
        w.bodies.iter().filter(|b| b.alive).count(),
        w.cracks,
        w.shards_made
    );
}
