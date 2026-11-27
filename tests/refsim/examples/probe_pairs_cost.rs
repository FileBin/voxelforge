// Стоимость генерации пар по сетке: сколько записей в ячейках, сколько
// итераций внутреннего цикла (O(k^2) на ячейку) и сколько из них вообще
// осмысленны (пары РАЗНЫХ тел).
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn stats(w: &mut Live) -> (usize, u64, u64) {
    let (starts, cells) = w.gpu.read_tet_grid();
    let mut entries = 0usize;
    let mut iters = 0u64;
    let mut cross = 0u64;
    for c in 0..starts.len().saturating_sub(1) {
        let s = starts[c] as usize;
        let e = starts[c + 1] as usize;
        let k = e.saturating_sub(s);
        entries += k;
        iters += (k * k.saturating_sub(1) / 2) as u64;
        for i in s..e.min(cells.len()) {
            for j in (i + 1)..e.min(cells.len()) {
                if cells[i][0] != cells[j][0] {
                    cross += 1;
                }
            }
        }
    }
    (entries, iters, cross)
}

fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    w.set_sleep(false);
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);

    for tick in 0..12 {
        w.step(1.0 / 20.0, 10, 18);
        if tick == 0 || tick == 2 || tick == 5 || tick == 11 {
            let (entries, iters, cross) =
                if std::env::var("VF_PAIRS").is_ok() { stats(&mut w) } else { (0, 0, 0) };
            println!(
                "такт {tick:2}: тел {:2}, трещин {}, записей в сетке {entries:6}, итераций пар {iters:9}, из них разных тел {cross:7}",
                w.bodies.iter().filter(|b| b.alive).count(),
                w.cracks
            );
        }
    }
}
