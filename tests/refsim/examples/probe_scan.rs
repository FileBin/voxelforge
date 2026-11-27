// Сверка префикс-скана сетки тетов: параллельный против старого однопоточного.
// Запускать дважды — VF_SCAN=serial и без него — и сравнивать отпечаток.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn fnv(data: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in data {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    w.shoot([p[0] + 4.0, p[1] + 0.3, p[2]], [-1.0, 0.0, 0.0], 60.0);
    w.set_sleep(false); // иначе всё уснёт и сетка опустеет
    for _ in 0..3 {
        w.step(1.0 / 20.0, 10, 18);
    }
    let diff = w.gpu.verify_scan();
    let (starts, cells) = w.gpu.read_tet_grid();
    let used = starts[starts.len() - 1];
    let nonempty = starts.windows(2).filter(|p| p[1] > p[0]).count();
    println!(
        "скан {}: стартов {}, итог {}, непустых ячеек {}, записей {}, отпечаток {:016x}",
        if std::env::var("VF_SCAN").is_ok() { "serial" } else { "parallel" },
        starts.len(),
        used,
        nonempty,
        cells.len(),
        fnv(&starts)
    );
    println!(
        "сверка сканов на одних данных: несовпавших ячеек {diff} (0 — реализации совпали)"
    );
}
