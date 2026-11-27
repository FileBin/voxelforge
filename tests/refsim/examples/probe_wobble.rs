// Дрожит ли САМА физика: у движущегося осколка шаг такта должен быть
// монотонным. Если знак приращения каждый такт скачет — это не интерполяция
// кадра, а солвер: тело качается вокруг траектории.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;

fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    // дать плитке расколоться
    for _ in 0..12 {
        w.step(1.0 / 20.0, 10, 18);
    }
    w.refresh_poses();
    let n = w.bodies.len().min(8);
    let mut prev: Vec<[f64; 3]> = w.bodies[..n].iter().map(|b| b.pos).collect();
    let mut flips = 0;
    let mut samples = 0;
    for t in 0..30 {
        for _ in 0..1 {
            w.step(1.0 / 20.0, 10, 18);
        }
        w.refresh_poses();
        let mut line = format!("такт {t:2}:");
        for i in 0..n {
            if !w.bodies[i].alive {
                continue;
            }
            let d = [
                w.bodies[i].pos[0] - prev[i][0],
                w.bodies[i].pos[1] - prev[i][1],
                w.bodies[i].pos[2] - prev[i][2],
            ];
            // шаг по модулю
            let step = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            line += &format!(" {i}:{step:.4}");
        }
        println!("{line}");
        prev = w.bodies[..n].iter().map(|b| b.pos).collect();
        samples += 1;
    }
    // отдельно: знак шага по x у осколка, который медленно едет
    for i in 0..n {
        if !w.bodies[i].alive {
            continue;
        }
        let mut signs = Vec::new();
        let mut p = w.bodies[i].pos;
        for _ in 0..20 {
            w.step(1.0 / 20.0, 10, 18);
            w.refresh_poses();
            if !w.bodies[i].alive {
                break;
            }
            let dx = w.bodies[i].pos[0] - p[0];
            if dx.abs() > 1e-4 {
                signs.push(if dx > 0.0 { 1 } else { -1 });
            }
            p = w.bodies[i].pos;
        }
        if signs.len() >= 4 {
            let mut f = 0;
            for k in 1..signs.len() {
                if signs[k] != signs[k - 1] {
                    f += 1;
                }
            }
            println!(
                "тело {i}: смен знака шага по x {f} из {} (частые смены = качание, а не движение)",
                signs.len() - 1
            );
        }
    }
    println!("(сэмплов {samples}, смен {flips})");
}
