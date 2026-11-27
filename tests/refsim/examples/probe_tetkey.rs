// Сетка тетов на GPU: число записей должно совпасть с расчётом на CPU.
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    for _ in 0..20 { w.step(1.0 / 20.0, 10, 18); }

    let local: Vec<Vec<[[f32; 3]; 4]>> = w.bodies.iter().map(|b| b.local_tets().clone()).collect();
    w.gpu.upload_geometry(&local);
    let gpu = w.gpu.run_tet_count();

    // то же на CPU: сколько занятых клеток даёт каждый тетраэдр
    let cell = w.params.cell;
    let lo = w.params.lo;
    let dims = w.gpu.dims;
    let idx = |c: [i32; 3]| ((c[2] * dims[1] + c[1]) * dims[0] + c[0]) as usize;
    let mut cnt = vec![0u32; (dims[0] * dims[1] * dims[2]) as usize];
    for i in 0..w.bodies.len() {
        let b = &w.bodies[i];
        if !b.alive { continue; }
        for t in b.rotated_tets() {
            let pos = [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32];
            let (mut tlo, mut thi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for v in t {
                for k in 0..3 {
                    let x = v[k] + pos[k];
                    tlo[k] = tlo[k].min(x);
                    thi[k] = thi[k].max(x);
                }
            }
            let c = |q: f32, k: usize| (((q - lo[k]) / cell).floor() as i32).clamp(0, dims[k] - 1);
            let a = [c(tlo[0], 0), c(tlo[1], 1), c(tlo[2], 2)];
            let z = [c(thi[0], 0), c(thi[1], 1), c(thi[2], 2)];
            for x in a[0]..=z[0] { for y in a[1]..=z[1] { for zz in a[2]..=z[2] { cnt[idx([x, y, zz])] += 1; } } }
        }
    }
    let cpu: usize = cnt.iter().map(|c| *c as usize).sum();
    println!("записей в сетке: GPU {gpu}, CPU {cpu}, разница {}", gpu as i64 - cpu as i64);
}
