// Раскладка тетов по ячейкам на GPU против CPU: тот же набор (тело, тет) в той же ячейке.
use std::collections::HashMap;
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
    let total = w.gpu.run_tet_grid(1.0 / 60.0);
    let (starts, cell_tets) = w.gpu.read_tet_grid();

    let cell = w.params.cell;
    let lo = w.params.lo;
    let dims = w.gpu.dims;
    let idx = |c: [i32; 3]| ((c[2] * dims[1] + c[1]) * dims[0] + c[0]) as usize;

    // CPU-эталон: ячейка -> множество (тело, глобальный индекс тета в пуле)
    let mut cpu: HashMap<usize, Vec<[u32; 2]>> = HashMap::new();
    let mut base = 0u32;
    for i in 0..w.bodies.len() {
        let b = &w.bodies[i];
        if !b.alive { continue; }
        let pos = [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32];
        for (ti, t) in b.rotated_tets().iter().enumerate() {
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
            for x in a[0]..=z[0] { for y in a[1]..=z[1] { for zz in a[2]..=z[2] {
                cpu.entry(idx([x, y, zz])).or_default().push([i as u32, base + ti as u32]);
            } } }
        }
        base += b.local_tets().len() as u32;
    }

    let mut bad = 0usize;
    let mut checked = 0usize;
    for (ci, want) in cpu.iter() {
        let (s, e) = (starts[*ci] as usize, starts[*ci + 1] as usize);
        let mut got: Vec<[u32; 2]> = cell_tets[s..e].to_vec();
        let mut want = want.clone();
        got.sort(); want.sort();
        checked += 1;
        if got != want { bad += 1; }
    }
    println!("записей {total}, ячеек с тетами {} (CPU) / проверено {checked}, расхождений {bad}", cpu.len());
}
