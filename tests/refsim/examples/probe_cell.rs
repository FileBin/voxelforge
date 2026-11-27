// Одна клетка: кто в ней по GPU и кто по CPU. Ищем клетку со снарядом.
use std::collections::HashMap;
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    let bullet = w.shoot([p[0] - 4.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    for _ in 0..20 { w.step(1.0 / 20.0, 10, 18); }

    let local: Vec<Vec<[[f32; 3]; 4]>> = w.bodies.iter().map(|b| b.local_tets().clone()).collect();
    w.gpu.upload_geometry(&local);
    let _ = w.gpu.run_tet_grid();
    let (starts, cells) = w.gpu.read_tet_grid();

    let cell = w.params.cell;
    let lo = w.params.lo;
    let dims = w.gpu.dims;
    let idx = |c: [i32; 3]| ((c[2] * dims[1] + c[1]) * dims[0] + c[0]) as usize;

    // CPU: та же сетка
    let mut cpu: HashMap<usize, Vec<u32>> = HashMap::new();
    for i in 0..w.bodies.len() {
        let b = &w.bodies[i];
        if !b.alive { continue; }
        let pos = [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32];
        for t in b.rotated_tets() {
            let (mut tlo, mut thi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for v in &t { for k in 0..3 { let x = v[k] + pos[k]; tlo[k] = tlo[k].min(x); thi[k] = thi[k].max(x); } }
            let c = |q: f32, k: usize| (((q - lo[k]) / cell).floor() as i32).clamp(0, dims[k] - 1);
            let a = [c(tlo[0],0), c(tlo[1],1), c(tlo[2],2)];
            let z = [c(thi[0],0), c(thi[1],1), c(thi[2],2)];
            for x in a[0]..=z[0] { for y in a[1]..=z[1] { for zz in a[2]..=z[2] { cpu.entry(idx([x,y,zz])).or_default().push(i as u32); } } }
        }
    }

    println!("снаряд: индекс тела {bullet}, тел всего {}, dims {:?}", w.bodies.len(), dims);
    let mut shown = 0;
    for c in 0..(starts.len() - 1) {
        let (s, e) = (starts[c] as usize, starts[c + 1] as usize);
        if e == s { continue; }
        let bodies_gpu: Vec<u32> = cells[s..e].iter().map(|v| v[0]).collect();
        if !bodies_gpu.contains(&(bullet as u32)) { continue; }
        let mut bg = bodies_gpu.clone(); bg.sort(); bg.dedup();
        let mut bc = cpu.get(&c).cloned().unwrap_or_default(); bc.sort(); bc.dedup();
        println!("клетка {c}: записей {}, тела GPU {:?} / CPU {:?}", e - s, bg, bc);
        shown += 1;
        if shown >= 6 { break; }
    }
    if shown == 0 { println!("клеток со снарядом в раскладке НЕТ"); }
}
