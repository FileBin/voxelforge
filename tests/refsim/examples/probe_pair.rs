// Одна пара тетраэдров: CPU против GPU.
use voxelforge_world::collide;
use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0], p[1], p[2]], [1.0, 0.0, 0.0], 1.0);
    for _ in 0..2 { w.step(1.0 / 60.0, 4, 10); }

    let local: Vec<Vec<[[f32; 3]; 4]>> = w.bodies.iter().map(|b| b.local_tets().clone()).collect();
    let rotated: Vec<Vec<[[f32; 3]; 4]>> = w.bodies.iter().map(|b| b.rotated_tets()).collect();
    let pos: Vec<[f32; 3]> = w.bodies.iter().map(|b| [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32]).collect();
    let walls = ([-8.0f32, 0.0, -8.0], [8.0f32, 10.0, 8.0]);
    let pairs = collide::build_pairs(&local, &pos, walls, 0.15);
    println!("пар {}", pairs.len());
    // сверка пула: то, что лежит на устройстве, против локальных тетов CPU
    w.gpu.upload_geometry(&local);
    let back = w.gpu.read_geometry(w.bodies.len());
    let mut flat_cpu: Vec<[[f32; 3]; 4]> = Vec::new();
    for b in &local { flat_cpu.extend(b.iter().cloned()); }
    let mut worst = 0.0f32;
    let mut bad_at = 0usize;
    for (k, t) in flat_cpu.iter().enumerate() {
        if k >= back.len() { break; }
        for v in 0..4 {
            for a in 0..3 {
                let d = (back[k][v][a] - t[v][a]).abs();
                if d > worst { worst = d; bad_at = k; }
            }
        }
    }
    println!("пул: тетов CPU {} против {} с устройства, худшее расхождение {:.5} (тет {})", flat_cpu.len(), back.len(), worst, bad_at);
    let gb = w.gpu.read_bodies();
    for b in 0..w.bodies.len() {
        println!("тело {b}: CPU pos y={:.3} quat=[{:.3},{:.3},{:.3},{:.3}] | GPU pos y={:.3} quat=[{:.3},{:.3},{:.3},{:.3}]",
            w.bodies[b].pos[1], w.bodies[b].quat[0], w.bodies[b].quat[1], w.bodies[b].quat[2], w.bodies[b].quat[3],
            gb[b].pos_r[1], gb[b].quat[0], gb[b].quat[1], gb[b].quat[2], gb[b].quat[3]);
    }
    w.gpu.upload_geometry(&local);

    let mut shown = 0;
    for &(b1, t1, b2, t2) in pairs.iter() {
        let (b1, t1, b2, t2) = (b1 as usize, t1 as usize, b2 as usize, t2 as usize);
        // мировой тетраэдр так, как его видит CPU (rotated + pos)
        let wa: [[f32; 3]; 4] = rotated[b1][t1].map(|v| [v[0] + pos[b1][0], v[1] + pos[b1][1], v[2] + pos[b1][2]]);
        let wb: [[f32; 3]; 4] = rotated[b2][t2].map(|v| [v[0] + pos[b2][0], v[1] + pos[b2][1], v[2] + pos[b2][2]]);
        let Some((cn, cpen, cp)) = collide::tet_tet(&wa, &wb) else { continue };
        let sh = collide::tet_tet_shader_style(&wa, &wb);

        w.gpu.set_pairs(&[(((b1 as u32) << 20) | t1 as u32, ((b2 as u32) << 20) | t2 as u32)]);
        w.gpu.run_sat();
        let g = w.gpu.read_contacts();
        let Some(gc) = g[b1].first() else { println!("GPU контакта нет, CPU есть (pen {cpen:.4})"); continue };
        println!("CPU: n=[{:.3},{:.3},{:.3}] pen={:.4} p=[{:.3},{:.3},{:.3}]",
            cn[0], cn[1], cn[2], cpen, cp[0], cp[1], cp[2]);
        match sh {
            Some((sn, sp, spp)) => println!("ШЕЙДЕР-В-RUST: n=[{:.3},{:.3},{:.3}] pen={:.4} p=[{:.3},{:.3},{:.3}]",
                sn[0], sn[1], sn[2], sp, spp[0], spp[1], spp[2]),
            None => println!("ШЕЙДЕР-В-RUST: нет контакта"),
        }
        let meta = w.gpu.read_contacts_meta();
        println!("GPU: n=[{:.3},{:.3},{:.3}] pen={:.4} p=[{:.3},{:.3},{:.3}]  meta(ia,ib,ось)={:?}",
            gc.n[0], gc.n[1], gc.n[2], gc.pen, gc.p[0], gc.p[1], gc.p[2],
            meta[b1].first().map(|m| (m[1], m[2], m[3])));
        println!("   ожидалось: ia={} ib={}", (0..b1).map(|x| local[x].len()).sum::<usize>() + t1,
            (0..b2).map(|x| local[x].len()).sum::<usize>() + t2);
        shown += 1;
        if shown >= 3 { break; }
    }
}
