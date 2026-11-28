// Сверка запекания поля осколка: значения узлов с устройства против той же
// формулы на CPU (max(f_родителя, плоскости)) и марчинг по ним.
//
//   VF_SPEEDS не нужен; печатает по осколкам: узлов, макс. расхождение, объём
//   меша против объёма тетов.
use voxelforge_world::sim::{find_shaders, GpuBody, GpuSim, Params};
use voxelforge_world::tet::{TetBody, Tet};

fn main() {
    let sh = find_shaders(None).unwrap();
    let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
    let parent = plate.sdf().clone();
    let params = Params {
        lo: [-8.0, 0.0, -8.0],
        hi: [8.0, 10.0, 8.0],
        ..Params::default()
    };
    let mut gb = GpuBody::default();
    gb.pos_r = [0.0, 0.6, 0.0, 0.5];
    gb.quat = [1.0, 0.0, 0.0, 0.0];
    gb.misc = [0.0, 0.0, 1.0 / 590.0, 1.0 / 60.0];
    gb.vel_m = [0.0, 0.0, 0.0, 1.0 / 590.0];
    gb.ang_i = [0.0, 0.0, 0.0, 1.0 / 60.0];
    gb.half_shape = [0.56, 0.56, 0.56, 1.0];
    let mut gpu = GpuSim::new(vec![gb], &sh, params);
    println!("устройство: {}", gpu.name);
    gpu.upload_sdf(&[Some((&parent, [0.0; 3]))]);

    let pts = plate.shatter_pts(6, 0xABCD_1234, [0.0, 0.0, 0.0]);
    let bakes = plate.shatter_bakes(&pts);
    println!("осколков: {}", bakes.len());
    for (k, b) in bakes.iter().enumerate() {
        println!("  GPU-бокс {k}: lo {:?} dims {:?} шаг {:.4} узлов {}", b.lo, b.dims, b.step,
                 b.dims[0]*b.dims[1]*b.dims[2]);
    }

    // поле родителя уже в слоте 0; осколки пишем в слоты 1..
    let refs: Vec<(usize, usize, &voxelforge_world::tet::ShardBake)> = bakes
        .iter()
        .enumerate()
        .map(|(k, b)| (0usize, k + 1, b))
        .collect();
    let vals = gpu.bake_shard_fields(&refs, true);
    assert_eq!(vals.len(), bakes.len());
    let vals: Vec<voxelforge_world::sim::BakedShard> =
        vals.into_iter().map(|v| v.expect("осколок без занятой области")).collect();

    let mut worst = 0.0f32;
    for (k, b) in bakes.iter().enumerate() {
        let bk = voxelforge_world::tet::ShardBake {
            lo: vals[k].lo,
            dims: vals[k].dims,
            step: vals[k].step,
            ..(*b).clone()
        };
        let b = &bk;
        let v = &vals[k].vals;
        assert_eq!(v.len(), b.dims[0] * b.dims[1] * b.dims[2]);
        let mut maxdiff = 0.0f32;
        for z in 0..b.dims[2] {
            for y in 0..b.dims[1] {
                for x in 0..b.dims[0] {
                    let p = [
                        (b.lo[0] + x as i32) as f32,
                        (b.lo[1] + y as i32) as f32,
                        (b.lo[2] + z as i32) as f32,
                    ];
                    // та же формула на CPU
                    let pstep = parent.step;
                    let pmin = [parent.lo[0], parent.lo[1], parent.lo[2]];
                    let pmax = [
                        parent.lo[0] + parent.dims[0] as i32 - 1,
                        parent.lo[1] + parent.dims[1] as i32 - 1,
                        parent.lo[2] + parent.dims[2] as i32 - 1,
                    ];
                    let pn = [
                        (p[0] * (b.step / pstep)).round(),
                        (p[1] * (b.step / pstep)).round(),
                        (p[2] * (b.step / pstep)).round(),
                    ];
                    let inside = (0..3).all(|a| pn[a] as i32 >= pmin[a] && pn[a] as i32 <= pmax[a]);
                    let parent_v = if inside {
                        let pnf = [
                            p[0] * (b.step / pstep),
                            p[1] * (b.step / pstep),
                            p[2] * (b.step / pstep),
                        ];
                        parent.sample(pnf) * (pstep / b.step)
                    } else {
                        let far = parent.dims.iter().copied().max().unwrap() as f32 * 2.0;
                        far
                    };
                    let pm = [p[0] * b.step, p[1] * b.step, p[2] * b.step];
                    let mut plane = f32::NEG_INFINITY;
                    for (n, d) in &b.planes {
                        plane = plane.max(n[0] * pm[0] + n[1] * pm[1] + n[2] * pm[2] - d);
                    }
                    let want = parent_v.max(plane / b.step);
                    let got = v[(z * b.dims[1] + y) * b.dims[0] + x];
                    maxdiff = maxdiff.max((want - got).abs());
                }
            }
        }
        worst = worst.max(maxdiff);
        // марчинг по запечённому полю
        let tb = TetBody {
            tets: b.tets.clone(),
            level_cap: 3,
            origin: plate.origin,
            sdf: Default::default(),
        };
        let mut tb = tb;
        let _ = tb.sdf.set(voxelforge_world::field::NodeField {
            lo: b.lo,
            dims: b.dims,
            val: v.clone(),
            step: b.step,
        });
        let m = tb.surface();
        let vol_tets: f32 = b.tets.iter().map(Tet::volume).sum();
        let vol_mesh = mesh_volume(&m);
        println!(
            "осколок {k}: тетов {:4} · узлов {:6} · шаг {:.4} м · макс. расхождение {:.2e} · объём меша {:.4} против тетов {:.4} ({:+.1}%)",
            b.tets.len(),
            v.len(),
            b.step,
            maxdiff,
            vol_mesh,
            vol_tets,
            100.0 * (vol_mesh - vol_tets) / vol_tets.max(1e-9)
        );
    }
    println!("максимум по всем осколкам: {worst:.2e}");

    /* --- для сравнения: тот же раскол, но поле считает CPU --- */
    let cpu = plate.shatter_fields(&pts);
    let mut tot = (0.0f32, 0.0f32);
    for (k, sh) in cpu.iter().enumerate() {
        let f0 = sh.sdf();
        println!("  CPU-бокс {k}: lo {:?} dims {:?} шаг {:.4}", f0.lo, f0.dims, f0.step);
        let m = sh.surface();
        let vol_tets: f32 = sh.tets.iter().map(Tet::volume).sum();
        let vol_mesh = mesh_volume(&m);
        let f = sh.sdf();
        let neg = f.val.iter().filter(|v| **v < 0.0).count();
        tot.0 += vol_mesh;
        tot.1 += vol_tets;
        println!("GPU  {k}: lo {:?} dims {:?} шаг {:.4}", bakes[k].lo, bakes[k].dims, bakes[k].step);
        println!(
            "CPU  {k}: тетов {:4} · узлов {:6} · шаг {:.4} · минусов {:6} · объём меша {:.4} против тетов {:.4} ({:+.1}%)",
            sh.tets.len(),
            f.val.len(),
            f.step,
            neg,
            vol_mesh,
            vol_tets,
            100.0 * (vol_mesh - vol_tets) / vol_tets.max(1e-9)
        );
    }
    println!("CPU всего: меш {:.4} против тетов {:.4} ({:+.1}%)", tot.0, tot.1, 100.0*(tot.0-tot.1)/tot.1);
    let gpu_total: f32 = vals
        .iter()
        .enumerate()
        .map(|(k, v)| {
            let b = &bakes[k];
            let mut tb = TetBody { tets: b.tets.clone(), level_cap: 3, origin: plate.origin, sdf: Default::default() };
            let _ = tb.sdf.set(voxelforge_world::field::NodeField { lo: v.lo, dims: v.dims, val: v.vals.clone(), step: v.step });
            mesh_volume(&tb.surface())
        })
        .sum();
    let tets_total: f32 = bakes.iter().map(|b| b.tets.iter().map(Tet::volume).sum::<f32>()).sum();
    println!("GPU всего: меш {gpu_total:.4} против тетов {tets_total:.4} ({:+.1}%)", 100.0*(gpu_total-tets_total)/tets_total);
}

fn mesh_volume(m: &voxelforge_world::tet::Mesh) -> f32 {
    let mut v = 0.0f64;
    for t in m.indices.chunks(3) {
        let a = m.positions[t[0] as usize];
        let b = m.positions[t[1] as usize];
        let c = m.positions[t[2] as usize];
        v += (a[0] as f64 * (b[1] as f64 * c[2] as f64 - b[2] as f64 * c[1] as f64)
            - a[1] as f64 * (b[0] as f64 * c[2] as f64 - b[2] as f64 * c[0] as f64)
            + a[2] as f64 * (b[0] as f64 * c[1] as f64 - b[1] as f64 * c[0] as f64))
            / 6.0;
    }
    v.abs() as f32
}
