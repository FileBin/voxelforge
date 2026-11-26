//! Калибровка воды: какая плотность получается при заданной упаковке.
use voxelforge_refsim::gpu::find_shaders;
use voxelforge_refsim::water::{WaterParams, WaterSim};

fn blob(side: usize, spacing: f32, c: [f32; 3]) -> Vec<[f32; 3]> {
    let mut v = Vec::new();
    let half = 0.5 * (side as f32 - 1.0) * spacing;
    for z in 0..side { for y in 0..side { for x in 0..side {
        v.push([c[0]-half+x as f32*spacing, c[1]-half+y as f32*spacing, c[2]-half+z as f32*spacing]);
    }}}
    v
}

fn main() {
    let shaders = find_shaders(None).expect("spv");
    for (h, sp) in [(0.14f32, 0.06f32), (0.12, 0.05), (0.16, 0.07)] {
        let mut p = WaterParams::default();
        p.h = h; p.stiffness = 0.0; p.k_near = 0.0; p.viscosity = 0.0;
        let mut sim = WaterSim::new(blob(10, sp, [0.0, 2.0, 0.0]), &shaders, p);
        sim.frame(1.0 / 60.0, 1);
        let r = sim.read();
        let mut ds: Vec<f32> = r.iter().map(|q| q[3]).collect();
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean: f32 = ds.iter().sum::<f32>() / ds.len() as f32;
        println!("h={h:.2} шаг={sp:.2}: плотность средняя {mean:.2}, медиана {:.2}, p90 {:.2}",
                 ds[ds.len()/2], ds[ds.len()*9/10]);
    }
}
