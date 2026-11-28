fn main() {
    for r in [0.14f32, 0.3] {
        let b = voxelforge_world::tet::TetBody::ball([0.0, 0.0, 0.0], r, 0);
        let f = b.sdf();
        println!("шар r={r}: dims {:?} · step {} · lo {:?} · узлов {}", f.dims, f.step, f.lo, f.val.len());
        println!("  sample в центре: {:+.4} узл · {:+.4} м", f.sample([0.0,0.0,0.0]), f.sample_m([0.0,0.0,0.0]));
    }
    let c = voxelforge_world::tet::TetBody::cube([0.0,0.0,0.0], 2, 1);
    let f = c.sdf();
    println!("куб 2: dims {:?} · step {} · lo {:?} · узлов {}", f.dims, f.step, f.lo, f.val.len());
}
