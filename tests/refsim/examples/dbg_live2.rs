use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    // та же камера, что в демке
    let pos = [3.6f64, 2.8, 4.6];
    let target = [0.0f64, 1.2, 0.0];
    let d = [target[0]-pos[0], target[1]-pos[1], target[2]-pos[2]];
    let l = (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).sqrt();
    let dir = [d[0]/l, d[1]/l, d[2]/l];
    w.shoot([pos[0]+dir[0]*0.6, pos[1]+dir[1]*0.6, pos[2]+dir[2]*0.6], dir, 26.0);
    for f in 0..40 {
        w.step(1.0/60.0, 4, 10);
        if f % 4 == 0 { println!("кадр {f}: снаряд {:?} v={:?}", w.bodies[1].pos.map(|v| (v*100.0).round()/100.0), w.bodies[1].vel.map(|v| (v*10.0).round()/10.0)); }
    }
    println!("трещин {}, тел {}", w.cracks, w.alive_count());
}
