use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    println!("плита: тетов {}, поз {:?}", w.bodies[0].tets.len(), w.bodies[0].pos);
    w.shoot([-4.0, 1.2, 0.0], [1.0, 0.0, 0.0], 26.0);
    println!("снаряд: тетов {}, поз {:?}", w.bodies[1].tets.len(), w.bodies[1].pos);
    for f in 0..40 {
        w.step(1.0 / 60.0, 4, 10);
        if f % 4 == 0 {
            println!("кадр {f}: снаряд {:?} v={:?}", w.bodies[1].pos, w.bodies[1].vel);
        }
    }
    println!("трещин {}, тел {}", w.cracks, w.alive_count());
}
