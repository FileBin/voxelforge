// Такт на GPU без единой пары: если скорость всё равно исчезает — дело не в контактах.
use voxelforge_world::live::{Kind, Live};
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0] - 3.0, p[1], p[2]], [1.0, 0.0, 0.0], 26.0);
    let slot = w.bodies.iter().position(|b| b.alive && b.kind == Kind::Bullet).unwrap();
    let local: Vec<Vec<[[f32; 3]; 4]>> = w.bodies.iter().map(|b| b.local_tets().clone()).collect();
    w.gpu.upload_geometry(&local);
    w.gpu.set_pairs(&[]);                 // ни одной пары → контактов быть не может
    w.gpu.frame_gpu_contacts(0.05, 6, 18);
    let gb = w.gpu.read_bodies();
    println!("после такта без пар: vel_m={:?} pos={:?}",
        [gb[slot].vel_m[0], gb[slot].vel_m[1], gb[slot].vel_m[2]],
        [gb[slot].pos_r[0], gb[slot].pos_r[1]]);
    let c = w.gpu.read_contacts();
    println!("контактов у снаряда: {} (у плиты {})", c[slot].len(), c[0].len());
}
