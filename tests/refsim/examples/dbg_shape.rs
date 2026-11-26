use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;
fn bounds(pts: &[[f32;3]]) -> ([f32;3],[f32;3]) {
    let mut lo=[f32::MAX;3]; let mut hi=[f32::MIN;3];
    for p in pts { for k in 0..3 { lo[k]=lo[k].min(p[k]); hi[k]=hi[k].max(p[k]); } }
    (lo,hi)
}
fn main(){
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0]-4.0,p[1],p[2]],[1.0,0.0,0.0],26.0);
    for _ in 0..60 { w.step(1.0/60.0,4,10); }
    println!("трещин {}", w.cracks);
    for i in 0..w.bodies.len().min(4) {
        if !w.bodies[i].alive { continue; }
        let s = w.shape_for(i).unwrap();
        let (mlo,mhi) = bounds(&s.mesh.positions);
        let (elo,ehi) = bounds(&s.edges);
        println!("тело {i}: меш x[{:.2},{:.2}] y[{:.2},{:.2}] z[{:.2},{:.2}]", mlo[0],mhi[0],mlo[1],mhi[1],mlo[2],mhi[2]);
        println!("        каркас x[{:.2},{:.2}] y[{:.2},{:.2}] z[{:.2},{:.2}]  точек {} / меш {}",
            elo[0],ehi[0],elo[1],ehi[1],elo[2],ehi[2], s.edges.len(), s.mesh.positions.len());
    }
}
