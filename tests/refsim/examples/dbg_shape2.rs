use voxelforge_world::tet::TetBody;
fn b(pts: &[[f32;3]]) -> ([f32;3],[f32;3]) { let mut lo=[f32::MAX;3]; let mut hi=[f32::MIN;3];
  for p in pts { for k in 0..3 { lo[k]=lo[k].min(p[k]); hi[k]=hi[k].max(p[k]); } } (lo,hi) }
fn main(){
    let tb = TetBody::cube([0.0, 0.56, 0.0], 2, 0);
    println!("centroid {:?}", tb.centroid());
    println!("world_centroid {:?}", tb.world_centroid());
    let (m, g) = tb.surface_grid();
    let (mlo,mhi) = b(&m.positions);
    println!("меш (как есть)     x[{:.2},{:.2}] y[{:.2},{:.2}]", mlo[0],mhi[0],mlo[1],mhi[1]);
    println!("grid lo {:?} hi {:?}", g.lo, g.hi);
    let e = tb.edge_lines();
    let (elo,ehi) = b(&e);
    println!("каркас (как есть)  x[{:.2},{:.2}] y[{:.2},{:.2}]", elo[0],ehi[0],elo[1],ehi[1]);
}
