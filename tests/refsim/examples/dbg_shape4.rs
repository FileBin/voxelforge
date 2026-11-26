use voxelforge_world::tet::TetBody;
fn b(pts: &[[f32;3]]) -> ([f32;3],[f32;3]) { let mut lo=[f32::MAX;3]; let mut hi=[f32::MIN;3];
  for p in pts { for k in 0..3 { lo[k]=lo[k].min(p[k]); hi[k]=hi[k].max(p[k]); } } (lo,hi) }
fn main(){
    let tb = TetBody::cube([0.0, 0.56, 0.0], 4, 0);
    let frags = tb.shatter_biased(4, 7, [0.0, 0.56, 0.0]);
    for (i, f) in frags.iter().enumerate().take(2) {
        let c = f.world_centroid();
        let (m, g) = f.surface_grid();
        let (mlo,mhi) = b(&m.positions);
        let e = f.edge_lines();
        let (elo,ehi) = b(&e);
        println!("осколок {i}: c {:?}", c);
        println!("   меш СЫРОЙ  x[{:.2},{:.2}] y[{:.2},{:.2}] z[{:.2},{:.2}]", mlo[0],mhi[0],mlo[1],mhi[1],mlo[2],mhi[2]);
        println!("   каркас СЫРОЙ x[{:.2},{:.2}] y[{:.2},{:.2}] z[{:.2},{:.2}]", elo[0],ehi[0],elo[1],ehi[1],elo[2],ehi[2]);
        println!("   grid lo {:?}  centroid {:?}", g.lo, f.centroid());
    }
}
