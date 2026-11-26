use voxelforge_world::tet::TetBody;
fn b(pts: &[[f32;3]]) -> ([f32;3],[f32;3]) { let mut lo=[f32::MAX;3]; let mut hi=[f32::MIN;3];
  for p in pts { for k in 0..3 { lo[k]=lo[k].min(p[k]); hi[k]=hi[k].max(p[k]); } } (lo,hi) }
fn main(){
    let tb = TetBody::cube([0.0, 0.56, 0.0], 4, 0);
    let frags = tb.shatter_biased(4, 7, [0.0, 0.56, 0.0]);
    for (i, f) in frags.iter().enumerate() {
        let c = f.world_centroid();
        let (mut m, _g) = f.surface_grid();
        for p in m.positions.iter_mut() { for k in 0..3 { p[k] -= c[k]; } }
        let (mlo,mhi) = b(&m.positions);
        let e = f.edge_lines();
        let (elo,ehi) = b(&e);
        println!("осколок {i}: меш x[{:.2},{:.2}] y[{:.2},{:.2}] | каркас x[{:.2},{:.2}] y[{:.2},{:.2}] | c {:?}",
            mlo[0],mhi[0],mlo[1],mhi[1], elo[0],ehi[0],elo[1],ehi[1],
            c.map(|v| (v*100.0).round()/100.0));
    }
}
