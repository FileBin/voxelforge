use voxelforge_world::live::Live;
use voxelforge_world::sim::find_shaders;
fn main() {
    let sh = find_shaders(None).unwrap();
    let mut w = Live::new(&sh);
    let p = w.bodies[0].pos;
    w.shoot([p[0]-4.0,p[1],p[2]],[1.0,0.0,0.0],26.0);
    for _ in 0..60 { w.step(1.0/60.0,4,10); }
    for f in 0..800 {
        w.step(1.0/60.0,4,10);
        if f % 200 == 0 || f == 799 {
            let mut vs: Vec<f64> = w.bodies.iter().filter(|b| b.alive)
                .map(|b| (b.vel[0]*b.vel[0]+b.vel[1]*b.vel[1]+b.vel[2]*b.vel[2]).sqrt()).collect();
            vs.sort_by(|a,b| b.partial_cmp(a).unwrap());
            println!("кадр {f}: скорость max {:.3} | пять самых быстрых {:?}", vs.first().copied().unwrap_or(0.0),
                vs.iter().take(5).map(|v| (v*1000.0).round()/1000.0).collect::<Vec<_>>());
        }
    }
    println!("проникновение {:.3} м, тел {}", w.max_pen(), w.alive_count());
    for (i,b) in w.bodies.iter().enumerate() {
        if !b.alive { continue; }
        let v = (b.vel[0]*b.vel[0]+b.vel[1]*b.vel[1]+b.vel[2]*b.vel[2]).sqrt();
        if v > 0.02 { println!("  тело {i}: v {:.3} м/с, поз {:?}", v, b.pos.map(|x| (x*100.0).round()/100.0)); }
    }
}
