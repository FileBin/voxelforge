//! Сцены для сверки эталона с GPU: собираются из CPU-солверов, поэтому живут
//! в тестах, а не в движке. Движок строит свой мир сам (см. `bin/voxelforge`).

use voxelforge_world::sim::{params_for, GpuBody, Scene};
use voxelforge_world::types::Body;

pub fn scene_pile(count: usize) -> Scene {
    let sim = crate::pile_demo(count);
    let bodies: Vec<GpuBody> = sim.bodies.iter().map(GpuBody::from_cpu).collect();
    let top = bodies
        .iter()
        .map(|b| b.pos_r[1] + b.half_shape[1])
        .fold(0.0f32, f32::max);
    let params = params_for(&bodies, [-0.05, 0.0, -0.05], [4.2, top + 0.5, 4.2], 0.7);
    Scene {
        name: "pile",
        cpu: sim.bodies,
        bodies,
        params,
    }
}

pub fn scene_drop() -> Scene {
    let sim = crate::ContactSim::new(vec![Body::block([0.0, 3.0, 0.0], [0.5, 0.5, 0.5])]);
    let bodies: Vec<GpuBody> = sim.bodies.iter().map(GpuBody::from_cpu).collect();
    // Трение выключено: эталонный ContactSim его не умеет, иначе сверять нечего.
    let params = params_for(&bodies, [-3.0, 0.0, -3.0], [3.0, 12.0, 3.0], 0.0);
    Scene {
        name: "drop",
        cpu: sim.bodies,
        bodies,
        params,
    }
}

pub fn scene_stack(count: usize) -> Scene {
    let half = [0.5f64, 0.2, 0.5];
    let mut v = Vec::new();
    for i in 0..count {
        let y = half[1] + i as f64 * (2.0 * half[1] + 0.002);
        v.push(Body::block([0.0, y, 0.0], half));
    }
    let sim = crate::ContactSim::new(v);
    let bodies: Vec<GpuBody> = sim.bodies.iter().map(GpuBody::from_cpu).collect();
    let params = params_for(&bodies, [-3.0, 0.0, -3.0], [3.0, 12.0, 3.0], 0.0);
    Scene {
        name: "stack",
        cpu: sim.bodies,
        bodies,
        params,
    }
}

