// Точка контакта: куб не должен катиться по полу как шар. Раньше солвер
// брал плечо как n * half_shape, и любое тело вело себя как сфера в ориджине.
use voxelforge_world::collide;
use voxelforge_world::sim::{find_shaders, GpuBody, GpuSim, Params};
use voxelforge_world::tet::TetBody;
use voxelforge_world::types::Body;

fn gpu_ready() -> bool {
    std::env::var("VK_ICD_FILENAMES").map(|v| !v.is_empty()).unwrap_or(false)
        || std::path::Path::new("/dev/dri").exists()
}

#[test]
fn sliding_box_does_not_roll_like_a_ball() {
    if !gpu_ready() {
        eprintln!("нет Vulkan — проверка пропущена");
        return;
    }
    let Some(shaders) = find_shaders(None) else {
        eprintln!("нет SPIR-V — проверка пропущена");
        return;
    };

    // куб 0.28 м лежит на полу и скользит вбок
    let cube = TetBody::cube([0.0, 0.0, 0.0], 2, 1);
    let mut b = Body::block([0.0, 0.141, 0.0], [0.14, 0.14, 0.14]);
    b.mass = 20.0;
    b.vel = [2.0, 0.0, 0.0];
    let gb = GpuBody::from_cpu(&b);
    // Границы ящика задаём явно: у Params::default() lo = [0,0,0], и куб
    // оказывается «внутри» стенки — оттуда ложные контакты и NaN.
    let params = Params {
        friction: 0.6,
        lo: [-8.0, 0.0, -8.0],
        hi: [8.0, 10.0, 8.0],
        ..Params::default()
    };
    let mut gpu = GpuSim::new(vec![gb], &shaders, params);
    let tets = vec![cube.local_tets()];

    let mut turned = 0.0f32;
    let mut travelled = 0.0f32;
    let mut prev_x = 0.0f32;
    for _ in 0..60 {
        let snap = gpu.read_bodies();
        let pos = [[snap[0].pos_r[0], snap[0].pos_r[1], snap[0].pos_r[2]]];
        let cs = collide::build(&tets, &pos, 0.0, (params.lo, params.hi), 0.14);
        gpu.write_contacts(&cs);
        gpu.frame_with_contacts(1.0 / 60.0, 4, 10);
        let snap = gpu.read_bodies();
        let w = [snap[0].ang_i[0], snap[0].ang_i[1], snap[0].ang_i[2]];
        turned += (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt() / 60.0;
        travelled += (snap[0].pos_r[0] - prev_x).abs();
        prev_x = snap[0].pos_r[0];
    }

    // Шар радиусом 0.14 при v = 2 м/с катился бы: ω = v/r ≈ 14 рад/с,
    // то есть за секунду ~14 рад поворота. Куб должен лишь чуть кивнуть.
    assert!(
        turned < 3.0,
        "куб катится как шар: повернулся на {turned:.1} рад (проехал {travelled:.2} м)"
    );
    assert!(
        travelled > 0.05,
        "куб вообще не сдвинулся ({travelled:.3} м) — проверка ничего не значит"
    );
}
