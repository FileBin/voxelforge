//! GPU-солвер: Vulkan compute через C-ядро, те же проходы, что в C-эталоне.
//!
//! Данные живут в VRAM: позиции, скорости, ориентации, списки контактов,
//! сетка пар. CPU за кадр делает ровно одно дело — отправляет пачку
//! dispatch'ей и (в тестовом режиме) читает обратно результат для сверки.

use std::ffi::{c_char, c_void, CStr};
use std::path::{Path, PathBuf};

#[repr(C)]
pub(crate) struct VfStatus {
    pub(crate) code: i32,
    msg: *const c_char,
}

#[repr(C)]
pub(crate) struct VfDispatch {
    pub(crate) pipeline: *mut c_void,
    pub(crate) buffers: *mut *mut c_void,
    pub(crate) n_buffers: u32,
    pub(crate) push: *const c_void,
    pub(crate) push_bytes: u32,
    pub(crate) groups: [u32; 3],
}

unsafe extern "C" {
    pub(crate) fn vf_init(out: *mut *mut c_void) -> VfStatus;
    pub(crate) fn vf_shutdown(c: *mut c_void);
    pub(crate) fn vf_device_info(c: *mut c_void, name: *mut c_char, len: usize, api: *mut u32) -> VfStatus;
    pub(crate) fn vf_buffer_new(c: *mut c_void, bytes: usize, host: i32, out: *mut *mut c_void) -> VfStatus;
    pub(crate) fn vf_buffer_free(c: *mut c_void, b: *mut c_void);
    pub(crate) fn vf_buffer_write(
        c: *mut c_void,
        b: *mut c_void,
        off: usize,
        src: *const c_void,
        n: usize,
    ) -> VfStatus;
    pub(crate) fn vf_buffer_read(
        c: *mut c_void,
        b: *mut c_void,
        off: usize,
        dst: *mut c_void,
        n: usize,
    ) -> VfStatus;
    pub(crate) fn vf_pipeline_new(
        c: *mut c_void,
        spirv: *const u32,
        bytes: usize,
        n_buf: u32,
        push: u32,
        name: *const c_char,
        out: *mut *mut c_void,
    ) -> VfStatus;
    pub(crate) fn vf_pipeline_free(c: *mut c_void, p: *mut c_void);
    pub(crate) fn vf_run(c: *mut c_void, list: *mut VfDispatch, count: u32) -> VfStatus;
}

pub(crate) fn check_status(s: VfStatus, what: &str) {
    if s.code != 0 {
        let msg = unsafe { CStr::from_ptr(s.msg) }.to_string_lossy();
        panic!("{what}: [{:#x}] {msg}", s.code);
    }
}

/// Раскладка должна совпадать с `Body` в shaders/common.glsl — 112 байт.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct GpuBody {
    pub pos_r: [f32; 4],
    pub vel_m: [f32; 4],
    pub ang_i: [f32; 4],
    pub quat: [f32; 4],
    /// x — таймер покоя, y — флаг сна, z/w — обратные масса и инерция.
    pub misc: [f32; 4],
    /// Позиция на прошлом снимке сна (сон по смещению, как в эталоне).
    pub anchor: [f32; 4],
    /// Полуразмеры и код формы: 0 — сфера, 1 — коробка.
    pub half_shape: [f32; 4],
}

impl GpuBody {
    pub fn from_cpu(b: &crate::Body) -> Self {
        let im = if b.mass > 0.0 { 1.0 / b.mass } else { 0.0 } as f32;
        let shape = match b.shape {
            crate::Shape::Sphere => 0.0,
            crate::Shape::Block => 1.0,
        };
        Self {
            pos_r: [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32, b.radius],
            vel_m: [b.vel[0] as f32, b.vel[1] as f32, b.vel[2] as f32, im],
            ang_i: [
                b.ang[0] as f32,
                b.ang[1] as f32,
                b.ang[2] as f32,
                b.inv_inertia as f32,
            ],
            quat: [b.rot[0] as f32, b.rot[1] as f32, b.rot[2] as f32, b.rot[3] as f32],
            misc: [0.0, 0.0, im, b.inv_inertia as f32],
            anchor: [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32, 0.0],
            half_shape: [b.half[0] as f32, b.half[1] as f32, b.half[2] as f32, shape],
        }
    }
}

/// Готовая сцена: тела плюс параметры ящика и сетки.
pub struct Scene {
    pub name: &'static str,
    /// Воксельный статический коллайдер (поле SDF), если он есть у сцены.
    pub voxel: Option<crate::VoxelBody>,
    /// Тела в представлении эталона: для записи `.vfr` и для сверки.
    pub cpu: Vec<crate::Body>,
    pub bodies: Vec<GpuBody>,
    pub params: Params,
}

fn params_for(bodies: &[GpuBody], lo: [f32; 3], hi: [f32; 3], friction: f32) -> Params {
    /* Ячейка обязана быть не меньше самого крупного тела: иначе пара,
     * которая уже пересекается, окажется в ячейках дальше чем за один шаг. */
    let max_half = bodies
        .iter()
        .map(|b| b.half_shape[0].max(b.half_shape[1]).max(b.half_shape[2]))
        .fold(0.0f32, f32::max);
    Params {
        lo,
        hi,
        cell: (2.0 * max_half * 1.05).max(0.16),
        friction,
        ..Params::default()
    }
}

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
        voxel: None,
        cpu: sim.bodies,
        bodies,
        params,
    }
}

pub fn scene_drop() -> Scene {
    let sim = crate::ContactSim::new(vec![crate::Body::block([0.0, 3.0, 0.0], [0.5, 0.5, 0.5])]);
    let bodies: Vec<GpuBody> = sim.bodies.iter().map(GpuBody::from_cpu).collect();
    // Трение выключено: эталонный ContactSim его не умеет, иначе сверять нечего.
    let params = params_for(&bodies, [-3.0, 0.0, -3.0], [3.0, 12.0, 3.0], 0.0);
    Scene {
        name: "drop",
        voxel: None,
        cpu: sim.bodies,
        bodies,
        params,
    }
}

/// Разрушение: воксельное тело пре-фрактурировано на 8 октантов. Осколки
/// спят в исходной упаковке, пока в них не врежется тяжёлый шар; после удара
/// они просыпаются и разлетаются.
pub fn scene_fracture() -> Scene {
    let center = [0.0f32, 0.5, 0.0];
    let half = [0.5f32, 0.5, 0.5];
    let vox = crate::VoxelBody::cube(center, half, 24, 0.05);
    // Плотность 100 кг/м³: осколок 12.5 кг против снаряда 20 кг — удар
    // пробивает тело, а не подбрасывает его целиком.
    let chunks = vox.fracture_octants(center, half, 100.0);

    let mut cpu = Vec::new();
    for (c, h, m) in &chunks {
        let mut b = crate::Body::block(
            [c[0] as f64, c[1] as f64, c[2] as f64],
            [h[0] as f64, h[1] as f64, h[2] as f64],
        );
        b.mass = *m;
        cpu.push(b);
    }
    /* Снаряд бьёт СБОКУ, а не сверху: удар сверху только прижимает куски
     * друг к другу и к полу — тело остаётся целым, и это физически верно.
     * Боковой удар разносит осколки в стороны, как и должно при расколе. */
    let ball = crate::Body::new([-2.5, 0.75, 0.0], [25.0, 0.0, 0.0], 20.0, 0.25);
    cpu.push(ball);

    let mut bodies: Vec<GpuBody> = cpu.iter().map(GpuBody::from_cpu).collect();
    // Осколки стартуют спящими: тело ещё целое, пока в него не попали.
    for b in bodies.iter_mut().take(8) {
        b.misc[1] = 1.0;
        b.vel_m[3] = 0.0;
        b.ang_i[3] = 0.0;
    }

    // При мягком выталкивании осколки остаются вложенными друг в друга:
    // удар сминает упаковку быстрее, чем солвер успевает её развести.
    let mut params = params_for(&bodies, [-3.0, 0.0, -3.0], [3.0, 5.0, 3.0], 0.5);
    params.max_push = 1.0;
    Scene {
        name: "fracture",
        voxel: None,
        cpu,
        bodies,
        params,
    }
}

/// Тело падает на воксельное тело: поле SDF как статический коллайдер.
pub fn scene_voxel_fall(speed: f32) -> Scene {
    let center = [0.0f32, 0.6, 0.0];
    let radius = 0.5f32;
    let mut body = crate::VoxelBody::sphere(center, radius, 24, 0.05);
    // Уточняем верхнюю часть — там, куда прилетит шар.
    body.refine_region(center, radius, [-0.3, 0.35, -0.3], [0.3, 1.15, 0.3]);

    let ball = crate::Body::new([0.0, 3.0, 0.0], [0.0, -speed as f64, 0.0], 1.0, 0.1);
    let bodies = vec![GpuBody::from_cpu(&ball)];
    // Трение нужно: без него шар на макушке сферы не удержится вовсе.
    let mut params = params_for(&bodies, [-2.0, 0.0, -2.0], [2.0, 4.0, 2.0], 0.8);
    // Мягкое выталкивание: при 1.0 м/с шар на сфере подпрыгивает вечно.
    params.max_push = 0.2;
    Scene {
        name: "voxel",
        voxel: Some(body),
        cpu: vec![ball],
        bodies,
        params,
    }
}

pub fn scene_stack(count: usize) -> Scene {
    let half = [0.5f64, 0.2, 0.5];
    let mut v = Vec::new();
    for i in 0..count {
        let y = half[1] + i as f64 * (2.0 * half[1] + 0.002);
        v.push(crate::Body::block([0.0, y, 0.0], half));
    }
    let sim = crate::ContactSim::new(v);
    let bodies: Vec<GpuBody> = sim.bodies.iter().map(GpuBody::from_cpu).collect();
    let params = params_for(&bodies, [-3.0, 0.0, -3.0], [3.0, 12.0, 3.0], 0.0);
    Scene {
        name: "stack",
        voxel: None,
        cpu: sim.bodies,
        bodies,
        params,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Push {
    prm0: [f32; 4], // dt, gravity, ang_damp, friction
    prm1: [f32; 4], // beta, max_push, margin, cell
    grid: [i32; 4], // nx, ny, nz, n_bodies
    lo: [f32; 4],
    hi: [f32; 4],
}

const P_INTEGRATE: usize = 0;
const P_CLEAR: usize = 1;
const P_COUNT: usize = 2;
const P_SCAN: usize = 3;
const P_FILL: usize = 4;
const P_STATIC: usize = 5;
const P_BROAD: usize = 6;
const P_SOLVE: usize = 7;
const P_SLEEP: usize = 8;
const P_POS: usize = 9;
const P_VOX: usize = 10;

const CAP: usize = 24;
/// Body = 5 × vec4.
const BODY_BYTES: usize = 112;
const WG: u32 = 64;
pub(crate) const BODY_WG: u32 = 64;

struct Step {
    pipe: usize,
    bufs: Vec<*mut c_void>,
    push: Push,
    groups: [u32; 3],
}

/// Параметры, которые обязаны совпадать с CPU-эталоном.
#[derive(Clone, Copy)]
pub struct Params {
    pub gravity: f32,
    pub beta: f32,
    pub max_push: f32,
    pub margin: f32,
    pub friction: f32,
    pub ang_damp: f32,
    pub lo: [f32; 3],
    pub hi: [f32; 3],
    pub cell: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            beta: 0.2,
            max_push: 3.0,
            margin: 5e-4,
            friction: 0.7,
            ang_damp: 1.5,
            lo: [-0.05, 0.0, -0.05],
            hi: [4.2, 3.0, 4.2],
            cell: 0.16,
        }
    }
}

/// Где лежат скомпилированные шейдеры: аргумент → `VF_SHADERS` → два обычных
/// места запуска (из корня репозитория и из каталога крейта).
pub fn find_shaders(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return p.join("solve.spv").exists().then(|| p.to_path_buf());
    }
    if let Ok(p) = std::env::var("VF_SHADERS") {
        let p = PathBuf::from(p);
        if p.join("solve.spv").exists() {
            return Some(p);
        }
    }
    for p in ["shaders/spv", "../../shaders/spv"] {
        let p = PathBuf::from(p);
        if p.join("solve.spv").exists() {
            return Some(p);
        }
    }
    None
}

pub struct GpuSim {
    ctx: *mut c_void,
    pub name: String,
    pub n: usize,
    pub dims: [i32; 3],
    pub ncells: i32,
    pub params: Params,
    bodies: [*mut c_void; 2],
    list: *mut c_void,
    lcount: *mut c_void,
    cellcnt: *mut c_void,
    starts: *mut c_void,
    cursor: *mut c_void,
    cbody: *mut c_void,
    vox_sdf: *mut c_void,
    vox_params: *mut c_void,
    has_voxel: bool,
    pipes: Vec<*mut c_void>,
}

impl GpuSim {
    pub fn new(
        mut bodies: Vec<GpuBody>,
        shader_dir: &Path,
        params: Params,
        voxel: Option<&crate::VoxelBody>,
    ) -> Self {
        let n = bodies.len();
        let mut ctx: *mut c_void = std::ptr::null_mut();
        unsafe { check_status(vf_init(&mut ctx), "vf_init") };

        let mut namebuf = [0i8; 256];
        let mut api = 0u32;
        unsafe { check_status(vf_device_info(ctx, namebuf.as_mut_ptr(), 256, &mut api), "device_info") };
        let name = unsafe { CStr::from_ptr(namebuf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let dims = [
            ((params.hi[0] - params.lo[0]) / params.cell).ceil() as i32,
            ((params.hi[1] - params.lo[1]) / params.cell).ceil() as i32,
            ((params.hi[2] - params.lo[2]) / params.cell).ceil() as i32,
        ];
        let ncells = dims[0] * dims[1] * dims[2];

        let mk = |bytes: usize, host: bool| {
            let mut b: *mut c_void = std::ptr::null_mut();
            unsafe {
                check_status(
                    vf_buffer_new(ctx, bytes, if host { 1 } else { 0 }, &mut b),
                    "buffer_new",
                )
            };
            b
        };
        let bodies_a = mk(n * BODY_BYTES, true);
        let bodies_b = mk(n * BODY_BYTES, true);
        let list = mk(n * CAP * 32, false);
        let lcount = mk(n * 4, true);
        let cellcnt = mk(ncells as usize * 4, false);
        let starts = mk((ncells as usize + 1) * 4, false);
        let cursor = mk(ncells as usize * 4, false);
        let cbody = mk(n * 4, false);
        let (vox_sdf, vox_params) = match voxel {
            Some(v) => {
                let sdf = mk((v.data.len() * 4).max(16), true);
                let prm = mk(64, true);
                unsafe {
                    check_status(
                        vf_buffer_write(ctx, sdf, 0, v.data.as_ptr() as *const c_void, v.data.len() * 4),
                        "upload sdf",
                    )
                };
                let mut params_bytes = [0u8; 64];
                let put4f = |bytes: &mut [u8], off: usize, v: [f32; 4]| {
                    for (k, x) in v.iter().enumerate() {
                        bytes[off + k * 4..off + k * 4 + 4].copy_from_slice(&x.to_le_bytes());
                    }
                };
                let put4i = |bytes: &mut [u8], off: usize, v: [i32; 4]| {
                    for (k, x) in v.iter().enumerate() {
                        bytes[off + k * 4..off + k * 4 + 4].copy_from_slice(&x.to_le_bytes());
                    }
                };
                put4f(&mut params_bytes, 0, [v.base.origin[0], v.base.origin[1], v.base.origin[2], v.base.cell]);
                put4i(&mut params_bytes, 16, [v.base.dim as i32, v.base.offset as i32, 0, 0]);
                match &v.fine {
                    Some(f) => {
                        put4f(&mut params_bytes, 32, [f.origin[0], f.origin[1], f.origin[2], f.cell]);
                        put4i(&mut params_bytes, 48, [f.dim as i32, f.offset as i32, 1, 0]);
                    }
                    None => {
                        put4f(&mut params_bytes, 32, [0.0, 0.0, 0.0, 1.0]);
                        put4i(&mut params_bytes, 48, [1, 0, 0, 0]);
                    }
                }
                unsafe {
                    check_status(
                        vf_buffer_write(ctx, prm, 0, params_bytes.as_ptr() as *const c_void, 64),
                        "upload vox params",
                    )
                };
                (sdf, prm)
            }
            None => (std::ptr::null_mut(), std::ptr::null_mut()),
        };

        unsafe {
            check_status(
                vf_buffer_write(
                    ctx,
                    bodies_a,
                    0,
                    bodies.as_mut_ptr() as *const c_void,
                    n * BODY_BYTES,
                ),
                "upload",
            )
        };

        let names = [
            ("integrate", 1u32),
            ("clear", 2),
            ("count", 2),
            ("scan", 3),
            ("fill", 4),
            ("static_contacts", 3),
            ("broad", 5),
            ("solve", 4),
            ("sleep", 1),
            ("integrate_pos", 1),
            ("voxel", 5),
        ];
        let mut pipes = Vec::new();
        for (nm, nb) in names.iter() {
            let path = shader_dir.join(format!("{nm}.spv"));
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|e| panic!("нет {path:?}: {e} (соберите: glslc)"));
            assert!(bytes.len() % 4 == 0, "SPIR-V не кратен 4 байтам");
            let mut p: *mut c_void = std::ptr::null_mut();
            let cname = std::ffi::CString::new(*nm).unwrap();
            unsafe {
                check_status(
                    vf_pipeline_new(
                        ctx,
                        bytes.as_ptr() as *const u32,
                        bytes.len(),
                        *nb,
                        80,
                        cname.as_ptr(),
                        &mut p,
                    ),
                    nm,
                )
            };
            pipes.push(p);
        }

        Self {
            ctx,
            name,
            n,
            dims,
            ncells,
            params,
            bodies: [bodies_a, bodies_b],
            list,
            lcount,
            cellcnt,
            starts,
            cursor,
            cbody,
            vox_sdf,
            vox_params,
            has_voxel: voxel.is_some(),
            pipes,
        }
    }

    fn push(&self, h: f32, cell_groups: u32) -> Push {
        let _ = cell_groups;
        let p = self.params;
        Push {
            prm0: [h, p.gravity, p.ang_damp, p.friction],
            prm1: [p.beta, p.max_push, p.margin, p.cell],
            grid: [self.dims[0], self.dims[1], self.dims[2], self.n as i32],
            lo: [p.lo[0], p.lo[1], p.lo[2], 0.0],
            hi: [p.hi[0], p.hi[1], p.hi[2], 0.0],
        }
    }

    /// Один кадр = substeps × (интегрирование → сетка → пары → итерации →
    /// позиции). Всё уходит одной командой в GPU, CPU ждёт только fence.
    pub fn frame(&mut self, dt: f32, substeps: u32, iterations: u32) {
        let h = dt / substeps as f32;
        let g_body = ((self.n as u32) + WG - 1) / WG;
        let biggest = (self.n as u32).max(self.ncells as u32);
        let g_cell = (biggest + WG - 1) / WG;
        let push = self.push(h, g_cell);

        let mut steps: Vec<Step> = Vec::new();
        let (a, b) = (self.bodies[0], self.bodies[1]);

        for _ in 0..substeps {
            steps.push(Step {
                pipe: P_INTEGRATE,
                bufs: vec![a],
                push,
                groups: [g_body, 1, 1],
            });
            steps.push(Step {
                pipe: P_CLEAR,
                bufs: vec![self.lcount, self.cellcnt],
                push,
                groups: [g_cell, 1, 1],
            });
            steps.push(Step {
                pipe: P_COUNT,
                bufs: vec![a, self.cellcnt],
                push,
                groups: [g_body, 1, 1],
            });
            steps.push(Step {
                pipe: P_SCAN,
                bufs: vec![self.cellcnt, self.starts, self.cursor],
                push,
                groups: [1, 1, 1],
            });
            steps.push(Step {
                pipe: P_FILL,
                bufs: vec![a, self.cursor, self.starts, self.cbody],
                push,
                groups: [g_body, 1, 1],
            });
            steps.push(Step {
                pipe: P_STATIC,
                bufs: vec![a, self.list, self.lcount],
                push,
                groups: [g_body, 1, 1],
            });
            if self.has_voxel {
                steps.push(Step {
                    pipe: P_VOX,
                    bufs: vec![a, self.list, self.lcount, self.vox_sdf, self.vox_params],
                    push,
                    groups: [g_body, 1, 1],
                });
            }
            steps.push(Step {
                pipe: P_BROAD,
                bufs: vec![a, self.starts, self.cbody, self.list, self.lcount],
                push,
                groups: [g_body, 1, 1],
            });
            // Итерации солвера: чётное число, поэтому данные возвращаются в a.
            for it in 0..iterations {
                let (src, dst) = if it % 2 == 0 { (a, b) } else { (b, a) };
                steps.push(Step {
                    pipe: P_SOLVE,
                    bufs: vec![src, self.list, self.lcount, dst],
                    push,
                    groups: [g_body, 1, 1],
                });
            }
            steps.push(Step {
                pipe: P_SLEEP,
                bufs: vec![a],
                push,
                groups: [g_body, 1, 1],
            });
            steps.push(Step {
                pipe: P_POS,
                bufs: vec![a],
                push,
                groups: [g_body, 1, 1],
            });
        }

        let mut bufs_keep: Vec<Vec<*mut c_void>> =
            steps.iter().map(|s| s.bufs.clone()).collect();
        let push_keep: Vec<Push> = steps.iter().map(|s| s.push).collect();
        let pipes = self.pipes.clone();
        let mut ds: Vec<VfDispatch> = steps
            .iter()
            .enumerate()
            .map(|(i, s)| VfDispatch {
                pipeline: pipes[s.pipe],
                buffers: bufs_keep[i].as_mut_ptr(),
                n_buffers: s.bufs.len() as u32,
                push: &push_keep[i] as *const Push as *const c_void,
                push_bytes: 80,
                groups: s.groups,
            })
            .collect();
        unsafe { check_status(vf_run(self.ctx, ds.as_mut_ptr(), ds.len() as u32), "vf_run") };
    }

    pub fn read_bodies(&mut self) -> Vec<GpuBody> {
        let mut out = vec![GpuBody::default(); self.n];
        unsafe {
            check_status(
                vf_buffer_read(
                    self.ctx,
                    self.bodies[0],
                    0,
                    out.as_mut_ptr() as *mut c_void,
                    self.n * BODY_BYTES,
                ),
                "read_bodies",
            )
        };
        out
    }

    /// Сколько контактов всего (диагностика: если 0 — сетка или пары сломаны).
    pub fn contact_count(&mut self) -> u64 {
        let counts = self.contact_counts();
        counts.iter().map(|&c| (c.min(CAP as u32)) as u64).sum()
    }

    /// Контактов на тело — чтобы видеть переполнение списка (CAP).
    pub fn contact_counts(&mut self) -> Vec<u32> {
        let mut counts = vec![0u32; self.n];
        unsafe {
            check_status(
                vf_buffer_read(
                    self.ctx,
                    self.lcount,
                    0,
                    counts.as_mut_ptr() as *mut c_void,
                    self.n * 4,
                ),
                "read_counts",
            )
        };
        counts
    }
}

/// Мешер на GPU: тот же марширующий тетраэдр, что в CPU-версии, но одним
/// потоком на ячейку. Возвращает число треугольников и вершины (без сварки —
/// для отрисовки она не нужна).
pub fn gpu_mesh(body: &crate::VoxelBody, shader_dir: &Path) -> (usize, Vec<[f32; 3]>) {
    let mut ctx: *mut c_void = std::ptr::null_mut();
    unsafe { check_status(vf_init(&mut ctx), "vf_init") };

    let dim = body.base.dim as usize;
    let cells = (dim - 1) * (dim - 1) * (dim - 1);
    let cap = cells * 36;

    let mk = |bytes: usize| {
        let mut b: *mut c_void = std::ptr::null_mut();
        unsafe { check_status(vf_buffer_new(ctx, bytes, 1, &mut b), "buffer_new") };
        b
    };
    let sdf = mk(body.data.len() * 4);
    let verts = mk((cap * 16).max(16));
    let cnt = mk(4);
    unsafe {
        check_status(
            vf_buffer_write(ctx, sdf, 0, body.data.as_ptr() as *const c_void, body.data.len() * 4),
            "upload sdf",
        );
        let zero = [0u8; 4];
        check_status(vf_buffer_write(ctx, cnt, 0, zero.as_ptr() as *const c_void, 4), "zero cnt");
    }

    let bytes = std::fs::read(shader_dir.join("mesh.spv")).expect("mesh.spv (соберите glslc)");
    let mut pipe: *mut c_void = std::ptr::null_mut();
    let cname = std::ffi::CString::new("mesh").unwrap();
    unsafe {
        check_status(
            vf_pipeline_new(
                ctx,
                bytes.as_ptr() as *const u32,
                bytes.len(),
                3,
                80,
                cname.as_ptr(),
                &mut pipe,
            ),
            "mesh pipeline",
        )
    };

    let mut push = [0u8; 80];
    let put = |buf: &mut [u8], off: usize, v: f32| {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    };
    put(&mut push, 0, body.base.cell);
    put(&mut push, 4, dim as f32);
    put(&mut push, 8, body.base.offset as f32);
    put(&mut push, 16, body.base.origin[0]);
    put(&mut push, 20, body.base.origin[1]);
    put(&mut push, 24, body.base.origin[2]);

    let mut bufs = [sdf, verts, cnt];
    let mut d = VfDispatch {
        pipeline: pipe,
        buffers: bufs.as_mut_ptr(),
        n_buffers: 3,
        push: push.as_ptr() as *const c_void,
        push_bytes: 80,
        groups: [((cells as u32) + 63) / 64, 1, 1],
    };
    unsafe { check_status(vf_run(ctx, &mut d, 1), "mesh run") };

    let mut count = [0u8; 4];
    unsafe {
        check_status(vf_buffer_read(ctx, cnt, 0, count.as_mut_ptr() as *mut c_void, 4), "read cnt")
    };
    let n_verts = u32::from_le_bytes(count) as usize;
    let mut flat = vec![0f32; n_verts * 4];
    unsafe {
        check_status(
            vf_buffer_read(ctx, verts, 0, flat.as_mut_ptr() as *mut c_void, n_verts * 16),
            "read verts",
        )
    };
    let mut out = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        out.push([flat[i * 4], flat[i * 4 + 1], flat[i * 4 + 2]]);
    }

    unsafe {
        vf_pipeline_free(ctx, pipe);
        for b in bufs {
            vf_buffer_free(ctx, b);
        }
        vf_shutdown(ctx);
    }
    (n_verts / 3, out)
}

impl Drop for GpuSim {
    fn drop(&mut self) {
        unsafe {
            for p in &self.pipes {
                vf_pipeline_free(self.ctx, *p);
            }
            if !self.vox_sdf.is_null() {
                vf_buffer_free(self.ctx, self.vox_sdf);
                vf_buffer_free(self.ctx, self.vox_params);
            }
            for b in [
                self.bodies[0],
                self.bodies[1],
                self.list,
                self.lcount,
                self.cellcnt,
                self.starts,
                self.cursor,
                self.cbody,
            ] {
                vf_buffer_free(self.ctx, b);
            }
            vf_shutdown(self.ctx);
        }
    }
}
