//! GPU-солвер: Vulkan compute через C-ядро, те же проходы, что в C-эталоне.
//!
//! Данные живут в VRAM: позиции, скорости, ориентации, списки контактов,
//! сетка пар. CPU за кадр делает ровно одно дело — отправляет пачку
//! dispatch'ей и (в тестовом режиме) читает обратно результат для сверки.

use crate::types::{Body, Shape};
use std::ffi::{c_char, c_void, CStr};
use std::path::{Path, PathBuf};

#[repr(C)]
pub struct VfStatus {
    pub(crate) code: i32,
    msg: *const c_char,
}

#[repr(C)]
pub struct VfDispatch {
    pub pipeline: *mut c_void,
    pub buffers: *mut *mut c_void,
    pub n_buffers: u32,
    pub push: *const c_void,
    pub push_bytes: u32,
    pub groups: [u32; 3],
}

unsafe extern "C" {
    pub fn vf_init(out: *mut *mut c_void) -> VfStatus;
    pub fn vf_shutdown(c: *mut c_void);
    pub fn vf_device_info(c: *mut c_void, name: *mut c_char, len: usize, api: *mut u32) -> VfStatus;
    pub fn vf_buffer_new(c: *mut c_void, bytes: usize, host: i32, out: *mut *mut c_void) -> VfStatus;
    pub fn vf_buffer_free(c: *mut c_void, b: *mut c_void);
    pub fn vf_buffer_write(
        c: *mut c_void,
        b: *mut c_void,
        off: usize,
        src: *const c_void,
        n: usize,
    ) -> VfStatus;
    pub fn vf_buffer_read(
        c: *mut c_void,
        b: *mut c_void,
        off: usize,
        dst: *mut c_void,
        n: usize,
    ) -> VfStatus;
    pub fn vf_pipeline_new(
        c: *mut c_void,
        spirv: *const u32,
        bytes: usize,
        n_buf: u32,
        push: u32,
        name: *const c_char,
        out: *mut *mut c_void,
    ) -> VfStatus;
    pub fn vf_pipeline_free(c: *mut c_void, p: *mut c_void);
    pub fn vf_run(c: *mut c_void, list: *mut VfDispatch, count: u32) -> VfStatus;
}

pub fn check_status(s: VfStatus, what: &str) {
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
    pub fn from_cpu(b: &Body) -> Self {
        let im = if b.mass > 0.0 { 1.0 / b.mass } else { 0.0 } as f32;
        let shape = match b.shape {
            Shape::Sphere => 0.0,
            Shape::Block => 1.0,
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
    /// Тела в представлении эталона: для записи `.vfr` и для сверки.
    pub cpu: Vec<Body>,
    pub bodies: Vec<GpuBody>,
    pub params: Params,
}

pub fn params_for(bodies: &[GpuBody], lo: [f32; 3], hi: [f32; 3], friction: f32) -> Params {
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

const CAP: usize = 24;
/// Body = 5 × vec4.
pub const BODY_BYTES: usize = 112;
const WG: u32 = 64;
pub const BODY_WG: u32 = 64;

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
    pipes: Vec<*mut c_void>,
}

impl GpuSim {
    pub fn new(
        mut bodies: Vec<GpuBody>,
        shader_dir: &Path,
        params: Params,
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
        // Тела обязаны попасть на устройство до первого dispatch.
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

        let names: [(&str, u32); 10] = [
            ("integrate", 1),
            ("clear", 2),
            ("count", 2),
            ("scan", 3),
            ("fill", 4),
            ("static_contacts", 3),
            ("broad", 5),
            ("solve", 4),
            ("sleep", 1),
            ("integrate_pos", 1),
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

    /// Записать тела обратно на устройство: нужно, когда мир меняется по ходу
    /// прогона (импульс раскола при ударе). Пишем в `bodies[0]` — кадр всегда
    /// начинается с него, а чётное число итераций солвера возвращает данные туда же.
    pub fn write_bodies(&mut self, bodies: &[GpuBody]) {
        assert_eq!(bodies.len(), self.n, "число тел изменилось посреди прогона");
        unsafe {
            check_status(
                vf_buffer_write(
                    self.ctx,
                    self.bodies[0],
                    0,
                    bodies.as_ptr() as *const c_void,
                    self.n * BODY_BYTES,
                ),
                "write_bodies",
            )
        };
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

impl Drop for GpuSim {
    fn drop(&mut self) {
        unsafe {
            for p in &self.pipes {
                vf_pipeline_free(self.ctx, *p);
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
