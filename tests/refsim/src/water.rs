//! Вода: частицы + пространственный хеш, без ограничения объёма.
//!
//! Почему не клеточный автомат: КА живёт в решётке, а решётка — это домен.
//! Как только вода выходит за его границы, её нужно либо обрезать, либо
//! расширять сетку (а это перевыделение памяти на каждом шаге). Частицы
//! такого домена не имеют вовсе: положение — просто координата, соседей ищем
//! хешем от координаты. Поэтому вода может уехать куда угодно и никогда не
//! «упирается в стенку мира».
//!
//! Схема — двойная релаксация плотности (Clavet и др.): давление давит наружу,
//! «ближнее» слагаемое не даёт частицам слипнуться, а коррекция позиционная,
//! поэтому шаг интегрирования можно держать крупным.

use std::ffi::c_void;
use std::path::Path;

use crate::gpu::{check_status, VfDispatch, BODY_WG};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct WPush {
    prm0: [f32; 4], // dt, h, rho0, stiffness
    prm1: [f32; 4], // k_near, sigma, floor_y, beta
    prm2: [f32; 4], // n, cap, 0, 0
}

pub struct WaterParams {
    pub h: f32,
    pub rho0: f32,
    pub stiffness: f32,
    pub k_near: f32,
    pub viscosity: f32,
    pub floor_y: f32,
    pub radius: f32,
}

impl Default for WaterParams {
    fn default() -> Self {
        Self {
            h: 0.14,
            /* rho0 измерен пробой при упаковке 0.06 м и h=0.14: среднее 2.79.
             * Жёсткости подобраны так, чтобы коррекция позиции за подшаг была
             * порядка миллиметра — иначе сумма по ~20 соседям телепортирует
             * частицу (первая версия с k=0.1 разнесла воду на 200 м). */
            rho0: 2.8,
            stiffness: 0.0015,
            k_near: 0.0008,
            viscosity: 0.10,
            floor_y: 0.0,
            radius: 0.05,
        }
    }
}

struct Step {
    pipe: usize,
    bufs: Vec<*mut c_void>,
    push: WPush,
    groups: [u32; 3],
}

const P_CLEAR: usize = 0;
const P_INSERT: usize = 1;
const P_PREDICT: usize = 2;
const P_RELAX: usize = 3;
const P_VEL: usize = 4;

pub struct WaterSim {
    ctx: *mut c_void,
    pub n: usize,
    pub params: WaterParams,
    pos: *mut c_void,
    prev: *mut c_void,
    vel: *mut c_void,
    counts: *mut c_void,
    buckets: *mut c_void,
    pipes: Vec<*mut c_void>,
}

impl WaterSim {
    pub fn new(mut positions: Vec<[f32; 3]>, shader_dir: &Path, params: WaterParams) -> Self {
        let n = positions.len();
        let mut ctx: *mut c_void = std::ptr::null_mut();
        unsafe {
            check_status(vf_init(&mut ctx), "vf_init");
        }

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

        // pos и prev — по vec4 на частицу; vel тоже.
        let mut pos4: Vec<[f32; 4]> = positions
            .drain(..)
            .map(|p| [p[0], p[1], p[2], 0.0])
            .collect();
        let pos = mk(n * 16, true);
        let prev = mk(n * 16, true);
        let vel = mk(n * 16, true);
        let counts = mk(32768 * 4, false);
        let buckets = mk(32768 * 12 * 4, false);

        unsafe {
            check_status(
                vf_buffer_write(ctx, pos, 0, pos4.as_mut_ptr() as *const c_void, n * 16),
                "upload pos",
            );
            let zeros = vec![0u8; n * 16];
            check_status(
                vf_buffer_write(ctx, prev, 0, zeros.as_ptr() as *const c_void, n * 16),
                "upload prev",
            );
            check_status(
                vf_buffer_write(ctx, vel, 0, zeros.as_ptr() as *const c_void, n * 16),
                "upload vel",
            );
        }

        let names = [
            ("w_clear", 1u32),
            ("w_insert", 3),
            ("w_predict", 3),
            ("w_relax", 3),
            ("w_vel", 5),
        ];
        let mut pipes = Vec::new();
        for (nm, nb) in names.iter() {
            let path = shader_dir.join(format!("{nm}.spv"));
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|e| panic!("нет {path:?}: {e} (соберите glslc)"));
            let mut p: *mut c_void = std::ptr::null_mut();
            let cname = std::ffi::CString::new(*nm).unwrap();
            unsafe {
                check_status(
                    vf_pipeline_new(
                        ctx,
                        bytes.as_ptr() as *const u32,
                        bytes.len(),
                        *nb,
                        64,
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
            n,
            params,
            pos,
            prev,
            vel,
            counts,
            buckets,
            pipes,
        }
    }

    fn push(&self, h: f32) -> WPush {
        WPush {
            prm0: [h, self.params.h, self.params.rho0, self.params.stiffness],
            prm1: [self.params.k_near, self.params.viscosity, self.params.floor_y, 0.0],
            prm2: [self.n as f32, 12.0, 0.0, 0.0],
        }
    }

    pub fn frame(&mut self, dt: f32, substeps: u32) {
        let sub = dt / substeps as f32;
        let n = self.n as u32;
        let g_particles = (n + BODY_WG - 1) / BODY_WG;
        let g_buckets = (32768u32 + 255) / 256;
        let push = self.push(sub);

        let mut steps: Vec<Step> = Vec::new();
        for _ in 0..substeps {
            steps.push(Step {
                pipe: P_CLEAR,
                bufs: vec![self.counts],
                push,
                groups: [g_buckets, 1, 1],
            });
            steps.push(Step {
                pipe: P_PREDICT,
                bufs: vec![self.pos, self.prev, self.vel],
                push,
                groups: [g_particles, 1, 1],
            });
            steps.push(Step {
                pipe: P_INSERT,
                bufs: vec![self.pos, self.counts, self.buckets],
                push,
                groups: [g_particles, 1, 1],
            });
            steps.push(Step {
                pipe: P_RELAX,
                bufs: vec![self.pos, self.counts, self.buckets],
                push,
                groups: [g_particles, 1, 1],
            });
            steps.push(Step {
                pipe: P_VEL,
                bufs: vec![self.pos, self.prev, self.vel, self.counts, self.buckets],
                push,
                groups: [g_particles, 1, 1],
            });
        }

        let mut bufs_keep: Vec<Vec<*mut c_void>> = steps.iter().map(|s| s.bufs.clone()).collect();
        let push_keep: Vec<WPush> = steps.iter().map(|s| s.push).collect();
        let pipes = self.pipes.clone();
        let mut ds: Vec<VfDispatch> = steps
            .iter()
            .enumerate()
            .map(|(i, s)| VfDispatch {
                pipeline: pipes[s.pipe],
                buffers: bufs_keep[i].as_mut_ptr(),
                n_buffers: s.bufs.len() as u32,
                push: &push_keep[i] as *const WPush as *const c_void,
                push_bytes: 48,
                groups: s.groups,
                stage: s.pipe as u32,
            })
            .collect();
        unsafe {
            check_status(vf_run(self.ctx, ds.as_mut_ptr(), ds.len() as u32), "vf_run");
        }
    }

    /// Позиции и плотности (плотность в w).
    pub fn read(&mut self) -> Vec<[f32; 4]> {
        let mut out = vec![[0.0f32; 4]; self.n];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.pos, 0, out.as_mut_ptr() as *mut c_void, self.n * 16),
                "read water",
            );
        }
        out
    }
}

impl Drop for WaterSim {
    fn drop(&mut self) {
        unsafe {
            for p in &self.pipes {
                vf_pipeline_free(self.ctx, *p);
            }
            for b in [self.pos, self.prev, self.vel, self.counts, self.buckets] {
                vf_buffer_free(self.ctx, b);
            }
            vf_shutdown(self.ctx);
        }
    }
}

use crate::gpu::{vf_buffer_free, vf_buffer_new, vf_buffer_read, vf_buffer_write, vf_init,
                 vf_pipeline_free, vf_pipeline_new, vf_run, vf_shutdown};
