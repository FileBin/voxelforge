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
    /// Ядро само включает surface-расширения, какие есть в драйвере, и
    /// создаёт девайс с графикой и swapchain — под окно.
    pub fn vf_init_for_window(out: *mut *mut c_void) -> VfStatus;
    pub fn vf_buffer_handle(b: *mut c_void) -> *mut c_void;
    pub fn vf_idle(c: *mut c_void) -> VfStatus;
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
    /// Отправить такт и не ждать: результат читается на следующем.
    pub fn vf_run_async(c: *mut c_void, list: *mut VfDispatch, count: u32) -> VfStatus;
    pub fn vf_wait_back(c: *mut c_void, back: u32) -> VfStatus;
}

pub fn check_status(s: VfStatus, what: &str) {
    if s.code != 0 {
        let msg = unsafe { CStr::from_ptr(s.msg) }.to_string_lossy();
        panic!("{what}: [{:#x}] {msg}", s.code);
    }
}

/// Раскладка должна совпадать с `Body` в shaders/common.glsl — 144 байта.
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
    /// Поза на начало такта — от неё рендер интерполирует между кадрами.
    pub prev_pos_r: [f32; 4],
    pub prev_quat: [f32; 4],
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
            prev_pos_r: [b.pos[0] as f32, b.pos[1] as f32, b.pos[2] as f32, b.radius],
            prev_quat: [b.rot[0] as f32, b.rot[1] as f32, b.rot[2] as f32, b.rot[3] as f32],
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
/// Потолок тетраэдров в пуле: 64 тела по 768.
pub const MAX_POOL_TETS: usize = 64 * 768;
const P_SOLVE_CONTACTS: usize = 10;
const P_SOLVE_APPLY: usize = 11;
const P_SAT: usize = 12;
const P_DEP: usize = 13;
const P_SELECT: usize = 14;
const P_TETKEY: usize = 15;
const P_TETSCAN: usize = 16;
const P_TETFILL: usize = 17;
const P_TETPAIRS: usize = 18;
const P_IMPACT: usize = 19;
const P_BODYAABB: usize = 20;
const P_SNAPSHOT: usize = 21;
const P_STATS: usize = 22;
const P_BODYINIT: usize = 23;
const P_TETBLOCKSUM: usize = 24;
const P_TETBLOCKPREF: usize = 25;
const P_TETSCANP: usize = 26;
const P_TETPAIRSFLAT: usize = 27;
/// Ячеек на блок в параллельном скане (совпадает с 16 * 64 в шейдере).
pub const SCAN_BLOCK: usize = 1024;
/// Потолок изменений тел за одно событие (спавн/деспавн).
pub const MAX_CHANGES: usize = 64;
/// Потолок записей «тет x ячейка» за такт.
pub const MAX_CELL_TETS: usize = 262144;
/// Потолок тел в буферах GPU (совпадает с live::MAX_BODIES).
/// Ёмкость буферов под тела. Число ДЕЙСТВУЮЩИХ тел приходит в pc.grid.w и
/// может быть любым до этой ёмкости: шейдеры уже масштабируются по n, ни
/// вариантов шейдеров, ни повторных запусков не нужно.
pub const MAX_BODIES: usize = 512;
/// Потолок событий раскола в кольце за такт (совпадает с MAX_EVENTS в common.glsl).
pub const MAX_EVENTS: usize = 128;
/// Порог энергии раскола, Дж. Финальное слово — у шейдера (pc.lo.w), CPU
/// страхует только на всякий случай.
pub const FRACTURE_ENERGY: f32 = 500.0;
/// Потолок пар тетраэдров за такт.
pub const MAX_PAIRS: usize = 65536;

/// Контактов на тело. Было 24 — в плотной куче этого мало: часть контактов
/// просто не попадала в список. Память: 64 тела x 64 контакта x 48 Б ≈ 200 КБ.
pub const CAP: usize = 64;
/// Body = 5 × vec4.
pub const BODY_BYTES: usize = 144;
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
    pub bytes_up: u64,
    pub bytes_down: u64,
    /// Отключить сон (для отладки: видно всё, что не может разъехаться).
    pub skip_sleep: bool,
    bodies: [*mut c_void; 2],
    list: *mut c_void,
    imp: *mut c_void,
    /// Геометрия тел НА УСТРОЙСТВЕ: локальные тетраэдры в пуле, поза берётся
    /// из текущего тела. Заливается один раз при спавне/расколе, а не каждый
    /// кадр — иначе коллизия на GPU не избавит от копий.
    tets: *mut c_void,
    /// На тело: смещение в пуле и число тетраэдров.
    span: *mut c_void,
    /// Счётчики сетки тетов (host-visible, для раскладки и сверки).
    tetcnt: *mut c_void,
    /// Префиксные суммы и курсоры раскладки по ячейкам.
    tetstart: *mut c_void,
    tetcursor: *mut c_void,
    /// Раскладка: (тело, индекс тета в пуле) по ячейкам.
    cell_tets: *mut c_void,
    /// Счётчик пар от tet_pairs.
    pair_count: *mut c_void,
    /// Энергия самого сильного удара на тело (2 vec4 на тело).
    impact: *mut c_void,
    /// Кольцо событий раскола (3 vec4 на событие) и его счётчик — CPU
    /// читает 4 Б и только при изменении сами события.
    impact_ring: *mut c_void,
    impact_cnt: *mut c_void,
    /// (uvec2): x — максимум |v| битами, y — сколько тел бодрствует. 8 байт
    /// за такт вместо чтения всех тел обратно.
    stat: *mut c_void,
    /// Очередь изменений тел (шаг 3): номера слотов и сами тела.
    bslots: *mut c_void,
    bstage: *mut c_void,
    /// Частичные суммы блоков для параллельного скана сетки тетов.
    tetsums: *mut c_void,
    /// Генерация пар по записям (равномерно), а не по ячейкам.
    /// VF_PAIRGEN=cell — старый вариант для сверки.
    flat_pairs: bool,
    /// Параллельный скан (по умолчанию). VF_SCAN=serial — старый однопоточный.
    par_scan: bool,
    /// AABB тел для префильтра пар (2 vec4 на тело).
    baabb: *mut c_void,
    /// Пары и их число известны только устройству (SAT берёт число из буфера).
    pub pairs_on_gpu: bool,
    /// Пары тетраэдров, посчитанные CPU один раз на такт.
    pairs: *mut c_void,
    pub n_pairs: usize,
    lcount: *mut c_void,
    cellcnt: *mut c_void,
    starts: *mut c_void,
    cursor: *mut c_void,
    cbody: *mut c_void,
    pipes: Vec<*mut c_void>,
}

impl GpuSim {
    /// Девайс ядра наружу: окно рендерит на нём же, иначе буфер тел
    /// физически недоступен графике (шаг 0 — один общий девайс).
    pub fn ctx(&self) -> *mut c_void {
        self.ctx
    }

    /// Буфер тел (`bodies[0]`) как VkBuffer: из него шейдер окна читает позу,
    /// поэтому CPU не присылает матрицы и не гоняет тела обратно.
    pub fn body_buffer(&self) -> *mut c_void {
        unsafe { vf_buffer_handle(self.bodies[0]) }
    }

    /// Мир без окна: девайс только под compute (тесты, демки, headless).
    pub fn new(bodies: Vec<GpuBody>, shader_dir: &Path, params: Params) -> Self {
        Self::create(bodies, shader_dir, params, false)
    }

    /// Мир с окном: instance сразу с surface-расширениями, девайс — с
    /// графической очередью и swapchain. Окно потом работает на нём же.
    pub fn new_windowed(bodies: Vec<GpuBody>, shader_dir: &Path, params: Params) -> Self {
        Self::create(bodies, shader_dir, params, true)
    }

    fn create(
        mut bodies: Vec<GpuBody>,
        shader_dir: &Path,
        params: Params,
        windowed: bool,
    ) -> Self {
        let n = bodies.len();
        let mut ctx: *mut c_void = std::ptr::null_mut();
        unsafe {
            let st = if windowed { vf_init_for_window(&mut ctx) } else { vf_init(&mut ctx) };
            check_status(st, "vf_init");
        };

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
        // host-visible: контакты может заливать и CPU (столкновения по тетраэдрам)
        let list = mk(n * CAP * 48, true);
        // Две vec4 на точку контакта: приращение скорости и угловое.
        let imp = mk(n * CAP * 32, false);
        // Пул геометрии: 4 vec4 на тетраэдр (a, b, c, d).
        let tets = mk(MAX_POOL_TETS * 64, true);
        let span = mk(n * 8, true);
        let pairs = mk(MAX_PAIRS * 8, true);
        let tetcnt = mk(ncells as usize * 4, true);
        let tetstart = mk((ncells as usize + 1) * 4, true);
        let tetcursor = mk(ncells as usize * 4, false);
        let cell_tets = mk(MAX_CELL_TETS * 8, true);
        let pair_count = mk(4, true);
        let impact = mk(MAX_BODIES * 32, true);
        let impact_ring = mk(MAX_EVENTS * 3 * 16, true);
        let impact_cnt = mk(4, true);
        let stat = mk(8, true);
        /* Очередь изменений тел: номер слота + само тело. Хост-видимая, чтобы
         * CPU клал туда только изменённое, а раскладку делал body_init. */
        let nblocks = (ncells as usize + SCAN_BLOCK - 1) / SCAN_BLOCK + 1;
        let tetsums = mk(nblocks * 4, true);
        let bslots = mk(MAX_CHANGES * 4, true);
        let bstage = mk(MAX_CHANGES * BODY_BYTES, true);
        let baabb = mk(MAX_BODIES * 32, false);
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

        let names: [(&str, u32); 28] = [
            ("integrate", 1),
            ("clear", 2),
            ("count", 2),
            ("scan", 3),
            ("fill", 4),
            ("static_contacts", 5),
            ("broad", 5),
            ("solve", 4),
            ("sleep", 1),
            ("integrate_pos", 1),
            ("solve_contacts", 4),
            ("solve_apply", 4),
            ("sat", 7),
            ("depenetrate", 3),
            ("contact_select", 2),
            ("tet_key", 4),
            ("tet_scan", 3),
            ("tet_fill", 6),
            ("tet_pairs", 8),
            ("impact", 6),
            ("body_aabb", 4),
            ("snapshot", 1),
            ("stats", 2),
            ("body_init", 3),
            ("tet_blocksum", 2),
            ("tet_blockprefix", 2),
            ("tet_scan_p", 4),
            ("tet_pairs_flat", 8),
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
            bytes_up: 0,
            bytes_down: 0,
            skip_sleep: false,
            bodies: [bodies_a, bodies_b],
            list,
            imp,
            tets,
            span,
            tetcnt,
            tetstart,
            tetcursor,
            cell_tets,
            pair_count,
            impact,
            impact_ring,
            impact_cnt,
            stat,
            bslots,
            bstage,
            /* Параллельный скан сетки тетов: раньше один поток перебирал все
             * 262 144 ячейки каждый такт. VF_SCAN=serial возвращает старый
             * проход — он нужен для сверки в probe_scan. */
            tetsums,
            /* Параллельный скан — по умолчанию. Сверка probe_scan на одних
             * данных: расхождений 0 и по start, и по ячейкам. VF_SCAN=serial
             * возвращает старый однопоточный проход (для сверки). */
            flat_pairs: !matches!(std::env::var("VF_PAIRGEN").as_deref(), Ok("cell")),
            par_scan: !matches!(std::env::var("VF_SCAN").as_deref(), Ok("serial")),
            baabb,
            pairs_on_gpu: false,
            pairs,
            n_pairs: 0,
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
            /* lo.w — порог энергии раскола: отдельного vec4 в push-константах
             * нет (128 Б — гарантированный минимум), а это поле пустовало. */
            lo: [p.lo[0], p.lo[1], p.lo[2], FRACTURE_ENERGY],
            hi: [p.hi[0], p.hi[1], p.hi[2], 0.0],
        }
    }

    /// Один кадр = substeps × (интегрирование → сетка → пары → итерации →
    /// позиции). Всё уходит одной командой в GPU, CPU ждёт только fence.
    pub fn frame(&mut self, dt: f32, substeps: u32, iterations: u32) {
        self.frame_ex(dt, substeps, iterations, false)
    }

    /// Кадр с контактами, посчитанными на CPU по тетраэдрам: проходы
    /// `static_contacts` и `broad` пропускаются, список контактов уже залит
    /// через `write_contacts`.
    pub fn frame_with_contacts(&mut self, dt: f32, substeps: u32, iterations: u32) {
        self.frame_ex(dt, substeps, iterations, true)
    }

    /// Залить контакты в буферы солвера. Формат C: `ivec4 meta` (x — партнёр),
    /// `vec4 n_pen` (xyz — нормаль к партнёру, w — проникновение).
    pub fn write_contacts(&mut self, contacts: &[Vec<crate::collide::Contact>]) {
        let cap = CAP;
        let mut lc = vec![0u32; self.n];
        let mut list = vec![0u8; self.n * cap * 48];
        for (i, cs) in contacts.iter().enumerate().take(self.n) {
            lc[i] = cs.len().min(cap) as u32;
            for (k, c) in cs.iter().take(cap).enumerate() {
                let off = (i * cap + k) * 48;
                list[off..off + 4].copy_from_slice(&c.other.to_le_bytes());
                list[off + 4..off + 8].copy_from_slice(&0i32.to_le_bytes());
                list[off + 8..off + 12].copy_from_slice(&0i32.to_le_bytes());
                list[off + 12..off + 16].copy_from_slice(&0i32.to_le_bytes());
                for (j, v) in [c.n[0], c.n[1], c.n[2], c.pen].iter().enumerate() {
                    list[off + 16 + j * 4..off + 20 + j * 4].copy_from_slice(&v.to_le_bytes());
                }
                for (j, v) in [c.p[0], c.p[1], c.p[2], 1.0f32].iter().enumerate() {
                    list[off + 32 + j * 4..off + 36 + j * 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        unsafe {
            check_status(
                vf_buffer_write(self.ctx, self.list, 0, list.as_ptr() as *const c_void, list.len()),
                "write contacts",
            );
            check_status(
                vf_buffer_write(self.ctx, self.lcount, 0, lc.as_ptr() as *const c_void, lc.len() * 4),
                "write contact counts",
            );
        }
    }

    /// Такт целиком на GPU: контакты считает `sat.comp` по парам, которые CPU
    /// залил один раз. Всё уходит ОДНИМ командным буфером, CPU читает тела
    /// один раз в конце — это и есть «убрать синхронизации».
    /// Такт на GPU, но по той же схеме, что CPU-путь: контакты пересобираются
    /// ОДИН раз на «кусок», а внутри куска идёт `per` подшагов с тем же
    /// списком. Раньше контакты пересобирались на каждом подшаге, и куча
    /// вела себя иначе, чем на CPU при том же числе подшагов.
    /// `substeps` — ВСЕГО подшагов за такт (не «на кусок»), `groups` — сколько
    /// раз внутри такта пересобираются контакты. Так число подшагов не зависит
    /// от скорости: раньше `per = substeps/chunks` терял остаток, и при смене
    /// числа кусков такт считался за 10 подшагов или за 7 — тело ехало рывками
    /// (проверено probe_wobble).
    pub fn frame_gpu_contacts(&mut self, dt: f32, groups: u32, substeps: u32, iterations: u32) {
        self.frame_gpu_contacts_impl(dt, groups, substeps, iterations, false)
    }

    /// Такт без ожидания: CPU отправляет и идёт дальше, а статистику и
    /// события читает на СЛЕДУЮЩЕМ такте (см. `wait_prev`).
    pub fn frame_gpu_contacts_async(&mut self, dt: f32, groups: u32, substeps: u32, iterations: u32) {
        self.frame_gpu_contacts_impl(dt, groups, substeps, iterations, true)
    }

    /// Дождаться последнего отправленного такта: перед любой записью тел с
    /// CPU (спавн, раскол, сброс) — иначе побежим по буферу, который считает GPU.
    pub fn wait_idle(&mut self) {
        unsafe { check_status(vf_wait_back(self.ctx, 0), "vf_wait_back") };
    }

    /// Дождаться такта, отправленного предыдущим (к этому моменту он обычно
    /// уже закрыт — физика идёт раз в 50 мс).
    pub fn wait_prev(&mut self) {
        unsafe { check_status(vf_wait_back(self.ctx, 1), "vf_wait_back") };
    }

    fn frame_gpu_contacts_impl(
        &mut self,
        dt: f32,
        chunks: u32,
        substeps: u32,
        iterations: u32,
        async_submit: bool,
    ) {
        /* Подшагов всегда ровно substeps: от скорости зависит только то, как
         * часто пересобираются контакты (группы), а не длина шага. */
        let total = substeps.max(1);
        let groups = chunks.clamp(1, total);
        let per_group = (total + groups - 1) / groups;
        let mut left = total;
        let h = dt / total as f32;
        let g_body = ((self.n as u32) + WG - 1) / WG;
        let g_pairs = if self.pairs_on_gpu {
            ((MAX_PAIRS as u32) + WG - 1) / WG
        } else {
            ((self.n_pairs as u32) + WG - 1) / WG
        };
        let push = self.push(h, g_body);
        {
            let zero = vec![0u8; MAX_BODIES * 32];
            unsafe {
                check_status(
                    vf_buffer_write(self.ctx, self.impact, 0, zero.as_ptr() as *const c_void, zero.len()),
                    "zero_impact",
                );
                /* счётчик событий — 4 байта вместо прежних 2 КБ на такт */
                vf_buffer_write(self.ctx, self.impact_cnt, 0, [0u8; 4].as_ptr() as *const c_void, 4);
                /* статистика такта: 8 Б на запись и 8 Б на чтение */
                vf_buffer_write(self.ctx, self.stat, 0, [0u8; 8].as_ptr() as *const c_void, 8);
            }
        }

        let mut steps: Vec<Step> = Vec::new();
        let (a, b) = (self.bodies[0], self.bodies[1]);

        /* Поза на начало такта: рендер интерполирует от неё, а не от снимка,
         * который CPU раньше читал обратно каждые 50 мс. */
        steps.push(Step {
            pipe: P_SNAPSHOT,
            bufs: vec![a],
            push,
            groups: [g_body, 1, 1],
        });
        /* Максимум скорости и число бодрствующих — для CPU вместо чтения тел. */
        steps.push(Step {
            pipe: P_STATS,
            bufs: vec![a, self.stat],
            push,
            groups: [g_body, 1, 1],
        });

        for _group in 0..groups {
            let n_steps = per_group.min(left);
            if n_steps == 0 {
                break;
            }
            left -= n_steps;
            // контакты — раз на кусок
            steps.push(Step {
                pipe: P_CLEAR,
                bufs: vec![self.lcount, self.cellcnt],
                push,
                groups: [g_body, 1, 1],
            });
            steps.push(Step {
                pipe: P_STATIC,
                bufs: vec![a, self.tets, self.span, self.list, self.lcount],
                push,
                groups: [g_body, 1, 1],
            });
            steps.push(Step {
                pipe: P_SAT,
                bufs: vec![a, self.tets, self.span, self.pairs, self.list, self.lcount, self.pair_count],
                push,
                groups: [g_pairs.max(1), 1, 1],
            });
            steps.push(Step {
                pipe: P_SELECT,
                bufs: vec![self.list, self.lcount],
                push,
                groups: [g_body, 1, 1],
            });
            /* Энергия удара — по снимку скоростей куска; проход идёт на каждом
             * куске и копит максимум (снаряд долетает не в первом). */
            steps.push(Step {
                pipe: P_IMPACT,
                bufs: vec![a, self.list, self.lcount, self.impact, self.impact_ring, self.impact_cnt],
                push,
                groups: [g_body, 1, 1],
            });

            steps.push(Step {
                pipe: P_DEP,
                bufs: vec![a, self.list, self.lcount],
                push,
                groups: [g_body, 1, 1],
            });
            // подшаги внутри куска — по тому же списку контактов
            for _ in 0..n_steps {
                steps.push(Step {
                    pipe: P_INTEGRATE,
                    bufs: vec![a],
                    push,
                    groups: [g_body, 1, 1],
                });
                for it in 0..iterations {
                    let (src, dst) = if it % 2 == 0 { (a, b) } else { (b, a) };
                    steps.push(Step {
                        pipe: P_SOLVE_CONTACTS,
                        bufs: vec![src, self.list, self.lcount, self.imp],
                        push,
                        groups: [((self.n * CAP) as u32 + WG - 1) / WG, 1, 1],
                    });
                    steps.push(Step {
                        pipe: P_SOLVE_APPLY,
                        bufs: vec![src, self.lcount, self.imp, dst],
                        push,
                        groups: [g_body, 1, 1],
                    });
                }
                steps.push(Step {
                    pipe: P_POS,
                    bufs: vec![a],
                    push,
                    groups: [g_body, 1, 1],
                });
            }
        }
        if !self.skip_sleep {
            steps.push(Step {
                pipe: P_SLEEP,
                bufs: vec![a],
                push,
                groups: [g_body, 1, 1],
            });
        }

        let mut keep: Vec<Vec<*mut c_void>> = steps.iter().map(|s| s.bufs.clone()).collect();
        let push_keep: Vec<Push> = steps.iter().map(|s| s.push).collect();
        let pipes = self.pipes.clone();
        let mut ds: Vec<VfDispatch> = steps
            .iter()
            .enumerate()
            .map(|(k, s)| VfDispatch {
                pipeline: pipes[s.pipe],
                buffers: keep[k].as_mut_ptr(),
                n_buffers: s.bufs.len() as u32,
                push: &push_keep[k] as *const Push as *const c_void,
                push_bytes: 80,
                groups: s.groups,
            })
            .collect();
        unsafe {
            let st = if async_submit {
                vf_run_async(self.ctx, ds.as_mut_ptr(), ds.len() as u32)
            } else {
                vf_run(self.ctx, ds.as_mut_ptr(), ds.len() as u32)
            };
            check_status(st, if async_submit { "vf_run_async" } else { "frame_gpu_contacts" });
        }
    }

    fn frame_ex(&mut self, dt: f32, substeps: u32, iterations: u32, external_contacts: bool) {
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
            if !external_contacts {
            steps.push(Step {
                pipe: P_CLEAR,
                bufs: vec![self.lcount, self.cellcnt],
                push,
                groups: [g_cell, 1, 1],
            });
            }
            if !external_contacts {
            steps.push(Step {
                pipe: P_COUNT,
                bufs: vec![a, self.cellcnt],
                push,
                groups: [g_body, 1, 1],
            });
            }
            if !external_contacts {
            steps.push(Step {
                pipe: P_SCAN,
                bufs: vec![self.cellcnt, self.starts, self.cursor],
                push,
                groups: [1, 1, 1],
            });
            }
            if !external_contacts {
            steps.push(Step {
                pipe: P_FILL,
                bufs: vec![a, self.cursor, self.starts, self.cbody],
                push,
                groups: [g_body, 1, 1],
            });
            }
            if !external_contacts {
                steps.push(Step {
                    pipe: P_STATIC,
                    bufs: vec![a, self.tets, self.span, self.list, self.lcount],
                    push,
                    groups: [g_body, 1, 1],
                });
            }
            if !external_contacts {
                steps.push(Step {
                    pipe: P_BROAD,
                    bufs: vec![a, self.starts, self.cbody, self.list, self.lcount],
                    push,
                    groups: [g_body, 1, 1],
                });
            }
            // Итерации солвера: чётное число, поэтому данные возвращаются в a.
            // Два прохода: A — по точкам контакта (работа одинаковая),
            // B — по телам (только сложение). Иначе у тела с 24 контактами
            // работы в 24 раза больше, чем у тела с одним.
            let g_contacts = ((self.n * CAP) as u32 + 63) / 64;
            // VF_SOLVE=bodies — вернуть солвер «по телам» для сравнения.
            let by_contacts = std::env::var("VF_SOLVE")
                .map(|v| v != "bodies")
                .unwrap_or(true);
            for it in 0..iterations {
                let (src, dst) = if it % 2 == 0 { (a, b) } else { (b, a) };
                if by_contacts {
                    // A: по точкам контакта (одинаковая работа), B: по телам (сумма)
                    steps.push(Step {
                        pipe: P_SOLVE_CONTACTS,
                        bufs: vec![src, self.list, self.lcount, self.imp],
                        push,
                        groups: [g_contacts, 1, 1],
                    });
                    steps.push(Step {
                        pipe: P_SOLVE_APPLY,
                        bufs: vec![src, self.lcount, self.imp, dst],
                        push,
                        groups: [g_body, 1, 1],
                    });
                } else {
                    steps.push(Step {
                        pipe: P_SOLVE,
                        bufs: vec![src, self.list, self.lcount, dst],
                        push,
                        groups: [g_body, 1, 1],
                    });
                }
            }
            if !self.skip_sleep {
                steps.push(Step {
                    pipe: P_SLEEP,
                    bufs: vec![a],
                    push,
                    groups: [g_body, 1, 1],
                });
            }
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
    /// Залить геометрию тел в пул на устройстве (локальные тетраэдры) и
    /// записать на каждое тело смещение и число. Вызывается при спавне и
    /// расколе, а не каждый кадр: поза берётся шейдером из самого тела.
    pub fn upload_geometry(&mut self, shapes: &[Vec<[[f32; 3]; 4]>]) {
        let mut flat: Vec<[f32; 4]> = Vec::new();
        let mut spans: Vec<[u32; 2]> = vec![[0, 0]; shapes.len()];
        for (bi, tets) in shapes.iter().enumerate() {
            let start = flat.len() / 4;
            for t in tets {
                for v in t {
                    flat.push([v[0], v[1], v[2], 0.0]);
                }
            }
            let count = flat.len() / 4 - start;
            spans[bi] = [start as u32, count as u32];
        }
        let n = flat.len().min(MAX_POOL_TETS * 4);
        self.bytes_up += (n * 16 + spans.len() * 8) as u64;
        unsafe {
            check_status(
                vf_buffer_write(self.ctx, self.tets, 0, flat.as_ptr() as *const c_void, n * 16),
                "upload_geometry",
            );
            check_status(
                vf_buffer_write(
                    self.ctx,
                    self.span,
                    0,
                    spans.as_ptr() as *const c_void,
                    spans.len() * 8,
                ),
                "upload_spans",
            );
        }
    }

    /// Обратное чтение пула — только для проверок, в кадре не используется.
    pub fn read_geometry(&self, bodies: usize) -> Vec<[[f32; 3]; 4]> {
        let mut spans = vec![[0u32; 2]; bodies];
        unsafe {
            check_status(
                vf_buffer_read(
                    self.ctx,
                    self.span,
                    0,
                    spans.as_mut_ptr() as *mut c_void,
                    bodies * 8,
                ),
                "read_spans",
            );
        }
        let total: usize = spans.iter().map(|s| s[1] as usize).sum();
        let mut flat = vec![[0.0f32; 4]; total * 4];
        unsafe {
            check_status(
                vf_buffer_read(
                    self.ctx,
                    self.tets,
                    0,
                    flat.as_mut_ptr() as *mut c_void,
                    total * 64,
                ),
                "read_tets",
            );
        }
        let mut out = Vec::with_capacity(total);
        let mut k = 0usize;
        for s in spans.iter() {
            for _ in 0..s[1] {
                let mut t = [[0.0f32; 3]; 4];
                for v in 0..4 {
                    t[v] = [flat[k * 4 + v][0], flat[k * 4 + v][1], flat[k * 4 + v][2]];
                }
                out.push(t);
                k += 1;
            }
        }
        out
    }

    /// Пары тетраэдров на такт: `(body<<20|tet, body<<20|tet)`. Добиваем до
    /// кратности 64 пустыми — шейдер их отбросит.
    pub fn set_pairs(&mut self, pairs: &[(u32, u32)]) {
        self.pairs_on_gpu = false;
        let mut flat: Vec<[u32; 2]> = pairs.iter().map(|(a, b)| [*a, *b]).collect();
        while flat.len() % 64 != 0 {
            flat.push([0xFFFF_FFFF, 0xFFFF_FFFF]);
        }
        self.n_pairs = flat.len();
        self.bytes_up += (flat.len() * 8) as u64;
        {
            let n = [flat.len() as u32];
            unsafe {
                check_status(
                    vf_buffer_write(self.ctx, self.pair_count, 0, n.as_ptr() as *const c_void, 4),
                    "set_pair_count",
                );
            }
        }
        unsafe {
            check_status(
                vf_buffer_write(self.ctx, self.pairs, 0, flat.as_ptr() as *const c_void, flat.len() * 8),
                "set_pairs",
            );
        }
    }

    /// Подсчёт тетов по ячейкам (шаг 1 GPU-broadphase). Возвращает общее
    /// число записей «тет × ячейка» — столько будет после раскладки.
    pub fn run_tet_count(&mut self) -> usize {
        let zero = vec![0u8; self.ncells as usize * 4];
        unsafe {
            check_status(
                vf_buffer_write(self.ctx, self.tetcnt, 0, zero.as_ptr() as *const c_void, zero.len()),
                "zero_cells",
            );
        }
        let g_body = ((self.n as u32) + WG - 1) / WG;
        let push = self.push(1.0 / 60.0, g_body);
        let mut steps = vec![Step {
            pipe: P_TETKEY,
            bufs: vec![self.bodies[0], self.tets, self.span, self.tetcnt],
            push,
            groups: [g_body, 1, 1],
        }];
        let mut keep: Vec<Vec<*mut c_void>> = steps.iter().map(|s| s.bufs.clone()).collect();
        let push_keep: Vec<Push> = steps.iter().map(|s| s.push).collect();
        let mut ds: Vec<VfDispatch> = steps
            .iter_mut()
            .enumerate()
            .map(|(k, s)| VfDispatch {
                pipeline: self.pipes[s.pipe],
                buffers: keep[k].as_mut_ptr(),
                n_buffers: keep[k].len() as u32,
                push: &push_keep[k] as *const Push as *const c_void,
                push_bytes: 80,
                groups: s.groups,
            })
            .collect();
        unsafe {
            check_status(vf_run(self.ctx, ds.as_mut_ptr(), ds.len() as u32), "run_tet_count");
        }
        let mut counts = vec![0u32; self.ncells as usize];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.tetcnt, 0, counts.as_mut_ptr() as *mut c_void, counts.len() * 4),
                "read_cells",
            );
        }
        counts.iter().map(|c| *c as usize).sum()
    }

    /// Сетка тетов целиком: подсчёт → префиксные суммы → раскладка.
    /// Возвращает число разложенных записей.
    pub fn run_tet_grid(&mut self, dt: f32) -> usize {
        let zero = vec![0u8; self.ncells as usize * 4];
        unsafe {
            check_status(
                vf_buffer_write(self.ctx, self.tetcnt, 0, zero.as_ptr() as *const c_void, zero.len()),
                "zero_cells",
            );
        }
        let g_body = ((self.n as u32) + WG - 1) / WG;
        let push = self.push(dt, g_body);
        let g_cells = ((self.ncells as u32) + WG - 1) / WG;
        let mut steps = vec![
            Step {
                pipe: P_BODYAABB,
                bufs: vec![self.bodies[0], self.tets, self.span, self.baabb],
                push,
                groups: [g_body, 1, 1],
            },
            Step {
                pipe: P_TETKEY,
                bufs: vec![self.bodies[0], self.tets, self.span, self.tetcnt],
                push,
                groups: [g_body, 1, 1],
            },
            ];
            steps.extend(self.scan_steps(push));
            steps.extend(vec![
            Step {
                pipe: P_TETFILL,
                bufs: vec![
                    self.bodies[0],
                    self.tets,
                    self.span,
                    self.tetstart,
                    self.tetcursor,
                    self.cell_tets,
                ],
                push,
                groups: [g_body, 1, 1],
            },
        ]);
        let _ = g_cells;
        let mut keep: Vec<Vec<*mut c_void>> = steps.iter().map(|s| s.bufs.clone()).collect();
        let push_keep: Vec<Push> = steps.iter().map(|s| s.push).collect();
        let mut ds: Vec<VfDispatch> = steps
            .iter_mut()
            .enumerate()
            .map(|(k, s)| VfDispatch {
                pipeline: self.pipes[s.pipe],
                buffers: keep[k].as_mut_ptr(),
                n_buffers: keep[k].len() as u32,
                push: &push_keep[k] as *const Push as *const c_void,
                push_bytes: 80,
                groups: s.groups,
            })
            .collect();
        unsafe {
            check_status(vf_run(self.ctx, ds.as_mut_ptr(), ds.len() as u32), "run_tet_grid");
        }
        let mut counts = vec![0u32; self.ncells as usize];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.tetcnt, 0, counts.as_mut_ptr() as *mut c_void, counts.len() * 4),
                "read_cells",
            );
        }
        counts.iter().map(|c| *c as usize).sum()
    }

    /// Сверка двух реализаций префикс-скана на ОДНИХ И ТОГ ЖЕ данных:
    /// прогоняет старый однопоточный проход и параллельный, сравнивает
    /// `tetstart` побайтово. Возвращает число несовпавших ячеек.
    pub fn verify_scan(&mut self) -> usize {
        let before = self.par_scan;
        let push = self.push(0.05, 1);
        let run = |this: &mut Self, steps: Vec<Step>| {
            let mut keep: Vec<Vec<*mut c_void>> = steps.iter().map(|s| s.bufs.clone()).collect();
            let push_keep: Vec<Push> = steps.iter().map(|s| s.push).collect();
            let mut ds: Vec<VfDispatch> = steps
                .iter()
                .enumerate()
                .map(|(k, s)| VfDispatch {
                    pipeline: this.pipes[s.pipe],
                    buffers: keep[k].as_mut_ptr(),
                    n_buffers: keep[k].len() as u32,
                    push: &push_keep[k] as *const Push as *const c_void,
                    push_bytes: 80,
                    groups: s.groups,
                })
                .collect();
            unsafe {
                check_status(
                    vf_run(this.ctx, ds.as_mut_ptr(), ds.len() as u32),
                    "verify_scan",
                );
            }
        };
        self.par_scan = false;
        run(self, self.scan_steps(push));
        let (serial, scells) = self.read_tet_grid();
        self.par_scan = true;
        run(self, self.scan_steps(push));
        let (parallel, pcells) = self.read_tet_grid();
        self.par_scan = before;
        assert_eq!(serial.len(), parallel.len());
        let bad_starts = serial.iter().zip(parallel.iter()).filter(|(a, b)| a != b).count();
        let bad_cells = scells.iter().zip(pcells.iter()).filter(|(a, b)| a != b).count();
        if bad_starts + bad_cells > 0 {
            eprintln!(
                "скан: расхождений start {bad_starts}, ячеек {bad_cells} из {}",
                serial.len()
            );
        }
        bad_starts + bad_cells
    }

    /// Раскладка по ячейкам (для сверки с CPU). Возвращает (start, cell_tets).
    pub fn read_tet_grid(&self) -> (Vec<u32>, Vec<[u32; 2]>) {
        let mut starts = vec![0u32; self.ncells as usize + 1];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.tetstart, 0, starts.as_mut_ptr() as *mut c_void, starts.len() * 4),
                "read_starts",
            );
        }
        let total = starts[self.ncells as usize] as usize;
        let mut cell_tets = vec![[0u32; 2]; MAX_CELL_TETS.min(total + 1)];
        unsafe {
            check_status(
                vf_buffer_read(
                    self.ctx,
                    self.cell_tets,
                    0,
                    cell_tets.as_mut_ptr() as *mut c_void,
                    cell_tets.len() * 8,
                ),
                "read_cell_tets",
            );
        }
        (starts, cell_tets)
    }

    /// Пары тетов целиком на устройстве: сетка -> пары. Возвращает их число.
    pub fn run_tet_pairs(&mut self, dt: f32) -> usize {
        let zero = vec![0u8; 4];
        unsafe {
            check_status(
                vf_buffer_write(self.ctx, self.pair_count, 0, zero.as_ptr() as *const c_void, 4),
                "zero_pairs",
            );
        }
        /* Вся сетка и пары — ОДНИМ submit'ом. Раньше было два (сетка, потом
         * пары), и за такт набегало пять ожиданий конвейера: на реальной карте
         * это и держало загрузку у 100% на пустом кадре. */
        let zero_cells = vec![0u8; self.ncells as usize * 4];
        unsafe {
            check_status(
                vf_buffer_write(self.ctx, self.tetcnt, 0, zero_cells.as_ptr() as *const c_void, zero_cells.len()),
                "zero_cells",
            );
        }
        let g_body = ((self.n as u32) + WG - 1) / WG;
        let g_cells = ((self.ncells as u32) + WG - 1) / WG;
        let push = self.push(dt, g_body);
        let mut steps = vec![
            Step {
                pipe: P_BODYAABB,
                bufs: vec![self.bodies[0], self.tets, self.span, self.baabb],
                push,
                groups: [g_body, 1, 1],
            },
            Step {
                pipe: P_TETKEY,
                bufs: vec![self.bodies[0], self.tets, self.span, self.tetcnt],
                push,
                groups: [g_body, 1, 1],
            },
            ];
            steps.extend(self.scan_steps(push));
            steps.extend(vec![
            Step {
                pipe: P_TETFILL,
                bufs: vec![
                    self.bodies[0],
                    self.tets,
                    self.span,
                    self.tetstart,
                    self.tetcursor,
                    self.cell_tets,
                ],
                push,
                groups: [g_body, 1, 1],
            },
            Step {
                pipe: if self.flat_pairs { P_TETPAIRSFLAT } else { P_TETPAIRS },
                bufs: vec![
                    self.bodies[0],
                    self.tets,
                    self.span,
                    self.tetstart,
                    self.cell_tets,
                    self.pairs,
                    self.pair_count,
                    self.baabb,
                ],
                push,
                /* По записям: групп на весь потолок (записей не больше
                 * MAX_CELL_TETS), лишние invocation выходят первой проверкой.
                 * По ячейкам: по одной группе на 64 ячейки, как раньше. */
                groups: [
                    if self.flat_pairs {
                        ((MAX_CELL_TETS as u32) + WG - 1) / WG
                    } else {
                        g_cells
                    },
                    1,
                    1,
                ],
            },
        ]);
        let mut keep: Vec<Vec<*mut c_void>> = steps.iter().map(|s| s.bufs.clone()).collect();
        let push_keep: Vec<Push> = steps.iter().map(|s| s.push).collect();
        let mut ds: Vec<VfDispatch> = steps
            .iter_mut()
            .enumerate()
            .map(|(k, s)| VfDispatch {
                pipeline: self.pipes[s.pipe],
                buffers: keep[k].as_mut_ptr(),
                n_buffers: keep[k].len() as u32,
                push: &push_keep[k] as *const Push as *const c_void,
                push_bytes: 80,
                groups: s.groups,
            })
            .collect();
        unsafe {
            check_status(vf_run(self.ctx, ds.as_mut_ptr(), ds.len() as u32), "run_tet_pairs");
        }
        /* Счётчик не читаем: SAT берёт число пар из буфера. */
        self.pairs_on_gpu = true;
        self.n_pairs = MAX_PAIRS;
        self.n_pairs
    }

    pub fn read_pairs(&self) -> Vec<[u32; 2]> {
        let n = self.n_pairs.max(1);
        let mut out = vec![[0u32; 2]; n];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.pairs, 0, out.as_mut_ptr() as *mut c_void, n * 8),
                "read_pairs",
            );
        }
        out
    }

    /// События раскола из кольца: читаем счётчик 4 Б и, только если он
    /// изменился, сами события (48 Б на событие). В спокойной сцене из такта
    /// уходит 4 байта вместо 2 КБ.
    /// События раскола, накопленные тактом: счётчик 4 Б, и только если он
    /// ненулевой — сами события (48 Б на событие). Счётчик обнуляется в начале
    /// каждого такта, поэтому «сколько уже видел» хранить не нужно.
    pub fn read_impact_events(&mut self) -> Vec<[f32; 12]> {
        let mut cnt = [0u32; 1];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.impact_cnt, 0, cnt.as_mut_ptr() as *mut c_void, 4),
                "read_impact_count",
            );
        }
        self.bytes_down += 4; /* единственный обмен в спокойном такте */
        let n = cnt[0].min(MAX_EVENTS as u32) as usize;
        if n == 0 {
            return Vec::new();
        }
        let mut raw = vec![0f32; MAX_EVENTS * 12];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.impact_ring, 0, raw.as_mut_ptr() as *mut c_void, raw.len() * 4),
                "read_impact_ring",
            );
        }
        self.bytes_down += (n * 48) as u64;
        raw.chunks(12).take(n).map(|c| std::array::from_fn(|k| c[k])).collect()
    }

    /// События удара: на тело — (энергия, партнёр, нормаль, точка).
    /// Оставлено для сверок; в такте не используется.
    pub fn read_impact(&self) -> Vec<[f32; 8]> {
        let mut raw = vec![0f32; MAX_BODIES * 8];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.impact, 0, raw.as_mut_ptr() as *mut c_void, raw.len() * 4),
                "read_impact",
            );
        }
        raw.chunks(8).map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]).collect()
    }

    /// Прогнать только SAT-проход (для сверки с CPU).
    pub fn run_sat(&mut self) {
        let mut zero = vec![0u8; self.n * 4];
        unsafe {
            check_status(
                vf_buffer_write(self.ctx, self.lcount, 0, zero.as_mut_ptr() as *const c_void, self.n * 4),
                "zero_counts",
            );
        }
        let push = self.push(1.0 / 60.0, 1);
        let groups = ((self.n_pairs as u32 + 63) / 64).max(1);
        let mut steps = vec![Step {
            pipe: P_SAT,
            bufs: vec![
                self.bodies[0],
                self.tets,
                self.span,
                self.pairs,
                self.list,
                self.lcount,
            ],
            push,
            groups: [groups, 1, 1],
        }];
        let mut keep: Vec<Vec<*mut c_void>> = steps.iter().map(|s| s.bufs.clone()).collect();
        let push_keep: Vec<Push> = steps.iter().map(|s| s.push).collect();
        let mut ds: Vec<VfDispatch> = steps
            .iter_mut()
            .enumerate()
            .map(|(k, s)| VfDispatch {
                pipeline: self.pipes[s.pipe],
                buffers: keep[k].as_mut_ptr(),
                n_buffers: keep[k].len() as u32,
                push: &push_keep[k] as *const Push as *const c_void,
                push_bytes: std::mem::size_of::<Push>() as u32,
                groups: s.groups,
            })
            .collect();
        unsafe {
            check_status(vf_run(self.ctx, ds.as_mut_ptr(), ds.len() as u32), "run_sat");
        }
    }

    /// Отладочные поля контакта: (партнёр, ia, ib, номер оси).
    pub fn read_contacts_meta(&self) -> Vec<Vec<[i32; 4]>> {
        let mut counts = vec![0u32; self.n];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.lcount, 0, counts.as_mut_ptr() as *mut c_void, self.n * 4),
                "read_counts",
            );
        }
        let mut raw = vec![0u8; self.n * CAP * 48];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.list, 0, raw.as_mut_ptr() as *mut c_void, raw.len()),
                "read_list",
            );
        }
        let mut out: Vec<Vec<[i32; 4]>> = vec![Vec::new(); self.n];
        for b in 0..self.n {
            for k in 0..(counts[b] as usize).min(CAP) {
                let off = (b * CAP + k) * 48;
                let mut m = [0i32; 4];
                for j in 0..4 {
                    m[j] = i32::from_le_bytes(raw[off + j * 4..off + j * 4 + 4].try_into().unwrap());
                }
                out[b].push(m);
            }
        }
        out
    }

    /// Обратное чтение списков контактов (для сверки).
    pub fn read_contacts(&self) -> Vec<Vec<crate::collide::Contact>> {
        let mut counts = vec![0u32; self.n];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.lcount, 0, counts.as_mut_ptr() as *mut c_void, self.n * 4),
                "read_counts",
            );
        }
        let mut raw = vec![0u8; self.n * CAP * 48];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.list, 0, raw.as_mut_ptr() as *mut c_void, raw.len()),
                "read_list",
            );
        }
        let mut out: Vec<Vec<crate::collide::Contact>> = vec![Vec::new(); self.n];
        for b in 0..self.n {
            for k in 0..(counts[b] as usize).min(CAP) {
                let off = (b * CAP + k) * 48;
                let other = i32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
                let n = [
                    f32::from_le_bytes(raw[off + 16..off + 20].try_into().unwrap()),
                    f32::from_le_bytes(raw[off + 20..off + 24].try_into().unwrap()),
                    f32::from_le_bytes(raw[off + 24..off + 28].try_into().unwrap()),
                ];
                let pen = f32::from_le_bytes(raw[off + 28..off + 32].try_into().unwrap());
                let p = [
                    f32::from_le_bytes(raw[off + 32..off + 36].try_into().unwrap()),
                    f32::from_le_bytes(raw[off + 36..off + 40].try_into().unwrap()),
                    f32::from_le_bytes(raw[off + 40..off + 44].try_into().unwrap()),
                ];
                out[b].push(crate::collide::Contact { other, n, pen, p });
            }
        }
        out
    }

    /// Шаги префикс-скана сетки тетов. Параллельный вариант: частичные суммы
    /// по блокам 64 потоками, префикс по 256 суммам, развёртка по блокам.
    /// Старый однопоточный проход оставлен под VF_SCAN=serial для сверки.
    fn scan_steps(&self, push: Push) -> Vec<Step> {
        if !self.par_scan {
            return vec![Step {
                pipe: P_TETSCAN,
                bufs: vec![self.tetcnt, self.tetstart, self.tetcursor],
                push,
                groups: [1, 1, 1],
            }];
        }
        let g = ((self.ncells as u32) + SCAN_BLOCK as u32 - 1) / SCAN_BLOCK as u32;
        vec![
            Step {
                pipe: P_TETBLOCKSUM,
                bufs: vec![self.tetcnt, self.tetsums],
                push,
                groups: [g, 1, 1],
            },
            Step {
                pipe: P_TETBLOCKPREF,
                bufs: vec![self.tetsums, self.tetstart],
                push,
                groups: [1, 1, 1],
            },
            Step {
                pipe: P_TETSCANP,
                bufs: vec![self.tetcnt, self.tetsums, self.tetstart, self.tetcursor],
                push,
                groups: [g, 1, 1],
            },
        ]
    }

    /// Записать только изменённые тела (спавн/деспавн) — вместо заливки
    /// всего буфера: CPU кладёт в очередь номер слота и само тело, а
    /// раскладку делает проход body_init на устройстве.
    pub fn write_slots(&mut self, items: &[(usize, GpuBody)]) {
        if items.is_empty() || self.n == 0 {
            return;
        }
        let n = items.len().min(MAX_CHANGES);
        let mut slots: Vec<u32> = Vec::with_capacity(n);
        let mut stage: Vec<u8> = Vec::with_capacity(n * BODY_BYTES);
        for (slot, b) in items.iter().take(n) {
            slots.push(*slot as u32);
            let bytes = unsafe {
                std::slice::from_raw_parts(b as *const GpuBody as *const u8, BODY_BYTES)
            };
            stage.extend_from_slice(bytes);
        }
        self.bytes_up += (n * BODY_BYTES) as u64;
        let mut push = self.push(0.0, 0);
        push.hi[3] = n as f32;
        push.lo[3] = MAX_BODIES as f32;
        unsafe {
            check_status(
                vf_buffer_write(
                    self.ctx,
                    self.bslots,
                    0,
                    slots.as_ptr() as *const c_void,
                    slots.len() * 4,
                ),
                "write_slots/slots",
            );
            check_status(
                vf_buffer_write(
                    self.ctx,
                    self.bstage,
                    0,
                    stage.as_ptr() as *const c_void,
                    stage.len(),
                ),
                "write_slots/stage",
            );
        }
        let mut bufs = [self.bslots, self.bstage, self.bodies[0]];
        let mut d = VfDispatch {
            pipeline: self.pipes[P_BODYINIT],
            buffers: bufs.as_mut_ptr(),
            n_buffers: 3,
            push: &push as *const Push as *const c_void,
            push_bytes: 80,
            groups: [((n as u32) + 15) / 16, 1, 1],
        };
        unsafe {
            check_status(vf_run(self.ctx, &mut d, 1), "write_slots");
        }
    }

    pub fn write_bodies(&mut self, bodies: &[GpuBody]) {
        self.bytes_up += (bodies.len() * BODY_BYTES) as u64;
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

    /// Две цифры такта (8 Б): максимум |v| и число бодрствующих тел.
    /// Заменяет чтение всех тел, когда позы CPU не нужны.
    pub fn read_stats(&mut self) -> (f32, u32) {
        let mut raw = [0u32; 2];
        self.bytes_down += 8;
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.stat, 0, raw.as_mut_ptr() as *mut c_void, 8),
                "read_stats",
            );
        }
        (f32::from_bits(raw[0]), raw[1])
    }

    /// Одно тело обратно — только по событию (раскол, включение каркаса),
    /// а не в такте.
    pub fn read_body(&mut self, i: usize) -> Option<GpuBody> {
        if i >= self.n {
            return None;
        }
        let mut out = GpuBody::default();
        self.bytes_down += BODY_BYTES as u64;
        unsafe {
            check_status(
                vf_buffer_read(
                    self.ctx,
                    self.bodies[0],
                    i * BODY_BYTES,
                    &mut out as *mut GpuBody as *mut c_void,
                    BODY_BYTES,
                ),
                "read_body",
            );
        }
        Some(out)
    }

    pub fn read_bodies(&mut self) -> Vec<GpuBody> {
        self.bytes_down += (self.n * BODY_BYTES) as u64;
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
            /* Такт отправляется асинхронно: сначала дождаться, потом трогать
             * буферы. Без этого драйвер падал на втором мире в одном процессе
             * (segfault в probe_fastshot после первого прогона). */
            check_status(vf_idle(self.ctx), "vf_idle");
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
                self.imp,
                self.tets,
                self.span,
                self.pairs,
                self.tetcnt,
                self.tetstart,
                self.tetcursor,
                self.cell_tets,
                self.pair_count,
                self.impact,
                self.impact_ring,
                self.impact_cnt,
                self.stat,
                self.baabb,
            ] {
                vf_buffer_free(self.ctx, b);
            }
            vf_shutdown(self.ctx);
        }
    }
}
