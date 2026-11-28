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

/// Ожидание значения таймлайна другого потока (см. `vf_run_stream`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VfSemWait {
    pub stream: u32,
    pub value: u64,
}

#[repr(C)]
pub struct VfDispatch {
    pub pipeline: *mut c_void,
    pub buffers: *mut *mut c_void,
    pub n_buffers: u32,
    pub push: *const c_void,
    pub push_bytes: u32,
    pub groups: [u32; 3],
    /// Номер стадии для диагностики (ядро возвращает его с таймстемпом).
    pub stage: u32,
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
    /// Submit в конкретный поток: 0 — такт (compute, приоритет 0.0), 1 — кадр
    /// (graphics, приоритет 1.0). `waits` — чужие значения таймлайнов, которых
    /// этот submit ждёт ДО старта (сам он сигналит свой номер).
    pub fn vf_run_stream(
        c: *mut c_void,
        list: *mut VfDispatch,
        count: u32,
        stream: u32,
        waits: *const VfSemWait,
        n_waits: u32,
        wait_now: i32,
    ) -> VfStatus;
    pub fn vf_wait_stream(c: *mut c_void, stream: u32, back: u32) -> VfStatus;
    /// Стадии последнего submit'а потока, мс: запись команд, ожидание замка
    /// очереди, vkQueueSubmit, ожидание fence.
    pub fn vf_submit_stats(
        c: *mut c_void,
        stream: u32,
        record_ms: *mut f32,
        lock_ms: *mut f32,
        submit_ms: *mut f32,
        fence_ms: *mut f32,
    );
    pub fn vf_stream_value(c: *mut c_void, stream: u32) -> u64;
    pub fn vf_wait_timeline(c: *mut c_void, stream: u32, value: u64, timeout_ns: u64) -> i32;
    pub fn vf_split(c: *mut c_void) -> i32;
    pub fn vf_ctx_semaphore(c: *mut c_void, stream: u32) -> *mut c_void;
    pub fn vf_note_bank_read(c: *mut c_void, bank: u32, frame_value: u64);
    pub fn vf_bank_read_value(c: *mut c_void, bank: u32) -> u64;
    pub fn vf_queue_info(
        c: *mut c_void,
        gfx_family: *mut u32,
        tick_family: *mut u32,
        prio_gfx: *mut f32,
        prio_tick: *mut f32,
    );

    /* Диагностика стадий: таймстемпы по диспатчам такта. Значения читаются
     * на СЛЕДУЮЩЕМ такте (после fence), поэтому сбор не тормозит пайплайн. */
    pub fn vf_perf_supported(c: *mut c_void) -> i32;
    pub fn vf_perf_enable(c: *mut c_void, on: i32) -> VfStatus;
    pub fn vf_perf_read(
        c: *mut c_void,
        back: u32,
        ms: *mut f32,
        stage: *mut u32,
        max: i32,
        tick_ms: *mut f32,
    ) -> i32;
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
/// Потолок тетраэдров в пуле. Плотный шар (шаг radius/12) занимает несколько
/// тысяч тетов, поэтому на тело держим 4096: 64 тела × 4096 × 64 Б ≈ 16 МБ.
pub const MAX_POOL_TETS: usize = 64 * 4096;
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
/// Заполнение сетки SDF аналитическим примитивом на устройстве.
const P_SDFFILL: usize = 28;
/// Контакты SDF↔SDF (шаг 2 плана docs/sdf-collisions.md).
const P_SDFCONTACT: usize = 29;
/// Широкий проход SDF: пары ТЕЛ (не тетов).
const P_SDFPAIRS: usize = 30;
/// Контакты со статикой, нарезанные по тетам: свёртка, выбор победителя и
/// выброс контактов (сброс свёрток — это сам `static_contacts`, проход 0).
/// Все четыре шага идут под номером стадии `P_STATIC`, поэтому таблица
/// F3/VF_PERF=1 показывает их одной строкой `static_contacts` — так же, как
/// в старой схеме с одним диспатчем на тело.
const P_STATWORK: usize = 31;
const P_STATTIE: usize = 32;
const P_STATEMIT: usize = 33;
/// Публикация состояния такта в банк кадра (двойная буферизация).
const P_PUBLISH: usize = 34;
/// Кадровый проход по банку: интерполяция позы на тело (аналог вершинной
/// выборки в окне, но headless — с результатом в буфер).
const P_FRAMEREAD: usize = 35;

/// Имена пайплайнов в порядке `pipes` — ТОТ ЖЕ порядок, что у `P_*` выше и у
/// массива `names` на устройстве. По этому индексу ядро возвращает номер
/// стадии в диагностике производительности, поэтому список — единственный
/// источник правды и для загрузки, и для таблицы.
pub const PIPE_NAMES: [&str; 36] = [
    "integrate",
    "clear",
    "count",
    "scan",
    "fill",
    "static_contacts",
    "broad",
    "solve",
    "sleep",
    "integrate_pos",
    "solve_contacts",
    "solve_apply",
    "sat",
    "depenetrate",
    "contact_select",
    "tet_key",
    "tet_scan",
    "tet_fill",
    "tet_pairs",
    "impact",
    "body_aabb",
    "snapshot",
    "stats",
    "body_init",
    "tet_blocksum",
    "tet_blockprefix",
    "tet_scan_p",
    "tet_pairs_flat",
    "sdf_fill",
    "sdf_contact",
    "sdf_pairs",
    "static_work",
    "static_tie",
    "static_emit",
    "publish",
    "frame_read",
];
/// Число storage-буферов у каждого пайплайна — в том же порядке.
const PIPE_BUFNS: [u32; 36] = [
    1, 2, 2, 3, 4, 2, 5, 4, 1, 1, 4, 4, 7, 3, 2, 4, 3, 6, 8, 6, 4, 1, 2, 3, 2, 2, 4, 8, 1, 7, 5, 6,
    5, 6, 2, 2,
];
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
pub const MAX_BODIES: usize = 64;
/// Потолок событий раскола в кольце за такт (совпадает с MAX_EVENTS в common.glsl).
pub const MAX_EVENTS: usize = 128;
/// Каналов контакта со статикой на тело: 4 диагонали опорной площадки пола и
/// 4 стенки. По этому числу считаются слоты свёрток (тело × канал).
pub const STATIC_CH: usize = 8;
/// Пустой элемент списка работы (хвост, добитый до кратности группы).
const WORK_TAIL: u32 = 0xFFFF_FFFF;
/// Порог энергии раскола, Дж. Финальное слово — у шейдера (pc.lo.w), CPU
/// страхует только на всякий случай.
pub const FRACTURE_ENERGY: f32 = 500.0;
/// Потолок пар тетраэдров за такт.
pub const MAX_PAIRS: usize = 65536;
/// Слов поля SDF на слот тела в общем пуле. Поле плиты (куб 8 ячеек уровня 1)
/// занимает ~29³ ≈ 24 К узлов, шар-снаряд — 19³ ≈ 7 К; потолок с двойным
/// запасом. Память: 64 слота × 65536 × 4 Б = 16 МБ (совпадает с SDF_STRIDE
/// в sdf_contact.comp).
pub const SDF_STRIDE: usize = 65536;
/// Потолок пар тел в широком проходе SDF (64 тела → 2016; совпадает с
/// MAX_SDF_PAIRS в common.glsl).
pub const MAX_SDF_PAIRS: usize = MAX_BODIES * (MAX_BODIES - 1) / 2;

/// Потоки submit'ов ядра (см. `vf_run_stream`): такт — compute с приоритетом
/// 0.0, кадр — graphics с приоритетом 1.0.
pub const VF_STREAM_TICK: u32 = 0;
pub const VF_STREAM_FRAME: u32 = 1;

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

/// Дескриптор SDF-поля тела на устройстве: 4 vec4, 64 байта без паддингов —
/// ровно `struct SdfDesc` в sdf_contact.comp.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SdfDescGpu {
    /// lo.xyz и dims.x
    lo_dims: [i32; 4],
    /// dims.y, dims.z, 0, 0
    dims2: [i32; 4],
    /// метров на узел
    step_pad: [f32; 4],
    /// центр поля относительно центроида тела (в системе тела)
    shift: [f32; 4],
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
    /// Нарезка контактов со статикой по РАБОТЕ: на каждый элемент — пара
    /// (тело << 16 | локальный тет), хвост добит `WORK_TAIL`. Строится при
    /// заливке геометрии, поэтому на такт её не пересобираем.
    work: *mut c_void,
    /// Свёртки по каналам: на (тет, канал) — максимум в монотонных битах float
    /// (0 — кандидата нет). Пишется `static_work`, читается `static_tie`.
    cand: *mut c_void,
    /// Победитель по каналу (тела × каналы): максимум в монотонных битах float.
    static_score: *mut c_void,
    /// Победитель по каналу: индекс тета в пуле (0xFFFFFFFF — пусто).
    static_win: *mut c_void,
    /// Слотов свёртки в буферах (тело × канал, кратно рабочей группе).
    static_slots: usize,
    /// Сколько рабочих групп шлёт static_work/static_tie (0 — работы нет).
    work_groups: u32,
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
    /// Пары тетов строятся ВНУТРИ такта (одним submit'ом с ним), а не
    /// отдельным submit'ом с ожиданием fence. Ставится один раз из `Live`.
    inline_pairs: bool,
    /// Пары тетраэдров, посчитанные CPU один раз на такт.
    pairs: *mut c_void,
    pub n_pairs: usize,
    /// Таблица SDF-полей на устройстве (шаг 2 коллизий): дескриптор на тело
    /// (lo/dims/step/shift — 4 vec4), общий пул значений (SDF_STRIDE слов на
    /// слот), пары тел из широкого прохода и их счётчик. Заливается при
    /// смене геометрии; поза и поворот берутся шейдером из буфера тел.
    sdf_desc: *mut c_void,
    sdf_vals: *mut c_void,
    sdf_pairs: *mut c_void,
    sdf_paircnt: *mut c_void,
    /// Контакты считает `sdf_contact` по полям вместо `sat` (VF_COLLIDE=sdf).
    pub sdf_collide: bool,
    lcount: *mut c_void,
    cellcnt: *mut c_void,
    starts: *mut c_void,
    cursor: *mut c_void,
    cbody: *mut c_void,
    pipes: Vec<*mut c_void>,
    /// Диагностика стадий такта: окно агрегации, строки для оверлея.
    perf: crate::perf::Perf,
    /// Переиспользуемые буферы под чтение таймстемпов (не аллоцируем каждый
    /// такт; размер с запасом над VF_PERF_MAX ядра).
    perf_ms: Vec<f32>,
    perf_ids: Vec<u32>,
    /// Точек контакта в последнем такте окна. Читается из счётчиков списков
    /// только на границе окна печати, а не каждый такт.
    perf_contacts: u32,
    /// Две очереди (такт на compute с приоритетом 0.0, кадр на graphics с 1.0)
    /// и двойной банк состояния тел. Выключено — прежний путь: одна очередь,
    /// кадр читает буфер физики напрямую.
    pub split: bool,
    /// Банки, которые ЧИТАЕТ кадр. Такт пишет в банк (tick % 2) последним
    /// проходом такта и сигналит таймлайн; перезапись банка начинается
    /// только после того, как кадр, читавший его, закрылся.
    bank: [*mut c_void; 2],
    /// Приёмник кадрового прохода (поза и кватернион на тело).
    draw: *mut c_void,
    /// Что опубликовано и доступно кадру: номер такта и банк (на такт позади).
    pub published: (u64, usize),
    /// Банк, который записал ПОСЛЕДНИЙ такт: (номер такта, банк). Для сверки
    /// «банк равен физике» сразу после такта.
    pub tick_bank: (u64, usize),
    /// Время кадрового submit'а: от отправки до закрытия fence, мс.
    pub frame_ms: f32,
    pub frame_submits: u64,
    frame_t0: std::time::Instant,
    /// Каталог SPIR-V: нужен лениво создаваемым пайплайнам (`components`).
    shader_dir: std::path::PathBuf,
    /// Пайплайн разметки компонент связности (`components.spv`), создаётся
    /// при первом вызове `run_components` — таблицу PIPE_NAMES не трогаем.
    components_pipe: *mut c_void,
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
        /* Работа контактов со статикой: список (тело, тет) + свёртки по
         * каналам. Свёртки — по 8 слотов НА ТЕЛО, поэтому размер считается от
         * числа тел сцены, а не от MAX_BODIES: в тестах куча — 1000 и 5000 тел.
         * Кандидатов — 8 u32 на элемент работы. */
        let static_slots = ((n * STATIC_CH) + WG as usize - 1) / WG as usize * WG as usize;
        let work = mk((MAX_POOL_TETS + WG as usize) * 4, true);
        let cand = mk((MAX_POOL_TETS + WG as usize) * STATIC_CH * 4, false);
        let static_score = mk(static_slots * 4, false);
        let static_win = mk(static_slots * 4, false);
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
        /* Таблица SDF-полей: дескриптор на тело, общий пул значений (слот на
         * тело), пары тел широкого прохода и счётчик. Пул создаётся пустым:
         * незанятые дескрипторы — нули (dims = 0 → поля нет), и шейдер пару
         * с таким телом пропускает. */
        let sdf_desc = mk(MAX_BODIES * 64, true);
        let sdf_vals = mk(MAX_BODIES * SDF_STRIDE * 4, true);
        let sdf_pairs = mk(MAX_SDF_PAIRS * 8, true);
        let sdf_paircnt = mk(4, true);
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
        /* Дескрипторы SDF обязаны стартовать нулями: dims = 0 значит «у тела
         * нет поля», иначе шейдер прочитает мусор буфера как размеры сетки. */
        unsafe {
            let zeros = vec![0u8; MAX_BODIES * 64];
            check_status(
                vf_buffer_write(ctx, sdf_desc, 0, zeros.as_ptr() as *const c_void, zeros.len()),
                "sdf_desc zero",
            );
            check_status(
                vf_buffer_write(ctx, sdf_paircnt, 0, [0u8; 4].as_ptr() as *const c_void, 4),
                "sdf_paircnt zero",
            );
        }

        /* Диагностика стадий: спрашиваем у драйвера, умеет ли он таймстемпы, и
         * включаем сбор, если попросили VF_PERF=1. По умолчанию выключено:
         * включённый сбор добавляет в такт count+1 метку и копирование. */
        /* Две очереди и двойной банк. Ядро решает по драйверу и VF_SPLIT:
         * на RADV такт уходит в отдельное compute-семейство (приоритет 0.0),
         * кадр остаётся на graphics (1.0). Банк — буфер, который читает кадр;
         * физика пишет его последним проходом такта. */
        let split = unsafe { vf_split(ctx) } != 0;
        let bank = [mk(n * BODY_BYTES, true), mk(n * BODY_BYTES, true)];
        // Поза и кватернион на тело: 2 vec4.
        let draw = mk(MAX_BODIES * 32, true);
        if split {
            unsafe {
                for b in bank {
                    let zeros = vec![0u8; n * BODY_BYTES];
                    check_status(
                        vf_buffer_write(ctx, b, 0, zeros.as_ptr() as *const c_void, zeros.len()),
                        "bank zero",
                    );
                }
            }
        }

        let perf_ok = unsafe { vf_perf_supported(ctx) } != 0;
        let mut perf = crate::perf::Perf::new(perf_ok, false);
        let want_perf = matches!(std::env::var("VF_PERF").as_deref(), Ok(v) if v != "0");
        if want_perf {
            let st = unsafe { vf_perf_enable(ctx, 1) };
            if st.code == 0 {
                perf.enabled = true;
            } else {
                let msg = unsafe { CStr::from_ptr(st.msg) }.to_string_lossy();
                eprintln!("[perf] VF_PERF=1 не включился: {msg} — диагностика выключена");
            }
        }

        let mut pipes = Vec::new();
        for (i, nm) in PIPE_NAMES.iter().enumerate() {
            let nb = PIPE_BUFNS[i];
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
                        nb,
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
            work,
            cand,
            static_score,
            static_win,
            static_slots,
            work_groups: 0,
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
            inline_pairs: false,
            pairs,
            n_pairs: 0,
            sdf_desc,
            sdf_vals,
            sdf_pairs,
            sdf_paircnt,
            /* По умолчанию — SAT (рабочий режим пользователя). VF_COLLIDE=sdf
             * переключает контакты на поля; старый путь остаётся для сверки. */
            sdf_collide: matches!(std::env::var("VF_COLLIDE").as_deref(), Ok("sdf")),
            lcount,
            cellcnt,
            starts,
            cursor,
            cbody,
            pipes,
            perf,
            perf_ms: vec![0.0; 4096],
            perf_ids: vec![0; 4096],
            perf_contacts: 0,
            split,
            bank,
            draw,
            published: (0, 0),
            tick_bank: (0, 0),
            frame_ms: 0.0,
            frame_submits: 0,
            frame_t0: std::time::Instant::now(),
            shader_dir: shader_dir.to_path_buf(),
            components_pipe: std::ptr::null_mut(),
        }
    }

    /// Включить/выключить сбор стадий (F3 в окне). Возвращает новое состояние.
    /// При неподдерживаемом драйвере сбор не включается и причина печатается в
    /// stderr — «нулевая» таблица вместо чисел не выводится.
    pub fn perf_toggle(&mut self) -> bool {
        let want = !self.perf.enabled;
        let st = unsafe { vf_perf_enable(self.ctx, want as i32) };
        if st.code != 0 {
            if !st.msg.is_null() {
                let msg = unsafe { CStr::from_ptr(st.msg) }.to_string_lossy();
                eprintln!("[perf] включить не удалось: {msg}");
            }
            self.perf.enabled = false;
            return false;
        }
        self.perf.enabled = want;
        self.perf.restart_window();
        if want {
            println!("[perf] диагностика стадий включена (F3 выключает)");
        }
        want
    }

    pub fn perf_on(&self) -> bool {
        self.perf.enabled
    }

    /// Строки последнего закрытого окна — для оверлея в окне.
    pub fn perf_rows(&self) -> &[crate::perf::PerfRow] {
        &self.perf.rows
    }

    /// Копия строк в C-раскладке для оверлея окна.
    pub fn perf_overlay(&self) -> Vec<crate::perf::VfPerfRow> {
        self.perf.overlay_rows()
    }

    /// Среднее время такта последнего окна, мс (заголовок оверлея).
    pub fn perf_tick_ms(&self) -> f32 {
        self.perf.tick_ms()
    }

    /// Минимум/среднее/максимум такта последнего окна, мс (строка ИТОГО).
    pub fn perf_tick_range(&self) -> (f32, f32, f32) {
        self.perf.tick_range_ms()
    }

    /// Счётчики сцены для строки тел в оверлее (всего/бодр/контактов).
    pub fn perf_scene(&self) -> crate::perf::VfSceneCounts {
        self.perf.scene_counts()
    }

    /// Сколько тактов физики завершилось с включения сбора: по дельте этого
    /// числа главный поток считает настоящий TPS (номинал ничего не говорит).
    pub fn perf_ticks(&self) -> u64 {
        self.perf.ticks_total()
    }

    /// Итог окна частот (кадры/такты) от главного потока — печатается строкой
    /// над таблицей стадий; сама таблица печатается из такта, одним местом.
    pub fn perf_set_rates(&mut self, r: crate::perf::RatesOut) {
        self.perf.set_rates(r);
    }

    /// Забрать таймстемпы последнего ЗАКРЫТОГО такта и раз в секунду напечатать
    /// таблицу. Вызывать сразу после ожидания fence и ДО любого нового submit:
    /// таймстемпы живут в слоте кольца и перезаписываются следующим тактом.
    /// `awake` — число бодрствующих тел из `stat`, `alive` — сколько тел в сцене
    /// вообще (оба уже известны вызывающему): отдельных чтений ради строки тел
    /// не нужно. Внимание: `self.n` — ёмкость буфера, а не число тел, поэтому
    /// в сцене живут и «мёртвые» слоты — их за тело считать нельзя.
    pub fn perf_sample(&mut self, awake: u32, alive: usize) {
        if !self.perf.enabled {
            return;
        }
        let now = std::time::Instant::now();
        /* Счётчики сцены. Тела — бесплатно: `n` на CPU, бодрствующие — из тех
         * же 8 Б `stat`, что такт читает и так. Контакты — сумма счётчиков
         * списков (`lc`), и читаем мы их ТОЛЬКО на границе окна печати: раз в
         * секунду 4·n Б вместо новой синхронизации в каждом такте. */
        if self.perf.window_due(now) {
            let c = self.contact_count();
            self.perf_contacts = c.min(u32::MAX as u64) as u32;
        }
        self.perf
            .set_scene(alive as u32, awake, self.perf_contacts);
        let mut tick_ms = 0.0f32;
        let n = unsafe {
            vf_perf_read(
                self.ctx,
                0,
                self.perf_ms.as_mut_ptr(),
                self.perf_ids.as_mut_ptr(),
                self.perf_ms.len() as i32,
                &mut tick_ms,
            )
        };
        if n <= 0 {
            return;
        }
        let n = n as usize;
        self.perf
            .record(&self.perf_ids[..n], &self.perf_ms[..n], tick_ms);
        self.perf.maybe_print(std::time::Instant::now());
    }

    /// Контакты со статикой: четыре прохода по списку РАБОТЫ (тетам), а не
    /// один диспатч по телам. Раньше invocation на тело обходила все его теты
    /// последовательно — на плите это десятки тысяч вершин одним потоком, и
    /// стадия съедала ~80 % такта при одной занятой рабочей группе.
    ///
    /// Все четыре шага помечены стадией `P_STATIC`, поэтому таблица
    /// F3/VF_PERF=1 показывает их одной строкой `static_contacts` — числа
    /// остаются сравнимыми со старой схемой.
    fn push_static(&self, steps: &mut Vec<Step>, a: *mut c_void, push: Push, g_body: u32) {
        steps.push(Step {
            pipe: P_STATIC, /* проход 0 — сброс свёрток (см. static_contacts.comp) */
            bufs: vec![self.static_score, self.static_win],
            push,
            groups: [(self.static_slots as u32) / WG, 1, 1],
        });
        let g_work = self.work_groups;
        if g_work == 0 {
            return; /* тетов нет — и контактов со статикой нет */
        }
        steps.push(Step {
            pipe: P_STATWORK,
            bufs: vec![a, self.tets, self.span, self.work, self.cand, self.static_score],
            push,
            groups: [g_work, 1, 1],
        });
        steps.push(Step {
            pipe: P_STATTIE,
            bufs: vec![self.span, self.work, self.cand, self.static_score, self.static_win],
            push,
            groups: [g_work, 1, 1],
        });
        steps.push(Step {
            pipe: P_STATEMIT,
            bufs: vec![a, self.tets, self.span, self.static_win, self.list, self.lcount],
            push,
            groups: [g_body, 1, 1],
        });
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

    /// Пары тетов — внутрь списка такта (один submit вместо двух). Такт при
    /// этом не ждёт GPU на построении пар: порядок обеспечивает очередь.
    pub fn set_inline_pairs(&mut self, on: bool) {
        self.inline_pairs = on;
        if on {
            self.pairs_on_gpu = true;
            self.n_pairs = MAX_PAIRS;
        }
    }

    /// Стадии последнего submit'а такта (мс): запись команд, замок очереди,
    /// vkQueueSubmit, ожидание fence. Ставит их ядро на пути submit'а.
    pub fn submit_stats(&self) -> (f32, f32, f32, f32) {
        let (mut rec, mut lock, mut sub, mut fen) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        unsafe { vf_submit_stats(self.ctx, VF_STREAM_TICK, &mut rec, &mut lock, &mut sub, &mut fen) };
        (rec, lock, sub, fen)
    }

    /// Записать фазы такта на CPU в диагностику (F3 / VF_PERF=1).
    /// `wait` — ожидание прошлого такта, `reads` — чтения обратно, `submit` —
    /// весь вызов submit'а, `total` — такт целиком на потоке физики.
    pub fn note_tick_cpu(&mut self, wait: f32, reads: f32, submit: f32, total: f32) {
        if !self.perf.enabled {
            return;
        }
        let (rec, lock, sub, fen) = self.submit_stats();
        self.perf.record_cpu(crate::perf::CpuPhases {
            wait,
            reads,
            submit,
            lock,
            record: rec,
            queue_submit: sub,
            fence: fen,
            total,
        });
    }

    /// Дождаться последнего отправленного такта: перед любой записью тел с
    /// CPU (спавн, раскол, сброс) — иначе побежим по буферу, который считает GPU.
    pub fn wait_idle(&mut self) {
        unsafe { check_status(vf_wait_stream(self.ctx, VF_STREAM_TICK, 0), "vf_wait_back") };
    }

    /// Опубликованный банк (номер такта, индекс банка) — то, что читает кадр.
    pub fn published_bank(&self) -> (u64, usize) {
        self.published
    }

    /// Буфер банка как VkBuffer: его привязывает окно (или кадровый проход).
    pub fn bank_buffer(&self, bank: usize) -> *mut c_void {
        unsafe { vf_buffer_handle(self.bank[bank & 1]) }
    }

    /// Кадровый проход по опубликованному банку: тот же читатель данных, что
    /// вершинный шейдер окна, но headless (результат — в буфер draw).
    ///
    /// Ждёт НА GPU «такт N опубликован» (значение таймлайна такта), а не весь
    /// текущий такт: если физика считает такт N+1, кадр всё равно стартует
    /// сразу на банке такта N. Возвращает номер кадрового submit'а.
    /// Одна очередь: кадр идёт в ту же очередь, что такт, и читает буфер
    /// физики напрямую (прежний путь). Две: кадр на graphics, читает банк.
    pub fn frame_submit(&mut self, alpha: f32, wait: bool) -> u64 {
        if self.split && self.published.0 == 0 {
            /* Публикации ещё нет: рисовать нечего. */
            self.frame_ms = 0.0;
            return 0;
        }
        let (src, stream, tick_no) = if self.split {
            let (tick_no, bank) = self.published;
            (self.bank[bank], VF_STREAM_FRAME, tick_no)
        } else {
            (self.bodies[0], VF_STREAM_TICK, 0u64)
        };
        let mut p = self.push(alpha, 0);
        p.prm0[0] = alpha;
        let bufs = [src, self.draw];
        let mut d = VfDispatch {
            pipeline: self.pipes[P_FRAMEREAD],
            buffers: bufs.as_ptr() as *mut *mut c_void,
            n_buffers: 2,
            push: &p as *const Push as *const c_void,
            push_bytes: 80,
            groups: [((self.n as u32) + WG - 1) / WG, 1, 1],
            stage: P_FRAMEREAD as u32,
        };
        let waits = [VfSemWait {
            stream: VF_STREAM_TICK,
            value: tick_no,
        }];
        let n_waits = if self.split { 1 } else { 0 };
        self.frame_t0 = std::time::Instant::now();
        unsafe {
            let st = vf_run_stream(
                self.ctx,
                &mut d,
                1,
                stream,
                if n_waits > 0 { waits.as_ptr() } else { std::ptr::null() },
                n_waits,
                if wait { 1 } else { 0 },
            );
            check_status(st, "frame_submit");
            if wait {
                self.frame_ms = self.frame_t0.elapsed().as_secs_f32() * 1000.0;
            }
            if self.split {
                /* Кадр прочитал этот банк: такт ждёт это значение, прежде чем
                 * писать в него снова. */
                let (_, bank) = self.published;
                let fv = vf_stream_value(self.ctx, VF_STREAM_FRAME);
                vf_note_bank_read(self.ctx, bank as u32, fv);
            }
        }
        if !self.split {
            self.frame_ms = self.frame_t0.elapsed().as_secs_f32() * 1000.0;
        }
        self.frame_submits += 1;
        unsafe { vf_stream_value(self.ctx, VF_STREAM_FRAME) }
    }

    /// Дождаться кадрового submit'а на хосте: время от отправки до fence, мс.
    pub fn frame_wait(&mut self) -> f32 {
        unsafe {
            check_status(vf_wait_stream(self.ctx, VF_STREAM_FRAME, 0), "frame_wait");
        }
        self.frame_ms = self.frame_t0.elapsed().as_secs_f32() * 1000.0;
        self.frame_ms
    }

    /// Дождаться на хосте публикации банка (окно, которому нужен буфер прямо
    /// сейчас). Обычно значение уже достигнуто — стоит ноль.
    pub fn wait_published(&mut self, tick_no: u64) {
        unsafe {
            let _ = vf_wait_timeline(self.ctx, VF_STREAM_TICK, tick_no, 2_000_000_000);
        }
    }

    /// Сколько очередей и с какими приоритетами реально создано.
    pub fn queues(&self) -> (u32, u32, f32, f32, bool) {
        let (mut g, mut t, mut pg, mut pt) = (0u32, 0u32, 0f32, 0f32);
        unsafe { vf_queue_info(self.ctx, &mut g, &mut t, &mut pg, &mut pt) };
        (g, t, pg, pt, self.split)
    }

    /// Тела из банка (для сверки «банк равен физике») и из буфера физики.
    pub fn read_bank(&self, bank: usize) -> Vec<GpuBody> {
        let mut out = vec![GpuBody::from_cpu(&Body::new([0.0; 3], [0.0; 3], 1.0, 0.05)); self.n];
        let bytes = self.n * BODY_BYTES;
        unsafe {
            let st = vf_buffer_read(
                self.ctx,
                self.bank[bank & 1],
                0,
                out.as_mut_ptr() as *mut c_void,
                bytes,
            );
            if st.code != 0 {
                return Vec::new();
            }
        }
        out
    }

    /// Таймлайн-семафор потока (окну): кадровый — сигналить, тактовый — ждать.
    pub fn stream_semaphore(&self, stream: u32) -> *mut c_void {
        unsafe { vf_ctx_semaphore(self.ctx, stream) }
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
        /* SDF: одна рабочая группа на пару тел (64 затравки), число пар шейдер
         * читает из буфера — поэтому групп всегда по потолку пар. */
        let g_sdf = ((MAX_SDF_PAIRS as u32) + WG - 1) / WG;
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
                if self.sdf_collide {
                    /* счётчик пар тел — 4 Б; пары строятся один раз на такт
                     * (внутри такта позы меняются, но пересборка пар каждый
                     * кусок стоила бы дороже, чем сами контакты) */
                    vf_buffer_write(
                        self.ctx,
                        self.sdf_paircnt,
                        0,
                        [0u8; 4].as_ptr() as *const c_void,
                        4,
                    );
                }
            }
        }

        let mut steps: Vec<Step> = Vec::new();
        let (a, b) = (self.bodies[0], self.bodies[1]);

        /* Пары тетов — первыми: контакты ниже читают их. Раньше это был
         * отдельный submit с ожиданием fence, то есть такт на своём же пути
         * стоял до GPU; теперь всё идёт одним submit'ом, а порядок держит
         * очередь. */
        if self.inline_pairs {
            let push_pairs = self.push(dt, g_body);
            let mut ps = self.tet_pair_steps(push_pairs);
            steps.append(&mut ps);
        }

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
        if self.sdf_collide {
            /* Широкий проход по ТЕЛАМ (не по тетам): AABB с запасом на путь за
             * такт и пары тел. Пары строятся один раз на такт — позы внутри
             * такта меняются, но пересборка пар на каждом куске стоила бы
             * дороже самих контактов. */
            steps.push(Step {
                pipe: P_BODYAABB,
                bufs: vec![a, self.tets, self.span, self.baabb],
                push,
                groups: [g_body, 1, 1],
            });
            steps.push(Step {
                pipe: P_SDFPAIRS,
                bufs: vec![a, self.baabb, self.sdf_pairs, self.sdf_paircnt, self.sdf_desc],
                push,
                groups: [g_body, 1, 1],
            });
        }

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
            self.push_static(&mut steps, a, push, g_body);
            if self.sdf_collide {
                /* Контакты по полям: пары тел уже лежат в буфере, число пар
                 * шейдер читает сам — CPU его не знает и не ждёт. */
                steps.push(Step {
                    pipe: P_SDFCONTACT,
                    bufs: vec![
                        a,
                        self.sdf_desc,
                        self.sdf_vals,
                        self.sdf_pairs,
                        self.list,
                        self.lcount,
                        self.sdf_paircnt,
                    ],
                    push,
                    groups: [g_sdf.max(1), 1, 1],
                });
            } else {
                steps.push(Step {
                    pipe: P_SAT,
                    bufs: vec![a, self.tets, self.span, self.pairs, self.list, self.lcount, self.pair_count],
                    push,
                    groups: [g_pairs.max(1), 1, 1],
                });
            }
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

        /* Публикация в банк кадра — последним проходом такта. Банк выбирается
         * по номеру такта (N % 2), и перед записью такт ждёт кадр, который
         * этот банк читал: иначе он перезапишет читаемое — ровно то, от чего
         * первая попытка двух очередей падала в GPUVM fault. */
        let mut waits: Vec<VfSemWait> = Vec::new();
        if self.split {
            let tick_no = unsafe { vf_stream_value(self.ctx, VF_STREAM_TICK) } + 1;
            let bank = (tick_no % 2) as usize;
            steps.push(Step {
                pipe: P_PUBLISH,
                bufs: vec![a, self.bank[bank]],
                push,
                groups: [g_body, 1, 1],
            });
            self.tick_bank = (tick_no, bank);
            let read_by = unsafe { vf_bank_read_value(self.ctx, bank as u32) };
            if read_by > 0 {
                waits.push(VfSemWait {
                    stream: VF_STREAM_FRAME,
                    value: read_by,
                });
            }
            /* Кадру доступен банк ПРОШЛОГО такта (N-1): он опубликован, и на
             * него есть таймлайн-значение N-1. Текущий такт кадр НЕ ждёт — в
             * этом весь смысл раздельных очередей. Стоит задержки в один такт
             * (50 мс при 20 Гц), зато кадр неуязвим к длительности такта.
             * Первому такту публиковать нечего: published остаётся нулём, и
             * кадр пропускает submits, пока такт не посчитался. */
            if tick_no >= 2 {
                let prev = tick_no - 1;
                self.published = (prev, (prev % 2) as usize);
            }
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
                /* Три прохода контактов со статикой (work/tie/emit) помечаем
                 * стадией `static_contacts`, чтобы таблица F3/VF_PERF=1
                 * показывала стадию ОДНОЙ суммой, как при одном диспатче. */
                stage: if s.pipe >= P_STATWORK {
                    P_STATIC as u32
                } else {
                    s.pipe as u32
                },
            })
            .collect();
        let wait_ptr = if waits.is_empty() { std::ptr::null() } else { waits.as_ptr() };
        let wait_n = waits.len() as u32;
        unsafe {
            let st = if self.split {
                /* Такт — на compute-очереди с приоритетом 0.0. Кадр на graphics
                 * (1.0) не ждёт его целиком: он берёт опубликованный банк. */
                vf_run_stream(
                    self.ctx,
                    ds.as_mut_ptr(),
                    ds.len() as u32,
                    VF_STREAM_TICK,
                    wait_ptr,
                    wait_n,
                    if async_submit { 0 } else { 1 },
                )
            } else if async_submit {
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
                self.push_static(&mut steps, a, push, g_body);
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
                /* Три прохода контактов со статикой (work/tie/emit) помечаем
                 * стадией `static_contacts`, чтобы таблица F3/VF_PERF=1
                 * показывала стадию ОДНОЙ суммой, как при одном диспатче. */
                stage: if s.pipe >= P_STATWORK {
                    P_STATIC as u32
                } else {
                    s.pipe as u32
                },
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
        /* Нарезка контактов со статикой по РАБОТЕ (один элемент = один тет).
         * Строится здесь, потому что геометрия меняется ровно здесь (спавн,
         * раскол, сброс), а в такте список только читается. Хвост добит до
         * кратности рабочей группы: шейдеру не нужен счётчик работы в
         * push-константах, он просто выходит на WORK_TAIL. */
        let wg = WG as usize;
        let mut work: Vec<u32> = Vec::with_capacity((flat.len() / 4).min(MAX_POOL_TETS) + wg);
        for (bi, s) in spans.iter().enumerate() {
            let start = s[0] as usize;
            let cnt = (s[1] as usize).min(0xFFFF);
            for t in 0..cnt {
                if start + t >= MAX_POOL_TETS {
                    break; /* пул кончился — сверх потолка тетов не читаем */
                }
                work.push(((bi as u32) << 16) | t as u32);
            }
        }
        while work.len() % wg != 0 {
            work.push(WORK_TAIL);
        }
        self.work_groups = (work.len() / wg) as u32;
        self.bytes_up += (n * 16 + spans.len() * 8 + work.len() * 4) as u64;
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
            if !work.is_empty() {
                check_status(
                    vf_buffer_write(
                        self.ctx,
                        self.work,
                        0,
                        work.as_ptr() as *const c_void,
                        work.len() * 4,
                    ),
                    "upload_work",
                );
            }
        }
    }

    /// Залить таблицу SDF-полей на устройство: дескриптор на тело и значения
    /// в общий пул. Вызывается при смене геометрии (спавн/раскол/сброс),
    /// а не каждый такт: поза и поворот берутся шейдером из буфера тел.
    ///
    /// Элемент — `Some((поле, shift))`, где `shift` = `origin - центроид` в
    /// системе тела (см. `LiveBody::sdf_shift`). `None` или поле, не влезающее
    /// в слот, оставляют дескриптор нулевым: шейдер такую пару пропускает.
    pub fn upload_sdf(&mut self, fields: &[Option<(&crate::field::NodeField, [f32; 3])>]) {
        let mut descs = vec![SdfDescGpu::default(); MAX_BODIES];
        for (i, item) in fields.iter().take(MAX_BODIES).enumerate() {
            let Some((f, shift)) = item else { continue };
            let n = f.val.len();
            if f.dims.contains(&0) || n > SDF_STRIDE {
                eprintln!(
                    "sdf: поле тела {i} ({n} узлов) не влезает в слот {SDF_STRIDE} — контакты SDF для него выключены"
                );
                continue;
            }
            descs[i] = SdfDescGpu {
                lo_dims: [f.lo[0], f.lo[1], f.lo[2], f.dims[0] as i32],
                dims2: [f.dims[1] as i32, f.dims[2] as i32, 0, 0],
                step_pad: [f.step, 0.0, 0.0, 0.0],
                shift: [shift[0], shift[1], shift[2], 0.0],
            };
            unsafe {
                check_status(
                    vf_buffer_write(
                        self.ctx,
                        self.sdf_vals,
                        i * SDF_STRIDE * 4,
                        f.val.as_ptr() as *const c_void,
                        n * 4,
                    ),
                    "upload_sdf values",
                );
            }
            self.bytes_up += (n * 4) as u64;
        }
        unsafe {
            check_status(
                vf_buffer_write(
                    self.ctx,
                    self.sdf_desc,
                    0,
                    descs.as_ptr() as *const c_void,
                    descs.len() * std::mem::size_of::<SdfDescGpu>(),
                ),
                "upload_sdf descs",
            );
        }
        self.bytes_up += (descs.len() * 64) as u64;
    }

    /// Пары тел из широкого прохода SDF (диагностика: сколько их нашлось и
    /// какие именно — если контактов нет, сразу видно, чья это вина).
    pub fn read_sdf_pairs(&mut self) -> Vec<[u32; 2]> {
        let mut cnt = [0u32; 1];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, self.sdf_paircnt, 0, cnt.as_mut_ptr() as *mut c_void, 4),
                "read_sdf_paircnt",
            );
        }
        let n = (cnt[0] as usize).min(MAX_SDF_PAIRS);
        let mut out = vec![[0u32; 2]; n.max(1)];
        if n > 0 {
            unsafe {
                check_status(
                    vf_buffer_read(
                        self.ctx,
                        self.sdf_pairs,
                        0,
                        out.as_mut_ptr() as *mut c_void,
                        n * 8,
                    ),
                    "read_sdf_pairs",
                );
            }
        }
        out.truncate(n);
        out
    }

    /// Переключить контакты на поля (или обратно на SAT) на ходу — для проб.
    pub fn set_sdf_collide(&mut self, on: bool) {
        self.sdf_collide = on;
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
                /* Три прохода контактов со статикой (work/tie/emit) помечаем
                 * стадией `static_contacts`, чтобы таблица F3/VF_PERF=1
                 * показывала стадию ОДНОЙ суммой, как при одном диспатче. */
                stage: if s.pipe >= P_STATWORK {
                    P_STATIC as u32
                } else {
                    s.pipe as u32
                },
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
                /* Три прохода контактов со статикой (work/tie/emit) помечаем
                 * стадией `static_contacts`, чтобы таблица F3/VF_PERF=1
                 * показывала стадию ОДНОЙ суммой, как при одном диспатче. */
                stage: if s.pipe >= P_STATWORK {
                    P_STATIC as u32
                } else {
                    s.pipe as u32
                },
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
                    stage: s.pipe as u32,
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

    /// Список диспатчей «сетка тетов -> пары» для ВСТРАИВАНИЯ в такт.
    ///
    /// Раньше это был отдельный submit (`run_tet_pairs`) с ожиданием fence:
    /// поток физики стоял, пока GPU строил пары, — то есть такт дважды
    /// дожидался конвейера. Пары нужны только следующему за ними такту, а он
    /// идёт в ТОЙ ЖЕ очереди, поэтому порядок гарантирует сама очередь: ждать
    /// fence не нужно вовсе.
    pub fn tet_pair_steps(&mut self, push: Push) -> Vec<Step> {
        /* Счётчики — в ноль с хоста: предыдущий такт уже закрыт (wait_idle),
         * иначе это была бы запись по считающемуся буферу. */
        unsafe {
            vf_buffer_write(self.ctx, self.pair_count, 0, [0u8; 4].as_ptr() as *const c_void, 4);
            let zero_cells = vec![0u8; self.ncells as usize * 4];
            vf_buffer_write(
                self.ctx,
                self.tetcnt,
                0,
                zero_cells.as_ptr() as *const c_void,
                zero_cells.len(),
            );
        }
        let g_body = ((self.n as u32) + WG - 1) / WG;
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
        steps.push(Step {
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
        });
        steps.push(Step {
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
        });
        self.pairs_on_gpu = true;
        self.n_pairs = MAX_PAIRS;
        steps
    }

    /// Пары тетов целиком на устройстве, отдельным submit'ом с ожиданием.
    /// В такте НЕ используется (там пары встроены в его список диспатчей) —
    /// остаётся для проб и тестов, которым нужен готовый буфер пар.
    pub fn run_tet_pairs(&mut self, dt: f32) -> usize {
        let g_body = ((self.n as u32) + WG - 1) / WG;
        let push = self.push(dt, g_body);
        let steps = self.tet_pair_steps(push);
        let mut keep: Vec<Vec<*mut c_void>> = steps.iter().map(|s| s.bufs.clone()).collect();
        let push_keep: Vec<Push> = steps.iter().map(|s| s.push).collect();
        let mut ds: Vec<VfDispatch> = steps
            .iter()
            .enumerate()
            .map(|(k, s)| VfDispatch {
                pipeline: self.pipes[s.pipe],
                buffers: keep[k].as_mut_ptr(),
                n_buffers: keep[k].len() as u32,
                push: &push_keep[k] as *const Push as *const c_void,
                push_bytes: 80,
                groups: s.groups,
                stage: if s.pipe >= P_STATWORK {
                    P_STATIC as u32
                } else {
                    s.pipe as u32
                },
            })
            .collect();
        unsafe {
            check_status(vf_run(self.ctx, ds.as_mut_ptr(), ds.len() as u32), "run_tet_pairs");
        }
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
                stage: s.pipe as u32,
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
        /* Запись в host-visible память, которую в этот момент читает GPU
         * (такт отправляется асинхронно), — гонка: на RADV такие записи
         * роняли процесс. Это событие, а не такт, поэтому ждём устройство. */
        self.wait_idle();
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
            stage: P_BODYINIT as u32,
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
            if !self.components_pipe.is_null() {
                vf_pipeline_free(self.ctx, self.components_pipe);
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
                self.sdf_desc,
                self.sdf_vals,
                self.sdf_pairs,
                self.sdf_paircnt,
            ] {
                vf_buffer_free(self.ctx, b);
            }
            vf_shutdown(self.ctx);
        }
    }
}

/// Примитив, из которого сетку SDF заполняет устройство.
#[derive(Clone, Copy, Debug)]
pub struct SdfPrim {
    /// 0 — коробка, 1 — шар, 2 — капсула, 3 — цилиндр вдоль Y.
    pub kind: u32,
    pub center: [f32; 3],
    pub radius: f32,
    pub half: [f32; 3],
    pub b: [f32; 3],
}

impl SdfPrim {
    pub fn sphere(c: [f32; 3], r: f32) -> Self {
        Self { kind: 1, center: c, radius: r, half: [0.0; 3], b: [0.0; 3] }
    }
    pub fn box_shape(c: [f32; 3], half: [f32; 3]) -> Self {
        Self { kind: 0, center: c, radius: 0.0, half, b: [0.0; 3] }
    }
    pub fn capsule(a: [f32; 3], b: [f32; 3], r: f32) -> Self {
        Self { kind: 2, center: a, radius: r, half: [0.0; 3], b }
    }
    pub fn cylinder(c: [f32; 3], r: f32, half_height: f32) -> Self {
        Self { kind: 3, center: c, radius: r, half: [half_height, 0.0, 0.0], b: [0.0; 3] }
    }
}

/// Раскладка push-констант для `sdf_fill.comp` (80 байт).
#[repr(C)]
#[derive(Clone, Copy)]
struct SdfPush {
    grid: [u32; 4],
    origin: [f32; 4],
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
}

impl GpuSim {
    /// Заполнить сетку SDF примитивом НА УСТРОЙСТВЕ и вернуть значения хосту.
    ///
    /// Порядок значений: `((z * ny) + y) * nx + x`, позиция узла —
    /// `origin + (x,y,z) * step`. Знак: внутри примитива минус, снаружи плюс
    /// (обычное соглашение SDF), значения в мировых единицах.
    pub fn fill_sdf_grid(
        &mut self,
        prim: &SdfPrim,
        dims: [u32; 3],
        origin: [f32; 3],
        step: f32,
    ) -> Vec<f32> {
        let n = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        assert!(n > 0, "пустая сетка SDF");
        let mut buf: *mut c_void = std::ptr::null_mut();
        unsafe {
            check_status(vf_buffer_new(self.ctx, n * 4, 1, &mut buf), "sdf: buffer") 
        };
        let push = SdfPush {
            grid: [dims[0], dims[1], dims[2], prim.kind],
            origin: [origin[0], origin[1], origin[2], step],
            a: [prim.center[0], prim.center[1], prim.center[2], prim.radius],
            b: [prim.half[0], prim.half[1], prim.half[2], 0.0],
            c: [prim.b[0], prim.b[1], prim.b[2], 0.0],
        };
        let mut bufs = [buf];
        let mut ds = [VfDispatch {
            pipeline: self.pipes[P_SDFFILL],
            buffers: bufs.as_mut_ptr(),
            n_buffers: 1,
            push: &push as *const SdfPush as *const c_void,
            push_bytes: std::mem::size_of::<SdfPush>() as u32,
            groups: [(n as u32).div_ceil(64), 1, 1],
            stage: P_SDFFILL as u32,
        }];
        unsafe {
            check_status(vf_run(self.ctx, ds.as_mut_ptr(), 1), "sdf_fill");
        }
        let mut out = vec![0.0f32; n];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, buf, 0, out.as_mut_ptr() as *mut c_void, n * 4),
                "sdf: read",
            );
            vf_buffer_free(self.ctx, buf);
        }
        self.bytes_down += (n * 4) as u64;
        out
    }
}

impl GpuSim {
    /// Выгрузить запечённое поле на устройство и прочитать обратно.
    ///
    /// Шаг 1 коллизий по SDF: физике нужно поле рядом с телами. Проверяем
    /// именно то, что данные доезжают без искажений (иначе контакты будут
    /// считаться по мусору, и это невозможно заметить по картинке).
    pub fn sdf_roundtrip(&mut self, f: &crate::field::NodeField) -> bool {
        let n = f.val.len();
        let bytes = n * 4;
        let mut buf: *mut c_void = std::ptr::null_mut();
        unsafe {
            check_status(vf_buffer_new(self.ctx, bytes, 1, &mut buf), "sdf: buffer");
            check_status(
                vf_buffer_write(self.ctx, buf, 0, f.val.as_ptr() as *const c_void, bytes),
                "sdf: write",
            );
        }
        let mut back = vec![0.0f32; n];
        unsafe {
            check_status(
                vf_buffer_read(self.ctx, buf, 0, back.as_mut_ptr() as *mut c_void, bytes),
                "sdf: read",
            );
            vf_buffer_free(self.ctx, buf);
        }
        back == f.val
    }
}

/// Push-константы `components.comp` (32 байта).
#[repr(C)]
#[derive(Clone, Copy)]
struct PartsPush {
    /// nx, ny, nz, всего узлов.
    grid: [u32; 4],
    /// режим, слоёв эрозии, ёмкость компонент, номер прохода.
    par: [u32; 4],
}

/// Проходов распространения в одной пачке `components`: один submit и одно
/// чтение флага на пачку. Один dispatch на RADV стоит ~0.3 мс, поэтому пачка
/// важнее точности «до последнего прохода».
const PARTS_BATCH: u32 = 4;

/// Сколько прыжков по меткам делает ОДИН поток за проход (`label[label[...]]`).
/// Это локальная работа без синхронизации: метки только убывают, поэтому
/// цепочка безопасна, а глобальных проходов нужно в `PARTS_JUMPS` раз меньше.
const PARTS_JUMPS: u32 = 32;

impl GpuSim {
    /// Разметить компоненты связности поля НА УСТРОЙСТВЕ (`components.comp`).
    ///
    /// Вход — поле занятости (`val < 0` внутри), `erode` — слоёв эрозии до
    /// разметки (0 или 1; глубже считает CPU-оракул `parts::label_field`).
    /// Алгоритм тот же, что у оракула: ядро → распространение минимума по 6
    /// соседям с pointer jumping → кожа → остаток без ядра → корни → сжатие →
    /// плоский список узлов. Хост заливает поле, читает метки и по несколько
    /// чисел на компоненту; сортировка корней (их единицы-сотни) — на хосте.
    ///
    /// `None` — нет `components.spv` или размеры не годятся.
    pub fn run_components(
        &mut self,
        f: &crate::field::NodeField,
        erode: u32,
    ) -> Option<crate::parts::GpuParts> {
        if erode > 1 || f.dims.contains(&0) {
            return None;
        }
        let t0 = std::time::Instant::now();
        let (nx, ny, nz) = (f.dims[0], f.dims[1], f.dims[2]);
        let n = nx * ny * nz;
        if n == 0 {
            return None;
        }
        /* Ёмкость по числу компонент: шумовые области у среза тоже считаются,
         * поэтому с запасом, но с потолком по памяти. */
        let cap = (n / 4).clamp(64, 8192);
        /* Пайплайн создаём лениво: общий цикл сборки и таблицу PIPE_NAMES не
         * трогаем (правки не должны пересекаться с другим переносом на GPU). */
        if self.components_pipe.is_null() {
            let path = self.shader_dir.join("components.spv");
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("components: нет {path:?}: {e}");
                    return None;
                }
            };
            let cname = std::ffi::CString::new("components").ok()?;
            let mut p: *mut c_void = std::ptr::null_mut();
            unsafe {
                check_status(
                    vf_pipeline_new(
                        self.ctx,
                        bytes.as_ptr() as *const u32,
                        bytes.len(),
                        12,
                        std::mem::size_of::<PartsPush>() as u32,
                        cname.as_ptr(),
                        &mut p,
                    ),
                    "components",
                )
            };
            self.components_pipe = p;
        }
        let ctx = self.ctx;
        let mk = |bytes: usize, host: bool| -> *mut c_void {
            let mut b: *mut c_void = std::ptr::null_mut();
            unsafe {
                check_status(
                    vf_buffer_new(ctx, bytes, host as i32, &mut b),
                    "components: buffer",
                );
            }
            b
        };
        /* Горячие буферы — device-local: atomicMin по host-visible памяти у
         * дискретной карты идёт через PCIe и стоит десятки миллисекунд. */
        let bval = mk(n * 4, false);
        let blab = mk(n * 4, false);
        let bmask = mk(n * 4, false);
        let bcnt = mk(16, true);
        let bmdev = mk(cap * 7 * 4, false);
        let broots = mk(cap * 4, true);
        let boffs = mk(cap * 4, true);
        let bcur = mk(cap * 4, false);
        let bnodes = mk(n * 4, false);
        let bmeta = mk(cap * 7 * 4, true);
        let bstage = mk(n * 4, true);
        let bact = mk(n * 4, false);
        let mut bufs = [
            bval, blab, bmask, bcnt, bmdev, broots, boffs, bcur, bnodes, bmeta, bstage, bact,
        ];
        let grid = [nx as u32, ny as u32, nz as u32, n as u32];
        let mut iterations = 0u32;
        let mut iters_stage = [0u32; 3];

        /* Поле — через staging: прямо в device-local хост писать не может. */
        let bits: Vec<u32> = f.val.iter().map(|v| v.to_bits()).collect();
        unsafe {
            check_status(
                vf_buffer_write(ctx, bstage, 0, bits.as_ptr() as *const c_void, n * 4),
                "components: stage val",
            );
            check_status(
                vf_buffer_write(ctx, bcnt, 0, [0u8; 16].as_ptr() as *const c_void, 16),
                "components: cnt",
            );
        }
        self.components_step(&mut bufs, grid, 9, erode, cap as u32, 0, 1, n);
        /* Шаг 0: ядро и стартовые метки + список занятых узлов. */
        self.components_step(&mut bufs, grid, 0, erode, cap as u32, 0, 1, n);
        let mut hdr0 = [0u32; 4];
        unsafe {
            check_status(
                vf_buffer_read(ctx, bcnt, 0, hdr0.as_mut_ptr() as *mut c_void, 16),
                "components: active",
            );
        }
        /* Дальше все проходы идут ТОЛЬКО по занятым узлам. */
        let active = (hdr0[3] as usize).min(n);
        if std::env::var("VF_COMP_PROF").is_ok() {
            eprintln!("components: hdr={hdr0:?} активных узлов {active}");
        }

        /* Шаги 1 и 2: распространение по ядру, затем по коже. Пачки по
         * `PARTS_BATCH` проходов: флаг изменений читается раз на пачку, а не
         * на проход (каждое чтение — синхронизация с устройством, и на
         * RADV один dispatch стоит сотни микросекунд). */
        for (si, mode) in [1u32, 2u32].into_iter().enumerate() {
            let mut layer = 0u32;
            loop {
                unsafe {
                    check_status(
                        vf_buffer_write(ctx, bcnt, 0, [0u8; 4].as_ptr() as *const c_void, 4),
                        "components: flag",
                    );
                }
                self.components_step(
                    &mut bufs, grid, mode, erode, cap as u32, PARTS_JUMPS, PARTS_BATCH, active,
                );
                layer += PARTS_BATCH;
                iterations += PARTS_BATCH;
                let mut hdr = [0u32; 4];
                unsafe {
                    check_status(
                        vf_buffer_read(ctx, bcnt, 0, hdr.as_mut_ptr() as *mut c_void, 16),
                        "components: hdr",
                    );
                }
                if hdr[0] == 0 || layer > 8 * (nx + ny + nz) as u32 {
                    break;
                }
            }
            iters_stage[si] = layer;
        }
        /* Шаг 3: остаток без ядра (тонкие пластины) — своим индексом. */
        self.components_step(&mut bufs, grid, 3, erode, cap as u32, 0, 1, active);
        /* Шаг 4: распространение внутри остатка. */
        {
            let mut layer = 0u32;
            loop {
                unsafe {
                    check_status(
                        vf_buffer_write(ctx, bcnt, 0, [0u8; 4].as_ptr() as *const c_void, 4),
                        "components: flag",
                    );
                }
                self.components_step(
                    &mut bufs, grid, 4, erode, cap as u32, PARTS_JUMPS, PARTS_BATCH, active,
                );
                layer += PARTS_BATCH;
                iterations += PARTS_BATCH;
                let mut hdr = [0u32; 4];
                unsafe {
                    check_status(
                        vf_buffer_read(ctx, bcnt, 0, hdr.as_mut_ptr() as *mut c_void, 16),
                        "components: hdr",
                    );
                }
                if hdr[0] == 0 || layer > 8 * (nx + ny + nz) as u32 {
                    break;
                }
            }
            iters_stage[2] = layer;
        }

        /* Шаг 5: корни компонент (узел, индекс которого равен метке). Счётчики
         * обнуляем ЧАСТИЧНО: cnt[3] (занятых узлов) нужен режимам по списку. */
        unsafe {
            let z = [0u32, 0, 0, active as u32];
            check_status(
                vf_buffer_write(ctx, bcnt, 0, z.as_ptr() as *const c_void, 16),
                "components: cnt",
            );
        }
        self.components_step(&mut bufs, grid, 5, erode, cap as u32, 0, 1, active);
        let mut hdr = [0u32; 4];
        unsafe {
            check_status(
                vf_buffer_read(ctx, bcnt, 0, hdr.as_mut_ptr() as *mut c_void, 16),
                "components: hdr",
            );
        }
        let found = hdr[1] as usize;
        let c = found.min(cap);
        if found > cap {
            eprintln!("components: компонент {found} больше ёмкости {cap} — хвост отброшен");
        }
        let mut roots = vec![0u32; cap];
        unsafe {
            check_status(
                vf_buffer_read(ctx, broots, 0, roots.as_mut_ptr() as *mut c_void, cap * 4),
                "components: roots",
            );
        }
        roots.truncate(c);
        /* Сортировка корней на хосте (их единицы-сотни, а не узлы поля): номер
         * компоненты = позиция в отсортированном списке, поэтому сжатие метки —
         * двоичный поиск в шейдере и никакой таблицы соответствий. */
        let mut order: Vec<usize> = (0..c).collect();
        order.sort_by_key(|&i| roots[i]);
        let sorted: Vec<u32> = order.iter().map(|&i| roots[i]).collect();
        /* Мелочь на компоненту: габариты (lo = максимум, hi = 0) и счётчики
         * узлов. Сначала в staging, затем копированием в device-local. */
        let mut meta = vec![0u32; cap * 7];
        for k in 0..cap {
            meta[6 * k] = u32::MAX;
            meta[6 * k + 1] = u32::MAX;
            meta[6 * k + 2] = u32::MAX;
        }
        unsafe {
            check_status(
                vf_buffer_write(ctx, broots, 0, sorted.as_ptr() as *const c_void, c * 4),
                "components: sorted roots",
            );
            let hdr2 = [0u32, 0, c as u32, active as u32];
            check_status(
                vf_buffer_write(ctx, bcnt, 0, hdr2.as_ptr() as *const c_void, 16),
                "components: cnt sorted",
            );
            check_status(
                vf_buffer_write(ctx, bmeta, 0, meta.as_ptr() as *const c_void, cap * 28),
                "components: meta",
            );
        }
        self.components_step(&mut bufs, grid, 13, erode, cap as u32, 0, 1, cap * 7);
        /* Шаг 6: метка -> номер компоненты плюс габарит и счётчик узлов. */
        self.components_step(&mut bufs, grid, 6, erode, cap as u32, 0, 1, active);
        /* Читаем габариты и счётчики обратно через staging. */
        self.components_step(&mut bufs, grid, 12, erode, cap as u32, 0, 1, cap * 7);
        let mut bnd = vec![0u32; cap * 6];
        let mut counts = vec![0u32; cap];
        unsafe {
            check_status(
                vf_buffer_read(ctx, bmeta, 0, bnd.as_mut_ptr() as *mut c_void, cap * 24),
                "components: bnd",
            );
            check_status(
                vf_buffer_read(
                    ctx,
                    bmeta,
                    cap * 24,
                    counts.as_mut_ptr() as *mut c_void,
                    cap * 4,
                ),
                "components: counts",
            );
        }
        /* Шаг 7: плоский список узлов по компонентам (начала блоков — на хосте:
         * компонент десятки, узлов миллионы). */
        let mut offsets = vec![0u32; c + 1];
        for k in 0..c {
            offsets[k + 1] = offsets[k] + counts[k];
        }
        let total_nodes = offsets[c] as usize;
        unsafe {
            check_status(
                vf_buffer_write(ctx, boffs, 0, offsets.as_ptr() as *const c_void, c * 4),
                "components: offs",
            );
        }
        self.components_step(&mut bufs, grid, 11, erode, cap as u32, 0, 1, cap);
        self.components_step(&mut bufs, grid, 7, erode, cap as u32, 0, 1, active);
        /* Список узлов — единственный крупный обратный трансфер: метки
         * восстанавливаются из него на хосте (каждая компонента занимает
         * непрерывный блок в `nodes`), а читать n*4 Б меток через PCIe
         * значило бы платить за пустые узлы, которых в поле большинство. */
        let mut nodes = vec![0u32; total_nodes.max(1)];
        if total_nodes > 0 {
            self.components_step(&mut bufs, grid, 10, erode, cap as u32, 0, 1, total_nodes);
            unsafe {
                check_status(
                    vf_buffer_read(
                        ctx,
                        bstage,
                        0,
                        nodes.as_mut_ptr() as *mut c_void,
                        total_nodes * 4,
                    ),
                    "components: nodes",
                );
            }
        }
        nodes.truncate(total_nodes);

        let mut labels_raw = vec![u32::MAX; n];
        for c in 0..c {
            for k in offsets[c] as usize..offsets[c + 1] as usize {
                if let Some(slot) = labels_raw.get_mut(nodes[k] as usize) {
                    *slot = c as u32;
                }
            }
        }
        let labels = crate::parts::Labels {
            label: labels_raw,
            bounds: (0..c)
                .map(|k| crate::parts::Bounds {
                    lo: [
                        f.lo[0] + bnd[6 * k] as i32,
                        f.lo[1] + bnd[6 * k + 1] as i32,
                        f.lo[2] + bnd[6 * k + 2] as i32,
                    ],
                    hi: [
                        f.lo[0] + bnd[6 * k + 3] as i32,
                        f.lo[1] + bnd[6 * k + 4] as i32,
                        f.lo[2] + bnd[6 * k + 5] as i32,
                    ],
                })
                .collect(),
            counts: counts[0..c].iter().map(|&v| v as usize).collect(),
        };
        self.bytes_up += (n * 4) as u64;
        self.bytes_down += (n * 8) as u64;
        unsafe {
            for b in bufs {
                vf_buffer_free(ctx, b);
            }
        }
        Some(crate::parts::GpuParts {
            labels,
            iterations,
            iters_stage,
            ms: t0.elapsed().as_secs_f32() * 1e3,
            nodes,
            offsets,
        })
    }

    /// Один режим `components.comp`: `rounds` проходов одним submit'ом.
    ///
    /// `elems` — сколько потоков нужно режиму (узлы поля или элементы `cap`),
    /// `jumps` — сколько локальных прыжков по меткам делает поток за проход
    /// (режимы распространения; остальные режимы его не читают).
    #[allow(clippy::too_many_arguments)]
    fn components_step(
        &self,
        bufs: &mut [*mut c_void],
        grid: [u32; 4],
        mode: u32,
        erode: u32,
        cap: u32,
        jumps: u32,
        rounds: u32,
        elems: usize,
    ) {
        let pushes = vec![
            PartsPush {
                grid,
                par: [mode, erode, cap, jumps]
            };
            rounds as usize
        ];
        let mut ds: Vec<VfDispatch> = pushes
            .iter()
            .map(|p| VfDispatch {
                pipeline: self.components_pipe,
                buffers: bufs.as_mut_ptr(),
                n_buffers: bufs.len() as u32,
                push: p as *const PartsPush as *const c_void,
                push_bytes: std::mem::size_of::<PartsPush>() as u32,
                groups: [(elems as u32).div_ceil(64).max(1), 1, 1],
                stage: 0,
            })
            .collect();
        unsafe {
            check_status(vf_run(self.ctx, ds.as_mut_ptr(), ds.len() as u32), "components");
        }
    }
}
