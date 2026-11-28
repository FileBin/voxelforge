// Живой мир: цельный объект, выстрелы, раскол по удару. Всё считается
// на ходу — ни одного заранее заготовленного осколка.
use crate::collide;
use crate::sim::{GpuSim, GpuBody, Params};
use crate::tet::{node_pos, Tet, TetBody};
use crate::types::Body;
use crate::vfw::Shape;

pub const MAX_BODIES: usize = 64;
pub const DENSITY: f64 = 420.0;
pub const MIN_TETS: usize = 20;
/// Сколько снарядов живёт одновременно; дальше самый старый уходит из мира.
pub const MAX_BULLETS: usize = 12;
/// Дешёвый порог CCD: если самое быстрое тело за такт смещается меньше этого,
/// читать буфер тел ради continuous не нужно вовсе (метры).
const CCD_MIN_MOVE: f64 = 0.05;

/// Длина вектора скорости из `GpuBody` (w — обратная масса, не участвует).
fn len3(v: [f32; 4]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Разность скоростей пары.
fn sub3(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], 0.0]
}

/// Полуразмер габарита ПОЛЯ тела (максимум по осям, метры). Именно он задаёт
/// «свой размер» для порога continuous: радиус тетов у шара занижен вдвое.
fn field_half(f: Option<&crate::field::NodeField>) -> f32 {
    let Some(f) = f else { return 0.0 };
    (0..3)
        .map(|k| (f.dims[k] as f32 - 1.0) * f.step * 0.5)
        .fold(0.0f32, f32::max)
}

/// Расстояние от точки `q` до отрезка `p .. p + v*dt` — грубое отсечение пар
/// перед бисекцией.
fn segment_point_dist(p: [f32; 4], v: [f32; 4], dt: f32, q: [f32; 4]) -> f32 {
    let d = [v[0] * dt, v[1] * dt, v[2] * dt];
    let w = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let t = if dd > 1e-12 {
        ((w[0] * d[0] + w[1] * d[1] + w[2] * d[2]) / dd).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let c = [w[0] - d[0] * t, w[1] - d[1] * t, w[2] - d[2] * t];
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
}

/// Подробная трассировка прохода CCD (VF_CCD_TRACE=1) — для проб и отладки.
fn ccd_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VF_CCD_TRACE").is_ok())
}

/// Обратная масса тела из буфера: у спящего она лежит в `misc[2]`.
fn gpu_im(g: &GpuBody) -> f32 {
    if g.misc[1] > 0.5 { g.misc[2] } else { g.vel_m[3] }
}

/// Обратная инерция тела из буфера: у спящего — в `misc[3]`.
fn gpu_ii(g: &GpuBody) -> f32 {
    if g.misc[1] > 0.5 { g.misc[3] } else { g.ang_i[3] }
}

/// Тело глазами CCD из буфера устройства и набора пробных точек.
fn ccd_view<'a>(g: GpuBody, b: &'a LiveBody, s: &crate::ccd::ProbeSet) -> crate::ccd::BodyView<'a> {
    crate::ccd::BodyView {
        field: b.sdf.as_ref().unwrap(),
        shift: b.sdf_shift,
        pos: [g.pos_r[0], g.pos_r[1], g.pos_r[2]],
        quat: g.quat,
        vel: [g.vel_m[0], g.vel_m[1], g.vel_m[2]],
        ang: [g.ang_i[0], g.ang_i[1], g.ang_i[2]],
        inv_mass: gpu_im(&g),
        inv_inertia: gpu_ii(&g),
        shape: s.bb,
        radius: s.bb.radius.max(1e-4),
        thin: s.thin,
        asleep: g.misc[1] > 0.5,
    }
}

/// Фазы одного CCD-прохода, микросекунды. Нужны, чтобы «такт стал 33 мс» не
/// приходилось угадывать: видно, какая фаза его съела.
#[derive(Clone, Copy, Debug, Default)]
pub struct CcdPhases {
    /// Сборка пробных точек (обход сетки поля) — на тела.
    pub probes_us: u32,
    /// Сколько точек в наборах, собранных за такт.
    pub probes_n: u32,
    /// Поиск интервала касания по парам (сканы внутри габарита).
    pub trace_us: u32,
    /// Бисекция.
    pub bisect_us: u32,
    /// Итераций бисекции за такт.
    pub bisect_iters: u32,
    /// Сколько пар дошло до поиска.
    pub pairs: u32,
    /// Весь проход: чтение тел + наборы + поиск + заливка.
    pub total_us: u32,
}

/// Окно CCD-статистики: последние 256 проходов (5 с при 20 Гц), min/mean/max и
/// p99 по полному времени прохода. Печать и строка оверлея — раз в секунду и
/// только при VF_PERF=1: диагностика не должна сама менять картину.
pub struct CcdStat {
    enabled: bool,
    ring: [CcdPhases; 256],
    n: usize,
    head: usize,
    window_start: std::time::Instant,
    /// Средние фазы последнего закрытого окна.
    pub mean: CcdPhases,
    pub max_total_ms: f32,
    pub p99_total_ms: f32,
    /// Строка для оверлея в окне.
    pub line: String,
}

impl CcdStat {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ring: [CcdPhases::default(); 256],
            n: 0,
            head: 0,
            window_start: std::time::Instant::now(),
            mean: CcdPhases::default(),
            max_total_ms: 0.0,
            p99_total_ms: 0.0,
            line: String::new(),
        }
    }

    /// Один проход CCD: в окно и, если окно закрылось, — посчитать и напечатать.
    fn record(&mut self, ph: CcdPhases, now: std::time::Instant) {
        self.ring[self.head] = ph;
        self.head = (self.head + 1) % self.ring.len();
        if self.n < self.ring.len() {
            self.n += 1;
        }
        if !self.enabled || now.duration_since(self.window_start).as_secs_f32() < 1.0 {
            return;
        }
        self.window_start = now;
        self.finish_window();
    }

    /// Закрыть окно: средние по фазам, max и p99 полного времени.
    fn finish_window(&mut self) {
        if self.n == 0 {
            return;
        }
        let k = self.n as f32;
        let mut tot = [0.0f32; 256];
        let (mut sum, mut mx) = (CcdPhases::default(), 0.0f32);
        let mut acc = [0f64; 7];
        for i in 0..self.n {
            let p = self.ring[i];
            tot[i] = p.total_us as f32;
            mx = mx.max(tot[i]);
            acc[0] += p.probes_us as f64;
            acc[1] += p.trace_us as f64;
            acc[2] += p.bisect_us as f64;
            acc[3] += p.probes_n as f64;
            acc[4] += p.bisect_iters as f64;
            acc[5] += p.pairs as f64;
            acc[6] += p.total_us as f64;
        }
        sum.probes_us = (acc[0] / k as f64) as u32;
        sum.trace_us = (acc[1] / k as f64) as u32;
        sum.bisect_us = (acc[2] / k as f64) as u32;
        sum.probes_n = (acc[3] / k as f64) as u32;
        sum.bisect_iters = (acc[4] / k as f64) as u32;
        sum.pairs = (acc[5] / k as f64) as u32;
        sum.total_us = (acc[6] / k as f64) as u32;
        self.mean = sum;
        self.max_total_ms = mx * 1e-3;
        /* p99: 99-й процентиль полного времени прохода. */
        let mut s: Vec<f32> = tot[..self.n].to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((s.len() as f32 * 0.99) as usize).min(s.len() - 1);
        self.p99_total_ms = s[idx] * 1e-3;

        self.line = format!(
            "ccd: probes {}  trace {:.2} ms  bisect {:.2} ms  total {:.2} ms",
            sum.probes_n,
            sum.trace_us as f32 * 1e-3,
            sum.bisect_us as f32 * 1e-3,
            sum.total_us as f32 * 1e-3,
        );
        println!(
            "[ccd] такт: средн {:.2} · макс {:.2} · p99 {:.2} мс | точки {:.3} мс ({} шт) · интервал {:.3} · бисекция {:.3} мс ({} итер) · пар {}",
            sum.total_us as f32 * 1e-3,
            self.max_total_ms,
            self.p99_total_ms,
            sum.probes_us as f32 * 1e-3,
            sum.probes_n,
            sum.trace_us as f32 * 1e-3,
            sum.bisect_us as f32 * 1e-3,
            sum.bisect_iters,
            sum.pairs,
        );
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Target,
    Shard,
    Bullet,
}

pub struct LiveBody {
    /// Устойчивый идентификатор: слоты переиспользуются, id — нет.
    pub id: u64,
    /// До какого кадра тело нельзя раскалывать: свежие осколки толкаются
    /// друг с другом ещё до того, как улеглись, и раньше это дробило их
    /// дальше по кругу — плита рассыпалась сама себя.
    pub crack_after: u64,
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub vel: [f64; 3],
    pub mass: f64,
    pub radius: f32,
    pub tets: Vec<Tet>,
    pub origin: [f32; 3],
    pub level_cap: u8,
    /// Запечённое поле расстояний (SDF) тела: им пользуется и меш, и физика.
    /// Считается ОДИН раз при создании тела, в такте не пересчитывается.
    pub sdf: Option<crate::field::NodeField>,
    /// Центр поля (`NodeField::lo` в узлах) относительно центроида тела, в
    /// системе тела. Позу и поворот тела физика берёт из буфера тел, поэтому
    /// в таблицу SDF уходит только эта постоянная часть преобразования.
    /// Считается в `rebuild()` — там же, где локальные тетраэдры.
    pub sdf_shift: [f32; 3],
    local: Vec<[[f32; 3]; 4]>,
    pub mesh: i32,
    pub lines: i32,
    pub kind: Kind,
    pub alive: bool,
}

impl LiveBody {
    fn rebuild(&mut self) {
        let tb = {
            let mut b = TetBody {
                tets: self.tets.clone(),
                level_cap: self.level_cap,
                origin: self.origin,
                sdf: Default::default(),
            };
            if let Some(f) = &self.sdf {
                let _ = b.sdf.set(f.clone());
            }
            b
        };
        {
            /* Меш здесь НЕ строится: раньше этот println марчил поле целиком
             * ради числа треугольников, и каждый осколок считался дважды
             * (здесь и в `shape_for`) — на мелком поле это удваивало время
             * раскола. Печатаем только то, что и так известно. */
            println!(
                "[body {}] поле SDF: {} · тетов {} · узлов {}",
                self.id,
                if tb.sdf.get().is_some() { "ЕСТЬ" } else { "НЕТ — вывод из тетов" },
                self.tets.len(),
                tb.sdf.get().map(|f| f.val.len()).unwrap_or(0)
            );
        }
        let c = tb.world_centroid();
        /* Поле задано в координатах узлов от `origin`; тело физики живёт от
         * центроида. Разница постоянная в системе тела — её и запоминаем. */
        self.sdf_shift = [
            self.origin[0] - c[0],
            self.origin[1] - c[1],
            self.origin[2] - c[2],
        ];
        self.local = self
            .tets
            .iter()
            .map(|t| {
                std::array::from_fn(|k| {
                    let p = node_pos(t.v[k], t.level, self.origin);
                    [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
                })
            })
            .collect();
    }

    pub fn radius_of(&self) -> f32 {
        let mut r = 0.0f32;
        for t in &self.local {
            for p in t {
                r = r.max((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt());
            }
        }
        r
    }

    /// Тетраэдры в системе тела, ПОВЁРНУТЫЕ, но без переноса: `collide::build`
    /// сам прибавляет позицию тела. Раньше здесь прибавлялась ещё и позиция —
    /// коллайдер уезжал на удвоенную высоту: плита падала сквозь пол на 28 см,
    /// спала там и «прыгала», а каркас не совпадал с мешем.
    /// Локальные (не повёрнутые) тетраэдры — их и держит пул на устройстве.
    pub fn local_tets(&self) -> &Vec<[[f32; 3]; 4]> {
        &self.local
    }

    pub fn rotated_tets(&self) -> Vec<[[f32; 3]; 4]> {
        self.local
            .iter()
            .map(|t| {
                std::array::from_fn(|k| {
                    let p = rot(self.quat, [t[k][0] as f64, t[k][1] as f64, t[k][2] as f64]);
                    [p[0] as f32, p[1] as f32, p[2] as f32]
                })
            })
            .collect()
    }
}

fn rot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[1], q[2], q[3]];
    let cx = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    let t = [cx[0] + q[0] * v[0], cx[1] + q[0] * v[1], cx[2] + q[0] * v[2]];
    let c2 = [u[1] * t[2] - u[2] * t[1], u[2] * t[0] - u[0] * t[2], u[0] * t[1] - u[1] * t[0]];
    [v[0] + 2.0 * c2[0], v[1] + 2.0 * c2[1], v[2] + 2.0 * c2[2]]
}

fn rot_inv(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    rot([q[0], -q[1], -q[2], -q[3]], v)
}

/// Матрица модели (column-major) из позиции и кватерниона.
pub fn model_matrix(pos: [f64; 3], quat: [f64; 4]) -> [f32; 16] {
    let q = quat;
    let (xx, yy, zz) = (q[1] * q[1], q[2] * q[2], q[3] * q[3]);
    let (xy, xz, yz) = (q[1] * q[2], q[1] * q[3], q[2] * q[3]);
    let (wx, wy, wz) = (q[0] * q[1], q[0] * q[2], q[0] * q[3]);
    [
        (1.0 - 2.0 * (yy + zz)) as f32, (2.0 * (xy + wz)) as f32, (2.0 * (xz - wy)) as f32, 0.0,
        (2.0 * (xy - wz)) as f32, (1.0 - 2.0 * (xx + zz)) as f32, (2.0 * (yz + wx)) as f32, 0.0,
        (2.0 * (xz + wy)) as f32, (2.0 * (yz - wx)) as f32, (1.0 - 2.0 * (xx + yy)) as f32, 0.0,
        pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0,
    ]
}

/* Live владеет девайсом Vulkan и его буферами (сырые указатели) и переезжает
 * в поток физики целиком: одновременно им пользуется ровно один поток, а
 * окно на том же девайсе ходит в ядро через собственные объекты. */
unsafe impl Send for Live {}

pub struct Live {
    pub bodies: Vec<LiveBody>,
    pub gpu: GpuSim,
    /// Зеркало поз на CPU. Нужно ТОЛЬКО CPU-солверу (эталон в тестах): на
    /// GPU-пути его больше никто не читает и не пишет. Вынос в отдельный
    /// модуль за фичу `cpu-solver` — следующая правка.
    pub mirror: Vec<GpuBody>,
    /// Следующий свободный идентификатор тела.
    next_id: u64,
    /// Номер текущего кадра.
    frames: u64,
    /// Живые снаряды в порядке выстрела: их слоты переиспользуются.
    pub bullets: std::collections::VecDeque<usize>,
    /// Контакты считает GPU (VF_COLLIDE=gpu). Пока сверка SAT не сошлась с
    /// CPU, по умолчанию выключено: включается флагом окружения.
    pub gpu_collide: bool,
    /// Контакты считает `sdf_contact` по запечённым полям (VF_COLLIDE=sdf):
    /// tет-пары и SAT из такта уходят, широкий проход — по AABB тел.
    pub sdf_collide: bool,
    /// Пары тетов строит устройство (VF_PAIRS=gpu): тогда из кадра уходит и
    /// заливка пар, и чтение позиций тел.
    pub gpu_pairs: bool,
    /// Пары отдельным submit'ом с ожиданием (старый путь, VF_PAIRS_OLD=1).
    pub pairs_old: bool,
    /// Геометрия изменилась (спавн/раскол/сброс) — нужно перезалить пул.
    geom_dirty: bool,
    /// Пробуждение по контакту на CPU — только для пути, где коллизию
    /// считает CPU. На GPU-пути этот проход не вызывается: будит SAT.
    pub wake_on_contact: bool,
    /// Раскол включён. В отладке столкновений его выключают: иначе не видно,
    /// останавливает ли контакт снаряд вообще, или он проходит насквозь.
    pub fracture_enabled: bool,
    /// Сон включён. Выключается флагом --no-sleep для отладки: если тела
    /// никогда не засыпают, сразу видно, кто не может разъехаться.
    pub sleep_enabled: bool,
    pub params: Params,
    pub cracks: usize,
    pub shards_made: usize,
    pub shots: usize,
    pub last_log: String,
    pub lines_on: bool,
    pub near: f32,
    pub far: f32,
    /// Статистика прошлого такта: максимум |v| и число бодрствующих тел.
    /// Вместо чтения всех тел обратно (9 КБ/такт) — 8 Б.
    pub last_vmax: f32,
    pub last_awake: u32,
    /// Скорость, которую CPU узнал от события (выстрел, осколки). Статистика
    /// прошлого такта её не знает — на один такт берём максимум из двух.
    pending_vmax: f32,
    /// Пробные точки CCD на тело (ключ — устойчивый `id`). Поле печётся один
    /// раз на тело, поэтому и точки переиспользуются; слот с новым id их не
    /// наследует.
    ccd_probes: Vec<(u64, crate::ccd::ProbeSet)>,
    /// Сколько раз за такт включился continuous-режим (для VF_PERF/диагностики).
    pub last_ccd: u32,
    /// Сколько миллисекунд занял CCD-проход (CPU: чтение тел + бисекция).
    pub last_ccd_ms: f32,
    /// Фазы последнего прохода и окно статистики (VF_PERF=1).
    pub ccd_ph: CcdPhases,
    pub ccd_stat: CcdStat,
    /// Замерять фазы CCD (VF_PERF=1): таймеры в горячем пути стоят денег.
    ccd_prof: bool,
    /// Удары, найденные CCD и ждущие раскола: (тело, партнёр, точка, энергия).
    /// Обрабатываются в НАЧАЛЕ следующего такта — там же, где события с
    /// устройства, и ДО заливки геометрии: осколки обязаны попасть в пул.
    ccd_impacts: Vec<(usize, usize, [f32; 3], f32)>,
}

/// Исходная сцена мира: ПЛИТА на полу, шар-снаряд при выстреле и пул спящих
/// тел. Нужна и конструктору, и сбросу (R), который раньше создавал новый Live:
/// теперь девайс один на всё, поэтому сброс только пересобирает состояние.
fn scene_plate() -> (Params, LiveBody, Vec<GpuBody>) {
    // Плита сразу лежит на полу: центр на высоте полуразмера, иначе она
    // падает с 1.2 м и к первому выстрелу оказывается не там, куда целятся.
    let plate0 = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
    let (lo0, hi0) = plate0.bounds();
    let rest_y = 0.5 * (hi0[1] - lo0[1]) + 0.001;
    let plate = TetBody::cube([0.0, rest_y, 0.0], 8, 1);
    let params = Params {
        lo: [-8.0, 0.0, -8.0],
        hi: [8.0, 10.0, 8.0],
        cell: 0.30,
        friction: 0.45,
        max_push: 0.55,
        ..Params::default()
    };

    let mut body = LiveBody {
        id: 1,
        crack_after: 0,
        pos: [0.0, rest_y as f64, 0.0],
        quat: [1.0, 0.0, 0.0, 0.0],
        vel: [0.0; 3],
        mass: DENSITY * plate.volume(),
        radius: 0.5,
        tets: plate.tets.clone(),
        origin: plate.origin,
        level_cap: plate.level_cap,
        /* Запечённое поле плиты переносим в тело мира: иначе rebuild()
         * строил TetBody без поля и меш считался BFS-ом по занятости —
         * отсюда гребни и шип на угле. */
        sdf: Some(plate.sdf().clone()),
        sdf_shift: [0.0; 3],
        local: Vec::new(),
        mesh: -1,
        lines: -1,
        kind: Kind::Target,
        alive: true,
    };
    body.rebuild();
    // Ставим плиту ровно на пол по ЛОКАЛЬНЫМ тетраэдрам (это и есть
    // коллайдер). bounds() считает в другой системе, из-за него плита
    // стартовала вдвое выше, падала на 28 см и подпрыгивала.
    let mut lowest_local = f32::MAX;
    for t in &body.local {
        for v in t {
            lowest_local = lowest_local.min(v[1]);
        }
    }
    if lowest_local.is_finite() {
        body.pos[1] = (-lowest_local) as f64 + 0.001;
    }

    let mut gpu_bodies: Vec<GpuBody> = Vec::with_capacity(MAX_BODIES);
    gpu_bodies.push(gpu_body(&body));
    for _ in 1..MAX_BODIES {
        let parked = Body::new([0.0, -400.0, 0.0], [0.0; 3], 1.0, 0.05);
        let mut g = GpuBody::from_cpu(&parked);
        g.misc[1] = 1.0;
        gpu_bodies.push(g);
    }
    (params, body, gpu_bodies)
}

impl Live {
    /// Девайс ядра: окно рендерит на нём же (один Vulkan-девайс на всё).
    pub fn ctx(&self) -> *mut std::ffi::c_void {
        self.gpu.ctx()
    }

    /// Мир без окна (тесты, пробы, headless-демки).
    pub fn new(shader_dir: &std::path::Path) -> Live {
        Self::with_window(shader_dir, false)
    }

    /// Мир под живое окно: девайс сразу создаётся с surface-расширениями и
    /// swapchain, чтобы окно жило на нём же (шаг 0 — один девайс).
    pub fn new_windowed(shader_dir: &std::path::Path) -> Live {
        Self::with_window(shader_dir, true)
    }

    fn with_window(shader_dir: &std::path::Path, windowed: bool) -> Live {
        let (params, body, gpu_bodies) = scene_plate();
        /* Замеры фаз CCD включаются диагностикой производительности: таймеры
         * в горячем пути сами стоят времени, поэтому по умолчанию их нет. */
        let ccd_prof = matches!(std::env::var("VF_PERF").as_deref(), Ok("1"))
            || std::env::var("VF_CCD_PROF").is_ok();

        let gpu = if windowed {
            GpuSim::new_windowed(gpu_bodies.clone(), shader_dir, params)
        } else {
            GpuSim::new(gpu_bodies.clone(), shader_dir, params)
        };
        let mut gpu = gpu;
        /* Режим контактов задаётся здесь один раз: у мира и у устройства он
         * обязан совпадать, иначе такт соберёт проходы одного режима. */
        gpu.set_sdf_collide(matches!(std::env::var("VF_COLLIDE").as_deref(), Ok("sdf")));
        /* Пары тетов строятся ВНУТРИ такта одним submit'ом: отдельный submit с
         * ожиданием fence держал поток физики до GPU на каждом такте. */
        let gpu_pairs = std::env::var("VF_PAIRS").map(|v| v == "gpu").unwrap_or(false);
        /* VF_PAIRS_OLD=1 — прежний путь: пары отдельным submit'ом с ожиданием
         * fence. Оставлен для сравнения «до/после» на одном бинаре. */
        let pairs_old = std::env::var("VF_PAIRS_OLD").is_ok();
        gpu.set_inline_pairs(gpu_pairs && !pairs_old);

        Live {
            next_id: 1 + (MAX_BODIES as u64),
            frames: 0,
            bullets: std::collections::VecDeque::new(),
            gpu_collide: matches!(
                std::env::var("VF_COLLIDE").as_deref(),
                Ok("gpu") | Ok("sdf")
            ),
            sdf_collide: matches!(std::env::var("VF_COLLIDE").as_deref(), Ok("sdf")),
            gpu_pairs,
            pairs_old,
            geom_dirty: true,
            wake_on_contact: true,
            fracture_enabled: true,
            sleep_enabled: true,
            bodies: vec![body],
            gpu,
            mirror: gpu_bodies,
            params,
            cracks: 0,
            shards_made: 0,
            shots: 0,
            last_log: String::new(),
            lines_on: false,
            near: 0.0,
            far: 60.0,
            last_vmax: 0.0,
            last_awake: 64,
            pending_vmax: 0.0,
            ccd_probes: Vec::new(),
            last_ccd: 0,
            last_ccd_ms: 0.0,
            ccd_ph: CcdPhases::default(),
            ccd_stat: CcdStat::new(ccd_prof),
            ccd_prof,
            ccd_impacts: Vec::new(),
        }
    }

    /// Расталкивает только что созданные осколки ЧИСТО ПОЗИЦИОННО.
    ///
    /// Осколки делят объём родителя и поэтому рождаются впритык, а местами
    /// перекрываясь. Солвер видит это как глубокое проникновение и выдаёт
    /// импульсы — кучу разбрасывает ещё до того, как она успела упасть.
    /// Здесь перекрытия снимаются сдвигом позиций (без скоростей), и только
    /// на столько, чтобы тела перестали пересекаться.
    fn settle_new_shards(&mut self, slots: &[usize]) {
        if slots.len() < 2 {
            return;
        }
        let mut boxes: Vec<[f64; 6]> = Vec::with_capacity(slots.len());
        for &s in slots {
            let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
            for t in self.bodies[s].rotated_tets() {
                for v in t {
                    for k in 0..3 {
                        lo[k] = lo[k].min(v[k] as f64);
                        hi[k] = hi[k].max(v[k] as f64);
                    }
                }
            }
            let p = self.bodies[s].pos;
            boxes.push([p[0] + lo[0], p[1] + lo[1], p[2] + lo[2], p[0] + hi[0], p[1] + hi[1], p[2] + hi[2]]);
        }

        for _ in 0..8 {
            let mut moved = false;
            for a in 0..slots.len() {
                for b in (a + 1)..slots.len() {
                    let (ba, bb) = (boxes[a], boxes[b]);
                    let mut ov = [0.0f64; 3];
                    let mut ok = true;
                    for k in 0..3 {
                        ov[k] = ba[k + 3].min(bb[k + 3]) - ba[k].max(bb[k]);
                        if ov[k] <= 0.0 {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        continue;
                    }
                    // расталкиваем по оси наименьшего перекрытия
                    let axis = (0..3).min_by(|x, y| ov[*x].partial_cmp(&ov[*y]).unwrap()).unwrap();
                    let step = (0.5 * ov[axis]).min(0.02);
                    let dir = if self.bodies[slots[a]].pos[axis] <= self.bodies[slots[b]].pos[axis] { -1.0 } else { 1.0 };
                    for k in 0..2 {
                        let s = slots[[a, b][k]];
                        self.bodies[s].pos[axis] += dir * step * if k == 0 { 1.0 } else { -1.0 };
                        let p = self.bodies[s].pos;
                        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
                        for t in self.bodies[s].rotated_tets() {
                            for v in t {
                                for q in 0..3 {
                                    lo[q] = lo[q].min(v[q] as f64);
                                    hi[q] = hi[q].max(v[q] as f64);
                                }
                            }
                        }
                        boxes[[a, b][k]] = [p[0] + lo[0], p[1] + lo[1], p[2] + lo[2], p[0] + hi[0], p[1] + hi[1], p[2] + hi[2]];
                    }
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }

        // и над полом: осколок не должен рождаться внутри пола
        for &s in slots {
            let low = boxes[slots.iter().position(|x| *x == s).unwrap()][1];
            if low < 0.0 {
                self.bodies[s].pos[1] -= low;
            }
            let p = self.bodies[s].pos;
            self.mirror[s].pos_r = [p[0] as f32, p[1] as f32, p[2] as f32, self.mirror[s].pos_r[3]];
            self.mirror[s].anchor[0] = p[0] as f32;
            self.mirror[s].anchor[1] = p[1] as f32;
            self.mirror[s].anchor[2] = p[2] as f32;
        }
    }

    /// Пробные точки CCD для тела `i` (кэш по устойчивому id: поле печётся
    /// один раз, поэтому точки тоже). Сборка — обход сетки поля, самая дорогая
    /// часть CCD, поэтому она ленивая: только для тел, прошедших порог.
    fn ccd_probe(&mut self, i: usize) -> crate::ccd::ProbeSet {
        let empty = || crate::ccd::ProbeSet {
            pts: Vec::new(),
            bb: crate::ccd::BoxShape::default(),
            half: [0.0; 3],
            neg: 0,
            thin: true,
        };
        if i >= self.bodies.len() {
            return empty();
        }
        let id = self.bodies[i].id;
        if let Some((_, s)) = self.ccd_probes.iter().find(|(cid, _)| *cid == id) {
            return s.clone();
        }
        let Some(f) = self.bodies[i].sdf.as_ref() else {
            return empty();
        };
        /* Габарит — по ТЕТАМ, а не по блоку поля: у блока поля есть запас в
         * несколько узлов (марчингу нужны внешние слои), и по нему касание
         * выходило бы раньше настоящего. */
        let shape = crate::ccd::box_of_tets(&self.bodies[i].local);
        let s = crate::ccd::probe_set(
            f,
            self.bodies[i].sdf_shift,
            Some(shape),
            crate::ccd::PROBE_POINTS,
        );
        self.ccd_probes.push((id, s.clone()));
        if self.ccd_probes.len() > 4 * MAX_BODIES {
            self.ccd_probes.remove(0);
        }
        s
    }

    /// Набор пробных точек тела `i` в кэше ТЕКУЩЕГО такта: одно тело может
    /// участвовать во многих парах, а обход поля должен случиться один раз.
    fn ccd_set_for(
        &mut self,
        i: usize,
        sets: &mut Vec<(usize, crate::ccd::ProbeSet)>,
        ph: &mut CcdPhases,
    ) -> usize {
        if let Some(p) = sets.iter().position(|(k, _)| *k == i) {
            return p;
        }
        let t = std::time::Instant::now();
        let s = self.ccd_probe(i);
        ph.probes_us += t.elapsed().as_micros() as u32;
        ph.probes_n += s.pts.len() as u32;
        sets.push((i, s));
        sets.len() - 1
    }

    /// Continuous-режим для быстрых тел (CCD по SDF и габаритам).
    ///
    /// Включается ТОЛЬКО когда относительное смещение за такт больше размера
    /// меньшего тела: медленные тела идут прежним путём и ничего не платят.
    /// Для такой пары ищем первый момент касания вдоль перемещения, ставим
    /// туда быстрое тело и гасим нормальную скорость в точке касания — иначе
    /// тело оказывается по другую сторону партнёра между подшагами
    /// (туннелирование), а на малой скорости ведёт себя нормально.
    ///
    /// Плоскости (пол и стены) идут отдельным проходом ДО пар: у них нет поля,
    /// и раньше CCD их не покрывал вовсе.
    fn ccd_pass(&mut self, dt: f32) {
        self.last_ccd = 0;
        self.last_ccd_ms = 0.0;
        self.ccd_ph = CcdPhases::default();
        if matches!(std::env::var("VF_CCD").as_deref(), Ok("0")) {
            return;
        }
        let t_ccd = std::time::Instant::now();
        let iters = std::env::var("VF_CCD_ITERS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(12)
            .clamp(1, 24);
        /* Дешёвый порог: без быстрых тел буфер тел не читаем вовсе. */
        let vmax = (self.pending_vmax as f64).max(self.last_vmax as f64);
        if vmax <= 0.0 || vmax * dt as f64 <= CCD_MIN_MOVE {
            return;
        }
        let snap = self.gpu.read_bodies();
        let n = self.bodies.len().min(snap.len());
        if n < 2 {
            return;
        }
        let mut state: Vec<GpuBody> = snap.clone();
        let mut changed: Vec<usize> = Vec::new();
        let mut done: Vec<(usize, usize)> = Vec::new();
        let mut sets: Vec<(usize, crate::ccd::ProbeSet)> = Vec::new();
        let mut ph = CcdPhases::default();
        /* Пол и стены: разрешено `m·p >= d`. Пол — y=0, стены — границы мира. */
        let planes: [([f32; 3], f32); 5] = [
            ([0.0, 1.0, 0.0], self.params.lo[1] as f32),
            ([1.0, 0.0, 0.0], self.params.lo[0] as f32),
            ([-1.0, 0.0, 0.0], -(self.params.hi[0] as f32)),
            ([0.0, 0.0, 1.0], self.params.lo[2] as f32),
            ([0.0, 0.0, -1.0], -(self.params.hi[2] as f32)),
        ];

        for i in 0..n {
            if !self.bodies[i].alive || self.bodies[i].sdf.is_none() {
                continue;
            }
            let gi = state[i];
            if gi.misc[1] > 0.5 {
                continue; /* спит — continuous не нужен */
            }
            let speed = len3(gi.vel_m);
            /* размер тела — по ГАБАРИТУ ПОЛЯ, а не по радиусу тетов: у шара
             * r=0.14 теты дают 0.2, и порог срабатывал раньше времени */
            let half = field_half(self.bodies[i].sdf.as_ref());
            if !crate::ccd::needs_continuous(speed, dt, half) {
                if ccd_trace() {
                    eprintln!("[ccd] тело {i}: порог скорости {speed:.1}*{dt} <= 2*{half:.3}");
                }
                continue;
            }
            let si = self.ccd_set_for(i, &mut sets, &mut ph);
            /* ---- плоскости (кандидат, ещё НЕ применяем) ---- */
            let plane_hit = {
                let pv = ccd_view(gi, &self.bodies[i], &sets[si].1);
                let mut best: Option<(f32, [f32; 3], [f32; 3])> = None;
                for (m, d) in planes.iter() {
                    if let Some((t, p)) = crate::ccd::plane_contact(&pv, *m, *d, dt) {
                        if best.map(|(bt, _, _)| t < bt).unwrap_or(true) {
                            best = Some((t, p, *m));
                        }
                    }
                }
                best
            };
            /* ---- пары тел ---- */
            let mut best: Option<(usize, crate::ccd::SweepHit)> = None;
            for j in 0..n {
                if j == i || !self.bodies[j].alive || self.bodies[j].sdf.is_none() {
                    continue;
                }
                if done.contains(&(i.min(j), i.max(j))) {
                    continue;
                }
                let li = self.bodies[i].sdf.as_ref().map(|f| f.val.len()).unwrap_or(0);
                let lj = self.bodies[j].sdf.as_ref().map(|f| f.val.len()).unwrap_or(0);
                /* пробу берём с МЕНЬШЕГО тела: у него точки плотнее, и именно
                 * его поверхность пересекает партнёра первой */
                let (probe_i, other_i) = if li <= lj { (i, j) } else { (j, i) };
                let (pk, ok) = if probe_i == i { (gi, state[j]) } else { (state[j], gi) };
                if pk.misc[1] > 0.5 && ok.misc[1] > 0.5 {
                    if ccd_trace() {
                        eprintln!("[ccd] пара {probe_i}/{other_i}: оба спят");
                    }
                    continue;
                }
                let pr = field_half(self.bodies[probe_i].sdf.as_ref());
                let or_ = field_half(self.bodies[other_i].sdf.as_ref());
                let rel = sub3(ok.vel_m, pk.vel_m);
                if !crate::ccd::needs_continuous(len3(rel), dt, pr.min(or_)) {
                    if ccd_trace() {
                        eprintln!(
                            "[ccd] пара {probe_i}/{other_i}: отн {:.1}*{dt} <= 2*{:.3} · pk vel {:?} ok vel {:?}",
                            len3(rel),
                            pr.min(or_),
                            pk.vel_m,
                            ok.vel_m
                        );
                    }
                    continue;
                }
                let (sp, so) = (
                    self.ccd_set_for(probe_i, &mut sets, &mut ph),
                    self.ccd_set_for(other_i, &mut sets, &mut ph),
                );
                if sets[sp].1.pts.is_empty() || sets[so].1.pts.is_empty() {
                    if ccd_trace() {
                        eprintln!("[ccd] пара {probe_i}/{other_i}: пустые наборы {} / {}", sets[sp].1.pts.len(), sets[so].1.pts.len());
                    }
                    continue;
                }
                /* Отсечение по сфере, заметающей путь. ОТНОСИТЕЛЬНАЯ скорость,
                 * не собственная пробного: пробу берём с МЕНЬШЕГО по полю тела,
                 * а оно может стоять (замер: пара «мелкий осколок / снаряд» —
                 * отрезок-точка 3.000 при зазоре 0 и отсечение съедало пару
                 * целиком). */
                let move_len = len3(rel) * dt;
                /* Относительное движение: расстояние от НАЧАЛА координат до
                 * отрезка «пробное минус партнёр». Раньше отрезок строили от
                 * позы пробного по его собственной скорости — у стоящего
                 * мелкого осколка он вырождался в точку, и пара «осколок /
                 * снаряд» отбрасывалась целиком (замер: 3.000 > 0.847 при
                 * зазоре 0 — снаряд проходил сквозь). */
                let dpos = [pk.pos_r[0] - ok.pos_r[0], pk.pos_r[1] - ok.pos_r[1], pk.pos_r[2] - ok.pos_r[2], 0.0];
                let dvel = [pk.vel_m[0] - ok.vel_m[0], pk.vel_m[1] - ok.vel_m[1], pk.vel_m[2] - ok.vel_m[2], 0.0];
                let seg = segment_point_dist(dpos, dvel, dt, [0.0; 4]);
                if seg > (pr + or_) * 1.5 + move_len * 0.0 + 0.05 {
                    if ccd_trace() {
                        eprintln!("[ccd] пара {probe_i}/{other_i}: мимо пути, отрезок-точка {seg:.3} > {:.3}", (pr + or_) * 1.5 + 0.05);
                    }
                    continue;
                }
                ph.pairs += 1;
                let (pset, oset) = (&sets[sp].1, &sets[so].1);
                let pv = ccd_view(pk, &self.bodies[probe_i], pset);
                let ov = ccd_view(ok, &self.bodies[other_i], oset);
                let mut c = if self.ccd_prof {
                    Some(crate::ccd::Cost::default())
                } else {
                    None
                };
                let hit = crate::ccd::first_contact_cost(&pset.pts, &pv, &ov, dt, iters, c.as_mut());
                if let Some(c) = c {
                    ph.trace_us += (c.scan_ns / 1000) as u32;
                    ph.bisect_us += (c.bisect_ns / 1000) as u32;
                    ph.bisect_iters += c.bisect_iters;
                }
                if ccd_trace() {
                    eprintln!("[ccd] пара {probe_i}/{other_i}: поиск → {hit:?}");
                }
                if let Some(hit) = hit {
                    let better = best.map(|(_, h)| hit.t < h.t).unwrap_or(true);
                    if better {
                        best = Some((j, hit));
                    }
                }
            }
            /* Чей удар раньше — плоскости или партнёра? Применяем ОДИН, самый
             * ранний: раньше плоскость применялась до пар, и снаряд, летящий
             * в осколок у стены, гасился о стену в конце того же такта, так и
             * не проверив осколок (найдено замером: 240 м/с, путь 12 м). */
            let plane_first = match (&best, &plane_hit) {
                (None, Some(_)) => true,
                (Some((_, h)), Some((t, _, _))) => *t < h.t,
                _ => false,
            };
            if plane_first {
                let (t, p, m) = plane_hit.unwrap();
                let pv = ccd_view(gi, &self.bodies[i], &sets[si].1);
                let mut gp = state[i];
                gp.pos_r[0] += gp.vel_m[0] * dt * t;
                gp.pos_r[1] += gp.vel_m[1] * dt * t;
                gp.pos_r[2] += gp.vel_m[2] * dt * t;
                let (dv, dw) = crate::ccd::plane_impulse(&pv, m, p);
                gp.vel_m[0] += dv[0];
                gp.vel_m[1] += dv[1];
                gp.vel_m[2] += dv[2];
                gp.ang_i[0] += dw[0];
                gp.ang_i[1] += dw[1];
                gp.ang_i[2] += dw[2];
                gp.prev_pos_r[0] = gp.pos_r[0];
                gp.prev_pos_r[1] = gp.pos_r[1];
                gp.prev_pos_r[2] = gp.pos_r[2];
                state[i] = gp;
                if !changed.contains(&i) {
                    changed.push(i);
                }
                self.last_ccd += 1;
                continue;
            }
            let Some((j, hit)) = best else { continue };
            let (fi, fj) = (
                self.bodies[i].sdf.as_ref().unwrap(),
                self.bodies[j].sdf.as_ref().unwrap(),
            );
            let (probe_i, other_i) = if fi.val.len() <= fj.val.len() { (i, j) } else { (j, i) };
            let (pi, oi) = (probe_i, other_i);
            let pk = if pi == i { gi } else { state[j] };
            let ok = if pi == i { state[j] } else { gi };
            let set_of = |k: usize| -> (crate::ccd::BoxShape, bool) {
                sets.iter()
                    .find(|(x, _)| *x == k)
                    .map(|(_, s)| (s.bb, s.thin))
                    .unwrap_or((crate::ccd::BoxShape::default(), true))
            };
            let (pbox, pthin) = set_of(pi);
            let (obox, _othin) = set_of(oi);
            let pv = crate::ccd::BodyView {
                field: if pi == i { fi } else { fj },
                shift: self.bodies[pi].sdf_shift,
                pos: [pk.pos_r[0], pk.pos_r[1], pk.pos_r[2]],
                quat: pk.quat,
                vel: [pk.vel_m[0], pk.vel_m[1], pk.vel_m[2]],
                ang: [pk.ang_i[0], pk.ang_i[1], pk.ang_i[2]],
                inv_mass: gpu_im(&pk),
                inv_inertia: gpu_ii(&pk),
                shape: pbox,
                radius: pk.prev_pos_r[3].max(1e-3),
                thin: pthin,
                asleep: pk.misc[1] > 0.5,
            };
            let ov = crate::ccd::BodyView {
                field: if pi == i { fj } else { fi },
                shift: self.bodies[oi].sdf_shift,
                pos: [ok.pos_r[0], ok.pos_r[1], ok.pos_r[2]],
                quat: ok.quat,
                vel: [ok.vel_m[0], ok.vel_m[1], ok.vel_m[2]],
                ang: [ok.ang_i[0], ok.ang_i[1], ok.ang_i[2]],
                inv_mass: gpu_im(&ok),
                inv_inertia: gpu_ii(&ok),
                shape: obox,
                radius: ok.prev_pos_r[3].max(1e-3),
                thin: false,
                asleep: ok.misc[1] > 0.5,
            };
            let imp = crate::ccd::contact_impulse(&pv, &ov, &hit);
            /* Энергию удара считаем ДО импульса (как impact.comp): по ней
             * решается раскол. Событие уедет в очередь и обработается в начале
             * следующего такта — до заливки геометрии осколков. */
            let energy = {
                let inv = pv.inv_mass + ov.inv_mass;
                if self.fracture_enabled && imp.vn < 0.0 && inv > 0.0 {
                    Some(0.5 * (1.0 / inv) * imp.vn * imp.vn)
                } else {
                    None
                }
            };
            /* поза пробного тела — в момент касания (t доля шага) */
            let mut gp = state[pi];
            gp.pos_r[0] += pk.vel_m[0] * dt * hit.t;
            gp.pos_r[1] += pk.vel_m[1] * dt * hit.t;
            gp.pos_r[2] += pk.vel_m[2] * dt * hit.t;
            gp.vel_m[0] += imp.dv_a[0];
            gp.vel_m[1] += imp.dv_a[1];
            gp.vel_m[2] += imp.dv_a[2];
            gp.ang_i[0] += imp.dw_a[0];
            gp.ang_i[1] += imp.dw_a[1];
            gp.ang_i[2] += imp.dw_a[2];
            gp.prev_pos_r[0] = gp.pos_r[0];
            gp.prev_pos_r[1] = gp.pos_r[1];
            gp.prev_pos_r[2] = gp.pos_r[2];
            let mut go = state[oi];
            if go.misc[1] > 0.5 {
                /* спящего будим: иначе обратная масса нулевая и импульс не
                 * передаётся (контактный шейдер делает то же самое) */
                go.misc[1] = 0.0;
                go.misc[0] = 0.0;
                go.vel_m[3] = go.misc[2];
                go.ang_i[3] = go.misc[3];
                go.anchor = [go.pos_r[0], go.pos_r[1], go.pos_r[2], 0.0];
            }
            go.vel_m[0] += imp.dv_b[0];
            go.vel_m[1] += imp.dv_b[1];
            go.vel_m[2] += imp.dv_b[2];
            go.ang_i[0] += imp.dw_b[0];
            go.ang_i[1] += imp.dw_b[1];
            go.ang_i[2] += imp.dw_b[2];
            state[pi] = gp;
            state[oi] = go;
            done.push((i.min(j), i.max(j)));
            if !changed.contains(&pi) {
                changed.push(pi);
            }
            if !changed.contains(&oi) {
                changed.push(oi);
            }
            if let Some(e) = energy {
                if e >= crate::sim::FRACTURE_ENERGY {
                    self.ccd_impacts.push((pi, oi, hit.p, e));
                }
            }
            self.last_ccd += 1;
        }
        ph.total_us = t_ccd.elapsed().as_micros() as u32;
        self.last_ccd_ms = ph.total_us as f32 * 1e-3;
        self.ccd_ph = ph;
        self.ccd_stat.record(ph, std::time::Instant::now());
        if changed.is_empty() {
            return;
        }
        let items: Vec<(usize, GpuBody)> = changed.iter().map(|&k| (k, state[k])).collect();
        self.gpu.write_slots(&items);
        /* CPU-зеркало этих тел: иначе следующее событие (раскол) зальёт
         * устаревшие позы и вернёт тело назад. */
        for k in changed {
            self.bodies[k].pos = [
                state[k].pos_r[0] as f64,
                state[k].pos_r[1] as f64,
                state[k].pos_r[2] as f64,
            ];
            self.bodies[k].vel = [
                state[k].vel_m[0] as f64,
                state[k].vel_m[1] as f64,
                state[k].vel_m[2] as f64,
            ];
        }
    }

    /// Такт полностью на GPU: CPU один раз строит пары и заливает их, дальше
    /// всё считает устройство, а тела читаются один раз в конце.
    fn step_gpu(&mut self, dt: f32, substeps: u32, iterations: u32) {
        use std::time::Instant;
        /* Фазы такта по запросу (VF_STEP_PROF=1): вопрос «куда ушли 33 мс в
         * кадре удара» должен решаться числами, а не догадками. По умолчанию
         * таймеров нет — они сами стоят времени. Фазы CPU (ожидание, чтения,
         * submit) меряются всегда: они бесплатны и нужны таблице. */
        let prof = std::env::var("VF_STEP_PROF").is_ok();
        let t_tick = Instant::now();
        let t0 = t_tick;
        let mut t_reads = 0.0f32;
        let mut marks = [0.0f32; 4];
        /* Такт N-1 отправлен и давно закрыт: ждём его (это ноль), забираем
         * статистику и события ПРОШЛОГО такта. Раскол пишет тела с CPU, поэтому
         * до него GPU обязан быть свободен — иначе пишем по считающемуся буферу. */
        let t_wait = {
            let t = Instant::now();
            self.gpu.wait_idle();
            t.elapsed().as_secs_f32() * 1000.0
        };
        /* Статистика прошлого такта (8 Б из `stat`) — ДО сэмпла перфа: строка
         * тел в таблице берёт бодрствующих оттуда, отдельного чтения нет. */
        let t = Instant::now();
        let (vmax_now, awake_now) = self.gpu.read_stats();
        t_reads += t.elapsed().as_secs_f32() * 1000.0;

        /* Таймстемпы стадий прошлого такта: он уже закрыт, а новый submit ещё
         * не отправлен — данные в слоте кольца целы. Включено только по F3 или
         * VF_PERF=1; иначе вызов пустой. */
        self.gpu.perf_sample(awake_now, self.alive_count());
        /* Статистика прошлого такта + то, что CPU узнал от события: иначе на
         * такте выстрела chunks считался бы по сцене без снаряда, и снаряд на
         * 120 м/с перескакивал плиту между подшагами (проверено probe_fastshot). */
        let vmax = (vmax_now as f64).max(self.pending_vmax as f64);
        self.pending_vmax = 0.0;
        self.last_vmax = vmax as f32;
        self.last_awake = awake_now;
        if self.fracture_enabled {
            if self.gpu_collide {
                /* События удара с устройства: 32 байта на тело вместо списка
                 * контактов, и энергия посчитана ДО солвера. Кольцо + счётчик:
                 * в спокойном такте из GPU уходит 4 байта. */
            let t = Instant::now();
                let ev = self.gpu.read_impact_events();
                t_reads += t.elapsed().as_secs_f32() * 1000.0;
                if !ev.is_empty() {
                    self.fracture_from_events(&ev);
                }
            } else {
                let t = Instant::now();
                let contacts = self.gpu.read_contacts();
                t_reads += t.elapsed().as_secs_f32() * 1000.0;
                self.fracture(&contacts);
            }
        }
        /* Удары, найденные CCD на прошлом такте: быстрый снаряд останавливается
         * ДО прохода контактов, поэтому в списке контактов его нет и событие
         * с устройства не придёт — энергию удара приносим сюда сами. */
        if self.fracture_enabled && !self.ccd_impacts.is_empty() {
            let hits = std::mem::take(&mut self.ccd_impacts);
            let ev: Vec<[f32; 12]> = hits
                .iter()
                .map(|(i, j, p, e)| {
                    [
                        *e, *i as f32, *j as f32, 0.0, 0.0, 0.0, 0.0, 0.0, p[0], p[1], p[2], 0.0,
                    ]
                })
                .collect();
            self.fracture_from_events(&ev);
        }

        if prof {
            marks[0] = t0.elapsed().as_secs_f32() * 1e3;
        }
        /* Геометрию заливаем ДО пары: иначе на такте спавна снаряда пул
         * ещё пуст, пар нет, и снаряд пролетает сквозь плиту (найдено на
         * 120 м/с: кадр 0 — 0 пар, кадр 1 — 22976, но уже поздно). */
        if self.geom_dirty {
            let local: Vec<Vec<[[f32; 3]; 4]>> = self.bodies.iter().map(|b| b.local_tets().clone()).collect();
            self.gpu.upload_geometry(&local);
            /* Таблица SDF-полей: дескриптор и значения на каждый слот. Заливаем
             * здесь же — вместе с геометрией, потому что набор тел меняется
             * ровно в этих точках (спавн, раскол, сброс). */
            let items: Vec<Option<(&crate::field::NodeField, [f32; 3])>> = self
                .bodies
                .iter()
                .map(|b| {
                    if b.alive {
                        b.sdf.as_ref().map(|f| (f, b.sdf_shift))
                    } else {
                        None
                    }
                })
                .collect();
            self.gpu.upload_sdf(&items);
            self.geom_dirty = false;
        }

        if prof {
            marks[1] = t0.elapsed().as_secs_f32() * 1e3;
        }
        if self.gpu_pairs {
            /* Пары рождаются на устройстве ВНУТРИ такта (см.
             * `tet_pair_steps`/`set_inline_pairs`): отдельных submit'ов и
             * ожиданий fence здесь больше нет, на CPU приходит только их
             * число — и то не в такте. VF_PAIRS_OLD=1 возвращает старый путь
             * (отдельный submit + fence) для сравнения. */
            if self.pairs_old {
                let t = Instant::now();
                self.gpu.run_tet_pairs(dt);
                t_reads += t.elapsed().as_secs_f32() * 1000.0;
            }
        } else if self.sdf_collide {
            /* SDF-контактам пары ТЕТОВ не нужны вовсе: широкий проход по AABB
             * тел делает сам такт. CPU-путь ниже строил бы миллионы тет-пар
             * ради ничего. */
        } else {
            /* Пары на CPU — отладочный путь: ему нужны позы, читаем буфер
             * целиком (в такте этого больше нет). */
            self.sync_all();
            let shapes: Vec<Vec<[[f32; 3]; 4]>> =
                self.bodies.iter().map(|b| b.rotated_tets()).collect();
            let pos: Vec<[f32; 3]> = self
                .mirror
                .iter()
                .map(|b| [b.pos_r[0], b.pos_r[1], b.pos_r[2]])
                .collect();
            // растягиваем AABB на путь за такт: иначе быстрый снаряд проскакивает
            let inflate: Vec<[f32; 3]> = self
                .bodies
                .iter()
                .map(|b| {
                    let r = (dt.abs() * 0.5) as f64;
                    [
                        (b.vel[0].abs() * r) as f32,
                        (b.vel[1].abs() * r) as f32,
                        (b.vel[2].abs() * r) as f32,
                    ]
                })
                .collect();
            let pairs = collide::build_pairs_swept(
                &shapes,
                &pos,
                (self.params.lo, self.params.hi),
                self.params.cell * 0.5,
                &inflate,
            );
            let packed: Vec<(u32, u32)> = pairs
                .iter()
                .take(crate::sim::MAX_PAIRS)
                .map(|(b1, t1, b2, t2)| ((b1 << 20) | t1, (b2 << 20) | t2))
                .collect();
            self.gpu.set_pairs(&packed);
        }
        /* Та же нарезка, что на CPU: куски по скорости, внутри куска — подшаги
         * с ОДНИМ списком контактов. Скорость берём из статистики ПРОШЛОГО
         * такта: она приходит на устройстве и стоит 8 Б, а не чтения всех тел.
         * Скорость меняется плавно, запаздывание на такт безопасно. */
        let cell = (self.params.cell * 0.5) as f64;
        let mut chunks = ((vmax * dt as f64 / (cell * 0.35)).ceil() as u32).clamp(2, 12);
        /* VF_CHUNKS=N — потолок кусков. Стоимость такта линейна по кускам:
         * один снаряд на 26 м/с уже упирается в 12, и такт дорожает вшестеро. */
        if let Ok(v) = std::env::var("VF_CHUNKS") {
            if let Ok(n) = v.parse::<u32>() {
                chunks = chunks.min(n.clamp(1, 12));
            }
        }
        /* Continuous-режим для быстрых тел: бисекция вдоль перемещения за такт
         * ДО отправки такта. Тело, которое за такт смещается дальше своего
         * размера, ставится в позу первого касания с импульсом там же — иначе
         * оно оказывается по другую сторону партнёра между подшагами
         * (туннелирование). Медленные тела этим путём не идут. */
        if prof {
            marks[2] = t0.elapsed().as_secs_f32() * 1e3;
        }
        self.ccd_pass(dt);
        if prof {
            marks[3] = t0.elapsed().as_secs_f32() * 1e3;
        }
        /* Такт отправляем БЕЗ ожидания: к моменту следующего такта (50 мс) GPU
         * давно свободен, и CPU не стоит ни миллисекунды. Результат прошлого
         * такта читаем здесь же — он уже закрыт. */
        let t_submit = {
            let t = Instant::now();
            self.gpu.frame_gpu_contacts_async(dt, chunks, substeps, iterations);
            t.elapsed().as_secs_f32() * 1000.0
        };
        /* Фазы такта по требованию: «отправка» — настоящий submit, а не
         * остаток времени до него (раньше печаталось ДО отправки). */
        if prof {
            println!(
                "[step] такт: ожидание+стат+раскол {:.2} · заливка {:.2} · пары {:.2} · CCD {:.2} · отправка {:.2} мс (тел {}, трещин {})",
                marks[0],
                marks[1] - marks[0],
                marks[2] - marks[1],
                marks[3] - marks[2],
                t_submit,
                self.alive_count(),
                self.cracks
            );
        }
        // Контакты нужны и геймплею: раскол и пробуждение по контакту решаются
        // на CPU, поэтому читаем их один раз за такт (73 КБ при 64 телах).
        /* Контакты нужны CPU ТОЛЬКО для раскола: пробуждение делает SAT на
         * устройстве. Значит при выключенном расколе читать их обратно не
         * нужно вовсе — минус ~196 КБ за такт. */
        /* Никаких поз обратно: рендер читает их из буфера тел на устройстве,
         * раскол приходит событиями, статистика — с прошлого такта. */
        /* Каркас тетраэдров — отладка: ему нужны позы на CPU. Читаем буфер
         * только когда он включён, и это единственный случай, кроме раскола. */
        if self.lines_on {
            self.sync_all();
        }
        /* Фазы такта на CPU — в диагностику: без них «такт долгий» не отличить
         * от «GPU занят», а «кадр дёргается» — от «поток физики держал замок». */
        self.gpu.note_tick_cpu(
            t_wait,
            t_reads,
            t_submit,
            t_tick.elapsed().as_secs_f32() * 1000.0,
        );
        self.frames += 1;
    }

    /// Будит уснувшее тело: возвращает обратную массу и инерцию, которые
    /// проход сна на GPU обнулил (спящее тело = бесконечная масса), и
    /// переставляет якорь, чтобы таймер покоя начал считаться заново.
    fn wake(&mut self, i: usize) {
        if i >= self.bodies.len() || self.mirror[i].misc[1] <= 0.5 {
            return;
        }
        let b = &self.bodies[i];
        let im = if b.mass > 0.0 { 1.0 / b.mass } else { 0.0 } as f32;
        let inertia = solid_inertia(b);
        let ii = if inertia > 0.0 { (1.0 / inertia) as f32 } else { 0.0 };
        let g = &mut self.mirror[i];
        g.misc[1] = 0.0;
        g.vel_m[3] = im;
        g.ang_i[3] = ii;
        g.misc[2] = im;
        g.misc[3] = ii;
        g.anchor = [g.pos_r[0], g.pos_r[1], g.pos_r[2], 0.0];
    }

    /// Спит одно тело пары, а другое — нет: будим спящее. Спящая куча сама
    /// себя не будит (все её члены спят), а пуля или ползущий осколок будит
    /// соседа сразу, как только коснулся.
    fn wake_on_contact_pass(&mut self, contacts: &[Vec<collide::Contact>]) {
        let mut woke = false;
        for i in 0..self.bodies.len().min(contacts.len()) {
            for c in &contacts[i] {
                if c.other < 0 {
                    continue;
                }
                let j = c.other as usize;
                if j >= self.bodies.len() {
                    continue;
                }
                let si = self.mirror[i].misc[1] > 0.5;
                let sj = self.mirror[j].misc[1] > 0.5;
                if si == sj {
                    continue;
                }
                /* Будим, только если бодрствующий сосед РЕАЛЬНО движется.
                 * Раньше хватало самого факта контакта: в куче один
                 * шевелящийся осколок будил соседей, те — следующих, и куча
                 * просыпалась заново каждый такт. Каждое пробуждение ещё и
                 * заливает все тела на GPU — отсюда «со сном медленнее». */
                let awake = if si { j } else { i };
                let g = &self.mirror[awake];
                let speed = (g.vel_m[0] * g.vel_m[0] + g.vel_m[1] * g.vel_m[1] + g.vel_m[2] * g.vel_m[2]).sqrt();
                if speed < 0.35 {
                    continue;
                }
                self.wake(if si { i } else { j });
                woke = true;
            }
        }
        if woke {
            self.gpu.write_bodies(&self.mirror);
        }
    }

    /// Включить или выключить контактное пробуждение на CPU.
    pub fn set_wake_on_contact(&mut self, on: bool) {
        self.wake_on_contact = on;
    }

    /// Включить или выключить раскол.
    pub fn set_fracture(&mut self, on: bool) {
        self.fracture_enabled = on;
    }

    /// Включить или выключить сон тел.
    pub fn set_sleep(&mut self, on: bool) {
        self.sleep_enabled = on;
        self.gpu.skip_sleep = !on;
        if !on {
            /* Позы на CPU могли устареть: без чтения такт их не обновляет.
             * Будить тела по устаревшему зеркалу — значит залить на устройство
             * старые координаты и телепортировать осколки. */
            self.gpu.wait_idle();
            self.sync_all();
            for i in 0..self.bodies.len() {
                if self.bodies[i].alive {
                    self.wake(i);
                }
            }
            self.gpu.write_bodies(&self.mirror);
        }
    }

    /// Новый устойчивый id. Слот переиспользуется, id — никогда.
    fn new_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn alive_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.alive).count()
    }

    fn free_slot(&mut self) -> Option<usize> {
        if let Some(i) = self.bodies.iter().position(|b| !b.alive) {
            return Some(i);
        }
        if self.bodies.len() < MAX_BODIES {
            let _ = Body::new([0.0, -400.0, 0.0], [0.0; 3], 1.0, 0.05);
            self.bodies.push(LiveBody {
                id: 0,
                crack_after: 0,
                pos: [0.0, -400.0, 0.0],
                quat: [1.0, 0.0, 0.0, 0.0],
                vel: [0.0; 3],
                mass: 1.0,
                radius: 0.05,
                tets: Vec::new(),
                origin: [0.0; 3],
                level_cap: 0,
                local: Vec::new(),
                mesh: -1,
                lines: -1,
                kind: Kind::Shard,
                alive: false,
                sdf: None,
                sdf_shift: [0.0; 3],
            });
            return Some(self.bodies.len() - 1);
        }
        None
    }

    pub fn step(&mut self, dt: f32, substeps: u32, iterations: u32) {
        self.frames += 1;
        // Контакты нельзя снимать один раз на кадр: за 1/60 с снаряд на 26 м/с
        // пролетает 0.43 м — больше тетраэдра, и просто проходит сквозь тело.
        // Дробим кадр так, чтобы самое быстрое тело смещалось не больше
        // трети ячейки за кусок, и пересчитываем контакты на каждом куске.
        let vmax = self
            .bodies
            .iter()
            .filter(|b| b.alive)
            .map(|b| (b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2]).sqrt())
            .fold(0.0f64, f64::max);
        let cell = (self.params.cell * 0.5) as f64;
        // Потолок кусков выше: на такте 20 Гц быстрый снаряд проходит 1.3 м за
        // шаг, и при шести кусках он перескакивал мелкие осколки целиком.
        let chunks = ((vmax * dt as f64 / (cell * 0.35)).ceil() as u32).clamp(2, 12);
        let per = (substeps / chunks).max(1);
        let sdt = dt / chunks as f32;

        if self.gpu_collide {
            self.step_gpu(dt, substeps, iterations);
            return;
        }
        let timing = std::env::var("VF_TIME").is_ok();
        let (mut t_collide, mut t_dep, mut t_solve, mut t_frac) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for _ in 0..chunks {
            let t0 = std::time::Instant::now();
            let mut tet_shapes: Vec<Vec<[[f32; 3]; 4]>> = Vec::with_capacity(MAX_BODIES);
            for b in &self.bodies {
                tet_shapes.push(if b.alive { b.rotated_tets() } else { Vec::new() });
            }
            while tet_shapes.len() < MAX_BODIES {
                tet_shapes.push(Vec::new());
            }
            let positions: Vec<[f32; 3]> = self
                .mirror
                .iter()
                .map(|b| [b.pos_r[0], b.pos_r[1], b.pos_r[2]])
                .collect();
            let mut contacts = collide::build(
                &tet_shapes,
                &positions,
                0.0,
                (self.params.lo, self.params.hi),
                self.params.cell * 0.5,
            );
            for i in self.bodies.len()..MAX_BODIES {
                contacts[i].clear();
            }

            if !self.sleep_enabled {
            // страховка на случай, если проход сна уже успел кого-то усыпить
            for i in 0..self.bodies.len() {
                if self.bodies[i].alive && self.mirror[i].misc[1] > 0.5 {
                    self.wake(i);
                }
            }
        }
        if timing { t_collide += t0.elapsed().as_secs_f64(); }
            let t1 = std::time::Instant::now();
            if self.wake_on_contact {
                self.wake_on_contact_pass(&contacts);
            }
            self.depenetrate(&contacts);
            if timing { t_dep += t1.elapsed().as_secs_f64(); }
            let t2 = std::time::Instant::now();
            if self.fracture_enabled {
                self.fracture(&contacts);
            }
            if timing { t_frac += t2.elapsed().as_secs_f64(); }

            let t3 = std::time::Instant::now();
            self.gpu.write_contacts(&contacts);
            self.gpu.frame_with_contacts(sdt, per, iterations);
            if timing { t_solve += t3.elapsed().as_secs_f64(); }
            let snap = self.gpu.read_bodies();

            for i in 0..self.bodies.len() {
                if !self.bodies[i].alive {
                    continue;
                }
                let g = &snap[i];
                let b = &mut self.bodies[i];
                b.pos = [g.pos_r[0] as f64, g.pos_r[1] as f64, g.pos_r[2] as f64];
                b.quat = [
                    g.quat[0] as f64,
                    g.quat[1] as f64,
                    g.quat[2] as f64,
                    g.quat[3] as f64,
                ];
                b.vel = [g.vel_m[0] as f64, g.vel_m[1] as f64, g.vel_m[2] as f64];
            }
            let n = self.bodies.len();
            self.mirror[..n].copy_from_slice(&snap[..n]);
        }

        if timing {
            eprintln!("[time] кадр {}: контакты {:.2} мс, развязка {:.2} мс, солвер {:.2} мс, раскол {:.2} мс, всего {:.2} мс",
                self.frames, t_collide * 1e3, t_dep * 1e3, t_solve * 1e3, t_frac * 1e3,
                (t_collide + t_dep + t_solve + t_frac) * 1e3);
        }
        // улетевшее за пределы ящика убираем
        let mut moved = false;
        for i in 0..self.bodies.len() {
            if !self.bodies[i].alive {
                continue;
            }
            let p = self.bodies[i].pos;
            if p[1] < -2.0
                || p[0].abs() > self.params.hi[0] as f64 + 1.5
                || p[2].abs() > self.params.hi[2] as f64 + 1.5
            {
                self.bodies[i].alive = false;
                let parked = Body::new([0.0, -400.0, 0.0], [0.0; 3], 1.0, 0.05);
                let mut g2 = GpuBody::from_cpu(&parked);
                g2.misc[1] = 1.0;
                self.mirror[i] = g2;
                moved = true;
            }
        }
        if moved {
            self.gpu.write_bodies(&self.mirror);
        }
    }

    /// Позиционная развязка — мягко и с потолком: жёсткий «выброс» на всю
    /// глубину контакта в кадр и давал дрожание, будто коллайдер шире меша.
    fn depenetrate(&mut self, contacts: &[Vec<collide::Contact>]) {
        let mut delta: Vec<[f32; 3]> = vec![[0.0; 3]; self.mirror.len()];
        // Сколько точек контакта двигают тело. Без этого деления четыре точки
        // пола поднимали тело в 4 раза сильнее нужного: куб отрывался вверх,
        // падал и снова отрывался — «не может успокоиться».
        let mut cnt: Vec<u32> = vec![0; self.mirror.len()];
        let mut any = false;
        for i in 0..self.bodies.len() {
            if !self.bodies[i].alive {
                continue;
            }
            for c in &contacts[i] {
                if c.pen < 0.006 {
                    continue;
                }
                let k = (0.6 * c.pen).min(0.015);
                if c.other < 0 {
                    for a in 0..3 {
                        delta[i][a] += c.n[a] * k;
                    }
                    cnt[i] += 1;
                    any = true;
                } else if (c.other as usize) < self.bodies.len() {
                    let j = c.other as usize;
                    if !self.bodies[j].alive {
                        continue;
                    }
                    let (wi, wj) = (self.mirror[i].vel_m[3], self.mirror[j].vel_m[3]);
                    if wi <= 0.0 || wj <= 0.0 {
                        continue;
                    }
                    let share = wi / (wi + wj);
                    for a in 0..3 {
                        delta[i][a] -= c.n[a] * k * share;
                        delta[j][a] += c.n[a] * k * share;
                    }
                    cnt[i] += 1;
                    cnt[j] += 1;
                    any = true;
                }
            }
        }
        if !any {
            return;
        }
        for i in 0..self.bodies.len() {
            if delta[i] == [0.0; 3] {
                continue;
            }
            /* Раньше деление на число контактов (cnt) давало телу с 24
             * контактами 1/24 коррекции — в плотной куче разъезд практически
             * останавливался. Теперь ограничиваем СУММАРНЫЙ сдвиг тела за
             * кусок: направление сохраняется, шаг не больше 2 см. Так куча
             * расходится, но тело всё равно не подбрасывает. */
            /* Деление на число контактов: контакты считались независимо и в
             * сумме перебирали коррекцию — коробку подбрасывало. Скорость
             * разъезда поднимаем ПОДШАГАМИ и ИТЕРАЦИЯМИ, а не этой
             * константой: на большом шаге пара успевает обменяться сторонами
             * и стопка проваливается друг в друга. */
            let n = cnt[i].max(1) as f32;
            for a in 0..3 {
                self.mirror[i].pos_r[a] += delta[i][a] / n;
                // Якорь сна сдвигаем вместе с телом: иначе GPU видит «прыжок»
                // каждый кадр, таймер покоя сбрасывается и осколки дрожат
                // вечно вместо того, чтобы уснуть.
                self.mirror[i].anchor[a] += delta[i][a] / n;
            }
        }
        self.gpu.write_bodies(&self.mirror);
    }

    /// Наибольшее проникновение в текущей позе (м) — для проверок и отладки.
    pub fn max_pen(&self) -> f32 {
        let mut tet_shapes: Vec<Vec<[[f32; 3]; 4]>> = Vec::with_capacity(MAX_BODIES);
        for b in &self.bodies {
            tet_shapes.push(if b.alive { b.rotated_tets() } else { Vec::new() });
        }
        while tet_shapes.len() < MAX_BODIES {
            tet_shapes.push(Vec::new());
        }
        let positions: Vec<[f32; 3]> = self
            .mirror
            .iter()
            .map(|b| [b.pos_r[0], b.pos_r[1], b.pos_r[2]])
            .collect();
        let contacts = collide::build(
            &tet_shapes,
            &positions,
            0.0,
            (self.params.lo, self.params.hi),
            self.params.cell * 0.5,
        );
        contacts
            .iter()
            .flatten()
            .map(|c| c.pen)
            .fold(0.0f32, f32::max)
    }

    /// Наибольшая скорость среди живых тел.
    pub fn max_speed(&self) -> f64 {
        self.bodies
            .iter()
            .filter(|b| b.alive)
            .map(|b| (b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2]).sqrt())
            .fold(0.0, f64::max)
    }

    fn fracture(&mut self, contacts: &[Vec<collide::Contact>]) {
        let mut jobs: Vec<(usize, [f32; 3], f32)> = Vec::new();
        for (i, list) in contacts.iter().enumerate() {
            if i >= self.bodies.len() {
                break;
            }
            for c in list {
                if c.other < 0 || c.other as usize >= self.bodies.len() {
                    continue;
                }
                let j = c.other as usize;
                if i >= j || !self.bodies[i].alive || !self.bodies[j].alive {
                    continue;
                }
                let a = self.bodies[i].vel;
                let b = self.bodies[j].vel;
                let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                let n = [c.n[0] as f64, c.n[1] as f64, c.n[2] as f64];
                let vrel = (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]).abs();
                if vrel < 4.0 {
                    continue;
                }
                // цель — не снаряд: снаряд мелкий и колется только он сам,
                // а нам нужен тот, в кого попали
                let (ti, oi) = match (self.bodies[i].kind == Kind::Bullet, self.bodies[j].kind == Kind::Bullet) {
                    (false, true) => (i, j),
                    (true, false) => (j, i),
                    (true, true) => continue,
                    (false, false) => {
                        if self.bodies[i].tets.len() >= self.bodies[j].tets.len() {
                            (i, j)
                        } else {
                            (j, i)
                        }
                    }
                };
                if self.bodies[ti].tets.len() < MIN_TETS || self.bodies[ti].kind == Kind::Bullet {
                    continue;
                }
                if self.frames < self.bodies[ti].crack_after {
                    continue;
                }
                let mu = 1.0 / (1.0 / self.bodies[ti].mass + 1.0 / self.bodies[oi].mass);
                let e = 0.5 * mu * vrel * vrel;
                if e < 500.0 {
                    continue;
                }
                let pi = self.bodies[ti].pos;
                let pn = [c.n[0] as f64, c.n[1] as f64, c.n[2] as f64];
                let point = [
                    (pi[0] - pn[0] * 0.2) as f32,
                    (pi[1] - pn[1] * 0.2) as f32,
                    (pi[2] - pn[2] * 0.2) as f32,
                ];
                jobs.push((ti, point, e as f32));
            }
        }
        self.apply_jobs(jobs);
    }

    /// Раскол по событиям удара С УСТРОЙСТВА: энергия посчитана до солвера,
    /// поэтому в неё попадает и удар, который солвер тут же погасил.
    /// Раскол по событиям из кольца на устройстве.
    /// Событие: (энергия, тело, партнёр), (нормаль), (точка) — 12 float.
    fn fracture_from_events(&mut self, ev: &[[f32; 12]]) {
        let mut jobs: Vec<(usize, [f32; 3], f32)> = Vec::new();
        for rec in ev {
            let (e, partner) = (rec[0], rec[2]);
            let i = rec[1] as usize;
            if partner < 0.0 || e < crate::sim::FRACTURE_ENERGY {
                continue;
            }
            if i >= self.bodies.len() || !self.bodies[i].alive {
                continue;
            }
            let j = partner as usize;
            if j >= self.bodies.len() || !self.bodies[j].alive {
                continue;
            }
            // цель — тот, в кого попали (не снаряд)
            let ti = match (self.bodies[i].kind == Kind::Bullet, self.bodies[j].kind == Kind::Bullet) {
                (true, false) => j,
                (false, true) => i,
                (true, true) => continue,
                (false, false) => i,
            };
            if self.bodies[ti].kind == Kind::Bullet || self.bodies[ti].tets.len() < MIN_TETS {
                continue;
            }
            if self.frames < self.bodies[ti].crack_after {
                continue;
            }
            jobs.push((ti, [rec[8], rec[9], rec[10]], e));
        }
        self.apply_jobs(jobs);
        self.refresh_vmax();
    }

    /// Общая часть: разложить заявки на раскол с учётом свободных слотов.
    fn apply_jobs(&mut self, jobs: Vec<(usize, [f32; 3], f32)>) {
        /* split() заливает буфер тел ЦЕЛИКОМ — значит зеркало обязано быть
         * свежим, иначе остальные тела уедут назад к моменту прошлого чтения.
         * Раскол — событие, поэтому полное чтение здесь допустимо. */
        if !jobs.is_empty() {
            self.gpu.wait_idle();
            self.sync_all();
        }
        for (ti, point, energy) in jobs {
            if !self.bodies[ti].alive {
                continue;
            }
            let free = self.bodies.iter().filter(|b| !b.alive).count()
                + (MAX_BODIES - self.bodies.len());
            if free < 2 {
                continue;
            }
            self.split(ti, point, energy, free.min(8));
        }
    }

    /// Разбить тело ti в точке point. Число осколков и разлёт — от энергии.
    /// Подтянуть позу одного тела с устройства. Только по событию (раскол,
    /// включение каркаса): в такте позы на CPU больше не нужны.
    fn sync_body(&mut self, i: usize) {
        let Some(g) = self.gpu.read_body(i) else { return };
        if i >= self.bodies.len() || !self.bodies[i].alive {
            return;
        }
        let b = &mut self.bodies[i];
        b.pos = [g.pos_r[0] as f64, g.pos_r[1] as f64, g.pos_r[2] as f64];
        b.quat = [g.quat[0] as f64, g.quat[1] as f64, g.quat[2] as f64, g.quat[3] as f64];
        b.vel = [g.vel_m[0] as f64, g.vel_m[1] as f64, g.vel_m[2] as f64];
    }

    /// Пересчитать максимум скорости по CPU-состоянию. Нужен сразу после
    /// событий (выстрел, осколки): иначе нарезка такта на куски берёт скорость
    /// прошлого такта, где снаряда ещё не было, chunks остаётся 2 — и быстрый
    /// снаряд пролетает плиту между подшагами.
    fn refresh_vmax(&mut self) {
        let v = self
            .bodies
            .iter()
            .filter(|b| b.alive)
            .map(|b| (b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2]).sqrt())
            .fold(0.0f64, f64::max) as f32;
        self.pending_vmax = self.pending_vmax.max(v);
        self.last_vmax = self.last_vmax.max(v);
    }

    /// Сколько тел бодрствует по последнему такту (для решения «шагать ли»).
    pub fn awake(&self) -> bool {
        self.last_awake > 0
    }

    /// Подтянуть ВСЕ позы с устройства. Только по событию: `write_bodies`
    /// заливает буфер целиком, и залить устаревшее зеркало — значит
    /// телепортировать все тела назад. В такте не вызывается.
    pub fn refresh_poses(&mut self) {
        self.sync_all();
    }

    /// Подтянуть ВСЕ позы с устройства. Только по событию (раскол, сброс):
    /// в такте этого нет. Зеркало обновляется для CPU-солвера — GPU-путь его
    /// не читает.
    fn sync_all(&mut self) {
        let snap = self.gpu.read_bodies();
        let n = self.mirror.len().min(snap.len());
        self.mirror[..n].copy_from_slice(&snap[..n]);
        let m = self.bodies.len().min(snap.len());
        for i in 0..m {
            self.bodies[i].pos = [snap[i].pos_r[0] as f64, snap[i].pos_r[1] as f64, snap[i].pos_r[2] as f64];
            self.bodies[i].quat = [
                snap[i].quat[0] as f64,
                snap[i].quat[1] as f64,
                snap[i].quat[2] as f64,
                snap[i].quat[3] as f64,
            ];
            self.bodies[i].vel = [
                snap[i].vel_m[0] as f64,
                snap[i].vel_m[1] as f64,
                snap[i].vel_m[2] as f64,
            ];
        }
    }

    pub fn split(&mut self, ti: usize, point: [f32; 3], energy: f32, max_new: usize) {
        /* Поза цели нужна для точки удара и начального положения осколков:
         * берём одно тело с устройства, а не весь буфер. */
        self.sync_body(ti);
        let (tq, tpos, tvel) = {
            let t = &self.bodies[ti];
            (t.quat, t.pos, t.vel)
        };
        /* ПОЛЕ РОДИТЕЛЯ — в осколки. Раньше здесь стоял `sdf: Default::default()`,
         * и `shatter_fields` считал поле заново BFS-ом по занятости: осколки
         * выходили «старой» грубой формы, пока целые тела были гладкими. */
        let mut tb = TetBody {
            tets: self.bodies[ti].tets.clone(),
            level_cap: self.bodies[ti].level_cap,
            origin: self.bodies[ti].origin,
            sdf: Default::default(),
        };
        if let Some(f) = &self.bodies[ti].sdf {
            let _ = tb.sdf.set(f.clone());
        }
        let c = tb.world_centroid();
        // точка удара в координатах решётки этого тела
        let p_local = rot_inv(
            tq,
            [
                point[0] as f64 - tpos[0],
                point[1] as f64 - tpos[1],
                point[2] as f64 - tpos[2],
            ],
        );
        let focus = [
            (p_local[0] + c[0] as f64) as f32,
            (p_local[1] + c[1] as f64) as f32,
            (p_local[2] + c[2] as f64) as f32,
        ];

        // clamp(3.0, max_new) паниковал, когда свободных слотов меньше трёх:
        // у clamp минимум оказывался больше максимума. Если мест мало —
        // просто режем на сколько есть.
        let want = 2.0 + energy / 700.0;
        let hi = max_new as f32;
        let n = if hi <= 3.0 {
            hi.max(1.0) as usize
        } else {
            want.clamp(3.0, hi) as usize
        };
        let mut frags = tb.shatter_biased(n, 0x51ED_2A17 ^ (self.cracks as u64 + 1), focus);
        frags.truncate(max_new);
        if frags.len() < 2 {
            return;
        }

        let mass_total: f64 = frags.iter().map(|f| DENSITY * f.volume()).sum();
        let v_rms = (2.0 * 0.20 * energy as f64 / mass_total.max(1e-6))
            .sqrt()
            .min(20.0);

        // Разлёт должен быть импульсно-нейтральным: куски разлетаются от
        // точки удара, но НЕ уносят всю кучу вперёд. Иначе плита «улетала»
        // вместе со снарядом: замер — осколки по 2–10 кг разгонялись до
        // 20 м/с, хотя импульс снаряда (90 кг × 26 м/с) на 590 кг плиты
        // даёт всего ~4 м/с.
        let mut net = [0.0f64; 3];
        for fr in frags.iter() {
            // центроид осколка в координатах РЕШЁТКИ; c — центроид родителя
            // в тех же координатах, поэтому смещение = разность
            let fc = fr.world_centroid();
            let off = rot(tq, [
                (fc[0] - c[0]) as f64,
                (fc[1] - c[1]) as f64,
                (fc[2] - c[2]) as f64,
            ]);
            let w = [off[0] + tpos[0], off[1] + tpos[1], off[2] + tpos[2]];
            let d = [w[0] - point[0] as f64, w[1] - point[1] as f64, w[2] - point[2] as f64];
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3);
            let m = DENSITY * fr.volume();
            for k in 0..3 {
                net[k] += m * (d[k] / l) as f64;
            }
        }
        let drift = [net[0] / mass_total, net[1] / mass_total, net[2] / mass_total];

        let mut rng = crate::tet::Lcg::new(0xB17E_5EED ^ (self.cracks as u64));
        let mut created = 0usize;
        let mut new_slots: Vec<usize> = Vec::new();

        for (k, fr) in frags.iter().enumerate() {
            let lc = fr.world_centroid();
            // Мировой центр осколка: `lc` — в координатах решётки, `c` — центроид
            // РОДИТЕЛЯ в тех же координатах (он же и есть tpos). Без вычитания
            // `c` позиция родителя прибавлялась дважды, и осколки рождались
            // выше на высоту исходного куба.
            let wp = rot(tq, [
                (lc[0] - c[0]) as f64,
                (lc[1] - c[1]) as f64,
                (lc[2] - c[2]) as f64,
            ]);
            let wp = [wp[0] + tpos[0], wp[1] + tpos[1], wp[2] + tpos[2]];
            let d = [wp[0] - point[0] as f64, wp[1] - point[1] as f64, wp[2] - point[2] as f64];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3);
            let speed = v_rms * (0.5 + 0.9 * rng.unit() as f64);

            let slot = if k == 0 { Some(ti) } else { self.free_slot() };
            let Some(slot) = slot else { break };

            let id = self.new_id();
            let mut body = LiveBody {
                id,
                crack_after: self.frames + 30,
                pos: wp,
                quat: tq,
                vel: [
                    d[0] / len * speed + tvel[0] * 0.4 - drift[0],
                    d[1] / len * speed + tvel[1] * 0.4 - drift[1],
                    d[2] / len * speed + tvel[2] * 0.4 - drift[2],
                ],
                mass: DENSITY * fr.volume(),
                radius: 0.2,
                tets: fr.tets.clone(),
                origin: fr.origin,
            sdf: fr.sdf.get().cloned(),
                sdf_shift: [0.0; 3],
                level_cap: fr.level_cap,
                local: Vec::new(),
                mesh: -1,
                lines: -1,
                kind: Kind::Shard,
                alive: true,
            };
            body.rebuild();
            body.radius = body.radius_of();
            if !self.gpu_collide {
                self.mirror[slot] = gpu_body(&body);
            }
            if slot < self.bodies.len() {
                self.bodies[slot] = body;
            } else {
                self.bodies.push(body);
            }
            new_slots.push(slot);
            created += 1;
        }

        // Слоты осколков ОБЯЗАНЫ уехать на GPU в этом же кадре: иначе снимок
        // вернёт в этот слот старое тело, и новый осколок «телепортируется»
        // на его место (выглядит как пинг). Раньше запись делалась только при
        // created >= 2, а при нехватке слотов выходило ровно один.
        if created > 0 {
            // сначала аккуратно расталкиваем новорождённых позициями,
            // и только потом отдаём их солверу
            self.settle_new_shards(&new_slots);
            /* Осколки уносят до 20 м/с: такт обязан узнать это ДО своего
             * CCD-порога, иначе первый такт после раскола идёт без
             * continuous и свежие мелкие куски пробиваются насквозь. */
            self.refresh_vmax();
            /* Только слоты осколков (первый из них — слот родителя):
             * остальные тела в буфере новее зеркала, их заливать нельзя. */
            let items: Vec<(usize, GpuBody)> =
                new_slots.iter().map(|&sl| (sl, gpu_body(&self.bodies[sl]))).collect();
            self.gpu.write_slots(&items);
            self.bodies[ti].crack_after = self.frames + 30;
        self.geom_dirty = true;
        }
        if created >= 2 {
            self.cracks += 1;
            self.shards_made += created;
            self.last_log = format!(
                "трещина #{}: {} осколков, удар {:.1} кДж",
                self.cracks,
                created,
                energy as f64 / 1000.0
            );
        }
    }

    /// Убирает тело из мира: паркует его далеко внизу и помечает спящим.
    /// Слот после этого свободен и его забирает следующий снаряд.
    fn despawn(&mut self, slot: usize) {
        if slot >= self.bodies.len() || !self.bodies[slot].alive {
            return;
        }
        self.bodies[slot].alive = false;
        let parked = Body::new([0.0, -400.0, 0.0], [0.0; 3], 1.0, 0.05);
        let mut g = GpuBody::from_cpu(&parked);
        g.misc[1] = 1.0;
        if !self.gpu_collide {
            /* CPU-солверу позы нужны на CPU — там зеркало ещё рабочее. */
            self.mirror[slot] = g;
        }
        /* Только этот слот: остальные тела в буфере трогать нельзя, их позы
         * новее зеркала. Ни ожидания, ни полного чтения больше не нужно. */
        self.gpu.write_slots(&[(slot, g)]);
    }

    /// Выстрел из точки from в направлении dir.
    pub fn shoot(&mut self, from: [f64; 3], dir: [f64; 3], speed: f64) -> bool {
        // Пул снарядов: мёртвые записи выбрасываем, а когда живых набралось
        // MAX_BULLETS — убираем самый старый из мира и забираем его слот.
        while let Some(&old) = self.bullets.front() {
            let stale = old >= self.bodies.len()
                || !self.bodies[old].alive
                || self.bodies[old].kind != Kind::Bullet;
            if stale {
                self.bullets.pop_front();
            } else {
                break;
            }
        }
        while self.bullets.len() >= MAX_BULLETS {
            match self.bullets.pop_front() {
                Some(old) => self.despawn(old),
                None => break,
            }
        }
        let Some(slot) = self.free_slot() else { return false };
        /* Снаряд — ШАР, заданный SDF (аналитическая сфера, запечённая в узлы).
         * Раньше здесь стоял гранёный куб, и никакая правка ball() до снаряда
         * не доходила — снаружи это и выглядело как угловатый бок с шипом. */
        let ball = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let id = self.new_id();
        let mut body = LiveBody {
            id,
            crack_after: 0,
            pos: from,
            quat: [1.0, 0.0, 0.0, 0.0],
            vel: [dir[0] * speed, dir[1] * speed, dir[2] * speed],
            // Снаряд-камень, а не ядро: 90 кг на 26 м/с давали 30 кДж и
            // прошивали плиту, разгоняя осколки до 20 м/с. 8 кг — попадание,
            // от которого плита колется, но не улетает.
            mass: 8.0,
            radius: 0.14,
            tets: ball.tets.clone(),
            origin: ball.origin,
            level_cap: ball.level_cap,
            sdf: ball.sdf.get().cloned(),
            sdf_shift: [0.0; 3],
            local: Vec::new(),
            mesh: -1,
            lines: -1,
            kind: Kind::Bullet,
            alive: true,
        };
        body.rebuild();
        body.radius = body.radius_of();
        if slot < self.bodies.len() {
            self.bodies[slot] = body;
        } else {
            self.bodies.push(body);
        }
        let gb = gpu_body(&self.bodies[slot]);
        if !self.gpu_collide {
            self.mirror[slot] = gb;
        }
        /* Спавн — запись одного слота; остальные тела в буфере не трогаем. */
        self.gpu.write_slots(&[(slot, gb)]);
        self.geom_dirty = true;
        // слот мог остаться в очереди от прошлого снаряда — иначе появляются
        // дубликаты и часть снарядов выпадает из пула навсегда
        self.bullets.retain(|&s| s != slot);
        self.bullets.push_back(slot);
        self.shots += 1;
        /* снаряд только что родился на CPU — нарезка такта обязана это знать */
        self.refresh_vmax();
        self.last_awake = self.last_awake.max(1);
        true
    }

    /// Сброс: снова один цельный объект.
    pub fn reset(&mut self) {
        /* ВАЖНО: не создавать новый Live. Девайс теперь один на физику и окно,
         * и свежий Live означал бы второй девайс, а прежний снёсся бы вместе с
         * буфером, на который смотрит окно. Сбрасываем состояние на месте. */
        self.gpu.wait_idle();
        let (params, body, gpu_bodies) = scene_plate();
        self.bodies = vec![body];
        self.mirror = gpu_bodies;
        self.gpu.write_bodies(&self.mirror);
        self.geom_dirty = true;
        self.bullets.clear();
        self.next_id = 1 + (MAX_BODIES as u64);
        self.cracks = 0;
        self.shards_made = 0;
        self.shots = 0;
        self.last_vmax = 0.0;
        self.last_awake = MAX_BODIES as u32;
        self.pending_vmax = 0.0;
        self.last_log.clear();
        /* id начинаются заново — кэш пробных точек по id обязан умереть
         * вместе со сценой, иначе новые тела получат точки старых. */
        self.ccd_probes.clear();
    }

    /// Формы для окна: у каждого тела свой меш и каркас тетраэдров.
    pub fn shape_for(&self, i: usize) -> Option<Shape> {
        let b = &self.bodies[i];
        if !b.alive || b.tets.is_empty() {
            return None;
        }
        /* ПОЛЕ ТЕЛА, а не BFS по занятости: раньше здесь стоял `sdf:
         * Default::default()`, и меш строился по BFS-расстояниям от границы
         * занятости. Те значения ступенчатые (|v| ≈ 0.5·шаг у границы, ±), ноль
         * поля попадал в фиксированную долю ребра — вершины «липли к узлам» и
         * стояли в постоянном отступе от них, независимо от формы. Запечённое
         * поле — настоящие расстояния, по нему интерполяция и работает. */
        let tb = {
            let mut tb = TetBody {
                tets: b.tets.clone(),
                level_cap: b.level_cap,
                origin: b.origin,
                sdf: Default::default(),
            };
            if let Some(f) = &b.sdf {
                let _ = tb.sdf.set(f.clone());
            }
            tb
        };
        let c = tb.world_centroid();
        // без сглаживания: у скола должны быть грани, а не блоб
        let (mut mesh, grid) = tb.surface_grid();
        for p in mesh.positions.iter_mut() {
            p[0] -= c[0];
            p[1] -= c[1];
            p[2] -= c[2];
        }
        // edge_lines() уже отдаёт точки относительно центроида — второй раз
        // вычитать нельзя, иначе каркас уезжает от меша
        let edges = tb.edge_lines();
        Some(Shape {
            level: mesh.level,
            mesh,
            grid_lo: grid.lo,
            grid_hi: grid.hi,
            grid_step: grid.step,
            edges,
        })
    }
}

/// Момент инерции по самим тетраэдрам.
///
/// Раньше тело строилось как сфера: `1/(0.4·m·r²)` с радиусом 0.5, хотя у
/// плиты полуразмер 0.56 и она куб. Инерция выходила вдвое меньше настоящей —
/// коробка крутилась слишком легко, качалась на грани и не могла успокоиться.
/// Осколки от этого почти не страдали: они мелкие и по форме близки к
/// компактным, а коробка — нет.
///
/// Для точечных масс I_xx = (2/3)·Σ mᵢ·|rᵢ|², что для куба со стороной a
/// даёт ровно m·a²/6.
fn solid_inertia(b: &LiveBody) -> f64 {
    if b.local.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f64;
    let mut moment = 0.0f64;
    let mut w = [0.0f64; 3];
    for t in &b.local {
        let a = [t[0][0] as f64, t[0][1] as f64, t[0][2] as f64];
        let bb = [t[1][0] as f64, t[1][1] as f64, t[1][2] as f64];
        let c = [t[2][0] as f64, t[2][1] as f64, t[2][2] as f64];
        let d = [t[3][0] as f64, t[3][1] as f64, t[3][2] as f64];
        let u = [bb[0] - a[0], bb[1] - a[1], bb[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let w2 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
        let det = u[0] * (v[1] * w2[2] - v[2] * w2[1]) - u[1] * (v[0] * w2[2] - v[2] * w2[0])
            + u[2] * (v[0] * w2[1] - v[1] * w2[0]);
        let vol = (det / 6.0).abs();
        if vol <= 0.0 {
            continue;
        }
        let m = DENSITY * vol;
        let cen = [
            0.25 * (a[0] + bb[0] + c[0] + d[0]),
            0.25 * (a[1] + bb[1] + c[1] + d[1]),
            0.25 * (a[2] + bb[2] + c[2] + d[2]),
        ];
        total += m;
        for k in 0..3 {
            w[k] += m * cen[k];
        }
    }
    if total <= 0.0 {
        return 0.0;
    }
    let com = [w[0] / total, w[1] / total, w[2] / total];
    for t in &b.local {
        let a = [t[0][0] as f64, t[0][1] as f64, t[0][2] as f64];
        let bb = [t[1][0] as f64, t[1][1] as f64, t[1][2] as f64];
        let c = [t[2][0] as f64, t[2][1] as f64, t[2][2] as f64];
        let d = [t[3][0] as f64, t[3][1] as f64, t[3][2] as f64];
        let u = [bb[0] - a[0], bb[1] - a[1], bb[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let w2 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
        let det = u[0] * (v[1] * w2[2] - v[2] * w2[1]) - u[1] * (v[0] * w2[2] - v[2] * w2[0])
            + u[2] * (v[0] * w2[1] - v[1] * w2[0]);
        let vol = (det / 6.0).abs();
        if vol <= 0.0 {
            continue;
        }
        let m = DENSITY * vol;
        let cen = [
            0.25 * (a[0] + bb[0] + c[0] + d[0]) - com[0],
            0.25 * (a[1] + bb[1] + c[1] + d[1]) - com[1],
            0.25 * (a[2] + bb[2] + c[2] + d[2]) - com[2],
        ];
        /* Собственный момент тетраэдра — разброс ЕГО вершин вокруг своего
         * центроида. Без этого у осколка из одного-двух тетраэдров все массы
         * лежат в одной точке, момент выходит почти нулевым, обратная
         * инерция — огромной, и любой импульс раскручивал кусок до
         * бесконечности («мелкие осколки неумолимо вращаются»). */
        let c0 = [
            0.25 * (a[0] + bb[0] + c[0] + d[0]),
            0.25 * (a[1] + bb[1] + c[1] + d[1]),
            0.25 * (a[2] + bb[2] + c[2] + d[2]),
        ];
        let mut s2 = 0.0f64;
        for v in [a, bb, c, d] {
            let dx = v[0] - c0[0];
            let dy = v[1] - c0[1];
            let dz = v[2] - c0[2];
            s2 += dx * dx + dy * dy + dz * dz;
        }
        s2 *= 0.25;
        moment += m * (cen[0] * cen[0] + cen[1] * cen[1] + cen[2] * cen[2] + 0.4 * s2);
    }
    (2.0 / 3.0) * moment
}

fn gpu_body(b: &LiveBody) -> GpuBody {
    let mut cpu = Body::new(b.pos, b.vel, b.mass, b.radius);
    cpu.rot = b.quat;
    let i = solid_inertia(b);
    cpu.inv_inertia = if i > 0.0 { 1.0 / i } else { 0.0 };
    let mut g = GpuBody::from_cpu(&cpu);
    // Настоящие полуразмеры, а не радиус: солвер берёт по ним плечо для
    // контактов, которые считает сам GPU. Точка контакта из SAT теперь
    // приходит отдельно, но габариты всё равно должны быть честными.
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for t in &b.local {
        for p in t {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    let half = if b.local.is_empty() {
        [b.radius, b.radius, b.radius]
    } else {
        [
            0.5 * (hi[0] - lo[0]),
            0.5 * (hi[1] - lo[1]),
            0.5 * (hi[2] - lo[2]),
        ]
    };
    g.half_shape = [half[0], half[1], half[2], 1.0];
    g
}
