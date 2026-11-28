//! Непрерывное обнаружение столкновений (CCD) по SDF.
//!
//! Зачем: обычный контакт ищется только в позах на концах шага. Если тело за
//! шаг смещается дальше, чем его собственный размер, оно может оказаться по
//! другую сторону партнёра, ни разу не попав внутрь на момент проверки —
//! «туннелирование». Лечение: для таких пар ищем ПЕРВЫЙ момент касания вдоль
//! перемещения за шаг и ставим тело в эту позу; импульс считается там же, как
//! у обычного контакта.
//!
//! Как ищется момент касания (порядок важен — каждая ступень чинит свой класс
//! промахов, найденных замерами):
//!  1. ГАБАРИТ. Сначала аналитически: отрезок движения центра пробного тела
//!     против ОСЕВОЙ КОРОБКИ партнёра, раздутой на габаритный радиус пробного
//!     (`sweep_sphere_box`). Это точный интервал `[t0, t1]`, где тела вообще
//!     могут соприкоснуться. Раньше здесь был сферный трейсинг по полю с
//!     шагом `|f|` и равномерный проход из 48 проб: вне блока поля `|f|`
//!     зажато к значению на границе, трейсинг топтался на месте, а 48 проб по
//!     12 м пути — это шаг 25 см. Мелкий осколок (3 см) такой шаг пролетал
//!     целиком, отсюда и остаточное туннелирование.
//!  2. ПОЛЕ. Внутри интервала ищем первую долю шага, где поверхность пробного
//!     тела дошла до поверхности партнёра (`g(s) < 0`), и уточняем бисекцией.
//!  3. ТОНКОЕ ПОЛЕ. Если поле партнёра тоньше трёх шагов сетки, оно Геометрию
//!     не описывает (у скола остаётся слой узлов), и шаг 2 может не найти
//!     ноль вовсе. Для такого партнёра контакт считается аналитически по его
//!     габаритной коробке — она и есть коллайдер, который видит GPU.
//!
//! Пробные точки — не только узлы поля: у мелкого осколка узлов мало и все они
//! в одном слое, поэтому добавляются ЦЕНТР габарита (с глубиной, равной
//! вписанному радиусу — он ловит случай «партнёр целиком внутри пробного») и
//! шесть точек по осям габарита для полей с малым числом узлов.
//!
//! Порог включения — ОТНОСИТЕЛЬНОЕ смещение за шаг больше размера меньшего
//! тела (`needs_continuous`): медленные тела идут прежним путём и ничего не
//! платят.

use std::time::Instant;

use crate::field::NodeField;

/// Сколько пробных точек берём с тела. Точки — узлы у ПОВЕРХНОСТИ (сортируем
/// по глубине) плюс центр и оси габарита. 64 вместо 256: цена бисекции линейна
/// по числу точек, а на точность момента касания влияют первые десятки.
pub const PROBE_POINTS: usize = 64;

/// Сколько узлов поля вообще просматриваем при сборке пробных точек: поле
/// крупного тела — 60 тысяч узлов, и обход всех на каждом новом теле давал
/// всплеск такта ровно в кадре раскола (появляются десятки новых тел сразу).
const PROBE_SCAN_NODES: usize = 8192;

/// Сканирующих проб внутри интервала габарита перед бисекцией.
const SCAN_STEPS: u32 = 16;

/// Допуск бисекции — доля перемещения за шаг.
pub const T_TOL: f32 = 1e-3;

/// Подробная трассировка одного поиска (VF_CCD_TRACE=1) — для проб.
fn trace() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VF_CCD_TRACE").is_ok())
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale(a: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

#[inline]
fn norm(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

/// Кватернион (w, x, y, z) — как в `common.glsl`/`live::rot`.
#[inline]
pub fn quat_rot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[1], q[2], q[3]];
    let t = scale(cross(u, v), 2.0);
    add(add(v, scale(t, q[0])), cross(u, t))
}

#[inline]
pub fn quat_inv_rot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    quat_rot([q[0], -q[1], -q[2], -q[3]], v)
}

/// Знаковое расстояние до осевой коробки (минус внутри), метры.
#[inline]
pub fn box_sdf(p: [f32; 3], h: [f32; 3]) -> f32 {
    let q = [p[0].abs() - h[0], p[1].abs() - h[1], p[2].abs() - h[2]];
    let out = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]).sqrt() + q[0].max(q[1]).max(q[2]).min(0.0)
}

/// Габарит тела в его системе координат: осевая коробка по тетам.
///
/// Это же и коллайдер, который видит GPU (`half_shape`), поэтому тонкий осколок
/// честно считается коробкой, а не полем, которое его не описывает.
#[derive(Clone, Copy, Debug)]
pub struct BoxShape {
    pub center: [f32; 3],
    pub half: [f32; 3],
    /// Радиус описанной сферы (максимум |вершина| от центра коробки).
    pub radius: f32,
}

impl Default for BoxShape {
    fn default() -> Self {
        Self { center: [0.0; 3], half: [0.0; 3], radius: 0.0 }
    }
}

impl BoxShape {
    #[inline]
    fn sdf(&self, p: [f32; 3]) -> f32 {
        box_sdf(sub(p, self.center), self.half)
    }
}

/// Точка внутри тела: положение в системе тела (от центроида) и глубина под
/// собственной поверхностью, метры.
#[derive(Clone, Copy, Debug)]
pub struct ProbePoint {
    pub local: [f32; 3],
    pub depth: f32,
}

/// Пробные точки тела и всё, что CCD про него знает.
#[derive(Clone, Debug)]
pub struct ProbeSet {
    pub pts: Vec<ProbePoint>,
    /// Габарит тела (местная система тела).
    pub bb: BoxShape,
    /// Полуразмер БЛОКА ПОЛЯ (метры) — по нему считается порог и обрезка.
    pub half: [f32; 3],
    /// Узлов внутри поля (v < 0). Ноль — поле тела не описывает.
    pub neg: u32,
    /// Поле тоньше трёх шагов сетки: контакт по нему недостоверен.
    pub thin: bool,
}

impl ProbeSet {
    /// Полуразмер габарита, по которому включается continuous (метры).
    pub fn gate_radius(&self) -> f32 {
        self.half[0].max(self.half[1]).max(self.half[2])
    }
}

/// Габарит по локальным тетам: центр, полуразмеры и описанная сфера.
pub fn box_of_tets(local: &[[[f32; 3]; 4]]) -> BoxShape {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for t in local {
        for p in t {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    if lo[0] > hi[0] {
        return BoxShape::default();
    }
    let center = [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ];
    let half = [
        0.5 * (hi[0] - lo[0]),
        0.5 * (hi[1] - lo[1]),
        0.5 * (hi[2] - lo[2]),
    ];
    BoxShape { center, half, radius: norm(half) }
}

/// Собрать пробные точки и метрики поля.
///
/// `body_box` — габарит по тетам (None: считаем по блоку поля). Точки:
///  * узлы поля у поверхности (глубина = расстояние до своей поверхности);
///  * центр габарита с глубиной вписанного радиуса;
///  * оси габарита (6 точек, глубина 0) — когда узлов мало.
pub fn probe_set(
    f: &NodeField,
    shift: [f32; 3],
    body_box: Option<BoxShape>,
    want: usize,
) -> ProbeSet {
    let (nx, ny, nz) = (f.dims[0], f.dims[1], f.dims[2]);
    let half = [
        (nx as f32 - 1.0) * f.step * 0.5,
        (ny as f32 - 1.0) * f.step * 0.5,
        (nz as f32 - 1.0) * f.step * 0.5,
    ];
    let boxshape = body_box.unwrap_or(BoxShape {
        center: [
            shift[0] + (nx as f32 - 1.0) * f.step * 0.5,
            shift[1] + (ny as f32 - 1.0) * f.step * 0.5,
            shift[2] + (nz as f32 - 1.0) * f.step * 0.5,
        ],
        half,
        radius: norm(half),
    });

    /* Прореживаем ОБХОД, а не список: поле крупного тела — 60k узлов, и полный
     * проход по нему на каждом новом теле (раскол рождает десятки тел сразу)
     * стоил больше, чем весь остальной CCD. */
    let mut stride = 1usize;
    while (nx.div_ceil(stride)) * (ny.div_ceil(stride)) * (nz.div_ceil(stride)) > PROBE_SCAN_NODES {
        stride += 1;
    }

    let mut neg = 0u32;
    let mut nbox = ([i32::MAX; 3], [i32::MIN; 3]);
    let mut all: Vec<ProbePoint> = Vec::new();
    let band = 2.0f32; /* |v| <= 2 шага — только слой у поверхности */
    for z in (0..nz).step_by(stride) {
        for y in (0..ny).step_by(stride) {
            for x in (0..nx).step_by(stride) {
                let v = f.val[(z * ny + y) * nx + x];
                if !(v < 0.0) {
                    continue;
                }
                neg += 1;
                for (k, c) in [x as i32, y as i32, z as i32].iter().enumerate() {
                    nbox.0[k] = nbox.0[k].min(*c);
                    nbox.1[k] = nbox.1[k].max(*c);
                }
                if v > band {
                    continue; /* глубоко внутри: касание даст далеко не первым */
                }
                let n = [
                    (f.lo[0] + x as i32) as f32,
                    (f.lo[1] + y as i32) as f32,
                    (f.lo[2] + z as i32) as f32,
                ];
                all.push(ProbePoint {
                    local: [
                        shift[0] + n[0] * f.step,
                        shift[1] + n[1] * f.step,
                        shift[2] + n[2] * f.step,
                    ],
                    depth: -v * f.step,
                });
            }
        }
    }
    let neg_total = /* оценка полного числа узлов внутри: обход был прорежен */
        if stride == 1 { neg } else { neg * (stride * stride * stride) as u32 };

    /* Толщина поля в узлах: самый тонкий размер ЗАНЯТОЙ области. Один-два узла
     * значат, что поле — слой, а не объём, и расстояние в нём геометрию не
     * описывает. */
    let thickness_vox = if nbox.0[0] > nbox.1[0] {
        0
    } else {
        (0..3)
            .map(|k| (nbox.1[k] - nbox.0[k]) * stride as i32 + 1)
            .min()
            .unwrap_or(0)
    };
    let thin = thickness_vox < 3;

    /* Узлы у поверхности — первыми: точка глубиной 0.5 шага даёт момент
     * касания с ошибкой не больше этой глубины (она вычитается в условии).
     * Прореживаем ПЕРВЫЕ по глубине, а не весь список: шаг по всему списку
     * дотягивался до центра тела и давал ложное «уже пересеклись». */
    all.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));
    let want = want.max(1);
    let cap = (want * 4).min(all.len());
    let pts: Vec<ProbePoint> = if cap <= want {
        all.truncate(cap);
        all
    } else {
        let stride = (cap / want).max(1);
        all[..cap].iter().step_by(stride).take(want).copied().collect()
    };

    let mut pts = pts;
    /* ЦЕНТР ГАБАРИТА с глубиной вписанного радиуса: ловит случай, когда
     * партнёр ЦЕЛИКОМ внутри пробного тела — там все поверхностные точки
     * снаружи партнёра, и условие «поверхность дошла до поверхности» молчит
     * (шар на 240 м/с проходил сквозь мелкий осколок ровно так). */
    let c_local = boxshape.center;
    let c_field = [
        (c_local[0] - shift[0]) / f.step,
        (c_local[1] - shift[1]) / f.step,
        (c_local[2] - shift[2]) / f.step,
    ];
    let d_center = -f.sample_m(c_field).min(0.0);
    pts.push(ProbePoint { local: c_local, depth: d_center });
    /* Оси габарита: у мелкого поля узлов единицы, и все они в одном слое —
     * без этих шести точек пробного покрытия по форме просто нет. */
    if neg_total < 32 {
        for k in 0..3 {
            for s in [-1.0f32, 1.0] {
                let mut p = c_local;
                p[k] += s * boxshape.half[k];
                pts.push(ProbePoint { local: p, depth: 0.0 });
            }
        }
    }

    ProbeSet { pts, bb: boxshape, half, neg: neg_total, thin }
}

/// Пробные точки тела: узлы поля внутри, ближайшие к поверхности — первыми.
pub fn probe_points(f: &NodeField, shift: [f32; 3], want: usize) -> Vec<ProbePoint> {
    probe_set(f, shift, None, want).pts
}

/// Включать ли continuous для пары: относительное смещение за шаг больше
/// размера меньшего тела.
pub fn needs_continuous(rel_speed: f32, dt: f32, radius: f32) -> bool {
    rel_speed * dt > 2.0 * radius
}

/// Тело глазами CCD: запечённое поле, габарит, поза, скорость, инерция.
pub struct BodyView<'a> {
    pub field: &'a NodeField,
    /// Центр поля относительно центроида в системе тела.
    pub shift: [f32; 3],
    pub pos: [f32; 3],
    pub quat: [f32; 4],
    pub vel: [f32; 3],
    pub ang: [f32; 3],
    pub inv_mass: f32,
    pub inv_inertia: f32,
    /// Габарит тела в системе тела (осевая коробка по тетам).
    pub shape: BoxShape,
    /// Полуразмер габарита (метры) — порог «смещается больше своего размера».
    pub radius: f32,
    /// Поле тоньше трёх шагов — контакт считаем по габаритной коробке.
    pub thin: bool,
    pub asleep: bool,
}

impl<'a> BodyView<'a> {
    #[inline]
    fn to_field(&self, p: [f32; 3]) -> [f32; 3] {
        [
            (p[0] - self.shift[0]) / self.field.step,
            (p[1] - self.shift[1]) / self.field.step,
            (p[2] - self.shift[2]) / self.field.step,
        ]
    }

    /// Точка (в узлах поля) внутри блока? Вне блока выборка зажата к краю и
    /// не описывает расстояние — там нельзя ни нормаль, ни проекцию считать.
    #[inline]
    fn inside_block(&self, n: [f32; 3]) -> bool {
        let (lo, dims) = (self.field.lo, self.field.dims);
        (0..3).all(|k| {
            let hi = (lo[k] + dims[k] as i32 - 1) as f32;
            n[k] >= lo[k] as f32 && n[k] <= hi
        })
    }

    /// Нормаль габаритной коробки в точке (мир) — запасной путь, когда поле в
    /// этой точке не определено.
    fn box_normal(&self, pos: [f32; 3], w: [f32; 3]) -> [f32; 3] {
        let q = sub(quat_inv_rot(self.quat, sub(w, pos)), self.shape.center);
        let h = self.shape.half;
        /* Ближайшая грань — та, у которой |q| - h максимально: снаружи это
         * самая «пробитая» ось, внутри — самая близкая к границе. Знак
         * обратный (h - |q|) выбирал САМУЮ ДАЛЁКУЮ грань: касание двух
         * осколков бок о бок получало нормаль ВВЕРХ и импульс уходил в пустоту
         * (замер: Δv цели 0.06 м/с вместо 60). */
        let mut best = 0usize;
        let mut best_d = f32::NEG_INFINITY;
        for k in 0..3 {
            let d = q[k].abs() - h[k];
            if d > best_d {
                best_d = d;
                best = k;
            }
        }
        let mut g = [0.0f32; 3];
        g[best] = if q[best] >= 0.0 { 1.0 } else { -1.0 };
        quat_rot(self.quat, g)
    }

    /// Значение поля (метры) в мировой точке при заданной позе тела.
    fn value_at(&self, pos: [f32; 3], w: [f32; 3]) -> f32 {
        let p = quat_inv_rot(self.quat, sub(w, pos));
        self.field.sample_m(self.to_field(p))
    }

    /// Расстояние до ПОВЕРХНОСТИ партнёра (метры, минус внутри): поле, а для
    /// тонкого поля — габаритная коробка (поле его геометрии не описывает).
    ///
    /// ВНЕ блока поля выборка зажата к краю и врёт: точка в трёх метрах от
    /// осколка получает «+0.009 м до поверхности». Из-за этого центр пробного
    /// шара (глубина 0.13 м) оказывался «уже в контакте», и CCD молча
    /// выключался — замер: 240 м/с, 12 м пути, касания не найдено. Вне блока
    /// берём расстояние до САМОГО БЛОКА: тело внутри блока, значит это честная
    /// нижняя оценка расстояния до его поверхности.
    fn surf(&self, pos: [f32; 3], w: [f32; 3]) -> f32 {
        if self.thin {
            let p = quat_inv_rot(self.quat, sub(w, pos));
            return self.shape.sdf(p);
        }
        let p = quat_inv_rot(self.quat, sub(w, pos));
        let n = self.to_field(p);
        let (lo, dims) = (self.field.lo, self.field.dims);
        let mut dblock = [0.0f32; 3];
        let mut outside = false;
        for k in 0..3 {
            let hi = lo[k] + dims[k] as i32 - 1;
            if n[k] < lo[k] as f32 {
                dblock[k] = lo[k] as f32 - n[k];
                outside = true;
            } else if n[k] > hi as f32 {
                dblock[k] = n[k] - hi as f32;
                outside = true;
            }
        }
        if outside {
            let d = (dblock[0] * dblock[0] + dblock[1] * dblock[1] + dblock[2] * dblock[2]).sqrt();
            return d * self.field.step;
        }
        self.field.sample_m(n)
    }

    /// Единичная нормаль поверхности (мир) в мировой точке.
    ///
    /// Вне блока поля градиент вырождается в направление вдоль края блока
    /// (все выборки зажаты) — так касание шара в грань получало нормаль ВНИЗ
    /// и отбрасывало тело назад. Поэтому вне блока берём нормаль габарита.
    fn normal_at(&self, pos: [f32; 3], w: [f32; 3]) -> [f32; 3] {
        if self.thin {
            return self.box_normal(pos, w);
        }
        let n = self.to_field(quat_inv_rot(self.quat, sub(w, pos)));
        if !self.inside_block(n) {
            return self.box_normal(pos, w);
        }
        let g = self.field.gradient(n);
        if norm(g) < 0.5 {
            return self.box_normal(pos, w);
        }
        quat_rot(self.quat, g)
    }

    /// Проекция мировой точки на поверхность партнёра (4 шага Ньютона по
    /// полю; для коробки — точный clamp).
    fn project(&self, pos: [f32; 3], w: [f32; 3]) -> [f32; 3] {
        let outside = self.thin
            || !self.inside_block(self.to_field(quat_inv_rot(self.quat, sub(w, pos))));
        if outside {
            let p = quat_inv_rot(self.quat, sub(w, pos));
            let q = sub(p, self.shape.center);
            let h = self.shape.half;
            let c = [
                q[0].clamp(-h[0], h[0]),
                q[1].clamp(-h[1], h[1]),
                q[2].clamp(-h[2], h[2]),
            ];
            return add(pos, quat_rot(self.quat, add(self.shape.center, c)));
        }
        let g = self.normal_at(pos, w);
        let mut q = w;
        for _ in 0..4 {
            let v = self.value_at(pos, q);
            q = sub(q, scale(g, v));
        }
        q
    }
}

/// Первое касание вдоль шага.
#[derive(Clone, Copy, Debug)]
pub struct SweepHit {
    /// Доля шага, где поверхности впервые соприкоснулись.
    pub t: f32,
    /// Точка касания (мир).
    pub p: [f32; 3],
    /// Нормаль от пробного тела к партнёру.
    pub n: [f32; 3],
    /// Заглубление на найденной позе, метры (может быть чуть больше нуля).
    pub depth: f32,
    /// Сколько итераций бисекции реально понадобилось до допуска.
    pub iters: u32,
}

/// Цена одного поиска: сколько проб и итераций ушло и сколько это заняло.
/// Заполняется только при включённой диагностике (VF_PERF=1), иначе — нули.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cost {
    pub scan_probes: u32,
    pub bisect_iters: u32,
    pub scan_ns: u64,
    pub bisect_ns: u64,
}

/// Отрезок `p0 -> p0 + d` против коробки `h`, раздутой на `r`.
///
/// Возвращает `(t_enter, t_exit)` в долях пути (без clamp к [0,1]: вызывающий
/// сам решает, что делать с уже начавшимся перекрытием) и номер оси входа.
fn seg_box(
    p0: [f32; 3],
    d: [f32; 3],
    h: [f32; 3],
    r: f32,
) -> Option<(f32, f32, usize)> {
    let (mut t0, mut t1) = (f32::NEG_INFINITY, f32::INFINITY);
    let mut axis = 0usize;
    for k in 0..3 {
        let hk = h[k] + r;
        if d[k].abs() < 1e-12 {
            if p0[k] < -hk || p0[k] > hk {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d[k];
        let mut ta = (-hk - p0[k]) * inv;
        let mut tb = (hk - p0[k]) * inv;
        if ta > tb {
            std::mem::swap(&mut ta, &mut tb);
        }
        if ta > t0 {
            t0 = ta;
            axis = k;
        }
        t1 = t1.min(tb);
        if t0 > t1 {
            return None;
        }
    }
    if t1 < t0 {
        return None;
    }
    Some((t0, t1, axis))
}

/// Аналитический контакт пробного тела (габаритная сфера) с коробкой партнёра.
///
/// Вход в раздутую коробку — это уже касание габаритов; точка и нормаль
/// берутся с грани коробки, в которую упёрся центр.
fn sweep_sphere_box(
    probe: &BodyView,
    other: &BodyView,
    dt: f32,
) -> Option<(f32, [f32; 3], [f32; 3])> {
    let r = probe.shape.radius.max(1e-4);
    let p0 = sub(
        quat_inv_rot(other.quat, sub(probe.pos, other.pos)),
        other.shape.center,
    );
    let d = quat_inv_rot(other.quat, scale(sub(probe.vel, other.vel), dt));
    let (t0, _t1, axis) = seg_box(p0, d, other.shape.half, r)?;
    if t0 > 1.0 {
        return None;
    }
    let t = t0.max(0.0);
    /* Точка на грани коробки: центр пробного тела на t, прижатый к коробке. */
    let c = add(p0, scale(d, t));
    let h = other.shape.half;
    let mut on = [
        c[0].clamp(-h[0], h[0]),
        c[1].clamp(-h[1], h[1]),
        c[2].clamp(-h[2], h[2]),
    ];
    /* Прижать к грани по оси входа: clamp может оставить точку внутри, если
     * вход был через ребро. */
    let sign = if p0[axis] + d[axis] * t >= 0.0 { 1.0 } else { -1.0 };
    on[axis] = sign * h[axis];
    let p = add(other.pos, quat_rot(other.quat, add(on, other.shape.center)));
    let mut g = [0.0f32; 3];
    g[axis] = sign;
    let g = quat_rot(other.quat, g);
    Some((t, p, scale(g, -1.0)))
}

/// Ищет первую точку касания поверхности `probe` с полем `other` на отрезке
/// перемещения за `dt`. Возвращает `None`, если касания по пути нет.
pub fn first_contact(
    pts: &[ProbePoint],
    probe: &BodyView,
    other: &BodyView,
    dt: f32,
    max_iters: u32,
) -> Option<SweepHit> {
    first_contact_cost(pts, probe, other, dt, max_iters, None)
}

/// То же, но с замером стоимости (диагностика VF_PERF=1).
pub fn first_contact_cost(
    pts: &[ProbePoint],
    probe: &BodyView,
    other: &BodyView,
    dt: f32,
    max_iters: u32,
    cost: Option<&mut Cost>,
) -> Option<SweepHit> {
    if pts.is_empty() || dt <= 0.0 {
        return None;
    }
    let mut cost = cost;
    let probe_pos = |s: f32| add(probe.pos, scale(probe.vel, dt * s));
    let other_pos = |s: f32| add(other.pos, scale(other.vel, dt * s));
    /* Глубочайший «заход» поверхности за шаг s: min по точкам (f_other - d).
     * < 0 — поверхности пересеклись. */
    let deepest = |s: f32| -> Option<(f32, ProbePoint, [f32; 3])> {
        let pp = probe_pos(s);
        let op = other_pos(s);
        let mut best: Option<(f32, ProbePoint, [f32; 3])> = None;
        for pt in pts.iter() {
            let w = add(pp, quat_rot(probe.quat, pt.local));
            let over = other.surf(op, w) - pt.depth;
            if best.map(|(b, _, _)| over < b).unwrap_or(true) {
                best = Some((over, *pt, w));
            }
        }
        best
    };

    let (over0, _, _) = deepest(0.0)?;
    if over0 < 0.0 {
        return None; /* уже в контакте — этим занимается обычный проход */
    }

    /* ТОНКОЕ ПОЛЕ: у скола остаётся слой узлов, ноль поля лежит внутри одной
     * ячейки, и по пути в 12 м его можно не пересечь ни одной пробой. Коробка
     * по тетам — это и есть коллайдер GPU, поэтому касание считаем по ней
     * аналитически: промахнуться нечем. */
    if other.thin {
        let (t, p, n) = sweep_sphere_box(probe, other, dt)?;
        if t <= 0.0 {
            return None;
        }
        let (over, _, _) = deepest(t)?;
        return Some(SweepHit { t, p, n, depth: (-over).max(0.0), iters: 0 });
    }

    /* ИНТЕРВАЛ ГАБАРИТА. Вне блока поля |f| зажато к границе блока, поэтому
     * трейсинг по |f| вне его не двигается; аналитика по коробке даёт точные
     * границы, где тела вообще могут соприкоснуться. */
    let p0 = sub(quat_inv_rot(other.quat, sub(probe.pos, other.pos)), other.shape.center);
    let d = quat_inv_rot(other.quat, scale(sub(probe.vel, other.vel), dt));
    let Some((te, tx, _)) = seg_box(p0, d, other.shape.half, probe.shape.radius.max(1e-4)) else {
        return None; /* габариты за шаг не встретились */
    };
    let lo0 = te.max(0.0);
    let hi0 = tx.min(1.0);
    if hi0 <= lo0 {
        return None;
    }

    let t_scan = cost.is_some().then(Instant::now);
    /* Первая проба, где поверхности пересеклись. Интервал короткий (коробка
     * партнёра + радиус пробного), поэтому 16 проб хватает, а внутри
     * пересечения функция отрицательна почти на всём интервале. */
    let mut bracket: Option<(f32, f32)> = None;
    let mut prev = lo0;
    for k in 1..=SCAN_STEPS {
        let s = lo0 + (hi0 - lo0) * (k as f32 / SCAN_STEPS as f32);
        let (g, _, _) = deepest(s)?;
        if g < 0.0 {
            bracket = Some((prev, s));
            break;
        }
        prev = s;
    }
    if let (Some(t), Some(c)) = (t_scan, cost.as_deref_mut()) {
        c.scan_ns += t.elapsed().as_nanos() as u64;
        c.scan_probes += SCAN_STEPS;
    }
    let Some((mut lo, mut hi)) = bracket else {
        if trace() {
            let (mut mn, mut at) = (f32::MAX, 0.0f32);
            for k in 0..=16 {
                let s = lo0 + (hi0 - lo0) * (k as f32 / 16.0);
                if let Some((g, _, _)) = deepest(s) { if g < mn { mn = g; at = s; } }
            }
            eprintln!("[ccd] в интервале {lo0:.4}..{hi0:.4} минимум {mn:.4} при {at:.4}");
        }
        return None; /* за шаг поверхности не встретились */
    };

    let t_bisect = cost.is_some().then(Instant::now);
    let (mut used, max_iters) = (0u32, max_iters.clamp(1, 32));
    while hi - lo > T_TOL && used < max_iters {
        let mid = 0.5 * (lo + hi);
        let (over, _, _) = deepest(mid)?;
        if over < 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
        used += 1;
    }
    if let (Some(t), Some(c)) = (t_bisect, cost.as_deref_mut()) {
        c.bisect_ns += t.elapsed().as_nanos() as u64;
        c.bisect_iters += used;
    }

    /* Контакт считаем на позе t=hi (первое найденное проникновение).
     *
     * Нормаль и точка — с поверхности ПАРТНЁРА. Так удар шара в грань даёт
     * нормаль ГРАНИ и плечо вдоль неё — без паразитного вращения, которое
     * получалось, если брать случайную внутреннюю точку: у неё нормаль
     * «своего» поля наклонена, и шар начинал крутиться от прямого удара. */
    let (over, _, _) = deepest(hi)?;
    let op = other_pos(hi);
    /* ТОЧКА КАСАНИЯ — центр пробного тела, спроецированный на поверхность
     * партнёра: плечо получается вдоль удара, без паразитного вращения от
     * случайной внутренней точки (проверено: так шар в грань не крутится). */
    let q = other.project(op, probe_pos(hi));
    /* НОРМАЛЬ — градиент поля В ТОЧКЕ КАСАНИЯ. Раньше её брали в «глубочайшей»
     * пробной точке: у мелкого осколка это внутренняя точка, где ближайшая
     * грань совсем не та, и удар двух осколков бок о бок получал нормаль ВВЕРХ
     * (Δv цели 0.06 м/с вместо 60 — осколки проходили друг сквозь друга). */
    let mut g = other.normal_at(op, q);
    if norm(g) < 0.5 {
        let dv = sub(q, op);
        g = if norm(dv) > 1e-6 { scale(dv, 1.0 / norm(dv)) } else { [0.0, 1.0, 0.0] };
    }
    /* Страховка: из градиента и грани габарита берём ту нормаль, что даёт
     * большее СБЛИЖЕНИЕ — удар обязан гасить скорость, а не раскручивать. */
    let g_box = other.box_normal(op, q);
    let vrel = sub(probe.vel, other.vel);
    if dot(g_box, vrel) < dot(g, vrel) {
        g = g_box;
    }
    let n = scale(g, -1.0);
    Some(SweepHit { t: hi, p: q, n, depth: (-over).max(0.0), iters: used })
}

/// Касание телом плоскости за шаг.
///
/// `n` — направление В РАЗРЕШЁННУЮ полуплоскость (куда тело выталкивают),
/// `d` — граница: разрешено `n·p >= d`. Пол — `n=+y, d=0`; стена `x=lo.x` —
/// `n=+x, d=lo.x`; `x=hi.x` — `n=-x, d=-hi.x`.
///
/// Пол и стены — статические плоскости, полей у них нет, и CCD их раньше не
/// покрывал вовсе: быстрое тело проскакивало пол между подшагами. Возвращает
/// долю шага первого касания и точку касания на теле.
pub fn plane_contact(body: &BodyView, n: [f32; 3], d: f32, dt: f32) -> Option<(f32, [f32; 3])> {
    if dt <= 0.0 {
        return None;
    }
    let vn = dot(body.vel, n);
    if vn >= -1e-9 {
        return None; /* летит вдоль или от плоскости */
    }
    /* Опора тела ПРОТИВ направления выталкивания: минимум по вершинам
     * габарита. По ней и меряется зазор до плоскости. */
    let h = body.shape.half;
    let c = body.shape.center;
    let mut gmin = f32::INFINITY;
    for dist in 0..8 {
        let p = [
            c[0] + if dist & 1 != 0 { h[0] } else { -h[0] },
            c[1] + if dist & 2 != 0 { h[1] } else { -h[1] },
            c[2] + if dist & 4 != 0 { h[2] } else { -h[2] },
        ];
        gmin = gmin.min(dot(quat_rot(body.quat, p), n));
    }
    let gap = dot(body.pos, n) + gmin - d;
    if gap <= 0.0 {
        return None; /* уже за плоскостью — это обычный контакт */
    }
    let travel = vn * dt; /* отрицательное: движется к плоскости */
    if gap + travel > 0.0 {
        return None; /* за шаг до плоскости не дошло */
    }
    let t = (gap / -travel).clamp(0.0, 1.0);
    let p = add(body.pos, scale(body.vel, dt * t));
    Some((t, p))
}

/// Импульс о статическую плоскость: гасит нормальную скорость в точке
/// касания (партнёр бесконечно массивен). Возвращает `(dv, dw)`.
pub fn plane_impulse(body: &BodyView, m: [f32; 3], p: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let r = sub(p, body.pos);
    let v = add(body.vel, cross(body.ang, r));
    let vn = dot(v, m);
    if vn >= 0.0 || body.asleep {
        return ([0.0; 3], [0.0; 3]);
    }
    let rn = cross(r, m);
    let wn = body.inv_mass + dot(rn, rn) * body.inv_inertia;
    if wn <= 0.0 {
        return ([0.0; 3], [0.0; 3]);
    }
    let imp = scale(m, -vn / wn);
    (scale(imp, body.inv_mass), scale(cross(r, imp), body.inv_inertia))
}

/// Импульс одного контакта — та же математика, что в `solve.comp`
/// (`dl = max(push - vn, 0)/wn`, деления на число контактов нет: контакт один).
#[derive(Clone, Copy, Debug, Default)]
pub struct Impulse {
    pub dv_a: [f32; 3],
    pub dw_a: [f32; 3],
    pub dv_b: [f32; 3],
    pub dw_b: [f32; 3],
    /// Относительная нормальная скорость до импульса (м/с, минус — сближение).
    pub vn: f32,
}

pub fn contact_impulse(a: &BodyView, b: &BodyView, hit: &SweepHit) -> Impulse {
    let ra = sub(hit.p, a.pos);
    let rb = sub(hit.p, b.pos);
    let va = add(a.vel, cross(a.ang, ra));
    let vb = add(b.vel, cross(b.ang, rb));
    let vn = dot(sub(vb, va), hit.n);
    if vn >= 0.0 || a.asleep && b.asleep {
        return Impulse { vn, ..Default::default() };
    }
    let ran = cross(ra, hit.n);
    let rbn = cross(rb, hit.n);
    let wn = a.inv_mass + b.inv_mass + dot(ran, ran) * a.inv_inertia + dot(rbn, rbn) * b.inv_inertia;
    if wn <= 0.0 {
        return Impulse { vn, ..Default::default() };
    }
    /* push = 0: к моменту касания сближение уже недопустимо, поэтому
     * нормальная скорость гасится целиком (жёсткий удар, без отскока). */
    let dl = -vn / wn;
    let p = scale(hit.n, dl);
    Impulse {
        dv_a: scale(p, -a.inv_mass),
        dw_a: scale(cross(ra, p), -a.inv_inertia),
        dv_b: scale(p, b.inv_mass),
        dw_b: scale(cross(rb, p), b.inv_inertia),
        vn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tet::TetBody;

    fn view<'a>(b: &'a TetBody, pos: [f32; 3], vel: [f32; 3], radius: f32) -> BodyView<'a> {
        BodyView {
            field: b.sdf(),
            shift: [0.0; 3],
            pos,
            quat: [1.0, 0.0, 0.0, 0.0],
            vel,
            ang: [0.0; 3],
            inv_mass: 1.0 / 8.0,
            inv_inertia: 1.0 / 0.06,
            shape: BoxShape {
                center: [0.0; 3],
                half: [radius; 3],
                radius: radius * 1.732,
            },
            radius,
            thin: false,
            asleep: false,
        }
    }

    /// Бисекция находит момент касания шара с плитой: аналитика для шара r=0.14,
    /// летящего в плиту 1.12 м с фронтом в x=-0.61 на 240 м/с за 50 мс.
    #[test]
    fn bisection_finds_first_touch() {
        let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
        let ball = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let pts = probe_points(ball.sdf(), [0.0; 3], PROBE_POINTS);
        assert!(pts.len() > 32, "пробных точек мало: {}", pts.len());

        let speed = 240.0f32;
        let dt = 0.05f32;
        let from = [-3.0, 0.0, 0.0];
        let a = view(&ball, from, [speed, 0.0, 0.0], 0.14);
        let b = view(&plate, [0.0, 0.0, 0.0], [0.0; 3], 0.56);
        assert!(needs_continuous(speed, dt, a.radius), "240 м/с обязан включить CCD");

        let hit = first_contact(&pts, &a, &b, dt, 12).expect("касание обязано найтись");
        /* аналитика: (3 - 0.56 - 0.14)/12 = 0.1917 */
        let want = (3.0 - 0.56 - 0.14) / 12.0;
        println!(
            "t={:.4} (аналитика {want:.4}) · глубина {:.4} м · итераций {} · n={:?}",
            hit.t, hit.depth, hit.iters, hit.n
        );
        assert!((hit.t - want).abs() < 0.02, "t={} вместо {want}", hit.t);
        assert!(hit.depth < 0.05, "глубина на позе касания {:.4} м", hit.depth);
        assert!(hit.n[0] > 0.9, "нормаль должна смотреть по движению: {:?}", hit.n);
    }

    /// Промах (шар летит выше плиты) касания не даёт — CCD не срабатывает.
    #[test]
    fn bisection_reports_no_hit_on_miss() {
        let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
        let ball = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let pts = probe_points(ball.sdf(), [0.0; 3], PROBE_POINTS);
        let a = view(&ball, [-3.0, 2.0, 0.0], [240.0, 0.0, 0.0], 0.14);
        let b = view(&plate, [0.0, 0.0, 0.0], [0.0; 3], 0.56);
        assert!(first_contact(&pts, &a, &b, 0.05, 12).is_none());
    }

    /// Порог: медленное тело continuous не включает (поведение как раньше).
    #[test]
    fn continuous_gate_uses_own_size() {
        assert!(!needs_continuous(5.0, 0.05, 0.14), "5 м/с * 50 мс = 0.25 < 0.28");
        assert!(needs_continuous(240.0, 0.05, 0.14));
        /* уже в контакте — CCD не нужен, обычный проход справится */
        let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
        let ball = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let pts = probe_points(ball.sdf(), [0.0; 3], PROBE_POINTS);
        let a = view(&ball, [0.0, 0.0, 0.0], [240.0, 0.0, 0.0], 0.14);
        let b = view(&plate, [0.0, 0.0, 0.0], [0.0; 3], 0.56);
        assert!(first_contact(&pts, &a, &b, 0.05, 12).is_none());
    }

    /// ТОНКИЙ осколок: поле в один-два узла (так выглядит скол после
    /// нескольких дроблений). Ноль поля лежит внутри одной ячейки, и по пути в
    /// 12 м его можно не пересечь ни одной пробой — касание обязано найтись по
    /// габариту, а не пропасть.
    #[test]
    fn thin_shard_is_hit_by_geometry() {
        use crate::field::{intersect_plane, NodeField};
        let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
        let f0 = plate.sdf();
        /* Слэб толщиной 2 узла: держим x <= c+1 и x >= c-0. */
        let c = f0.lo[0] + 6;
        let mut f: NodeField = intersect_plane(f0, [1.0, 0.0, 0.0], c as f32 + 1.5);
        f = intersect_plane(&f, [-1.0, 0.0, 0.0], -(c as f32 - 0.5));

        /* Габарит — по узлам внутри (как `box_of_tets` у настоящего осколка). */
        let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
        for z in 0..f.dims[2] {
            for y in 0..f.dims[1] {
                for x in 0..f.dims[0] {
                    if f.val[(z * f.dims[1] + y) * f.dims[0] + x] < 0.0 {
                        let p = [f.lo[0] + x as i32, f.lo[1] + y as i32, f.lo[2] + z as i32];
                        for k in 0..3 {
                            lo[k] = lo[k].min(p[k]);
                            hi[k] = hi[k].max(p[k]);
                        }
                    }
                }
            }
        }
        let bb = BoxShape {
            center: [
                0.5 * (lo[0] + hi[0]) as f32 * f.step,
                0.5 * (lo[1] + hi[1]) as f32 * f.step,
                0.5 * (lo[2] + hi[2]) as f32 * f.step,
            ],
            half: [
                0.5 * (hi[0] - lo[0]) as f32 * f.step,
                0.5 * (hi[1] - lo[1]) as f32 * f.step,
                0.5 * (hi[2] - lo[2]) as f32 * f.step,
            ],
            radius: 0.0,
        };
        let mut bb = bb;
        bb.radius = norm(bb.half);
        let ps = probe_set(&f, [0.0; 3], Some(bb), PROBE_POINTS);
        println!(
            "слэб: {}×{}×{} узлов, внутри {}, тонкий {}, проб {}",
            hi[0] - lo[0] + 1,
            hi[1] - lo[1] + 1,
            hi[2] - lo[2] + 1,
            ps.neg,
            ps.thin,
            ps.pts.len()
        );
        assert!(ps.thin, "поле толщиной 2 узла обязано считаться тонким");
        assert!(ps.neg > 0, "внутри поля нет узлов");

        let ball = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let bp = probe_set(ball.sdf(), [0.0; 3], None, PROBE_POINTS);
        let speed = 240.0f32;
        let dt = 0.05f32;
        /* Шар летит по x через центр слэба: путь 12 м, толщина слэба 7 см. */
        let from = [bb.center[0] - 3.0, bb.center[1], bb.center[2]];
        let a = view(&ball, from, [speed, 0.0, 0.0], 0.14);
        let mut b = BodyView {
            field: &f,
            shift: [0.0; 3],
            pos: [0.0; 3],
            quat: [1.0, 0.0, 0.0, 0.0],
            vel: [0.0; 3],
            ang: [0.0; 3],
            inv_mass: 1.0 / 590.0,
            inv_inertia: 1.0 / 60.0,
            shape: bb,
            radius: bb.radius,
            thin: true,
            asleep: false,
        };
        b.thin = true;
        let hit = first_contact(&bp.pts, &a, &b, dt, 12).expect("тонкий осколок обязан поймать шар");
        println!(
            "тонкий: t={:.4} · точка {:.3},{:.3},{:.3} · n={:?} · глубина {:.4}",
            hit.t, hit.p[0], hit.p[1], hit.p[2], hit.n, hit.depth
        );
        /* Касание не позже, чем шар прошёл 3 м до центра слэба. */
        assert!(hit.t * speed * dt <= 3.0, "t={} — точка касания за осколком", hit.t);
        assert!(hit.n[0] > 0.7, "нормаль не по движению: {:?}", hit.n);
    }

    /// Плоскость пола: тело, летящее вниз, касается её за шаг; летящее вверх —
    /// нет.
    #[test]
    fn plane_catches_falling_body() {
        let ball = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let a = view(&ball, [0.0, 2.0, 0.0], [0.0, -240.0, 0.0], 0.14);
        let (t, _) = plane_contact(&a, [0.0, 1.0, 0.0], 0.0, 0.05).expect("пол обязан поймать");
        /* 2 м над полом, радиус 0.14: (2-0.14)/12 */
        assert!((t - (2.0 - 0.14) / 12.0).abs() < 0.02, "t={t}");
        let up = view(&ball, [0.0, 2.0, 0.0], [0.0, 240.0, 0.0], 0.14);
        assert!(plane_contact(&up, [0.0, 1.0, 0.0], 0.0, 0.05).is_none());
    }

    /// Импульс гасит нормальную скорость и сохраняет импульс пары.
    #[test]
    fn impulse_kills_normal_velocity_and_conserves_momentum() {
        let plate = TetBody::cube([0.0, 0.0, 0.0], 8, 1);
        let ball = TetBody::ball([0.0, 0.0, 0.0], 0.14, 0);
        let a = view(&ball, [-0.7, 0.0, 0.0], [240.0, 0.0, 0.0], 0.14);
        let mut b = view(&plate, [0.0, 0.0, 0.0], [0.0; 3], 0.56);
        b.inv_mass = 1.0 / 590.0;
        b.inv_inertia = 1.0 / 60.0;
        let hit = SweepHit {
            t: 0.2,
            p: [-0.61, 0.0, 0.0],
            n: [1.0, 0.0, 0.0],
            depth: 0.001,
            iters: 12,
        };
        let im = contact_impulse(&a, &b, &hit);
        let va = add(a.vel, im.dv_a);
        let vb = add(b.vel, im.dv_b);
        println!("vn={:.2} · снаряд {:.2} м/с · плита {:.3} м/с", im.vn, va[0], vb[0]);
        /* после удара тела движутся вместе: относительная нормальная скорость 0,
         * а не «снаряд остановился в лабораторной системе» (импульс общий) */
        assert!((vb[0] - va[0]).abs() < 0.5, "сближение не погашено: {:.2}", vb[0] - va[0]);
        assert!(va[0] < 10.0, "снаряд должен потерять почти всю скорость: {:.2}", va[0]);
        assert!(vb[0] > 0.1, "плита обязана получить импульс: {:.3}", vb[0]);
        /* импульс: суммарный до и после совпадает (8 кг * 240 м/с) */
        let mom = 8.0 * va[0] + 590.0 * vb[0];
        assert!((mom - 8.0 * 240.0).abs() < 2.0, "импульс не сохранился: {mom:.3}");
    }
}
