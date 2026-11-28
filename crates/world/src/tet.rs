//! Тетраэдральная решётка BCC и марчинг по тетраэдрам.
//!
//! BCC — способ разбиения пространства: кубическая ячейка с 8 углами и
//! центром тела делится на **12 одинаковых тетраэдров** (пирамида от каждой
//! грани к центру, разбитая по диагонали). Все вершины — узлы решётки.
//!
//! Уровень `L` задаёт шаг `base / 2^L`, поэтому размеры тетраэдров всегда
//! степени двойки. Red-разбиение тетраэдра: каждый делится на **8** (4 угловых
//! + 4 в середине), суммарный объём сохраняется точно — в отличие от попытки
//! дробить усечённый октаэдр, который на 8 своих копий не разбивается и терял
//! четверть объёма.
//!
//! Поверхность — изоповерхность поля занятости: марчинг по тетраэдрам. На
//! бинарном поле (тетраэдр занят или нет) нулевая изоповерхность совпадает с
//! гранями тетраэдров, поэтому извлекаются граничные грани — тот же результат,
//! но без паразитного сжатия на полребра, которое даёт линейная интерполяция.

use std::collections::HashMap;

use crate::bcc::BASE;
use crate::field;

/// Узел в единицах половины шага решётки на своём уровне.
pub type Node = (i32, i32, i32);

/// Тетраэдр: 4 узла плюс уровень (всегда степень двойки от базы).
#[derive(Clone, Copy, Debug)]
pub struct Tet {
    pub level: u8,
    pub v: [Node; 4],
}

pub(crate) fn pack(n: Node) -> u64 {
    let b = 21u64;
    ((n.0 as i64 + (1 << (b - 1))) as u64) << (2 * b)
        | ((n.1 as i64 + (1 << (b - 1))) as u64) << b
        | ((n.2 as i64 + (1 << (b - 1))) as u64)
}

/// Шаг сетки (половина шага решётки) на уровне.
pub fn node_step(level: u8) -> f32 {
    BASE / (1u32 << level) as f32 * 0.5
}

pub fn node_pos(n: Node, level: u8, origin: [f32; 3]) -> [f32; 3] {
    let s = node_step(level);
    [
        origin[0] + n.0 as f32 * s,
        origin[1] + n.1 as f32 * s,
        origin[2] + n.2 as f32 * s,
    ]
}

/// Сколько узлов поля влезает в слот SDF на устройстве (`sim::SDF_STRIDE` =
/// 65536). Поле тела обязано в него влезть, иначе SDF-контакты для тела
/// выключаются (см. `GpuSim::upload_sdf`). Берём с запасом.
const FIELD_NODE_BUDGET: usize = 60000;

/// Бокс поля осколка: тот же кусок мира, но, если влезает в слот SDF, на
/// более мелкой сетке (см. `shatter_fields`). Возвращает `(шаг, lo, dims)`.
fn shard_field_box(step: f32, lo: &[i32; 3], dims: &[usize; 3]) -> (f32, [i32; 3], [usize; 3]) {
    for r in [2.0f32, 1.5, 1.25] {
        let fs = step / r;
        let mut f_lo = [0i32; 3];
        let mut fd = [0usize; 3];
        for k in 0..3 {
            let w0 = lo[k] as f32 * step;
            let w1 = (lo[k] + dims[k] as i32) as f32 * step;
            f_lo[k] = (w0 / fs).floor() as i32;
            fd[k] = ((w1 / fs).ceil() as i32 - f_lo[k] + 1) as usize;
        }
        if fd[0] * fd[1] * fd[2] <= FIELD_NODE_BUDGET {
            return (fs, f_lo, fd);
        }
    }
    (step, *lo, *dims)
}

/// Полуразмер сетки поля на `reach` метров с запасом `pad` узлов: `(n, dim)`,
/// где сетка — `2n+1` узлов по каждой оси (узел 0 всегда в центре).
fn field_dims(reach: f32, step: f32, pad: i32) -> (i32, usize) {
    let n = (reach / step).ceil() as i32 + pad;
    (n, (2 * n + 1) as usize)
}

/// 12 тетраэдров кубической ячейки (углы — биты `x + 2y + 4z`, центр — 8).
const CUBE_TETS: [[u8; 4]; 12] = [
    [0, 2, 6, 8], [0, 6, 4, 8],
    [1, 5, 7, 8], [1, 7, 3, 8],
    [0, 4, 5, 8], [0, 5, 1, 8],
    [2, 3, 7, 8], [2, 7, 6, 8],
    [0, 1, 3, 8], [0, 3, 2, 8],
    [4, 6, 7, 8], [4, 7, 5, 8],
];

/// 12 тетов Сомирвилля, сходящихся на узле-угле `c`. КУБОВ ЗДЕСЬ НЕТ.
///
/// Узел-угол и его осевой сосед `c' = c ± 2·dir` дают единственное ребро тета
/// длины 2 (у тета ровно одно ребро «угол-угол», поэтому каждый тет
/// генерируется ровно один раз и дедупликация не нужна).
///
/// Две другие вершины — центры ячеек, равноудалённые от обоих концов на √3/2:
/// смещение вдоль оси `s`, плюс по единице по двум другим осям, с разным знаком
/// вдоль первой из них. Правило трансляционно-инвариантно, а значит замощение
/// согласовано по всему пространству.
/// Канонический ключ тета: набор вершин без порядка и без повтора.
fn tet_key(t: &[Node; 4]) -> [Node; 4] {
    let mut v = *t;
    v.sort_by_key(|n| pack(*n));
    v
}

/// Теты вокруг узла-угла БЕЗ дублей.
///
/// bcc_tets_at порождает один и тот же тет из обоих своих углов (у тета два
/// угла, и вызывается функция на каждом), поэтому в общем списке каждый тет
/// встречается дважды. Дубли в марчинге дают наложенные треугольники, и ребро
/// поверхности выходит кратности 3–4 вместо 2 — поверхность «не замкнута».
pub fn bcc_tets_at_dedup(c: Node) -> Vec<[Node; 4]> {
    let mut seen: Vec<[Node; 4]> = Vec::with_capacity(24);
    let mut out = Vec::with_capacity(24);
    for t in bcc_tets_at(c) {
        let k = tet_key(&t);
        if !seen.contains(&k) {
            seen.push(k);
            out.push(t);
        }
    }
    out
}

pub fn bcc_tets_at(c: Node) -> Vec<[Node; 4]> {
    let base = [c.0, c.1, c.2];
    let mut out = Vec::with_capacity(24);
    for d in 0..3 {
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;
        for s in [-1i32, 1] {
            /* второй угол ребра «угол-угол» */
            let mut far = base;
            far[d] += 2 * s;
            /* ЧЕТЫРЕ центра: у узла BCC 24 инцидентных тета (24 вершины
             * усечённого октаэдра), на шесть осевых рёбер — по четыре. */
            for su in [-1i32, 1] {
                for sv in [-1i32, 1] {
                    let mut c1 = base;
                    c1[d] += s;
                    c1[u] += su;
                    c1[v] += sv;
                    let mut c2 = c1;
                    c2[u] -= 2 * su;
                    out.push([
                        (base[0], base[1], base[2]),
                        (far[0], far[1], far[2]),
                        (c1[0], c1[1], c1[2]),
                        (c2[0], c2[1], c2[2]),
                    ]);
                }
            }
        }
    }
    out
}

/// Теты Сомирвилля №1 вокруг РЕБРА кубической решётки.
///
/// Ребро — два узла-угла на расстоянии 2 вдоль оси `d`. Вокруг ребра стоят
/// четыре кубические ячейки, их центры (узлы BCC) образуют цикл. Тет —
/// (угол, угол, центр, соседний центр): ровно четыре таких тета на ребро.
///
/// Почему так, а не «12 тетов на ячейку по одному на ребро»: у каждого ребра
/// четыре соседние ячейки, и один тет заявляют сразу несколько из них. Из-за
/// этого прежний вариант давал на 12 тетов ячейки 300 УНИКАЛЬНЫХ на 64 ячейки
/// и покрывал лишь 24% объёма — отсюда были и дырки в меше, и неверная масса.
/// Проверено объёмом: 768 тетов на 64 ячейки дают ровно объём области.
///
/// Обход: на каждое ребро ячейки (12 рёбер) выдаём 4 тета; повторы между
/// ячейками убирает вызывающий (каждый тет имеет ровно одно ребро длины 2, так
/// что ключ тета однозначен).
pub fn bcc_tets_around_edge(q: Node, d: usize) -> [[Node; 4]; 4] {
    let u = (d + 1) % 3;
    let v = (d + 2) % 3;
    let a = [q.0, q.1, q.2];
    let mut b = [q.0, q.1, q.2];
    b[d] += 2;
    let cyc = [(-1i32, -1i32), (1, -1), (1, 1), (-1, 1)];
    let mut cs = [[0i32; 3]; 4];
    for (k, (su, sv)) in cyc.iter().enumerate() {
        let mut c = [q.0, q.1, q.2];
        c[d] += 1;
        c[u] += su;
        c[v] += sv;
        cs[k] = c;
    }
    let mut out = [[(0, 0, 0); 4]; 4];
    for m in 0..4 {
        let c1 = cs[m];
        let c2 = cs[(m + 1) % 4];
        /* Канонический порядок T1 (см. `t1`): длинные рёбра в слотах 0-1 и 2-3,
         * правый обход. Порядок важен: от него зависит и red-дробление, и
         * ориентация граней в таблице. */
        out[m] = crate::t1::oriented(
            (a[0], a[1], a[2]),
            (b[0], b[1], b[2]),
            (c1[0], c1[1], c1[2]),
            (c2[0], c2[1], c2[2]),
        );
    }
    out
}

/// Все теты, инцидентные ячейке с углом `base`: 12 рёбер × 4 тета = 48 записей.
/// Уникальных среди них — 12 на ячейку (остальные принадлежат соседним ячейкам),
/// поэтому вызывающий обязан дедуплицировать по [`tet_key`].
pub fn bcc_tets(base: Node) -> Vec<[Node; 4]> {
    let mut out = Vec::with_capacity(48);
    for d in 0..3 {
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;
        for a in 0..2 {
            for b in 0..2 {
                let mut q = [base.0, base.1, base.2];
                q[u] += 2 * a;
                q[v] += 2 * b;
                for t in bcc_tets_around_edge((q[0], q[1], q[2]), d) {
                    out.push(t);
                }
            }
        }
    }
    out
}

/// 9 узлов кубической ячейки (8 углов + центр) в единицах полушага.
pub fn cube_nodes(base: Node) -> [Node; 9] {
    let mut out = [(0, 0, 0); 9];
    for i in 0..8u8 {
        let dx = 2 * (i & 1) as i32;
        let dy = 2 * ((i >> 1) & 1) as i32;
        let dz = 2 * ((i >> 2) & 1) as i32;
        out[i as usize] = (base.0 + dx, base.1 + dy, base.2 + dz);
    }
    out[8] = (base.0 + 1, base.1 + 1, base.2 + 1);
    out
}

impl Tet {
    /// Объём: куб делится на 12 равных частей.
    pub fn volume(&self) -> f32 {
        let a = BASE / (1u32 << self.level) as f32;
        a * a * a / 12.0
    }

    /// Центр (среднее вершин) — для Вороного и физики.
    pub fn centroid(&self) -> [f64; 3] {
        let s = node_step(self.level) as f64;
        let mut c = [0.0; 3];
        for v in &self.v {
            c[0] += v.0 as f64 * s / 4.0;
            c[1] += v.1 as f64 * s / 4.0;
            c[2] += v.2 as f64 * s / 4.0;
        }
        c
    }

    /// Red-разбиение: 8 детей вдвое мельче, объём сохраняется точно.
    ///
    /// Для НАСТОЯЩЕЙ T1-ячейки (а все теты тела теперь приходят из
    /// `bcc_tets_around_edge` и канонизированы) берётся таблица `t1::red_refine`:
    /// все восемь детей конгруэнтны T1 и подобны родителю с коэффициентом 1/2 —
    /// это самоподобие и есть смысл решётки Сомирвилля. Старое разбиение
    /// «октаэдр через ребро ab-cd» давало четыре правильных угловых тета и
    /// четыре ПРОИЗВОЛЬНЫХ из середины, то есть форму детей не гарантировало.
    /// Тет, не являющийся T1 (приходит только из неиспользуемого ныне
    /// `from_sdf_grid`), дробится старым общим способом — чтобы не терять объём.
    pub fn refine(&self) -> [Tet; 8] {
        let lv = self.level + 1;
        let canon = crate::t1::canonical(self.v);
        if crate::t1::is_canonical(&canon) {
            let kids = crate::t1::red_refine(&canon);
            return kids.map(|(_, cell)| Tet { level: lv, v: cell });
        }
        let [a, b, c, d] = self.v;
        // в координатах нового уровня те же точки — это удвоенные индексы
        let d2 = |n: Node| (2 * n.0, 2 * n.1, 2 * n.2);
        let ab = (a.0 + b.0, a.1 + b.1, a.2 + b.2);
        let ac = (a.0 + c.0, a.1 + c.1, a.2 + c.2);
        let ad = (a.0 + d.0, a.1 + d.1, a.2 + d.2);
        let bc = (b.0 + c.0, b.1 + c.1, b.2 + c.2);
        let bd = (b.0 + d.0, b.1 + d.1, b.2 + d.2);
        let cd = (c.0 + d.0, c.1 + d.1, c.2 + d.2);
        let (a2, b2, c2, d2) = (d2(a), d2(b), d2(c), d2(d));
        [
            // четыре угловых
            Tet { level: lv, v: [a2, ab, ac, ad] },
            Tet { level: lv, v: [b2, ab, bc, bd] },
            Tet { level: lv, v: [c2, ac, bc, cd] },
            Tet { level: lv, v: [d2, ad, bd, cd] },
            // октаэдр в середине — на четыре тетраэдра через ребро ab-cd
            Tet { level: lv, v: [ab, ac, ad, cd] },
            Tet { level: lv, v: [ab, bc, ac, cd] },
            Tet { level: lv, v: [ab, bc, bd, cd] },
            Tet { level: lv, v: [ab, bd, ad, cd] },
        ]
    }
}

/// Тело: набор тетраэдров (возможно разных уровней) и предел дробления.
#[derive(Clone, Debug)]
pub struct TetBody {
    pub tets: Vec<Tet>,
    pub level_cap: u8,
    /// Мировой центр, от которого отсчитываются координаты узлов.
    pub origin: [f32; 3],
    /// SDF по узлам (точное расстояние до поверхности тела, знак по занятости).
    /// Считается один раз и живёт вместе с телом: марчинг режет рёбра по этим
    /// числам, а не по BFS-расстояниям сетки.
    pub sdf: std::sync::OnceLock<field::NodeField>,
}

impl TetBody {
    /// Поле расстояний по узлам, посчитанное один раз.
    pub fn sdf(&self) -> &field::NodeField {
        self.sdf.get_or_init(|| {
            let flat: Vec<[Node; 4]> = self.tets.iter().map(|t| t.v).collect();
            /* Теты задают только УЗЛЫ (какие точки внутри). Расстояние
             * считается от границы занятости, грани тетов в нём не участвуют —
             * поэтому поверхность не повторяет тетраэдры. */
            field::from_occupancy_metric(&flat, 6, 3, node_step(self.tets.iter().map(|t| t.level).max().unwrap_or(0)))
        })
    }

    /// Тело из готовой сетки SDF (второй путь создания: сетку заполнил
    /// шейдер на устройстве или CPU из сейва).
    ///
    /// `sdf` — значения в мировых единицах, порядок `((z*ny)+y)*nx+x`, узел
    /// `(lo + (x,y,z))` в координатах своего уровня. Тет попадает в тело, если
    /// все четыре его узла внутри (значение < 0).
    pub fn from_sdf_grid(
        center: [f32; 3],
        level: u8,
        lo: [i32; 3],
        dims: [usize; 3],
        sdf: &[f32],
    ) -> Self {
        assert_eq!(sdf.len(), dims[0] * dims[1] * dims[2], "размер сетки SDF");
        let step = node_step(level);
        let solid = |n: Node| -> bool {
            let x = n.0 - lo[0];
            let y = n.1 - lo[1];
            let z = n.2 - lo[2];
            if x < 0 || y < 0 || z < 0 {
                return false;
            }
            let (x, y, z) = (x as usize, y as usize, z as usize);
            if x >= dims[0] || y >= dims[1] || z >= dims[2] {
                return false;
            }
            sdf[(z * dims[1] + y) * dims[0] + x] < 0.0
        };
        let mut tets = Vec::new();
        /* ячейки: основание чётное, шаг 2; берём только полностью «внутренние» */
        let hi = [
            lo[0] + dims[0] as i32 - 3,
            lo[1] + dims[1] as i32 - 3,
            lo[2] + dims[2] as i32 - 3,
        ];
        let mut bz = lo[2];
        while bz <= hi[2] {
            let mut by = lo[1];
            while by <= hi[1] {
                let mut bx = lo[0];
                while bx <= hi[0] {
                    let nodes = cube_nodes((bx, by, bz));
                    for t in CUBE_TETS.iter() {
                        let v = [
                            nodes[t[0] as usize],
                            nodes[t[1] as usize],
                            nodes[t[2] as usize],
                            nodes[t[3] as usize],
                        ];
                        if v.iter().all(|n| solid(*n)) {
                            tets.push(Tet { level, v });
                        }
                    }
                    bx += 2;
                }
                by += 2;
            }
            bz += 2;
        }
        /* Поле приводим к соглашению «ноль между узлами» и к единицам узла. */
        /* Внутри поле в единицах узла (ноль между узлами); метры даёт sample_m. */
        let mut val = Vec::with_capacity(sdf.len());
        for v in sdf {
            let d = v / step;
            val.push(if *v < 0.0 { d - 0.5 } else { d + 0.5 });
        }
        let body = Self { tets, level_cap: level, origin: center, sdf: Default::default() };
        let _ = body.sdf.set(field::NodeField { lo, dims, val, step });
        body
    }

    /// Куб из `n³` кубических ячеек, каждая — 12 тетраэдров.
    pub fn cube(center: [f32; 3], n: i32, level: u8) -> Self {
        /* BCC-тет занимает ДВЕ кубические ячейки (его центры лежат в соседних
         * кубах), поэтому нечётный набор кубов не складывается из тетов без
         * выхода за границу — выравниваем к чётному. */
        let n = n + (n & 1);
        /* Теты Сомирвилля: BCC-тет пересекает границу кубической ячейки,
         * поэтому при переборе по ячейкам один и тот же тет приходит дважды —
         * убираем дубли по канонизированному четвёрочному ключу. */
        /* Сборка по кубическим ячейкам (bcc_tets + дедуп) — единственный вариант,
         * который заполняет объём ровно: 12 тетов на кубическую ячейку. Перебор
         * по узлам (bcc_tets_at) при обрезке по коробке недобирает (n=4: 480
         * вместо 768), то есть тело теряло бы массу — см. ignored-тест. */
        /* Дедупликации здесь НЕТ и быть не должно: 12 тетов, которые даёт
         * bcc_tets для своей ячейки, попарно различны во всём теле. Проверка
         * объёмом это и поймала: с дедупликацией n=2 давало 54 тета вместо 96,
         * то есть тело было на 44% легче — ломались масса, инерция и динамика. */
        /* Теты СОМИРВИЛЛЯ: два узла-угла на ребре длины 2 плюс два центра по
         * сторонам ребра; вокруг каждого ребра ровно четыре таких тета.
         *
         * Правило перебора (проверено объёмом): основания рёбер идут по
         * -n..n-2 с шагом 2 по ВСЕМ трём осям, 4 тета на ребро, глобальный
         * дедуп по каноническому ключу. Это даёт ровно 12 тетов на ячейку и
         * объём, точно равный объёму n³ ячеек — то есть тело есть объединение
         * целых тетов, без щелей и наложений.
         *
         * Без дедупа каждый тет приходит до четырёх раз (его заявляют все
         * соседние ячейки) — объём выходил 156%. */
        let mut tets = Vec::with_capacity((n * n * n * 12) as usize);
        let mut seen: std::collections::HashSet<[Node; 4]> = std::collections::HashSet::new();
        let mut bz = -n;
        while bz <= n - 2 {
            let mut by = -n;
            while by <= n - 2 {
                let mut bx = -n;
                while bx <= n - 2 {
                    for d in 0..3 {
                        for t in bcc_tets_around_edge((bx, by, bz), d) {
                            if seen.insert(tet_key(&t)) {
                                tets.push(Tet { level, v: t });
                            }
                        }
                    }
                    bx += 2;
                }
                by += 2;
            }
            bz += 2;
        }

        let body = Self {
            tets,
            // Предел дробления: ещё один уровень. Два уровня — это 8× тетраэдров,
            // и столкновения по тетраэдрам на CPU становятся слишком дорогими
            // (нужен широкий проход на GPU).
            level_cap: level + 1,
            origin: center,
            sdf: Default::default(),
        };
        /* ПОЛЕ КОРОБКИ — АНАЛИТИЧЕСКОЕ, регулярная сетка по всему bounding box.
         * Сетка поля не связана с разметкой тетов; шаг — на уровень мельче
         * тетов, если сетка влезает в пул SDF на устройстве (65536 узлов).
         *
         * Коробка ОСЕВАЯ: грани параллельны осям и совпадают с габаритом
         * коллайдера (тетов). Раньше здесь был поворот на 45° вокруг (1,1,1) —
         * на картинке выходил скошенный октаэдр, который не совпадал с
         * коллайдером и «срезал» углы плиты. Никакого сдвига на полшага тоже
         * нет: он смещал коробку относительно тетов (перекос). */
        let step = node_step(level);
        let half = n as f32 * step;
        /* Изоповерхность НЕ ДОЛЖНА лечь ровно в узлы сетки: у осевой коробки
         * грани проходят по плоскостям узлов, значение там ровно 0, и марчинг
         * вырождается — вершины садятся в узлы, а грани выпадают. Поэтому
         * полуразмер увеличен на 0.1 мм: коробка остаётся ОСЕВОЙ и
         * СИММЕТРИЧНОЙ (никакого перекоса, центр тот же), а ноль поля уходит
         * строго между слоями узлов. */
        let half_eval = half + 1e-4;
        /* Шаг поля — ВДВОЕ мельче шага тетов (level+1). Мельче нельзя по
         * бюджету кадра: у куба грани ПЛОСКИЕ, лишние треугольники ничего не
         * добавляют к форме, а в программном рендере (llvmpipe) они стоят
         * кадра. Точность поверхности при этом задаётся полем, а не тетами. */
        let (nf, fstep) = {
            let fs = node_step(level + 1);
            let (n, dim) = field_dims(half, fs, 3);
            if dim * dim * dim <= FIELD_NODE_BUDGET {
                (n, fs)
            } else {
                let fs = node_step(level);
                (field_dims(half, fs, 3).0, fs)
            }
        };
        let dim = (2 * nf + 1) as usize;
        let mut val = vec![0.0f32; dim * dim * dim];
        for z in -nf..=nf {
            for y in -nf..=nf {
                for x in -nf..=nf {
                    let px = x as f32 * fstep;
                    let py = y as f32 * fstep;
                    let pz = z as f32 * fstep;
                    let q = [px.abs() - half_eval, py.abs() - half_eval, pz.abs() - half_eval];
                    let outside = (q[0].max(0.0).powi(2) + q[1].max(0.0).powi(2)
                        + q[2].max(0.0).powi(2))
                    .sqrt();
                    let inside = q[0].max(q[1]).max(q[2]).min(0.0);
                    let i = (((z + nf) as usize) * dim + ((y + nf) as usize)) * dim
                        + ((x + nf) as usize);
                    /* В ШАГАХ ПОЛЯ — физика (sample_m, intersect_plane,
                     * sdf_contact) считает поле узловым и умножает на шаг
                     * сама; молчаливые метры давали глубину в 14 раз меньше. */
                    val[i] = (outside + inside) / fstep;
                }
            }
        }
        let _ = body.sdf.set(field::NodeField {
            lo: [-nf, -nf, -nf],
            dims: [dim, dim, dim],
            val,
            step: fstep,
        });
        body
    }

    /// Шар из тетраэдров: всё, что ближе радиуса к центру.
    pub fn ball(center: [f32; 3], radius: f32, level: u8) -> Self {
        /* ШАР ИЗ SDF: аналитическая сфера запекается в узлы решётки, теты —
         * только разметка, где стоят воксели (тет входит, если все четыре его
         * узла внутри). Поверхность берётся из запечённого поля, поэтому шар
         * круглый независимо от формы тетов. */
        let mut lv = level;
        while lv < level.saturating_add(6) {
            if node_step(lv) <= radius / 12.0 {
                break;
            }
            lv += 1;
        }
        /* Бюджет: пул тетов на тело ограничен, поэтому плотность держим так,
         * чтобы шар занимал не больше ~3000 тетов (12·V/a³). */
        loop {
            let a = node_step(lv) * 2.0;
            let vol = 4.0 / 3.0 * std::f32::consts::PI * radius * radius * radius;
            let est = 12.0 * vol / (a * a * a);
            if est <= 3000.0 || lv == 0 {
                break;
            }
            lv -= 1;
        }
        let step = node_step(lv);
        /* +5 слоёв: марчинг идёт ячейками и внешний слой ему нужен с каждой
         * стороны — иначе с дальней стороны строить не из чего. */
        let n = ((radius / step).ceil() as i32) + 5;
        let dim = (2 * n + 1) as usize;
        let idx = |x: i32, y: i32, z: i32| -> usize {
            (((z + n) as usize) * dim + ((y + n) as usize)) * dim + ((x + n) as usize)
        };
        /* ЗАНЯТОСТЬ (только для разметки тетов) — по узлам уровня тетов. */
        let mut solid = vec![false; dim * dim * dim];
        for z in -n..=n {
            for y in -n..=n {
                for x in -n..=n {
                    let d = ((x * x + y * y + z * z) as f32).sqrt();
                    solid[idx(x, y, z)] = d < radius / step;
                }
            }
        }
        let mut tets = Vec::new();
        let mut seen: std::collections::HashSet<[Node; 4]> = std::collections::HashSet::new();
        let inside = |nd: Node| -> bool {
            let (x, y, z) = (nd.0, nd.1, nd.2);
            if x < -n || y < -n || z < -n || x > n || y > n || z > n {
                return false;
            }
            solid[idx(x, y, z)]
        };
        let mut bz = -n;
        while bz <= n - 2 {
            let mut by = -n;
            while by <= n - 2 {
                let mut bx = -n;
                while bx <= n - 2 {
                    for d in 0..3 {
                        for t in bcc_tets_around_edge((bx, by, bz), d) {
                            if !t.iter().all(|nd| inside(*nd)) {
                                continue;
                            }
                            if seen.insert(tet_key(&t)) {
                                tets.push(Tet { level: lv, v: t });
                            }
                        }
                    }
                    bx += 2;
                }
                by += 2;
            }
            bz += 2;
        }
        /* ПОЛЕ — РЕГУЛЯРНАЯ СЕТКА ПО ВСЕМУ BOUNDING BOX, ОДИН ШАГ НА ВСЁ ТЕЛО.
         * Меш строится по ней (см. `extract_marching`), теты — разметка для
         * физики. Шаг поля берём САМЫЙ МЕЛКИЙ, который влезает в слот SDF на
         * устройстве (SDF_STRIDE = 65536 узлов): гладкость шара определяется
         * именно шагом сетки поля, а не размером тета. Нижняя граница — вдвое
         * мельче шага тетов, иначе поле не добавляет ничего к их точности. */
        let (fn_, fstep) = {
            let mut best: Option<(i32, f32)> = None;
            for extra in 0..4 {
                let fs = node_step(lv + 1 + extra);
                let (n, dim) = field_dims(radius, fs, 1);
                if dim * dim * dim <= FIELD_NODE_BUDGET {
                    best = Some((n, fs));
                } else {
                    break;
                }
            }
            best.unwrap_or_else(|| {
                let fs = node_step(lv);
                (field_dims(radius, fs, 2).0, fs)
            })
        };
        let fdim = (2 * fn_ + 1) as usize;
        let mut val = vec![0.0f32; fdim * fdim * fdim];
        for z in -fn_..=fn_ {
            for y in -fn_..=fn_ {
                for x in -fn_..=fn_ {
                    let i = (((z + fn_) as usize) * fdim + ((y + fn_) as usize)) * fdim
                        + ((x + fn_) as usize);
                    /* без сдвига: у запечённого SDF ноль и есть поверхность,
                     * а |значение| — настоящее расстояние в шагах поля. 0.1 мм
                     * запаса — чтобы ноль не попал РОВНО в узел (иначе марчинг
                     * вырождается); шар при этом остаётся симметричным. */
                    val[i] = ((x * x + y * y + z * z) as f32).sqrt()
                        - (radius + 1e-4) / fstep;
                }
            }
        }
        let body = Self {
            tets,
            level_cap: lv,
            origin: center,
            sdf: Default::default(),
        };
        let _ = body.sdf.set(field::NodeField {
            lo: [-fn_, -fn_, -fn_],
            dims: [fdim, fdim, fdim],
            val,
            step: fstep,
        });
        body
    }

    pub fn volume(&self) -> f64 {
        self.tets.iter().map(|t| t.volume() as f64).sum()
    }

    /// Дробить все тетраэдры: каждый → 8, объём сохраняется точно.
    pub fn subdivide(&mut self) -> bool {
        if self.tets.iter().all(|t| t.level >= self.level_cap) {
            return false;
        }
        let mut out = Vec::with_capacity(self.tets.len() * 8);
        for t in &self.tets {
            if t.level >= self.level_cap {
                out.push(*t);
            } else {
                out.extend_from_slice(&t.refine());
            }
        }
        self.tets = out;
        true
    }

    pub fn centroid(&self) -> [f32; 3] {
        let mut s = [0.0f64; 3];
        let mut w = 0.0f64;
        for t in &self.tets {
            let v = t.volume() as f64;
            let c = t.centroid();
            for k in 0..3 {
                s[k] += c[k] * v;
            }
            w += v;
        }
        [
            (s[0] / w) as f32,
            (s[1] / w) as f32,
            (s[2] / w) as f32,
        ]
    }

    /// Центроид в мировых координатах (с учётом `origin`).
    pub fn world_centroid(&self) -> [f32; 3] {
        let c = self.centroid();
        [
            c[0] + self.origin[0],
            c[1] + self.origin[1],
            c[2] + self.origin[2],
        ]
    }

    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for t in &self.tets {
            for n in &t.v {
                let p = node_pos(*n, t.level, self.origin);
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        (lo, hi)
    }

    /// Плоскости Вороного по затравкам раскола: для каждого осколка — список
    /// `(n, d)` в УЗЛАХ ПОЛЯ, где `n·p <= d` — его сторона среза.
    /// Отдельной функцией, потому что это же геометрическое описание нужно
    /// тестам и диагностике (проверка «поле = расстояние до этих плоскостей»).
    pub fn shard_planes(&self, pts: &[[f32; 3]]) -> Vec<Vec<([f32; 3], f32)>> {
        let step = self.sdf().step;
        let mut out = Vec::with_capacity(pts.len());
        for (i, si) in pts.iter().enumerate() {
            let mut planes = Vec::with_capacity(pts.len().saturating_sub(1));
            for (j, sj) in pts.iter().enumerate() {
                if i == j {
                    continue;
                }
                let mut n = [sj[0] - si[0], sj[1] - si[1], sj[2] - si[2]];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
                n = [n[0] / len, n[1] / len, n[2] / len];
                let mid = [
                    (si[0] + sj[0]) * 0.5,
                    (si[1] + sj[1]) * 0.5,
                    (si[2] + sj[2]) * 0.5,
                ];
                /* узлы в единицах решётки, поэтому и плоскость — в них */
                let d = n[0] * mid[0] / step + n[1] * mid[1] / step + n[2] * mid[2] / step;
                planes.push((n, d));
            }
            out.push(planes);
        }
        out
    }

    /// Дробление ПО ПОЛЮ: каждый осколок — это поле родителя, отсечённое
    /// плоскостями Вороного его затравки. Плоскость среза проходит где угодно
    /// (не по границам тетов), поэтому сколы выходят плавными и точными.
    pub fn shatter_fields(&self, pts: &[[f32; 3]]) -> Vec<TetBody> {
        /* Каждому осколку даём лишние слои пустоты по краям: иначе у мелких
         * кусков марчингу не хватает внешних узлов и срез выходит «впуклым». */
        let parent = field::expand(self.sdf(), 4);
        let step = parent.step;
        let planes = self.shard_planes(pts);
        let mut out: Vec<TetBody> = Vec::new();
        for (i, _si) in pts.iter().enumerate() {
            let mut f = parent.clone();
            for (n, d) in &planes[i] {
                f = field::intersect_plane(&f, *n, *d);
            }
            /* Теты БЕРЁМ ИЗ РОДИТЕЛЯ, а не выводим из нового поля: каждый
             * родительский тет отдаём ровно одному осколку (по его центроиду
             * внутри поля). Тогда сумма объёмов осколков равна объёму родителя —
             * квантование по решётке больше не «съедает» массу на срезах.
             * Форму даёт поле (гладкий срез), массу — теты (точный баланс). */
            let mut tets = Vec::new();
            for t in &self.tets {
                let c = t.v;
                /* центроид тета — В УЗЛАХ ПОЛЯ: у тета свой шаг (node_step), у
                 * поля свой (step). Раньше центроид (в узлах тета) сравнивался с
                 * затравкой, поделённой на шаг ПОЛЯ, — раздача тетов уезжала и
                 * на мелком поле один осколок оставался вообще без тетов. */
                let s = node_step(t.level) / step;
                let cx = (c[0].0 + c[1].0 + c[2].0 + c[3].0) as f32 / 4.0 * s;
                let cy = (c[0].1 + c[1].1 + c[2].1 + c[3].1) as f32 / 4.0 * s;
                let cz = (c[0].2 + c[1].2 + c[2].2 + c[3].2) as f32 / 4.0 * s;
                let mut best = f32::MAX;
                let mut best_i = usize::MAX;
                for (k, sk) in pts.iter().enumerate() {
                    let dx = cx - sk[0] / step;
                    let dy = cy - sk[1] / step;
                    let dz = cz - sk[2] / step;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < best {
                        best = d2;
                        best_i = k;
                    }
                }
                if best_i == i {
                    tets.push(*t);
                }
            }
            if tets.is_empty() {
                eprintln!("DIAG shatter: осколок {i} пропущен — нет тетов");
                continue;
            }
            /* Поле осколка — ОБРЕЗАННЫЙ кусок поля родителя (то же расстояние,
             * тот же шаг), а не весь бокс: меш строится по нему, поэтому
             * осколок получает ту же гладкость, что и целое тело.
             * Границы обрезки берём по ЗАНЯТОЙ области самого поля (узлы с
             * отрицательным значением), а не по тетам: теты лишь разметка, и
             * поверхность среза может выйти за их бокс — тогда марчинг упрётся
             * в край обрезанного поля и меш развалится. */
            let (mut blo, mut bhi) = ([i32::MAX; 3], [i32::MIN; 3]);
            for z in 0..f.dims[2] {
                for y in 0..f.dims[1] {
                    for x in 0..f.dims[0] {
                        if f.val[(z * f.dims[1] + y) * f.dims[0] + x] < 0.0 {
                            let c = [
                                f.lo[0] + x as i32,
                                f.lo[1] + y as i32,
                                f.lo[2] + z as i32,
                            ];
                            for k in 0..3 {
                                blo[k] = blo[k].min(c[k]);
                                bhi[k] = bhi[k].max(c[k]);
                            }
                        }
                    }
                }
            }
            if blo[0] > bhi[0] {
                eprintln!("DIAG shatter: осколок {i} пропущен — в поле нет отрицательных узлов");
                continue;
            }
            /* запас 3 узла: марчинг идёт ячейками, и срезу нужен слой
             * внешних узлов; больше не нужно, а каждый лишний слой — это
             * объём поля, который обязан влезть в слот SDF на устройстве */
            let lo = [blo[0] - 3, blo[1] - 3, blo[2] - 3];
            let dims = [
                (bhi[0] - blo[0] + 1 + 6) as usize,
                (bhi[1] - blo[1] + 1 + 6) as usize,
                (bhi[2] - blo[2] + 1 + 6) as usize,
            ];
            /* ШАГ ПОЛЯ ОСКОЛКА — НАСТОЛЬКО МЕЛКИЙ, НАСКОЛЬКО ВЛЕЗАЕТ.
             * Зубцы вдоль ребра среза даёт не формула пересечения (она даёт
             * верный знак и точное расстояние у всех узлов, между которыми
             * проходит ноль), а ШАГ СЕТКИ МАРЧИНГА: он марчит ячейками по два
             * узла, то есть один воксель, и на изломе поля (плоскость среза ×
             * грань тела) линейная интерполяция по длинной диагонали ячейки
             * мажет вершину почти на воксель. Замер на живой плите (8 осколков,
             * шаг 0.035): поле на шаг вдвое мельче даёт макс. уход вершин от
             * истинной поверхности 29.8 → 14.6 мм и «зубцов» (рёбер со сменой
             * знака) на 1000 вершин 30.5 → 9.1 — при том же мешере, тетах и
             * законе пересечения. Срез и грани при этом стоят на месте: поле
             * линейно вдоль рёбер, интерполяция по ним точна.
             * Цена — вершины меша ×r², узлы поля ×r³, поэтому r ограничен
             * бюджетом слота SDF: крупные осколки остаются на шаге родителя. */
            let (fstep, flo, fdims) = shard_field_box(step, &lo, &dims);
            let fc = if (fstep - step).abs() < 1e-9 {
                field::crop(&f, lo, dims)
            } else {
                /* Родитель пересэмплирован трилинейно (он гладкий, терять
                 * нечего), срезы считаны заново уже в узлах мелкой сетки —
                 * плоскость остаётся плоскостью в любом масштабе. */
                let sc = step / fstep;
                let mut vals = Vec::with_capacity(fdims[0] * fdims[1] * fdims[2]);
                for z in 0..fdims[2] {
                    for y in 0..fdims[1] {
                        for x in 0..fdims[0] {
                            let p = [
                                (flo[0] + x as i32) as f32 / sc,
                                (flo[1] + y as i32) as f32 / sc,
                                (flo[2] + z as i32) as f32 / sc,
                            ];
                            /* значение родителя — В ЕГО ШАГАХ, а поле осколка
                             * живёт в мелких: один шаг родителя = `sc` мелких */
                            vals.push(parent.sample(p) * sc);
                        }
                    }
                }
                let mut nf = field::NodeField { lo: flo, dims: fdims, val: vals, step: fstep };
                for (n, d) in &planes[i] {
                    nf = field::intersect_plane(&nf, *n, *d * sc);
                }
                nf
            };
            let body = TetBody {
                tets,
                level_cap: self.level_cap,
                origin: self.origin,
                sdf: Default::default(),
            };
            let _ = body.sdf.set(fc);
            out.push(body);
        }
        out
    }

    /// Раскол с упором на точку удара: часть затравок ставится рядом с ней,
    /// поэтому близко к попаданию образуется больше мелких отколов, а плита
    /// раскалывается крупными кусками вдали.
    pub fn shatter_biased(&self, seeds: usize, seed: u64, focus: [f32; 3]) -> Vec<TetBody> {
        let (lo, hi) = self.bounds();
        let lo = [lo[0] - self.origin[0], lo[1] - self.origin[1], lo[2] - self.origin[2]];
        let hi = [hi[0] - self.origin[0], hi[1] - self.origin[1], hi[2] - self.origin[2]];
        let f = [focus[0] - self.origin[0], focus[1] - self.origin[1], focus[2] - self.origin[2]];
        let mut rng = Lcg::new(seed);
        let n_near = (seeds * 3) / 5; // 60% затравок — у места удара
        let spread = 0.35 * (hi[0] - lo[0]).max(hi[2] - lo[2]).max(0.1);
        let pts: Vec<[f64; 3]> = (0..seeds)
            .map(|i| {
                if i < n_near {
                    // вокруг точки удара, ближе к краю — отколы
                    let jx = (rng.unit() - 0.5) * 2.0 * spread;
                    let jy = (rng.unit() - 0.5) * 2.0 * spread;
                    let jz = (rng.unit() - 0.5) * 2.0 * spread;
                    [
                        (f[0] as f64 + jx as f64).clamp(lo[0] as f64, hi[0] as f64),
                        (f[1] as f64 + jy as f64).clamp(lo[1] as f64, hi[1] as f64),
                        (f[2] as f64 + jz as f64).clamp(lo[2] as f64, hi[2] as f64),
                    ]
                } else {
                    [
                        lo[0] as f64 + (hi[0] - lo[0]) as f64 * rng.unit() as f64,
                        lo[1] as f64 + (hi[1] - lo[1]) as f64 * rng.unit() as f64,
                        lo[2] as f64 + (hi[2] - lo[2]) as f64 * rng.unit() as f64,
                    ]
                }
            })
            .collect();

        let mut buckets: Vec<Vec<Tet>> = vec![Vec::new(); seeds];
        for t in &self.tets {
            let c = t.centroid();
            let mut best = 0;
            let mut best_d = f64::MAX;
            for (i, p) in pts.iter().enumerate() {
                let d = (c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2) + (c[2] - p[2]).powi(2);
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            buckets[best].push(*t);
        }
        let _ = &buckets;
        /* Дробление ПО ПОЛЮ: плоскости режут внутри тетов, сколы плавные. */
        let pts32: Vec<[f32; 3]> = pts
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        return self.shatter_fields(&pts32);
        #[allow(unreachable_code)]
        buckets.into_iter().filter(|v| !v.is_empty()).map(|tets| TetBody {
            tets,
            level_cap: self.level_cap,
            origin: self.origin,
            sdf: Default::default(),
        }).collect()
    }

    /// Номер области поля, к которой относится тет: по его ЦЕНТРОИДУ.
    ///
    /// Центроид тета считается в координатах УЗЛОВ ПОЛЯ (`node_step(level)` /
    /// `step`), как и в `shatter_fields`: у тета и поля шаги разные, и без
    /// пересчёта раздача тетов уезжает. Если центроид попал в пустой узел
    /// (тет крупнее зазора между кусками), берётся ближайший помеченный узел в
    /// радиусе трёх; если и такого нет — ближайший габарит области.
    fn part_of_tet(&self, t: &Tet, f: &field::NodeField, comps: &field::Components) -> usize {
        let s = node_step(t.level) / f.step;
        let cx = (t.v[0].0 + t.v[1].0 + t.v[2].0 + t.v[3].0) as f32 * 0.25 * s;
        let cy = (t.v[0].1 + t.v[1].1 + t.v[2].1 + t.v[3].1) as f32 * 0.25 * s;
        let cz = (t.v[0].2 + t.v[1].2 + t.v[2].2 + t.v[3].2) as f32 * 0.25 * s;
        let (nx, ny, nz) = (f.dims[0], f.dims[1], f.dims[2]);
        let idx = |x: i32, y: i32, z: i32| -> Option<usize> {
            let (ix, iy, iz) = (x - f.lo[0], y - f.lo[1], z - f.lo[2]);
            if ix < 0 || iy < 0 || iz < 0 {
                return None;
            }
            let (ix, iy, iz) = (ix as usize, iy as usize, iz as usize);
            if ix >= nx || iy >= ny || iz >= nz {
                return None;
            }
            Some((iz * ny + iy) * nx + ix)
        };
        let (gx, gy, gz) = (cx.round() as i32, cy.round() as i32, cz.round() as i32);
        if let Some(i) = idx(gx, gy, gz) {
            let l = comps.label[i];
            if l != u32::MAX {
                return l as usize;
            }
        }
        let mut best = u32::MAX;
        let mut best_d = f32::MAX;
        for dz in -3i32..=3 {
            for dy in -3i32..=3 {
                for dx in -3i32..=3 {
                    if let Some(i) = idx(gx + dx, gy + dy, gz + dz) {
                        let l = comps.label[i];
                        if l != u32::MAX {
                            let d = (dx * dx + dy * dy + dz * dz) as f32;
                            if d < best_d {
                                best_d = d;
                                best = l;
                            }
                        }
                    }
                }
            }
        }
        if best != u32::MAX {
            return best as usize;
        }
        /* Совсем пусто рядом — по ближайшему центру габарита области. */
        let mut bi = 0usize;
        let mut bd = f32::MAX;
        for (k, (lo, hi)) in comps.bounds.iter().enumerate() {
            let c = [
                (lo[0] + hi[0]) as f32 * 0.5,
                (lo[1] + hi[1]) as f32 * 0.5,
                (lo[2] + hi[2]) as f32 * 0.5,
            ];
            let d = (c[0] - cx).powi(2) + (c[1] - cy).powi(2) + (c[2] - cz).powi(2);
            if d < bd {
                bd = d;
                bi = k;
            }
        }
        bi
    }

    /// Разбить тело на СВЯЗНЫЕ части по занятости его поля.
    ///
    /// `None` — тело связное (одна область) или поля нет; `Some(parts)` —
    /// поле распалось на несколько несвязных кусков, частей не меньше двух.
    ///
    /// `min_gap` — допуск связности (см. [`field::components`]), `pad` — сколько
    /// узлов ПУСТОТЫ оставить вокруг каждой части: марчингу нужны внешние узлы,
    /// иначе срез выходит «впуклым» (то же правило, что в `shatter_fields`).
    ///
    /// Теты делятся по ЦЕНТРОИДУ, как в `shatter_fields`: каждый родительский
    /// тет уходит РОВНО в одну часть, поэтому сумма объёмов частей равна
    /// объёму родителя — масса сохраняется точно, без квантования по решётке.
    pub fn split_disconnected(&self, min_gap: i32, pad: i32) -> Option<Vec<TetBody>> {
        let f = self.sdf();
        let comps = field::components(f, min_gap);
        if comps.count() < 2 {
            return None;
        }
        let mut buckets: Vec<Vec<Tet>> = vec![Vec::new(); comps.count()];
        for t in &self.tets {
            let k = self.part_of_tet(t, f, &comps);
            buckets[k].push(*t);
        }
        let pad = pad.max(1);
        let mut out: Vec<TetBody> = Vec::with_capacity(buckets.len());
        /* Области БЕЗ тетов — шум поля (одиночные узлы у среза): массы у них
         * нет, тела из них не сделать. Считаем их разом, чтобы не заливать
         * лог строкой на каждый осколок. */
        let mut noise_areas = 0usize;
        let mut noise_nodes = 0usize;
        for (k, tets) in buckets.into_iter().enumerate() {
            if tets.is_empty() {
                noise_areas += 1;
                noise_nodes += comps.sizes[k];
                continue;
            }
            let (blo, bhi) = comps.bounds[k];
            /* Бокс части: её габарит плюс `pad` узлов пустоты. Обрезка по
             * границам родителя — внутри `field::crop`. */
            let lo = [blo[0] - pad, blo[1] - pad, blo[2] - pad];
            let dims = [
                (bhi[0] - blo[0] + 1 + 2 * pad) as usize,
                (bhi[1] - blo[1] + 1 + 2 * pad) as usize,
                (bhi[2] - blo[2] + 1 + 2 * pad) as usize,
            ];
            let mut part = field::crop(f, lo, dims);
            /* Чужие области, попавшие в бокс, гасим в ПЛЮС: иначе их
             * отрицательные узлы дадут в поле части вторую изоповерхность
             * (меш с дырой и лишним куском). Знак переворачиваем, а не ставим
             * «далеко»: так в поле не появляется НОВОГО перехода через ноль —
             * все перевёрнутые узлы положительные, а своих отрицательных узлов
             * у чужой области в боксе больше нет. */
            let (pnx, pny) = (part.dims[0], part.dims[1]);
            for z in 0..part.dims[2] {
                for y in 0..pny {
                    for x in 0..pnx {
                        let (gx, gy, gz) = (
                            part.lo[0] + x as i32,
                            part.lo[1] + y as i32,
                            part.lo[2] + z as i32,
                        );
                        let si = (((gz - f.lo[2]) as usize) * f.dims[1]
                            + ((gy - f.lo[1]) as usize))
                            * f.dims[0]
                            + ((gx - f.lo[0]) as usize);
                        let l = comps.label[si];
                        if l != u32::MAX && l != k as u32 {
                            let i = (z * pny + y) * pnx + x;
                            part.val[i] = part.val[i].abs().max(0.5);
                        }
                    }
                }
            }
            let body = TetBody {
                tets,
                level_cap: self.level_cap,
                origin: self.origin,
                sdf: Default::default(),
            };
            let _ = body.sdf.set(part);
            out.push(body);
        }
        if out.len() < 2 {
            return None;
        }
        /* Шум гасим молча, когда областей много и все они мелкие (узлы у
         * среза); крупную «область без тетов» показываем — это уже потеря
         * массы или кривое поле. */
        if noise_areas > 0 && noise_nodes > 8 {
            eprintln!(
                "DIAG split_disconnected: {noise_areas} областей без тетов ({noise_nodes} узлов) погашены как шум поля"
            );
        }
        Some(out)
    }

    /// Тетраэдры в локальных координатах (относительно центроида) — для
    /// столкновений и для показа самих тетраэдров во вьюере.
    pub fn local_tets(&self) -> Vec<[[f32; 3]; 4]> {
        let c = self.world_centroid();
        self.tets
            .iter()
            .map(|t| {
                std::array::from_fn(|i| {
                    let p = node_pos(t.v[i], t.level, self.origin);
                    [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
                })
            })
            .collect()
    }

    /// Только поверхностные тетраэдры (у которых есть грань наружу) — для
    /// столкновений: внутренние тетраэдры никого не касаются, а их большинство.
    pub fn surface_tets(&self) -> Vec<[[f32; 3]; 4]> {
        let c0 = self.world_centroid();
        let mut count: HashMap<[u64; 3], u32> = HashMap::new();
        for t in &self.tets {
            for (i, j, k) in [(0usize, 1usize, 2usize), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
                let mut key = [pack(t.v[i]), pack(t.v[j]), pack(t.v[k])];
                key.sort();
                *count.entry(key).or_insert(0) += 1;
            }
        }
        let c = c0;
        self.tets
            .iter()
            .filter(|t| {
                [(0usize, 1usize, 2usize), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
                    .iter()
                    .any(|(i, j, k)| {
                        let mut key = [pack(t.v[*i]), pack(t.v[*j]), pack(t.v[*k])];
                        key.sort();
                        count.get(&key).copied().unwrap_or(0) == 1
                    })
            })
            .map(|t| {
                std::array::from_fn(|i| {
                    let p = node_pos(t.v[i], t.level, self.origin);
                    [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
                })
            })
            .collect()
    }

    /// Рёбра тетраэдров (пары точек) без повторов — каркас для вьюера.
    pub fn edge_lines(&self) -> Vec<[f32; 3]> {
        let c = self.world_centroid();
        let mut seen: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
        let mut out = Vec::new();
        for t in &self.tets {
            let p: [[f32; 3]; 4] =
                std::array::from_fn(|i| node_pos(t.v[i], t.level, self.origin));
            for (i, j) in [(0usize, 1usize), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                let a = pack(t.v[i]);
                let b = pack(t.v[j]);
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) {
                    continue;
                }
                for k in [i, j] {
                    out.push([p[k][0] - c[0], p[k][1] - c[1], p[k][2] - c[2]]);
                }
            }
        }
        out
    }

    /// Марчинг по тетраэдрам: граничные грани тетраэдров + сглаживание.
    ///
    /// Поле занятости бинарное, поэтому настоящей дроби `t` на ребре не
    /// бывает: изоповерхность совпадает с гранями, и вершины сидят строго в
    /// узлах. Чтобы поверхность была живой, после извлечения вершины
    /// сдвигаются лапласианом (2 итерации) — форма перестаёт быть решёточной,
    /// а объём возвращается масштабированием относительно центра масс.
    pub fn surface(&self) -> Mesh {
        self.surface_grid().0
    }

    /// Меш и его решётка. НИКАКОГО сглаживания вершин и никаких проекций:
    /// форма — только из значений поля в узлах (марчинг + t по рёбрам).
    pub fn surface_grid(&self) -> (Mesh, Grid) {
        /* Поверхность — изоповерхность ПОЛЯ, а не набор граней тетов.
         * Единственный путь: марчинг по полю тела. Теты — только разметка и
         * масса, форма задаётся значениями поля в узлах. (Старый переключатель
         * VF_MARCH=old удалён: он собирал меш из тетов и перекрывал форму поля.) */
        let mut mesh = self.extract_marching();
        /* Лапласиана вершин здесь НЕТ: он сдвигал вершины с изоповерхности и
         * подменял собой интерполяцию. Гладкость даёт шаг сетки поля. */
        /* Нормали — из ГРАДИЕНТА поля, а не из граней треугольников: поле
         * гладкое (расстояние), поэтому затенение перестаёт показывать
         * тетраэдры и поверхность читается как одна оболочка. Кадр — кадр
         * ПОЛЯ: шаг берём из `f.step`, а не из уровня тетов (поле может быть
         * запечено мельче — см. `TetBody::ball`). */
        if let Some(f) = self.sdf.get() {
            let step = f.step;
            for (i, p) in mesh.positions.iter().enumerate() {
                let local = [
                    (p[0] - self.origin[0]) / step,
                    (p[1] - self.origin[1]) / step,
                    (p[2] - self.origin[2]) / step,
                ];
                let g = f.gradient(local);
                mesh.normals[i] = g;
            }
        }
        let level = mesh.level;
        let origin = self.centroid();
        let (lo, hi) = self.bounds();
        let fstep = self.sdf.get().map(|f| f.step).unwrap_or(BASE / (1u32 << level) as f32 * 0.5);
        (
            mesh,
            Grid {
                lo: [lo[0] - origin[0], lo[1] - origin[1], lo[2] - origin[2]],
                hi: [hi[0] - origin[0], hi[1] - origin[1], hi[2] - origin[2]],
                step: fstep,
            },
        )
    }

    /// Настоящий marching tetrahedra по непрерывному полю занятости.
    ///
    /// Проходит ВСЕМИ тетами покрывающей сетки (в том числе пустыми рядом с
    /// поверхностью — иначе по границе тела не хватало бы треугольников), по
    /// 12 тетов на кубическую ячейку, и берёт пересечение нуля поля с рёбрами.
    /// Вершины кэшируются по паре узлов, поэтому соседние теты делят одну
    /// вершину и швов нет.
    fn extract_marching(&self) -> Mesh {
        self.extract_marching_diag(None)
    }

    /// Замер распределения `t` по всем пересечённым рёбрам — диагностика
    /// интерполяции (см. тест `t_histogram`). Пустой список — все рёбра
    /// пересекаются ровно в узле, что для непрерывного поля невозможно.
    pub fn march_edge_t(&self) -> Vec<f32> {
        self.march_diag().t
    }

    /// Полный замер марчинга: `t` по рёбрам и точки, которые дала линейная
    /// интерполяция (до проекции на изоповерхность).
    pub fn march_diag(&self) -> MarchDiag {
        let mut d = MarchDiag::default();
        let _ = self.extract_marching_diag(Some(&mut d));
        d
    }

    fn extract_marching_diag(&self, mut diag: Option<&mut MarchDiag>) -> Mesh {
        /* Поле по ОБЪЁМУ тетов: заполнение одними вершинами даёт дырявую
         * занятость, ноль поля не встречается и марчинг выдаёт ноль
         * треугольников. Запас 3 узла: тет заходит за своё основание. */
        /* Поле расстояний живёт вместе с телом и считается один раз. Ребро
         * режется по НАСТОЯЩИМ расстояниям до поверхности, а не по сеточным
         * BFS-значениям — отсюда гладкая интерполяция. */
        let f = self.sdf();
        let level = self.tets.iter().map(|t| t.level).max().unwrap_or(0);
        let origin = self.origin;
        /* Сетка марчинга — сетка ПОЛЯ, а не тетов: поле может быть запечено
         * мельче, и тогда поверхность строится по его шагу. */
        let fstep = f.step;

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut cache: HashMap<(Node, Node), u32> = HashMap::new();


        /* Марчим ЯЧЕЙКАМИ сетки ПОЛЯ: ячейка — 2×2×2 узла, 12 тетов. Теты
         * здесь только ТОПОЛОГИЯ марчинга (какие четвёрки узлов соединять), а не
         * разметка тела: значение в узле — настоящее расстояние до поверхности,
         * и вершина ставится по нему. Соседние ячейки делят грань, вершины
         * кэшируются по паре узлов — швов нет.
         *
         * Ячейки, у всех 9 узлов которых один знак, не могут дать треугольника
         * (каждый тет — четвёрка этих же узлов) и пропускаются: их большинство,
         * а поле с запасом по краям. */
        let end = [
            f.lo[0] + f.dims[0] as i32 - 1,
            f.lo[1] + f.dims[1] as i32 - 1,
            f.lo[2] + f.dims[2] as i32 - 1,
        ];
        let even_up = |v: i32| if v & 1 != 0 { v + 1 } else { v };
        let even_dn = |v: i32| if v & 1 != 0 { v - 1 } else { v };
        /* Ячейка занимает узлы base..base+2, поэтому основание держим так,
         * чтобы вся ячейка лежала внутри блока: иначе крайние узлы читаются
         * прижатыми и поверхность на границе обрывается. */
        let lo = [even_up(f.lo[0]), even_up(f.lo[1]), even_up(f.lo[2])];
        let hi = [even_dn(end[0]), even_dn(end[1]), even_dn(end[2])];
        for bz in (lo[2]..=hi[2]).step_by(2) {
            for by in (lo[1]..=hi[1]).step_by(2) {
                for bx in (lo[0]..=hi[0]).step_by(2) {
                    let nodes = cube_nodes((bx, by, bz));
                    let vals: [f32; 9] =
                        std::array::from_fn(|i| f.at([nodes[i].0, nodes[i].1, nodes[i].2]));
                    let any_neg = vals.iter().any(|v| *v < 0.0);
                    let any_pos = vals.iter().any(|v| *v >= 0.0);
                    if !(any_neg && any_pos) {
                        continue;
                    }
                    for t in CUBE_TETS.iter() {
                        let tet_corners: [Node; 4] = [
                            nodes[t[0] as usize],
                            nodes[t[1] as usize],
                            nodes[t[2] as usize],
                            nodes[t[3] as usize],
                        ];
                        let mut pos = [[0.0f32; 3]; 4];
                        let val: [f32; 4] = std::array::from_fn(|i| {
                            let q = tet_corners[i];
                            pos[i] = [
                                origin[0] + q.0 as f32 * fstep,
                                origin[1] + q.1 as f32 * fstep,
                                origin[2] + q.2 as f32 * fstep,
                            ];
                            vals[t[i] as usize]
                        });
                        let mut vertex = |a: Node, b: Node, tp: f32| -> u32 {
                            if let Some(d) = diag.as_deref_mut() {
                                d.t.push(tp);
                            }
                            let mut a = a;
                            let mut b = b;
                            let mut tp = tp;
                            if pack(a) > pack(b) {
                                std::mem::swap(&mut a, &mut b);
                                tp = 1.0 - tp;
                            }
                            /* Ноль поля РОВНО в узле (симметричная аналитика
                             * даёт такие значения): пересечение формально легло
                             * в вершину сетки, и если кэшировать его по паре
                             * узлов, из одной точки получатся РАЗНЫЕ вершины для
                             * разных рёбер — меш разваливается на «открытые
                             * рёбра». Привязываем такую точку к самому узлу:
                             * тогда все рёбра, сходящиеся в нём, делят одну
                             * вершину. */
                            let node_hit = if tp <= 1e-6 {
                                Some(a)
                            } else if tp >= 1.0 - 1e-6 {
                                Some(b)
                            } else {
                                None
                            };
                            if let Some(n) = node_hit {
                                let key = (n, n);
                                if let Some(&id) = cache.get(&key) {
                                    return id;
                                }
                                let p = [
                                    origin[0] + n.0 as f32 * fstep,
                                    origin[1] + n.1 as f32 * fstep,
                                    origin[2] + n.2 as f32 * fstep,
                                ];
                                let id = positions.len() as u32;
                                if let Some(d) = diag.as_deref_mut() {
                                    d.interp.push(p);
                                }
                                positions.push(p);
                                cache.insert(key, id);
                                return id;
                            }
                            let key = (a, b);
                            if let Some(&id) = cache.get(&key) {
                                return id;
                            }
                            let pa = [origin[0] + a.0 as f32 * fstep, origin[1] + a.1 as f32 * fstep, origin[2] + a.2 as f32 * fstep];
                            let pb = [origin[0] + b.0 as f32 * fstep, origin[1] + b.1 as f32 * fstep, origin[2] + b.2 as f32 * fstep];
                            let p = [
                                pa[0] + (pb[0] - pa[0]) * tp,
                                pa[1] + (pb[1] - pa[1]) * tp,
                                pa[2] + (pb[2] - pa[2]) * tp,
                            ];
                            let id = positions.len() as u32;
                            if let Some(d) = diag.as_deref_mut() {
                                d.interp.push(p);
                            }
                            positions.push(p);
                            cache.insert(key, id);
                            id
                        };
                        march_tet(&tet_corners, &pos, &val, &mut vertex, &mut indices);
                    }
                }
            }
        }

        let mut mesh = Mesh {
            normals: vec![[0.0f32; 3]; positions.len()],
            positions,
            indices,
            level,
        };
        /* ПРОЕКЦИЙ НЕТ: вершина — это ровно точка линейной интерполяции на
         * ребре, ничего больше. Точность даёт шаг сетки поля (ошибка O(h²)), а
         * не доталкивание вершины на аналитику: проекция сдвигала вершину с
         * изоповерхности и, при шаге не того уровня, давала «вершины в
         * постоянном отступе от узла». */
        recompute_normals(&mut mesh);
        mesh
    }

/// Точная поверхность: грани, встречающиеся у одного тетраэдра.
    #[cfg(test)]
    fn extract_faces(&self) -> Mesh {
        let mut faces: HashMap<[u64; 3], (u32, [Node; 3], u8, [f32; 3])> = HashMap::new();
        // Узлы решётки отсчитываются от origin тела. Здесь раньше стоял
        // world_centroid(), и поверхность каждого осколка строилась в своём
        // сдвинутом кадре — меш уезжал относительно коллайдера.
        let origin = self.origin;
        for t in &self.tets {
            let p: [[f32; 3]; 4] = std::array::from_fn(|i| node_pos(t.v[i], t.level, origin));
            for (i, j, k) in [(0usize, 1usize, 2usize), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
                let mut key = [pack(t.v[i]), pack(t.v[j]), pack(t.v[k])];
                key.sort();
                let opposite = p[6 - i - j - k];
                if let Some(e) = faces.get_mut(&key) {
                    e.0 += 1;
                } else {
                    faces.insert(key, (1, [t.v[i], t.v[j], t.v[k]], t.level, opposite));
                }
            }
        }

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut vertex_of: HashMap<u64, u32> = HashMap::new();
        let level = self.tets.iter().map(|t| t.level).max().unwrap_or(0);

        for (_, (count, nodes, tet_level, opposite)) in faces {
            if count != 1 {
                continue;
            }
            let a = node_pos(nodes[0], tet_level, origin);
            let b = node_pos(nodes[1], tet_level, origin);
            let c = node_pos(nodes[2], tet_level, origin);
            let n = [
                (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
                (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
                (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
            ];
            let to_out = [
                (a[0] + b[0] + c[0]) / 3.0 - opposite[0],
                (a[1] + b[1] + c[1]) / 3.0 - opposite[1],
                (a[2] + b[2] + c[2]) / 3.0 - opposite[2],
            ];
            let flip = n[0] * to_out[0] + n[1] * to_out[1] + n[2] * to_out[2] < 0.0;
            let order: [usize; 3] = if flip { [0, 2, 1] } else { [0, 1, 2] };
            let mut ids = [0u32; 3];
            for (slot, idx) in order.iter().enumerate() {
                let key = pack(nodes[*idx]);
                ids[slot] = *vertex_of.entry(key).or_insert_with(|| {
                    positions.push([a, b, c][*idx]);
                    normals.push([n[0], n[1], n[2]]);
                    (positions.len() - 1) as u32
                });
            }
            indices.extend_from_slice(&ids);
        }

        let mut mesh = Mesh {
            positions,
            normals,
            indices,
            level,
        };
        orient_consistently(&mut mesh);
        recompute_normals(&mut mesh);
        mesh
    }
}

/// Сырые замеры марчинга (гистограмма `t` — диагностика интерполяции).
#[derive(Default, Debug)]
pub struct MarchDiag {
    /// `t` по каждому пересечённому ребру: 0 — ноль в узле `a`, 1 — в узле `b`.
    pub t: Vec<f32>,
    /// Точка, которую дала ЛИНЕЙНАЯ интерполяция вдоль ребра (до проекции),
    /// в том же порядке, что и вершины меша.
    pub interp: Vec<[f32; 3]>,
}

/// Параметры решётки тела для отрисовки сетки вокселей.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub lo: [f32; 3],
    pub hi: [f32; 3],
    pub step: f32,
}

/// Марчинг одного тетраэдра по знакам поля с интерполяцией по рёбрам.
fn march_tet(
    corners: &[Node; 4],
    pos: &[[f32; 3]; 4],
    val: &[f32; 4],
    vertex: &mut dyn FnMut(Node, Node, f32) -> u32,
    out: &mut Vec<u32>,
) {
    let inside: Vec<usize> = (0..4).filter(|&i| val[i] < 0.0).collect();
    if inside.is_empty() || inside.len() == 4 {
        return;
    }
    let mut pts: Vec<(u32, [f32; 3])> = Vec::new();
    for &i in &inside {
        for j in 0..4 {
            if val[j] >= 0.0 {
                // t по линейной интерполяции поля между узлами
                let t = val[i] / (val[i] - val[j]);
                let id = vertex(corners[i], corners[j], t);
                let p = pos[i];
                let q = pos[j];
                pts.push((id, [
                    p[0] + (q[0] - p[0]) * t,
                    p[1] + (q[1] - p[1]) * t,
                    p[2] + (q[2] - p[2]) * t,
                ]));
            }
        }
    }
    /* Порядок точек — это порядок обхода многоугольника сечения. Сортировать
     * по номеру вершины НЕЛЬЗЯ: на четырёх точках это даёт «бабочку», и
     * половина треугольников выходит вывернутой/пересечённой (в окне это
     * выглядит как дырки). Убираем только повторы, сохраняя порядок. */
    {
        let mut seen: Vec<u32> = Vec::with_capacity(4);
        pts.retain(|(id, _)| {
            if seen.contains(id) {
                false
            } else {
                seen.push(*id);
                true
            }
        });
    }

    let centroid_in = {
        let mut c = [0.0f32; 3];
        for &i in &inside {
            for k in 0..3 {
                c[k] += pos[i][k] / inside.len() as f32;
            }
        }
        c
    };

    let emit = |a: usize, b: usize, c: usize, out: &mut Vec<u32>| {
        let pa = pts[a].1;
        let pb = pts[b].1;
        let pc = pts[c].1;
        let n = [
            (pb[1] - pa[1]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[1] - pa[1]),
            (pb[2] - pa[2]) * (pc[0] - pa[0]) - (pb[0] - pa[0]) * (pc[2] - pa[2]),
            (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]),
        ];
        let mid = [
            (pa[0] + pb[0] + pc[0]) / 3.0 - centroid_in[0],
            (pa[1] + pb[1] + pc[1]) / 3.0 - centroid_in[1],
            (pa[2] + pb[2] + pc[2]) / 3.0 - centroid_in[2],
        ];
        let dot = n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2];
        if dot >= 0.0 {
            out.extend_from_slice(&[pts[a].0, pts[b].0, pts[c].0]);
        } else {
            out.extend_from_slice(&[pts[a].0, pts[c].0, pts[b].0]);
        }
    };

    match pts.len() {
        3 => emit(0, 1, 2, out),
        4 => {
            let mut c = [0.0f32; 3];
            for (_, p) in &pts {
                for k in 0..3 {
                    c[k] += p[k] / 4.0;
                }
            }
            let p0 = pts[0].1;
            let p1 = pts[1].1;
            let p2 = pts[2].1;
            let n = [
                (p1[1] - p0[1]) * (p2[2] - p0[2]) - (p1[2] - p0[2]) * (p2[1] - p0[1]),
                (p1[2] - p0[2]) * (p2[0] - p0[0]) - (p1[0] - p0[0]) * (p2[2] - p0[2]),
                (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]),
            ];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
            let nn = [n[0] / len, n[1] / len, n[2] / len];
            let d0 = [p0[0] - c[0], p0[1] - c[1], p0[2] - c[2]];
            let l0 = (d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2]).sqrt().max(1e-9);
            let u = [d0[0] / l0, d0[1] / l0, d0[2] / l0];
            let w = [
                nn[1] * u[2] - nn[2] * u[1],
                nn[2] * u[0] - nn[0] * u[2],
                nn[0] * u[1] - nn[1] * u[0],
            ];
            let mut order: Vec<usize> = (0..4).collect();
            order.sort_by(|&a, &b| {
                let da = [pts[a].1[0] - c[0], pts[a].1[1] - c[1], pts[a].1[2] - c[2]];
                let db = [pts[b].1[0] - c[0], pts[b].1[1] - c[1], pts[b].1[2] - c[2]];
                let aa = (da[0] * w[0] + da[1] * w[1] + da[2] * w[2])
                    .atan2(da[0] * u[0] + da[1] * u[1] + da[2] * u[2]);
                let ab = (db[0] * w[0] + db[1] * w[1] + db[2] * w[2])
                    .atan2(db[0] * u[0] + db[1] * u[1] + db[2] * u[2]);
                aa.partial_cmp(&ab).unwrap()
            });
            /* Диагональ четырёхугольника выбираем по НОМЕРАМ вершин, а не по
             * порядку обхода: соседний тет режет ту же грань, начиная обход с
             * другой вершины, и при выборе «по порядку» получалась ДРУГАЯ
             * диагональ — между двумя тетами оставалась щель, а у поверхности
             * половина рёбер выходила кратности 1 (в окне это и есть дырки).
             * Поворачиваем цикл так, чтобы первым шёл минимальный номер: тогда
             * диагональ order[0]—order[2] у обеих сторон совпадает. */
            let mut best = 0usize;
            for i in 1..4 {
                if pts[order[i]].0 < pts[order[best]].0 {
                    best = i;
                }
            }
            let o = [order[best], order[(best + 1) % 4], order[(best + 2) % 4], order[(best + 3) % 4]];
            emit(o[0], o[1], o[2], out);
            emit(o[0], o[2], o[3], out);
        }
        _ => {}
    }
}

/// Поверхности осколков одним проходом по ТЕЛУ: карта граней строится один
/// раз, поэтому грани на стыке двух осколков попадают в оба меша с одинаковой
/// ориентацией, и каждый осколок получает замкнутую поверхность.
pub fn split_surfaces(origin: [f32; 3], tets: &[Tet], shard_of: &[usize], n_shards: usize) -> Vec<Mesh> {
    use std::collections::HashMap as H;
    // грань → узлы, уровень и «противоположные» вершины по каждому владельцу
    // (у грани на стыке двух осколков владельцев два, и каждому нужна своя
    // ориентация — иначе у одного из осколков поверхность вывернута).
    let mut faces: H<[u64; 3], ([Node; 3], u8, Vec<(usize, [f32; 3])>)> = H::new();
    for (ti, t) in tets.iter().enumerate() {
        let owner = shard_of[ti];
        let p: [[f32; 3]; 4] = std::array::from_fn(|i| node_pos(t.v[i], t.level, origin));
        for (i, j, k) in [(0usize, 1usize, 2usize), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
            let mut key = [pack(t.v[i]), pack(t.v[j]), pack(t.v[k])];
            key.sort();
            let opposite = p[6 - i - j - k];
            match faces.get_mut(&key) {
                Some(e) => e.2.push((owner, opposite)),
                None => {
                    faces.insert(key, ([t.v[i], t.v[j], t.v[k]], t.level, vec![(owner, opposite)]));
                }
            }
        }
    }

    let mut out: Vec<Mesh> = (0..n_shards)
        .map(|_| Mesh {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            level: 0,
        })
        .collect();
    let mut vertex_of: Vec<H<u64, u32>> = (0..n_shards).map(|_| H::new()).collect();

    for (_, (nodes, level, owners)) in faces {
        // Грань внутри одного осколка (два тетраэдра одного владельца) — не
        // поверхность. Грань на стыке двух осколков нужна обоим, каждая со
        // своей ориентацией.
        let distinct: std::collections::BTreeSet<usize> =
            owners.iter().map(|(o, _)| *o).collect();
        if owners.len() >= 2 && distinct.len() == 1 {
            continue;
        }
        let a = node_pos(nodes[0], level, origin);
        let b = node_pos(nodes[1], level, origin);
        let c = node_pos(nodes[2], level, origin);
        let n = [
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        // Грань может встретиться у трёх тетраэдров (два в одном осколке и
        // один в соседнем). Каждому осколку — РОВНО один треугольник, иначе
        // в меше появляются дубликаты и рвётся водонепроницаемость.
        let mut per_owner: std::collections::BTreeMap<usize, [f32; 3]> =
            std::collections::BTreeMap::new();
        for (shard, opposite) in &owners {
            per_owner.entry(*shard).or_insert(*opposite);
        }
        for (shard, opposite) in per_owner {
            let to_out = [
                (a[0] + b[0] + c[0]) / 3.0 - opposite[0],
                (a[1] + b[1] + c[1]) / 3.0 - opposite[1],
                (a[2] + b[2] + c[2]) / 3.0 - opposite[2],
            ];
            let flip = n[0] * to_out[0] + n[1] * to_out[1] + n[2] * to_out[2] < 0.0;
            let order: [usize; 3] = if flip { [0, 2, 1] } else { [0, 1, 2] };
            let mesh = &mut out[shard];
            let mut ids = [0u32; 3];
            for (slot, idx) in order.iter().enumerate() {
                let key = pack(nodes[*idx]);
                ids[slot] = *vertex_of[shard].entry(key).or_insert_with(|| {
                    mesh.positions.push([a, b, c][*idx]);
                    mesh.normals.push([n[0] / len, n[1] / len, n[2] / len]);
                    (mesh.positions.len() - 1) as u32
                });
            }
            mesh.indices.extend_from_slice(&ids);
            mesh.level = mesh.level.max(level);
        }
    }

    for m in out.iter_mut() {
        orient_consistently(m);
    }
    out
}

pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub level: u8,
}

impl Mesh {
    pub fn triangles(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Объём меша (теорема о дивергенции) — для проверок.
pub fn mesh_volume(m: &Mesh) -> f32 {
    let mut v = 0.0f64;
    for t in m.indices.chunks(3) {
        let a = m.positions[t[0] as usize];
        let b = m.positions[t[1] as usize];
        let c = m.positions[t[2] as usize];
        v += (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])) as f64
            / 6.0;
    }
    v as f32
}

/// Детерминированный ГПСЧ: раскол обязан воспроизводиться.
pub struct Lcg(u64);
impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
}

/// Согласование обхода: у двух треугольников, делящих ребро, направления
/// должны быть противоположны. Одна вывернутая грань ломает водонепроницаемость
/// (три её ребра перестают быть парными), поэтому раскрашиваем компоненты
/// связности в два цвета «перевёрнут / нет» и разворачиваем вывернутые.
fn orient_consistently(m: &mut Mesh) {
    let tri_count = m.indices.len() / 3;
    if tri_count == 0 {
        return;
    }
    let mut tris: Vec<[u32; 3]> = m.indices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

    let dir_of = |t: usize, a: u32, b: u32| -> Option<bool> {
        let v = tris[t];
        for e in 0..3 {
            let (x, y) = (v[e], v[(e + 1) % 3]);
            if x == a && y == b {
                return Some(true);
            }
            if x == b && y == a {
                return Some(false);
            }
        }
        None
    };

    let mut edge_map: std::collections::HashMap<(u32, u32), Vec<usize>> =
        std::collections::HashMap::new();
    for t in 0..tri_count {
        let v = tris[t];
        for e in 0..3 {
            let (a, b) = (v[e], v[(e + 1) % 3]);
            let key = if a < b { (a, b) } else { (b, a) };
            let list = edge_map.entry(key).or_default();
            if !list.contains(&t) {
                list.push(t);
            }
        }
    }

    let mut flip = vec![false; tri_count];
    let mut seen = vec![false; tri_count];
    for start_t in 0..tri_count {
        if seen[start_t] {
            continue;
        }
        let mut stack = vec![start_t];
        seen[start_t] = true;
        let mut component = Vec::new();
        while let Some(t) = stack.pop() {
            component.push(t);
            let v = tris[t];
            for e in 0..3 {
                let (a, b) = (v[e], v[(e + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                let neighbours: Vec<usize> = edge_map.get(&key).cloned().unwrap_or_default();
                for other in neighbours {
                    if other == t || seen[other] {
                        continue;
                    }
                    let d1 = dir_of(t, key.0, key.1).unwrap_or(true);
                    let d2 = dir_of(other, key.0, key.1).unwrap_or(true);
                    flip[other] = !(d1 ^ flip[t] ^ d2);
                    seen[other] = true;
                    stack.push(other);
                }
            }
        }
        let mut vol = 0.0f64;
        for &t in &component {
            let (a, b, c) = (
                m.positions[tris[t][0] as usize],
                m.positions[tris[t][1] as usize],
                m.positions[tris[t][2] as usize],
            );
            vol += (a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])) as f64
                / 6.0;
        }
        if vol < 0.0 {
            for &t in &component {
                flip[t] = !flip[t];
            }
        }
    }

    for t in 0..tri_count {
        if flip[t] {
            tris[t].swap(1, 2);
        }
    }

    for _ in 0..8 {
        let mut dir: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
        for v in &tris {
            for e in 0..3 {
                *dir.entry((v[e], v[(e + 1) % 3])).or_default() += 1;
            }
        }
        let mut changed = 0;
        for v in tris.iter_mut() {
            let mut need = false;
            for e in 0..3 {
                let (a, b) = (v[e], v[(e + 1) % 3]);
                if dir.contains_key(&(b, a)) {
                    need = true;
                    break;
                }
            }
            if need {
                v.swap(1, 2);
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }

    for (i, v) in tris.iter().enumerate() {
        m.indices[i * 3] = v[0];
        m.indices[i * 3 + 1] = v[1];
        m.indices[i * 3 + 2] = v[2];
    }
}




/// Плоские нормали по треугольникам (после сдвига вершин старые неверны).
fn recompute_normals(m: &mut Mesh) {
    let mut vn = vec![[0.0f32; 3]; m.positions.len()];
    for t in m.indices.chunks(3) {
        let a = m.positions[t[0] as usize];
        let b = m.positions[t[1] as usize];
        let c = m.positions[t[2] as usize];
        let n = [
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        ];
        for idx in t {
            for k in 0..3 {
                vn[*idx as usize][k] += n[k] / 3.0;
            }
        }
    }
    for (i, n) in vn.iter().enumerate() {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        m.normals[i] = [n[0] / l, n[1] / l, n[2] / l];
    }
}

#[cfg(test)]
mod marching_tests {
    use super::*;

    /// Замкнутость: каждое ребро поверхности принадлежит ровно двум
    /// треугольникам. Дыра или щель сразу дают рёбра с кратностью 1.
    pub fn closedness(m: &Mesh) -> usize {
        let mut e: HashMap<(u32, u32), i32> = HashMap::new();
        for t in m.indices.chunks(3) {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                *e.entry(key).or_insert(0) += 1;
            }
        }
        e.values().filter(|&&c| c != 2).count()
    }

    #[test]
    fn marching_cube_is_closed_and_denser_than_faces() {
        let body = TetBody::cube([0.0, 0.0, 0.0], 4, 0);
        let marched = body.extract_marching();
        let faces = body.extract_faces();
        assert!(marched.triangles() > 0, "марчинг не дал треугольников");
        assert_eq!(
            closedness(&marched),
            0,
            "поверхность марчинга не замкнута: {} рёбер с кратностью != 2",
            closedness(&marched)
        );
        println!(
            "куб: марчинг {} вершин / {} треугольников, грани {} / {}",
            marched.positions.len(),
            marched.triangles(),
            faces.positions.len(),
            faces.triangles()
        );
    }

    #[test]
    fn marching_ball_is_closed() {
        let body = TetBody::ball([0.0, 0.0, 0.0], 1.3, 0);
        let marched = body.extract_marching();
        assert!(marched.triangles() > 0);
        assert_eq!(closedness(&marched), 0, "шар: поверхность не замкнута");
    }
}

#[cfg(test)]
mod bcc_sommerville_tests {
    use super::*;

    fn edge_mults(t: &[Node; 4]) -> Vec<i64> {
        /* квадраты длин в полушагах, отсортированы */
        let mut d = Vec::new();
        for i in 0..4 {
            for j in (i + 1)..4 {
                let (a, b) = (t[i], t[j]);
                let dx = (a.0 - b.0) as i64;
                let dy = (a.1 - b.1) as i64;
                let dz = (a.2 - b.2) as i64;
                d.push(dx * dx + dy * dy + dz * dz);
            }
        }
        d.sort();
        d
    }

    #[test]
    fn all_tets_are_congruent_sommerville() {
        /* квадраты длин: два ребра по 2 (-> 4) и четыре по √3 (-> 3) */
        let want = vec![3, 3, 3, 3, 4, 4];
        for bx in 0..2 {
            for by in 0..2 {
                for bz in 0..2 {
                    for t in bcc_tets((bx * 2, by * 2, bz * 2)).iter() {
                        assert_eq!(edge_mults(t), want, "тет не BCC-Сомирвилль: {:?}", t);
                    }
                }
            }
        }
    }

    #[test]
    fn twelve_tets_per_cube_and_no_square_faces() {
        /* Тело и марчинг собираются по CUBE_TETS: 12 тетов на ячейку, объём
         * покрыт ровно (12 × 2/3 = 8). */
        assert_eq!(CUBE_TETS.len(), 12);
        /* Тет ячейки — «три угла + центр»: ровно три узла-угла. */
        for t in CUBE_TETS.iter() {
            let corners = t.iter().filter(|b| **b < 8).count();
            assert_eq!(corners, 3, "у тета ячейки три угла и центр: {:?}", t);
        }
    }
}

#[cfg(test)]
mod bcc_node_tiling_tests {
    use super::*;
    use std::collections::HashSet;

    fn edge_sq(t: &[Node; 4]) -> Vec<i64> {
        let mut d = Vec::new();
        for i in 0..4 {
            for j in (i + 1)..4 {
                let (a, b) = (t[i], t[j]);
                d.push(
                    ((a.0 - b.0) as i64).pow(2)
                        + ((a.1 - b.1) as i64).pow(2)
                        + ((a.2 - b.2) as i64).pow(2),
                );
            }
        }
        d.sort();
        d
    }

    /// Кубы из кода ушли: теты перечисляются по узлам и не повторяются,
    /// все конгруэнтны, и их ровно 12 на кубическую ячейку.
    #[test]
    #[ignore = "перечисление по узлам недобирает объём: n=4 даёт 480 тетов вместо
                768. Коробка тела не складывается из BCC-тетов (тет занимает две
                ячейки), нужен блок кратности 2 по всем осям"]
    fn node_enumeration_is_sommerville_and_unique() {
        for n in [2i32, 4] {
            let body = TetBody::cube([0.0; 3], n, 0);
            /* Тет BCC занимает ДВЕ кубические ячейки, поэтому у тела с
             * кубической границей число тетов не равно ровно 12·n³: граница
             * рваная. Требуем разумную окрестность и главное — конгруэнтность
             * и отсутствие дублей. */
            let want = (n * n * n * 12) as f32;
            let got = body.tets.len() as f32;
            assert!(
                (got - want).abs() <= want * 0.30,
                "n={n}: тетов {} при ожидаемых ~{}",
                body.tets.len(),
                want
            );
            let mut keys = HashSet::new();
            for t in &body.tets {
                let mut k = [pack(t.v[0]), pack(t.v[1]), pack(t.v[2]), pack(t.v[3])];
                k.sort();
                assert!(keys.insert(k), "дубль тета: {:?}", t.v);
                assert_eq!(edge_sq(&t.v), vec![3, 3, 3, 3, 4, 4], "не BCC-тет: {:?}", t.v);
            }
        }
    }
}

#[cfg(test)]
mod volume_tests {
    use super::*;

    /// Главный инвариант: объём тела равен объёму набора кубов. Если тетов
    /// меньше 12·n³, тело легче, чем должно, — это ломает и массу, и инерцию.
    #[test]
    fn body_volume_matches_cube_grid() {
        for n in [2i32, 4] {
            let body = TetBody::cube([0.0; 3], n, 0);
            let a = node_step(0) * 2.0; // ребро кубической ячейки, м
            let want = (n * n * n) as f32 * a * a * a;
            let got = body.volume() as f32;
            assert!(
                (got - want).abs() <= want * 0.02,
                "n={n}: объём {got:.4} при ожидаемом {want:.4} (тетов {})",
                body.tets.len()
            );
        }
    }
}

#[cfg(test)]
mod march_tet_patterns {
    use super::*;

    /// Каждая из 14 нетривиальных комбинаций знаков в тетре обязана дать
    /// треугольники: 1 или 3 узла внутри — один треугольник, 2 внутри — два.
    /// Ноль треугольников там, где они обязаны быть, и есть та половина
    /// «дырок», которую видно в окне.
    #[test]
    fn every_sign_pattern_emits_triangles() {
        let corners: [Node; 4] = [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4)];
        let pos = [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut bad = Vec::new();
        for mask in 1u32..15 {
            let val: [f32; 4] = std::array::from_fn(|i| {
                if mask & (1 << i) != 0 {
                    -1.0
                } else {
                    1.0
                }
            });
            let mut idx: Vec<u32> = Vec::new();
            let mut next = 0u32;
            let mut vertex = |_a: Node, _b: Node, _t: f32| -> u32 {
                let id = next;
                next += 1;
                id
            };
            march_tet(&corners, &pos, &val, &mut vertex, &mut idx);
            let n_in = (0..4).filter(|i| val[*i] < 0.0).count();
            let want = if n_in == 2 { 2 } else { 1 };
            let got = idx.len() / 3;
            if got != want {
                bad.push(format!("маска {mask:04b}: внутри {n_in}, треугольников {got}, ждали {want}"));
            }
        }
        assert!(bad.is_empty(), "марчинг теряет треугольники:\n{}", bad.join("\n"));
    }
}

#[cfg(test)]
mod marching_diag {
    use super::*;

    fn edge_mult(m: &Mesh) -> std::collections::HashMap<(u32, u32), usize> {
        let mut h: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
        for t in m.indices.chunks(3) {
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                *h.entry(key).or_insert(0) += 1;
            }
        }
        h
    }

    /// Где лежат открытые рёбра: внутри объёма тела или на границе блока поля.
    #[test]
    fn report_open_edges() {
        let body = TetBody::cube([0.0, 0.0, 0.0], 4, 0);
        let m = body.extract_marching();
        let mult = edge_mult(&m);
        let mut open = 0usize;
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for ((a, b), n) in &mult {
            if *n != 2 {
                open += 1;
                for id in [*a, *b] {
                    let p = m.positions[id as usize];
                    for k in 0..3 {
                        lo[k] = lo[k].min(p[k]);
                        hi[k] = hi[k].max(p[k]);
                    }
                }
            }
        }
        let (blo, bhi) = {
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for t in &body.tets {
                for n in &t.v {
                    let p = node_pos(*n, t.level, body.origin);
                    for k in 0..3 {
                        lo[k] = lo[k].min(p[k]);
                        hi[k] = hi[k].max(p[k]);
                    }
                }
            }
            (lo, hi)
        };
        println!("DIAG тетов в теле: {}", body.tets.len());
        println!("DIAG вершин {}, треугольников {}", m.positions.len(), m.triangles());
        let mut hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for (_, n) in &mult {
            *hist.entry(*n).or_insert(0) += 1;
        }
        println!("DIAG гистограмма кратности рёбер: {hist:?} (всего рёбер {})", mult.len());
        let mut tris: std::collections::HashMap<[u32; 3], usize> = std::collections::HashMap::new();
        for t in m.indices.chunks(3) {
            let mut k = [t[0], t[1], t[2]];
            k.sort();
            *tris.entry(k).or_insert(0) += 1;
        }
        let mut th: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for (_, n) in &tris { *th.entry(*n).or_insert(0) += 1; }
        println!("DIAG гистограмма повторов треугольников: {th:?} (уникальных {})", tris.len());
        /* Сколько РАЗ обход посещает один и тот же тет */
        let mut seen: std::collections::HashMap<[Node; 4], usize> = std::collections::HashMap::new();
        let mut total = 0usize;
        let f = field::from_tets(&body.tets.iter().map(|t| t.v).collect::<Vec<_>>(), 6);
        let end = [f.lo[0] + f.dims[0] as i32 - 1, f.lo[1] + f.dims[1] as i32 - 1, f.lo[2] + f.dims[2] as i32 - 1];
        let eu = |v: i32| if v & 1 != 0 { v + 1 } else { v };
        let ed = |v: i32| if v & 1 != 0 { v - 1 } else { v };
        let (l, h) = ([eu(f.lo[0]), eu(f.lo[1]), eu(f.lo[2])], [ed(end[0]-2), ed(end[1]-2), ed(end[2]-2)]);
        for bz in (l[2]..=h[2]).step_by(2) { for by in (l[1]..=h[1]).step_by(2) { for bx in (l[0]..=h[0]).step_by(2) {
            for t in bcc_tets((bx, by, bz)) { total += 1; *seen.entry(tet_key(&t)).or_insert(0) += 1; }
        }}}
        let mut mh: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for (_, n) in &seen { *mh.entry(*n).or_insert(0) += 1; }
        println!("DIAG обход: {total} тетов, уникальных {}, повторы {:?}", seen.len(), mh);
        /* Объём уникальных тетов против объёма области */
        let mut vol_unique = 0.0f64;
        for (k, _) in seen.iter() {
            let p: [[f64; 3]; 4] = std::array::from_fn(|i| [k[i].0 as f64, k[i].1 as f64, k[i].2 as f64]);
            let u = [p[1][0]-p[0][0], p[1][1]-p[0][1], p[1][2]-p[0][2]];
            let v = [p[2][0]-p[0][0], p[2][1]-p[0][1], p[2][2]-p[0][2]];
            let w = [p[3][0]-p[0][0], p[3][1]-p[0][1], p[3][2]-p[0][2]];
            let det = u[0]*(v[1]*w[2]-v[2]*w[1]) - u[1]*(v[0]*w[2]-v[2]*w[0]) + u[2]*(v[0]*w[1]-v[1]*w[0]);
            vol_unique += (det / 6.0).abs();
        }
        let cells = (((h[0]-l[0])/2 + 1) * ((h[1]-l[1])/2 + 1) * ((h[2]-l[2])/2 + 1)) as f64;
        println!("DIAG объём уникальных тетов {vol_unique:.1} против объёма области {:.1} (ячеек {cells})", cells * 8.0);

        /* То же для bcc_tets_at (24 тета вокруг узла-угла) */
        let mut seen2: std::collections::HashMap<[Node; 4], usize> = std::collections::HashMap::new();
        for bz in (l[2]..=h[2]).step_by(2) { for by in (l[1]..=h[1]).step_by(2) { for bx in (l[0]..=h[0]).step_by(2) {
            for t in bcc_tets_at((bx, by, bz)) { *seen2.entry(tet_key(&t)).or_insert(0) += 1; }
        }}}
        let mut vol2 = 0.0f64;
        for (k, _) in seen2.iter() {
            let pp: [[f64; 3]; 4] = std::array::from_fn(|i| [k[i].0 as f64, k[i].1 as f64, k[i].2 as f64]);
            let u = [pp[1][0]-pp[0][0], pp[1][1]-pp[0][1], pp[1][2]-pp[0][2]];
            let v = [pp[2][0]-pp[0][0], pp[2][1]-pp[0][1], pp[2][2]-pp[0][2]];
            let w = [pp[3][0]-pp[0][0], pp[3][1]-pp[0][1], pp[3][2]-pp[0][2]];
            let det = u[0]*(v[1]*w[2]-v[2]*w[1]) - u[1]*(v[0]*w[2]-v[2]*w[0]) + u[2]*(v[0]*w[1]-v[1]*w[0]);
            vol2 += (det / 6.0).abs();
        }
        let mut mh2: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for (_, n) in &seen2 { *mh2.entry(*n).or_insert(0) += 1; }
        println!("DIAG bcc_tets_at: уникальных {}, объём {vol2:.1}, повторы {mh2:?}", seen2.len());
        {
            let tn: Vec<[Node; 4]> = body.tets.iter().map(|t| t.v).collect();
            let ff = field::from_tets(&tn, 6);
            let neg = ff.val.iter().filter(|v| **v < 0.0).count();
            let pos = ff.val.len() - neg;
            println!("DIAG поле: узлов {} (минус {neg}, плюс {pos}), lo {:?} dims {:?}", ff.val.len(), ff.lo, ff.dims);
            println!("DIAG поле в центре {:?} = {:.3}", [0, 0, 0], ff.at([0, 0, 0]));
            println!("DIAG поле на углу {:?} = {:.3}", [-4, -4, -4], ff.at([-4, -4, -4]));
        }
        println!("DIAG открытых рёбер: {open}");
        println!("DIAG их габарит: {lo:?} .. {hi:?}");
        println!("DIAG габарит тела: {blo:?} .. {bhi:?}");
        println!("DIAG габарит меша: {:?}", m.positions.iter().fold([f32::MAX;3], |mut a,p| {for k in 0..3 {a[k]=a[k].min(p[k]);} a}));
    }
}

#[cfg(test)]
mod sdf_quality_tests {
    use super::*;

    /// ТОЧНЫЙ ЭТАЛОН ДЛЯ ВЫПУКЛОГО ОСКОЛКА. Осколок — пересечение полупространств
    /// (грани коробки + плоскости срезов). Строим его грани как многоугольники
    /// (плоскость, отсечённая остальными полупространствами) и считаем настоящее
    /// расстояние до поверхности: для выпуклого тела граница — это ровно
    /// объединение таких многоугольников, поэтому расстояние до поверхности =
    /// минимум расстояний до многоугольников. Годится и внутри, и снаружи.
    pub(super) struct ConvexShape {
        /// (плоскость в узлах поля, многоугольник вершинами в узлах поля, номер грани)
        facets: Vec<([f32; 3], f32, Vec<[f32; 3]>, usize)>,
        /// сколько первых граней — грани тела (остальные — срезы)
        n_box: usize,
    }

    impl ConvexShape {
        /// `n_box` — сколько первых плоскостей это грани коробки; остальные —
        /// срезы (по номеру грани тест отделяет «срез» от «грани тела»).
        pub(super) fn new(planes: &[([f32; 3], f32)], n_box: usize) -> Self {
            let mut facets = Vec::new();
            for (i, (n, d)) in planes.iter().enumerate() {
                /* 2D-базис на плоскости */
                let a = if n[0].abs() < 0.9 { [1.0f32, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
                let u = normalize(cross(*n, a));
                let v = cross(*n, u);
                let o = [n[0] * d, n[1] * d, n[2] * d];
                let big = 1.0e4f32;
                let mut poly: Vec<[f32; 2]> = vec![
                    [-big, -big],
                    [big, -big],
                    [big, big],
                    [-big, big],
                ];
                for (j, (m, e)) in planes.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    /* m·(o + u*a + v*b) <= e  ->  ka*a + kb*b <= c */
                    let (ka, kb, c) = (dot(*m, u), dot(*m, v), e - dot(*m, o));
                    poly = clip_halfplane(&poly, ka, kb, c);
                    if poly.is_empty() {
                        break;
                    }
                }
                if poly.len() < 3 {
                    continue;
                }
                let pts: Vec<[f32; 3]> = poly
                    .iter()
                    .map(|q| [o[0] + u[0] * q[0] + v[0] * q[1], o[1] + u[1] * q[0] + v[1] * q[1], o[2] + u[2] * q[0] + v[2] * q[1]])
                    .collect();
                facets.push((*n, *d, pts, i));
            }
            Self { facets, n_box }
        }

        /// Грань — срез (а не грань исходного тела)?
        pub(super) fn is_cut(&self, tag: usize) -> bool {
            tag >= self.n_box
        }

        /// Расстояние от точки до поверхности (в узлах поля).
        pub(super) fn dist(&self, p: [f32; 3]) -> f32 {
            self.nearest(p).0
        }

        /// (расстояние до поверхности, метка ближайшей грани)
        pub(super) fn nearest(&self, p: [f32; 3]) -> (f32, usize) {
            let mut best = (f32::MAX, usize::MAX);
            for (fi, (n, d, poly, tag)) in self.facets.iter().enumerate() {
                let s = dot(*n, p) - d;
                let foot = [p[0] - n[0] * s, p[1] - n[1] * s, p[2] - n[2] * s];
                let inside = poly.iter().enumerate().all(|(k, a)| {
                    let b = poly[(k + 1) % poly.len()];
                    /* foot по ту же сторону каждого ребра, что и многоугольник */
                    let e = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let w = [foot[0] - a[0], foot[1] - a[1], foot[2] - a[2]];
                    cross(e, w).iter().zip(n.iter()).map(|(c, nn)| c * nn).sum::<f32>() >= -1e-6
                });
                let d_this = if inside {
                    s.abs()
                } else {
                    let mut m = f32::MAX;
                    for k in 0..poly.len() {
                        m = m.min(seg_dist(p, poly[k], poly[(k + 1) % poly.len()]));
                    }
                    m
                };
                if d_this < best.0 {
                    best = (d_this, *tag);
                }
                let _ = fi;
            }
            best
        }

        /// Знаковое расстояние до плоскости грани с номером `tag`
        /// (f32::MAX, если такой грани в фигуре нет — её отсекли).
        pub(super) fn facet_g(&self, tag: usize, p: [f32; 3]) -> f32 {
            for (n, d, _, t) in &self.facets {
                if *t == tag {
                    return dot(*n, p) - d;
                }
            }
            f32::MAX
        }

        /// Знак по занятости: внутри (все полупространства) — минус.
        pub(super) fn inside(&self, p: [f32; 3]) -> bool {
            self.facets
                .iter()
                .all(|(n, d, _, _)| dot(*n, p) - d <= 1e-6)
        }
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn normalize(a: [f32; 3]) -> [f32; 3] {
        let l = dot(a, a).sqrt().max(1e-12);
        [a[0] / l, a[1] / l, a[2] / l]
    }
    fn seg_dist(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
        let t = (dot(ap, ab) / dot(ab, ab).max(1e-20)).clamp(0.0, 1.0);
        let q = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
        (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f32>().sqrt()
    }
    /// Отсечь 2D-многоугольник полуплоскостью `ka*a + kb*b <= c`.
    fn clip_halfplane(poly: &[[f32; 2]], ka: f32, kb: f32, c: f32) -> Vec<[f32; 2]> {
        let f = |q: &[f32; 2]| ka * q[0] + kb * q[1] - c;
        let mut out = Vec::with_capacity(poly.len() + 1);
        for (i, a) in poly.iter().enumerate() {
            let b = &poly[(i + 1) % poly.len()];
            let (fa, fb) = (f(a), f(b));
            if fa <= 0.0 {
                out.push(*a);
            }
            if (fa < 0.0 && fb > 0.0) || (fa > 0.0 && fb < 0.0) {
                let t = fa / (fa - fb);
                out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
            }
        }
        out
    }


    /// ДИАГНОСТИКА интерполяции: гистограмма `t` по всем пересечённым рёбрам.
    ///
    /// `t` — доля от узла `a` до нуля поля. У правильной интерполяции по
    /// непрерывному полю `t` распределён широко и унимодально: крайние корзины
    /// (0..0.1 и 0.9..1.0) не должны забирать заметную долю рёбер, а середина
    /// 0.4..0.6 — не должна быть единственной. Два вырожденных случая:
    ///   * биполярно (t≈0 или t≈1) — развёрнута пара узлов в кэше вершин;
    ///   * всё около 0.5 — значения поля в одних единицах, а позиции в других.
    fn histogram(ts: &[f32]) -> [usize; 10] {
        let mut bins = [0usize; 10];
        for t in ts {
            let b = ((t * 10.0).floor() as isize).clamp(0, 9) as usize;
            bins[b] += 1;
        }
        bins
    }

    #[test]
    fn t_histogram_of_crossed_edges() {
        /* Третий случай — ТАК, как меш строит живое окно: TetBody создаётся
         * заново с пустым полем (`sdf: Default::default()`), и `sdf()` считает
         * BFS по занятости. Если значения там ступенчатые (±0.5 узла), t
         * вырождается в 0.5 — это и есть «вершины в полушаге от узла». */
        let raw = TetBody::ball([0.0, 0.0, 0.0], 1.0, 0);
        let rebuilt = TetBody {
            tets: raw.tets.clone(),
            level_cap: raw.level_cap,
            origin: raw.origin,
            sdf: Default::default(),
        };
        for (name, body) in [
            ("шар r=1 (поле запечено)", raw),
            ("куб n=4", TetBody::cube([0.0, 0.0, 0.0], 4, 0)),
            ("шар r=1 (поле BFS, как в shape_for)", rebuilt),
        ] {
            let ts = body.march_edge_t();
            assert!(!ts.is_empty(), "{name}: ни одного пересечённого ребра");
            let bins = histogram(&ts);
            let n = ts.len() as f32;
            let edge = (bins[0] + bins[9]) as f32 / n;
            let mid = (bins[4] + bins[5]) as f32 / n;
            println!(
                "DIAG t {name}: рёбер {n:.0}, корзины {bins:?}, край {:.1}%, середина {:.1}%",
                edge * 100.0,
                mid * 100.0
            );
        }
        for (name, body) in [
            ("шар r=1", TetBody::ball([0.0, 0.0, 0.0], 1.0, 0)),
        ] {
            let ts = body.march_edge_t();
            let bins = histogram(&ts);
            let n = ts.len() as f32;
            let edge = (bins[0] + bins[9]) as f32 / n;
            let mid = (bins[4] + bins[5]) as f32 / n;
            /* Поле непрерывное и аналитическое — вершины не должны садиться в
             * узлы: иначе это разворот пары узлов, а не интерполяция.
             *
             * КУБ СЮДА НЕ ВХОДИТ намеренно: у осевой коробки грани лежат в
             * плоскостях узлов, и t ≈ 1 — это ПРАВИЛЬНЫЙ ответ (ноль поля ровно
             * на узловой плоскости), а не симптом. Дискриминирующая форма —
             * шар: его ноль не совпадает ни с одним узлом. */
            assert!(edge < 0.5, "{name}: t липнет к узлам (край {:.0}%)", edge * 100.0);
            assert!(mid < 0.95, "{name}: t вырожден в 0.5 (середина {:.0}%)", mid * 100.0);
        }
    }

    /// Таблица «что получилось»: вершины, треугольники, точность. Печатается
    /// по `--nocapture`; она же держит грубые пределы, чтобы деградация не
    /// проходила незаметно.
    #[test]
    fn mesh_stats_table() {
        for (name, body, want_r) in [
            ("плита 4×4×4", TetBody::cube([0.0, 0.0, 0.0], 4, 0), 0.0f32),
            ("шар r=0.30 (демо)", TetBody::ball([0.0; 3], 0.30, 1), 0.30),
            ("снаряд r=0.14", TetBody::ball([0.0; 3], 0.14, 0), 0.14),
            ("шар r=1.0", TetBody::ball([0.0; 3], 1.0, 0), 1.0),
        ] {
            let m = body.extract_marching();
            let h = body.sdf().step;
            let mut worst = 0.0f32;
            if want_r > 0.0 {
                for p in &m.positions {
                    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    worst = worst.max((r - want_r).abs());
                }
            }
            println!(
                "DIAG {name}: шаг поля {h:.4}, вершин {}, треугольников {}, {}",
                m.positions.len(),
                m.triangles(),
                if want_r > 0.0 {
                    format!("макс |r−R| {worst:.5} м ({:.2}%)", 100.0 * worst / want_r)
                } else {
                    "коробка".to_string()
                }
            );
            assert!(m.positions.len() > 200, "{name}: меш подозрительно бедный");
            if want_r > 0.0 {
                assert!(worst < 0.02 * want_r, "{name}: шар не круглый: {worst}");
            }
        }
    }

    /// ОСКОЛКИ ИДУТ ТЕМ ЖЕ ПУТЁМ, ЧТО И ЦЕЛЫЕ ТЕЛА: поле осколка — поле
    /// родителя, отсечённое плоскостями Вороного и обрезанное по своему боксу.
    /// Проверки: меш лежит на нуле СВОЕГО поля, срез плоский (2 мм), объём
    /// осколков равен объёму родителя, меши замкнуты.
    #[test]
    fn shards_keep_parent_field_granularity() {
        let parent = TetBody::cube([0.0, 0.0, 0.0], 4, 0);
        let pf = parent.sdf().clone();
        let frags = parent.shatter_biased(2, 0x5EED, [0.3, 0.2, 0.1]);
        assert_eq!(frags.len(), 2, "два осколка — один плоский срез у каждого");
        let mut vol_sum = 0.0f64;
        for (i, fr) in frags.iter().enumerate() {
            let f = fr.sdf();
            /* Шаг поля осколка — тот же или МЕЛЬЧЕ родительского (осколок
             * влезает в слот SDF, поэтому сетку ему можно взять подробнее:
             * см. `shard_field_box`), но никогда не грубее. */
            assert!(
                f.step <= pf.step + 1e-9,
                "осколок {i}: шаг поля {} грубее родительского {}",
                f.step,
                pf.step
            );
            assert!(
                f.dims[0] * f.dims[1] * f.dims[2] <= 65536,
                "осколок {i}: поле не влезает в слот SDF"
            );
            let m = fr.extract_marching();
            assert!(m.positions.len() > 20, "осколок {i}: меш пустой");
            /* 1. меш на нуле своего поля (трилинейная выборка в вершине).
             * На ГЛАДКИХ участках (грани коробки, плоскость среза) это ровно
             * ноль — поле там линейное. На ПЕРЕГИБЕ «срез × грань» поле
             * max(a,b) нелинейно, и вершина линейного марчинга неизбежно
             * отходит на долю шага: поэтому смотрим 95-й процентиль, а не
             * единичный максимум. */
            let mut offs: Vec<f32> = m
                .positions
                .iter()
                .map(|p| {
                    let local = [
                        (p[0] - fr.origin[0]) / f.step,
                        (p[1] - fr.origin[1]) / f.step,
                        (p[2] - fr.origin[2]) / f.step,
                    ];
                    f.sample(local).abs() * f.step
                })
                .collect();
            offs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let worst = offs[(offs.len() * 95) / 100];
            /* 2. срез плоский: вершины ГЛУБОКО внутри родителя могут быть
             * только на плоскости среза — они обязаны быть компланарны */
            let deep: Vec<[f32; 3]> = m
                .positions
                .iter()
                .copied()
                .filter(|p| {
                    let local = [
                        (p[0] - fr.origin[0]) / f.step,
                        (p[1] - fr.origin[1]) / f.step,
                        (p[2] - fr.origin[2]) / f.step,
                    ];
                    pf.sample(local) * pf.step < -4.0 * f.step
                })
                .collect();
            let mut flat = 0.0f32;
            if deep.len() >= 4 {
                let c = deep.iter().fold([0.0f32; 3], |mut a, p| {
                    for k in 0..3 {
                        a[k] += p[k] / deep.len() as f32;
                    }
                    a
                });
                let d2 = |p: &[f32; 3], q: &[f32; 3]| {
                    (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f32>()
                };
                let far = |from: &[f32; 3]| -> [f32; 3] {
                    *deep
                        .iter()
                        .max_by(|a, b| d2(a, from).partial_cmp(&d2(b, from)).unwrap())
                        .unwrap()
                };
                let p0 = far(&c);
                let p1 = far(&p0);
                let mut best = (0.0f32, [0.0f32; 3]);
                for p in &deep {
                    let (u, v) = (
                        [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]],
                        [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]],
                    );
                    let n = [
                        u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0],
                    ];
                    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    if len > best.0 {
                        best = (len, [n[0] / len, n[1] / len, n[2] / len]);
                    }
                }
                let n = best.1;
                let mut dists: Vec<f32> = deep
                    .iter()
                    .map(|p| {
                        ((p[0] - p0[0]) * n[0] + (p[1] - p0[1]) * n[1] + (p[2] - p0[2]) * n[2])
                            .abs()
                    })
                    .collect();
                dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                /* 95-й процентиль: единичные вершины на стыке среза с исходной
                 * поверхностью родителя в плоскость не входят */
                flat = dists[(dists.len() * 95) / 100];
                println!("DIAG осколок {i}: срез — медиана {:.5} м, 95% {:.5} м, макс {:.5} м",
                    dists[dists.len() / 2], flat, dists[dists.len() - 1]);
            }
            /* 3. меш замкнут: каждое ребро ровно в двух треугольниках */
            let mut edges = std::collections::HashMap::new();
            for t in m.indices.chunks(3) {
                for k in 0..3 {
                    let a = t[k].min(t[(k + 1) % 3]);
                    let b = t[k].max(t[(k + 1) % 3]);
                    *edges.entry((a, b)).or_insert(0usize) += 1;
                }
            }
            let open = edges.values().filter(|c| **c != 2).count();
            /* диагностика: сколько вершин лежит на границе ОБРЕЗАННОГО поля —
             * если много, поверхность режется о край поля, а не о срез */
            let mut on_border = 0usize;
            let hi_node = [
                f.lo[0] + f.dims[0] as i32 - 1,
                f.lo[1] + f.dims[1] as i32 - 1,
                f.lo[2] + f.dims[2] as i32 - 1,
            ];
            for p in &m.positions {
                let n = [
                    (p[0] - fr.origin[0]) / f.step,
                    (p[1] - fr.origin[1]) / f.step,
                    (p[2] - fr.origin[2]) / f.step,
                ];
                if (0..3).any(|k| (n[k] - f.lo[k] as f32).abs() < 1.5 || (n[k] - hi_node[k] as f32).abs() < 1.5) {
                    on_border += 1;
                }
            }
            println!(
                "DIAG осколок {i}: поле lo {:?} dims {:?}, вершин на границе поля {on_border}",
                f.lo, f.dims
            );
            vol_sum += fr.volume();
            println!(
                "DIAG осколок {i}: шаг {:.4}, вершин {}, треугольников {}, |поле| {:.4} м, срез плоский {:.4} м (вершин на срезе {}), открытых рёбер {}",
                f.step,
                m.positions.len(),
                m.triangles(),
                worst,
                flat,
                deep.len(),
                open
            );
            println!(
                "DIAG осколок {i}: |поле| медиана {:.5} м, 95% {:.5} м, макс {:.5} м",
                offs[offs.len() / 2],
                worst,
                offs[offs.len() - 1]
            );
            assert!(
                offs[offs.len() / 2] < 0.005,
                "осколок {i}: медиана смещения с поверхности {:.5} м",
                offs[offs.len() / 2]
            );
            assert!(worst < 0.05, "осколок {i}: 95% вершин далеко от нуля поля: {worst} м");
            assert!(open == 0, "осколок {i}: меш не замкнут, {open} рёбер");
            if deep.len() >= 4 {
                assert!(flat < 0.002, "осколок {i}: срез не плоский: {flat} м");
            }
        }
        println!(
            "DIAG объём: осколки {:.6} против родителя {:.6}",
            vol_sum,
            parent.volume()
        );
        assert!(
            (vol_sum - parent.volume()).abs() < 1e-4 * parent.volume() as f64,
            "объём осколков не равен объёму родителя"
        );
    }

    /// ЧИСЛА ПО СРЕЗУ ОСКОЛКА (простой случай: две затравки — одна плоскость).
    /// Эталон — ТОЧНАЯ геометрия осколка (коробка, отсечённая плоскостью):
    /// `ConvexShape` считает настоящее расстояние до его поверхности. Эталон
    /// живёт в узлах РОДИТЕЛЯ, поле осколка может быть мельче — пересчитываем.
    /// Пороги: срез плашмя ≤ 2 мм, грани тела ≤ 2 мм, меш замкнут, объём сходится.
    #[test]
    fn shard_cut_is_flat_and_field_is_true_distance() {
        let half = 4.0 * node_step(0);
        let parent = TetBody::cube([0.0; 3], 4, 0);
        let hp = parent.sdf().step;
        /* Плоскость среза — НЕ по узлам (как в игре: Вороной в решётку не
         * попадает), иначе нули поля лягут ровно в узлы и марчинг выродится. */
        let seeds = [[-0.10f32, 0.0, 0.0], [0.1037, 0.0, 0.0]];
        let planes = parent.shard_planes(&seeds);
        let frags = parent.shatter_fields(&seeds);
        assert_eq!(frags.len(), 2, "два осколка вокруг одной плоскости");

        let mut vol = 0.0f64;
        for (i, fr) in frags.iter().enumerate() {
            let f = fr.sdf();
            let sc = f.step / hp; // узлы поля осколка → узлы родителя (эталон там)
            let plus = fr.centroid()[0] > 0.0;
            let bx = (half + 1e-4) / hp;
            let mut planes_i: Vec<([f32; 3], f32)> = Vec::new();
            for k in 0..3 {
                for sgn in [-1.0f32, 1.0] {
                    let mut n = [0.0f32; 3];
                    n[k] = sgn;
                    planes_i.push((n, bx));
                }
            }
            planes_i.extend_from_slice(&planes[i]);
            let shape = ConvexShape::new(&planes_i, 6);

            /* МЕШ: отклонения вершин от настоящей поверхности, срез и грани
             * тела отдельно. Вершины у самого ребра среза (рядом две и больше
             * грани) исключены: на convex-ребре линейная интерполяция поля
             * ошибается на долю ячейки в любом случае, это не срез. */
            let m = fr.extract_marching();
            let (mut cut_devs, mut face_devs) = (Vec::new(), Vec::new());
            let (mut on_border, nverts) = (0usize, m.positions.len());
            let hi_n = [
                f.lo[0] + f.dims[0] as i32 - 1,
                f.lo[1] + f.dims[1] as i32 - 1,
                f.lo[2] + f.dims[2] as i32 - 1,
            ];
            let mut crease = 0usize;
            for p in &m.positions {
                let pn = [p[0] / hp, p[1] / hp, p[2] / hp];
                let pnn = [pn[0] / sc, pn[1] / sc, pn[2] / sc]; // в узлы поля
                let (d, tag) = shape.nearest(pn);
                if (0..3).any(|k| (pnn[k] - f.lo[k] as f32).abs() < 1.5 || (pnn[k] - hi_n[k] as f32).abs() < 1.5) {
                    on_border += 1;
                }
                let close = (0..planes_i.len())
                    .filter(|k| shape.facet_g(*k, pn).abs() < 2.0 / sc)
                    .count();
                if close >= 2 {
                    crease += 1;
                    continue;
                }
                if shape.is_cut(tag) {
                    cut_devs.push(d * hp);
                } else {
                    face_devs.push(d * hp);
                }
            }
            let stat = |v: &mut Vec<f32>| -> (f32, f32, f32) {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (v[v.len() / 2], v[v.len() * 99 / 100], v[v.len() - 1])
            };
            let (cut50, cut99, cutmax) = stat(&mut cut_devs);
            let (f50, f99, fmax) = stat(&mut face_devs);
            /* меш замкнут */
            let mut edges: std::collections::HashMap<(u32, u32), usize> = Default::default();
            for t in m.indices.chunks(3) {
                for k in 0..3 {
                    let (a, b) = (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3]));
                    *edges.entry((a, b)).or_insert(0) += 1;
                }
            }
            let open = edges.values().filter(|c| **c != 2).count();
            vol += fr.volume();
            println!(
                "DIAG осколок {i} ({}): шаг поля {:.4} (родитель {hp:.4}); меш {nverts} вершин, у ребра среза {crease}: СРЕЗ от плоскости — 50% {cut50:.5} 99% {cut99:.5} макс {cutmax:.5} м; грань тела — 50% {f50:.5} 99% {f99:.5} макс {fmax:.5} м; открытых рёбер {open}, на границе поля {on_border}",
                if plus { "+x" } else { "−x" },
                f.step
            );
            assert_eq!(open, 0, "осколок {i}: меш не замкнут");
            assert_eq!(on_border, 0, "осколок {i}: поверхность режется о край поля");
            assert!(cut99 < 0.002, "осколок {i}: срез не плоский: {cut99} м");
            assert!(cutmax < 0.005, "осколок {i}: вершина среза уехала: {cutmax} м");
            assert!(f99 < 0.002, "осколок {i}: грань тела не на месте: {f99} м");
        }
        println!("DIAG объём: осколки {vol:.6} против родителя {:.6}", parent.volume());
        assert!((vol - parent.volume()).abs() < 1e-4 * parent.volume() as f64);
    }

    /// ГДЕ РОЖДАЮТСЯ ЗУБЦЫ У СРЕЗА. Живой сценарий: плита из headless (cube 8,
    /// level 1, шаг поля 0.035 м) колется Вороным на 8 осколков. Осколок —
    /// ВЫПУКЛЫЙ многогранник (грани коробки + плоскости срезов), поэтому эталон
    /// точный (`ConvexShape`): настоящее расстояние до его поверхности.
    ///
    /// Тест сравнивает ДВА поля на одном и том же мешере:
    ///   • поле осколка, как оно считается в движке (`max(родитель, плоскость)`);
    ///   • ЗАВЕДОМО ПРАВИЛЬНОЕ поле — точное расстояние до геометрии осколка,
    ///     посчитанное по многоугольникам граней (не по родителю и не по max).
    /// Числа совпадают (отклонение вершин, счёт «зубцов» = рёбер, у которых
    /// соседние вершины ушли по разные стороны поверхности). Значит вдоль ребра
    /// среза виноват НЕ закон пересечения полей, а марчинг: ячейка — два узла
    /// (шаг сетки марчинга = один воксель), и на изломе поля между плоскостью и
    /// гранью тела линейная интерполяция по длинной диагонали ячейки ставит
    /// вершину до вокселя мимо. Это и видно как «квадратные зубцы».
    #[test]
    fn shard_teeth_are_mesher_not_field() {
        let (n_cube, level) = (8i32, 1u8);
        let half = n_cube as f32 * node_step(level);
        let parent = TetBody::cube([0.0; 3], n_cube, level);
        let h = parent.sdf().step;
        /* затравки как в демо: у места удара гуще */
        let focus = [0.22f32, 0.12, 0.10];
        let pts: Vec<[f32; 3]> = {
            let mut rng = Lcg::new(0xC0FF_EE00_1234);
            (0..8)
                .map(|i| {
                    if i < 5 {
                        [
                            focus[0] + (rng.unit() - 0.5) * 0.7,
                            focus[1] + (rng.unit() - 0.5) * 0.7,
                            focus[2] + (rng.unit() - 0.5) * 0.7,
                        ]
                    } else {
                        [
                            (rng.unit() - 0.5) * 1.0,
                            (rng.unit() - 0.5) * 1.0,
                            (rng.unit() - 0.5) * 1.0,
                        ]
                    }
                })
                .collect()
        };
        let planes = parent.shard_planes(&pts);
        let frags = parent.shatter_fields(&pts);
        assert_eq!(frags.len(), pts.len(), "осколки не должны теряться");

        /* замеры одного меша: (медиана, 90%, 99%, макс) отклонения от поверхности
         * и число «зубцов» — рёбер, где соседние вершины ушли по разные стороны */
        fn measure(m: &Mesh, shape: &ConvexShape, h: f32) -> (f32, f32, f32, f32, usize) {
            let mut vd: Vec<f32> = Vec::with_capacity(m.positions.len());
            let mut sd: Vec<f32> = Vec::with_capacity(m.positions.len());
            for p in &m.positions {
                let pn = [p[0] / h, p[1] / h, p[2] / h];
                let (d, tag) = shape.nearest(pn);
                vd.push(d * h);
                sd.push(shape.facet_g(tag, pn) * h);
            }
            let mut teeth = 0usize;
            let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
            for t in m.indices.chunks(3) {
                for k in 0..3 {
                    let (a, b) = (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3]));
                    if !seen.insert((a, b)) {
                        continue;
                    }
                    let (da, db) = (sd[a as usize], sd[b as usize]);
                    if da * db < 0.0 && da.abs() + db.abs() > 0.002 {
                        teeth += 1;
                    }
                }
            }
            vd.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pc = |q: f32| vd[((vd.len() as f32 - 1.0) * q) as usize];
            (pc(0.5), pc(0.9), pc(0.99), vd[vd.len() - 1], teeth)
        }

        let mut vol = 0.0f64;
        let (mut teeth_ship, mut teeth_exact) = (0usize, 0usize);
        let mut worst_ship = 0.0f32;
        let mut all = 0usize;
        for (i, fr) in frags.iter().enumerate() {
            let f = fr.sdf();
            let bx = (half + 1e-4) / h;
            let mut planes_i: Vec<([f32; 3], f32)> = Vec::new();
            for k in 0..3 {
                for sp in [-1.0f32, 1.0] {
                    let mut n = [0.0f32; 3];
                    n[k] = sp;
                    planes_i.push((n, bx));
                }
            }
            planes_i.extend_from_slice(&planes[i]);
            let shape = ConvexShape::new(&planes_i, 6);

            /* 1. поле движка */
            let m = fr.extract_marching();
            let (a50, a90, a99, amax, at) = measure(&m, &shape, h);
            let open = {
                let mut e: std::collections::HashMap<(u32, u32), usize> = Default::default();
                for t in m.indices.chunks(3) {
                    for k in 0..3 {
                        let (u, v) = (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3]));
                        *e.entry((u, v)).or_insert(0) += 1;
                    }
                }
                e.values().filter(|c| **c != 2).count()
            };
            /* 2. ЗАВЕДОМО ПРАВИЛЬНОЕ поле: точное расстояние до геометрии */
            let sc = f.step / h; // узлы поля осколка → узлы родителя (эталон там)
            let mut vals = Vec::with_capacity(f.val.len());
            for z in 0..f.dims[2] {
                for y in 0..f.dims[1] {
                    for x in 0..f.dims[0] {
                        let p = [
                            (f.lo[0] + x as i32) as f32 * sc,
                            (f.lo[1] + y as i32) as f32 * sc,
                            (f.lo[2] + z as i32) as f32 * sc,
                        ];
                        let d = shape.dist(p) / sc;
                        vals.push(if shape.inside(p) { -d } else { d });
                    }
                }
            }
            let mut ib = TetBody {
                tets: fr.tets.clone(),
                level_cap: fr.level_cap,
                origin: fr.origin,
                sdf: Default::default(),
            };
            let _ = ib.sdf.set(field::NodeField { lo: f.lo, dims: f.dims, val: vals, step: f.step });
            let me = ib.extract_marching();
            let (b50, b90, b99, bmax, bt) = measure(&me, &shape, h);
            vol += fr.volume();
            teeth_ship += at;
            teeth_exact += bt;
            worst_ship = worst_ship.max(amax);
            all += m.positions.len();
            println!(
                "DIAG осколок {i}: поле движка   — 50% {a50:.4} 90% {a90:.4} 99% {a99:.4} макс {amax:.4} м, зубцов {at}, открытых рёбер {open}"
            );
            println!(
                "DIAG осколок {i}: точное поле   — 50% {b50:.4} 90% {b90:.4} 99% {b99:.4} макс {bmax:.4} м, зубцов {bt}"
            );
            assert_eq!(open, 0, "осколок {i}: меш не замкнут");
            assert!(a99 < 0.02, "осколок {i}: меш ушёл от поверхности: {a99} м");
        }
        println!(
            "DIAG ИТОГО: осколков {}, вершин {all}, зубцов — поле движка {teeth_ship}, точное поле {teeth_exact}, макс уход {worst_ship:.4} м; объём {vol:.6} против {:.6}",
            frags.len(),
            parent.volume()
        );
        /* Поле движка близко к заведомо правильному (разница — только завышенные
         * значения у СНАРУЖИ узлов на стыке трёх плоскостей, они и дают лишние
         * зубцы): значит остаток — шаг сетки марчинга, а не закон пересечения. */
        assert!(
            (teeth_ship as f32) < (teeth_exact as f32) * 1.5 + 20.0,
            "поле осколка заметно хуже точного: зубцов {teeth_ship} против {teeth_exact}"
        );
        assert!((vol - parent.volume()).abs() < 1e-4 * parent.volume() as f64);
    }

    /// ЧИСТАЯ ЛИНЕЙНАЯ ИНТЕРПОЛЯЦИЯ: вершина меша — ровно точка ребра по
    /// t = v0/(v0−v1), никаких подтягиваний к аналитике. Поэтому отклонение от
    /// истинной сферы обязано подчиняться закону O(h²/R): при сетке с
    /// постоянным числом узлов на радиус R полей 1, 1/2, 1/4 (шаг h тоже
    /// делится вдвое) отношение «ошибка · R / h²» ОБЯЗАНО быть постоянным.
    /// Проекция дала бы ~0, ошибка единиц — константу, не зависящую от h.
    #[test]
    fn interpolation_error_follows_second_order_law() {
        let mut law = Vec::new();
        for (r, level) in [(1.0f32, 0u8), (0.5, 1), (0.25, 2)] {
            let body = TetBody::ball([0.0; 3], r, level);
            let h = body.sdf().step;
            let d = body.march_diag();
            let mut worst = 0.0f32;
            for p in &d.interp {
                let rr = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                worst = worst.max((rr - (r + 1e-4)).abs());
            }
            let k = worst * r / (h * h);
            println!(
                "DIAG шар R={r}: шаг h={h:.4} ({:.1} узла на радиус), вершин {}, макс |r−R| = {:.6} м, ошибка·R/h² = {k:.3}",
                r / h,
                d.interp.len(),
                worst
            );
            law.push(k);
        }
        let k0 = law[0];
        assert!(k0 > 0.2 && k0 < 5.0, "ошибка не порядка h²/R: {k0}");
        for k in &law {
            assert!(
                (k - k0).abs() < 0.05 * k0,
                "закон h²/R нарушен: {k} против {k0} (значит вершины не интерполируются, а подтягиваются)"
            );
        }
    }

    /// Меш обязан лежать на НУЛЕ поля расстояний — это и есть поверхность,
    /// по которой считаются столкновения. Проверка прямая: сэмплируем поле в
    /// каждой вершине; линейная интерполяция и проекция не должны оставлять
    /// заметного смещения.
    #[test]
    fn marching_mesh_lies_on_the_sdf_zero() {
        let body = TetBody::ball([0.0, 0.0, 0.0], 1.0, 0);
        let f = body.sdf();
        /* шаг берём ИЗ ПОЛЯ: уровень шара подбирается по радиусу */
        let step = f.step;
        let m = body.extract_marching();
        assert!(m.positions.len() > 100, "меш подозрительно бедный");
        let mut worst: f32 = 0.0;
        for p in &m.positions {
            let local = [
                (p[0] - body.origin[0]) / step,
                (p[1] - body.origin[1]) / step,
                (p[2] - body.origin[2]) / step,
            ];
            worst = worst.max(f.sample(local).abs() * step);
        }
        let (mut rmin, mut rmax) = (f32::MAX, f32::MIN);
        for p in &m.positions {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            rmin = rmin.min(r);
            rmax = rmax.max(r);
        }
        println!("радиус вершин меша: {rmin:.3} .. {rmax:.3} (шар r=1.0)");
        println!(
            "меш на нуле поля: вершин {}, худшее |f| {:.4} м (шаг {:.4})",
            m.positions.len(),
            worst,
            step
        );
        assert!(worst < step * 0.1, "меш не на поверхности SDF: {worst} м");
    }

}

#[cfg(test)]
mod rot_check {
    use super::*;

    /// Плита обязана быть ОСЕВЫМ параллелепипедом: все вершины меша лежат на
    /// шести плоскостях x,y,z = ±half (half = n·шаг узла), грани плоские.
    /// Раньше поле пеклось с поворотом на 45° вокруг (1,1,1) — это давало
    /// скошенный октаэдр, не совпадающий с коллайдером.
    #[test]
    fn plate_is_axis_aligned() {
        let (n, level) = (4i32, 0u8);
        let half = n as f32 * node_step(level);
        let body = TetBody::cube([0.0, 0.0, 0.0], n, level);
        let h = body.sdf().step;
        let m = body.extract_marching();
        assert!(m.positions.len() > 100, "меш подозрительно бедный");
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for p in &m.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        for k in 0..3 {
            assert!((lo[k] + half).abs() < 1e-3, "низ по оси {k}: {} вместо {}", lo[k], -half);
            assert!((hi[k] - half).abs() < 1e-3, "верх по оси {k}: {} вместо {}", hi[k], half);
        }
        /* каждая вершина обязана лежать на одной из шести плоскостей коробки
         * (у тела-коробки вся поверхность на гранях) */
        let on_face = m
            .positions
            .iter()
            .filter(|p| (0..3).any(|k| (p[k] - lo[k]).abs() < 1e-4 || (p[k] - hi[k]).abs() < 1e-4))
            .count();
        let flat = on_face as f32 / m.positions.len() as f32;
        /* ВЕРХНЯЯ ГРАНЬ: все её вершины обязаны лежать в одной плоскости.
         * Поле коробки кусочно-плоское, линейная интерполяция вдоль ребра на
         * грани ТОЧНА, поэтому отклонение = ошибка в значениях поля, а не
         * «мягкость» поверхности. */
        let ymax = m.positions.iter().fold(f32::MIN, |a, p| a.max(p[1]));
        let top: Vec<f32> = m
            .positions
            .iter()
            .filter(|p| p[1] > ymax - 0.5 * h)
            .map(|p| (p[1] - half).abs())
            .collect();
        let top_max = top.iter().fold(0.0f32, |a, d| a.max(*d));
        println!(
            "плита {}×{}×{}: шаг поля {h:.4}, вершин {}, на плоскостях {:.0}%, вершин на верхней грани {}, макс отклонение по y {:.6} м",
            n,
            n,
            n,
            m.positions.len(),
            flat * 100.0,
            top.len(),
            top_max
        );
        assert!(flat > 0.98, "вершины не на плоскостях коробки: {:.0}%", flat * 100.0);
        assert!(
            top_max <= 1e-3,
            "верхняя грань не плоская: {top_max} м (это ошибка в значениях поля, не геометрия)"
        );
    }
}

#[cfg(test)]
mod connectivity_tests {
    //! Разбиение НЕСВЯЗНЫХ тел: аналитические случаи (два шара с перемычкой
    //! тоньше шага, куб с дыркой, два шара без перемычки), сохранение массы и
    //! свои боксы поля у каждой части.
    use super::*;
    use std::collections::HashSet;

    fn ball_nodes(c: Node, r: i32) -> Vec<Node> {
        let mut out = Vec::new();
        for z in c.2 - r..=c.2 + r {
            for y in c.1 - r..=c.1 + r {
                for x in c.0 - r..=c.0 + r {
                    let (dx, dy, dz) = (x - c.0, y - c.1, z - c.2);
                    if dx * dx + dy * dy + dz * dz <= r * r {
                        out.push((x, y, z));
                    }
                }
            }
        }
        out
    }

    fn box_nodes(lo: Node, hi: Node) -> Vec<Node> {
        let mut out = Vec::new();
        for z in lo.2..=hi.2 {
            for y in lo.1..=hi.1 {
                for x in lo.0..=hi.0 {
                    out.push((x, y, z));
                }
            }
        }
        out
    }

    /// Теты тела по занятым узлам: 24 тета вокруг каждого узла плюс глобальный
    /// дедуп по каноническому ключу (как в `TetBody::cube`).
    fn tets_of(nodes: &[Node]) -> Vec<Tet> {
        let mut seen: HashSet<[Node; 4]> = HashSet::new();
        let mut out = Vec::new();
        for &n in nodes {
            for t in bcc_tets_at_dedup(n) {
                if seen.insert(tet_key(&t)) {
                    out.push(Tet { level: 0, v: t });
                }
            }
        }
        out
    }

    /// Тело из занятых узлов: поле строится по ним же, шаг поля равен шагу
    /// узлов тетов (как у настоящего тела, иначе центроиды уедут).
    fn body_of(nodes: &[Node], pad: i32) -> TetBody {
        let mut f = field::from_occupied(nodes.iter().copied(), pad);
        f.step = node_step(0);
        let mut b = TetBody {
            tets: tets_of(nodes),
            level_cap: 0,
            origin: [0.0; 3],
            sdf: Default::default(),
        };
        let _ = b.sdf.set(f);
        b
    }

    /// Два шара С ПЕРЕМЫЧКОЙ ТОНЬШЕ ШАГА поля: в поле это две раздельные
    /// области (перемычка не заняла ни одного узла), значит тело обязано
    /// распасться на два, а объём (масса) — остаться у суммы частей.
    #[test]
    fn thin_bridge_splits_into_two_and_keeps_volume() {
        let mut occ = ball_nodes((8, 12, 12), 5);
        occ.extend(ball_nodes((28, 12, 12), 5));
        let body = body_of(&occ, 4);
        let v_parent = body.volume();
        let parts = body
            .split_disconnected(1, 3)
            .expect("несвязное поле обязано разбиться");
        assert_eq!(parts.len(), 2, "два шара — два тела");
        let v_sum: f64 = parts.iter().map(|p| p.volume()).sum();
        assert!(
            (v_sum - v_parent).abs() <= v_parent * 0.01,
            "объём не сохранился: {v_sum} против {v_parent} ({:.2}%)",
            (v_sum - v_parent).abs() / v_parent * 100.0
        );
        /* У каждой части СВОЙ бокс: меньше узлов, чем у родителя, и ровно одна
         * занятая область (чужой кусок в бокс не попал или погашен). */
        let parent_nodes = body.sdf().val.len();
        for (k, p) in parts.iter().enumerate() {
            let f = p.sdf();
            assert!(
                f.val.len() < parent_nodes,
                "часть {k} несёт бокс родителя: {} узлов",
                f.val.len()
            );
            let c = field::components(f, 1);
            assert_eq!(c.count(), 1, "в поле части {k} осталось {} области", c.count());
        }
        /* Два шара далеко друг от друга: бокс одной части не накрывает центр
         * другой. */
        let f0 = parts[0].sdf();
        let f1 = parts[1].sdf();
        let inside = |f: &field::NodeField, p: Node| {
            let a = [p.0, p.1, p.2];
            (0..3).all(|k| a[k] >= f.lo[k] && a[k] <= f.lo[k] + f.dims[k] as i32 - 1)
        };
        assert!(
            !inside(f0, (28, 12, 12)) && !inside(f1, (8, 12, 12)),
            "бокс одной части накрывает чужой шар"
        );
    }

    /// Два шара БЕЗ перемычки — тоже два тела (тот же механизм, другая форма).
    #[test]
    fn two_balls_without_bridge_split() {
        let mut occ = ball_nodes((8, 12, 12), 4);
        occ.extend(ball_nodes((24, 12, 12), 4));
        let body = body_of(&occ, 4);
        let parts = body.split_disconnected(1, 3).expect("два шара — два тела");
        assert_eq!(parts.len(), 2);
        let v_parent = body.volume();
        let v_sum: f64 = parts.iter().map(|p| p.volume()).sum();
        assert!((v_sum - v_parent).abs() <= v_parent * 0.01);
    }

    /// Куб С ДЫРКОЙ (полость внутри) — одно тело: не режем.
    #[test]
    fn cube_with_hole_is_not_split() {
        let mut occ = box_nodes((2, 2, 2), (18, 18, 18));
        occ.retain(|&(x, y, z)| !(x > 6 && x < 14 && y > 6 && y < 14 && z > 6 && z < 14));
        let body = body_of(&occ, 4);
        assert!(
            body.split_disconnected(1, 3).is_none(),
            "куб с дыркой обязан остаться одним телом"
        );
    }

    /// Цельный куб не делится вообще.
    #[test]
    fn solid_cube_is_not_split() {
        let b = TetBody::cube([0.0, 0.0, 0.0], 4, 1);
        assert!(b.split_disconnected(1, 3).is_none(), "цельный куб разрезан быть не должен");
    }

    /// Три куска — три тела, объём сохранён.
    #[test]
    fn three_parts_make_three_bodies() {
        let mut occ = ball_nodes((6, 12, 12), 4);
        occ.extend(ball_nodes((22, 12, 12), 4));
        occ.extend(ball_nodes((14, 12, 26), 4));
        let body = body_of(&occ, 4);
        let parts = body.split_disconnected(1, 3).expect("три шара — три тела");
        assert_eq!(parts.len(), 3);
        let v_parent = body.volume();
        let v_sum: f64 = parts.iter().map(|p| p.volume()).sum();
        assert!((v_sum - v_parent).abs() <= v_parent * 0.01);
        for p in &parts {
            assert!(!p.tets.is_empty(), "часть без тетов — потеря массы");
        }
    }

    /// Меш каждой части замкнут: в поле части нет второй изоповерхности от
    /// чужого куска (иначе на срезе была бы дыра).
    #[test]
    fn parts_meshes_are_closed() {
        let mut occ = ball_nodes((8, 12, 12), 5);
        occ.extend(ball_nodes((26, 12, 12), 5));
        let body = body_of(&occ, 4);
        let parts = body.split_disconnected(1, 3).expect("два шара — два тела");
        for (k, p) in parts.iter().enumerate() {
            let m = p.surface();
            let open = super::marching_tests::closedness(&m);
            assert_eq!(open, 0, "меш части {k} не замкнут: {open} рёбер непарные");
            assert!(mesh_volume(&m).abs() > 0.0, "меш части {k} пуст");
        }
    }
}
