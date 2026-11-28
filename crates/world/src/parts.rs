//! Связность поля и разбиение тела на НЕСВЯЗНЫЕ куски.
//!
//! ЗАЧЕМ. Поле тела рвётся там, где детали тоньше шага сетки: в ОДНОМ теле
//! оказываются два физически раздельных куска. Физика тянет их друг за
//! другом, меш и коллайдер считаются как одно тело — такое тело обязано
//! распасться на отдельные.
//!
//! КАК. Занятость читается по ЗНАКУ поля (`val < 0`). Сначала — эрозия на
//! `erode` слоёв по 6 соседям: в «ядро» входят только узлы, у которых все
//! шесть соседей тоже заняты. Компоненты считаются ПО ЯДРУ, поэтому перемычка
//! тоньше одного шага решётки (угловой контакт узел-в-узел, нитка в один узел)
//! куски не склеивает. Затем «кожа» (эродированные узлы) приписывается
//! ближайшей компоненте распространением минимальной метки по 6 соседям, а
//! куски БЕЗ ядра (тонкие пластины) размечаются отдельным проходом — иначе
//! они исчезли бы совсем.
//!
//! ЭТО ОРАКУЛ. Тот же алгоритм посчитан на устройстве
//! (`shaders/components.comp`, запуск `sim::GpuSim::run_components`): тесты и
//! проба сверяют число компонент и разбиение узлов один-в-один. Здесь —
//! эталон, по которому проверяется GPU.
//!
//! Теты при разбиении берутся ИЗ РОДИТЕЛЯ (по центроиду тета), поэтому сумма
//! объёмов частей равна объёму родителя: масса сохраняется точно, без
//! квантования по решётке (тот же приём, что в `TetBody::shatter_fields`).

use crate::field::{self, NodeField};
use crate::tet::{node_step, Tet, TetBody};

/// Габарит компоненты в координатах узлов поля, включительно (как `lo..=hi`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub lo: [i32; 3],
    pub hi: [i32; 3],
}

impl Bounds {
    /// Размер в узлах.
    pub fn dims(&self) -> [usize; 3] {
        [
            (self.hi[0] - self.lo[0] + 1).max(1) as usize,
            (self.hi[1] - self.lo[1] + 1).max(1) as usize,
            (self.hi[2] - self.lo[2] + 1).max(1) as usize,
        ]
    }

    /// Центр в координатах узлов поля.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.lo[0] + self.hi[0]) as f32 * 0.5,
            (self.lo[1] + self.hi[1]) as f32 * 0.5,
            (self.lo[2] + self.hi[2]) as f32 * 0.5,
        ]
    }
}

/// Метки компонент на узлы поля: `u32::MAX` — узел пуст.
#[derive(Clone, Debug, Default)]
pub struct Labels {
    /// Метка на узел блока (`f.val.len()` элементов).
    pub label: Vec<u32>,
    /// Габарит каждой компоненты (индекс = метка).
    pub bounds: Vec<Bounds>,
    /// Сколько занятых узлов в каждой компоненте.
    pub counts: Vec<usize>,
}

impl Labels {
    /// Число компонент: 1 — тело связное.
    pub fn count(&self) -> usize {
        self.bounds.len()
    }

    /// Метка в узле с мировыми координатами (для тестов и диагностики).
    pub fn at(&self, f: &NodeField, p: [i32; 3]) -> u32 {
        let (x, y, z) = (p[0] - f.lo[0], p[1] - f.lo[1], p[2] - f.lo[2]);
        if x < 0 || y < 0 || z < 0 {
            return u32::MAX;
        }
        let (x, y, z) = (x as usize, y as usize, z as usize);
        if x >= f.dims[0] || y >= f.dims[1] || z >= f.dims[2] {
            return u32::MAX;
        }
        self.label[(z * f.dims[1] + y) * f.dims[0] + x]
    }
}

/// Часть тела: свой бокс поля, свои теты, своя метка компоненты.
#[derive(Clone, Debug)]
pub struct BodySeed {
    /// Метка компоненты в родительском поле.
    pub label: u32,
    /// Габарит компоненты (в координатах узлов поля).
    pub bounds: Bounds,
    /// Теты, центроид которых попал в эту компоненту (ровно один раз).
    pub tets: Vec<Tet>,
    /// Поле части: вырезка родителя с погашенными чужими областями.
    pub sdf: NodeField,
}

/// Шаг эрозии по умолчанию: один слой. Перемычка в один узел и угловой
/// контакт перестают держать куски вместе (см. модульный комментарий).
pub const DEFAULT_ERODE: u32 = 1;
/// Запас пустоты вокруг части по умолчанию (узлов): марчингу нужны внешние
/// узлы, иначе срез выходит «впуклым».
pub const DEFAULT_PAD: i32 = 3;

/// Итог GPU-разметки: готовые метки, цена и сходимость.
#[derive(Clone, Debug, Default)]
pub struct GpuParts {
    pub labels: Labels,
    /// Сколько проходов распространения минимума понадобилось (ядро + кожа +
    /// остаток) — «итераций до стабилизации».
    pub iterations: u32,
    /// Разбивка проходов: [ядро, кожа, остаток].
    pub iters_stage: [u32; 3],
    /// Время целиком: заливка, проходы, чтение, мс.
    pub ms: f32,
    /// Плоский список узлов по компонентам и начала блоков (диагностика).
    pub nodes: Vec<u32>,
    pub offsets: Vec<u32>,
}

/// Связные компоненты занятости поля: `(метка, габарит)` на компоненту.
///
/// `erode` — сколько слоёв эрозии по 6 соседям применить до разметки:
/// `0` — обычная 6-связность (угловой контакт уже не связь), `1` — плюс
/// разрыв нитки в один узел.
pub fn components(f: &NodeField, erode: u32) -> Vec<(u32, Bounds)> {
    let lab = label_field(f, erode);
    lab.bounds
        .iter()
        .enumerate()
        .map(|(i, b)| (i as u32, *b))
        .collect()
}

/// Полная разметка: метка на каждый узел плюс габариты и размеры компонент.
pub fn label_field(f: &NodeField, erode: u32) -> Labels {
    let dims = f.dims;
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let n = nx * ny * nz;
    let mut out = Labels::default();
    if n == 0 {
        return out;
    }
    /* 1. Ядро: занятость, пережившая `erode` слоёв эрозии по 6 соседям. */
    let mut core = vec![false; n];
    for i in 0..n {
        core[i] = f.val[i] < 0.0;
    }
    for _ in 0..erode {
        let prev = core.clone();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = (z * ny + y) * nx + x;
                    core[i] = prev[i]
                        && x > 0
                        && prev[i - 1]
                        && x + 1 < nx
                        && prev[i + 1]
                        && y > 0
                        && prev[i - nx]
                        && y + 1 < ny
                        && prev[i + nx]
                        && z > 0
                        && prev[i - nx * ny]
                        && z + 1 < nz
                        && prev[i + nx * ny];
                }
            }
        }
    }
    /* 2. Разметка ядра: каждая компонента получает минимальный индекс узла
     *    (распространение минимума по 6 соседям — та же схема, что на GPU). */
    let mut label = vec![u32::MAX; n];
    for i in 0..n {
        if core[i] {
            label[i] = i as u32;
        }
    }
    propagate_min(&mut label, &core, &core, dims);
    /* 3. Кожа: занятые узлы вне ядра получают метку ближайшего ядра. Пишем
     *    ТОЛЬКО в кожу: метки ядра неприкосновенны, иначе распространение
     *    «уменьшило» бы метку одного куска меткой другого и склеило их. Куски
     *    без ядра вовсе (тонкая пластина) остаются без метки. */
    let mut solid = vec![false; n];
    let mut skin = vec![false; n];
    for i in 0..n {
        solid[i] = f.val[i] < 0.0;
        skin[i] = solid[i] && !core[i];
    }
    propagate_min(&mut label, &skin, &core, dims);
    /* 4. Куски без ядра: своя 6-связность ВНУТРИ остатка. */
    let mut rest = vec![false; n];
    let mut any = false;
    for i in 0..n {
        if solid[i] && label[i] == u32::MAX {
            rest[i] = true;
            label[i] = i as u32;
            any = true;
        }
    }
    if any {
        propagate_min(&mut label, &rest, &rest, dims);
    }
    /* 5. Сжатие меток в 0..K-1 в порядке первого появления узла (устойчиво) и
     *    габариты/размеры. */
    let mut remap = vec![u32::MAX; n];
    for i in 0..n {
        let r = label[i];
        if r == u32::MAX {
            continue;
        }
        let r = r as usize;
        if remap[r] == u32::MAX {
            remap[r] = out.bounds.len() as u32;
            out.bounds.push(Bounds {
                lo: [i32::MAX; 3],
                hi: [i32::MIN; 3],
            });
            out.counts.push(0);
        }
        let id = remap[r];
        label[i] = id;
        let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
        let c = [
            f.lo[0] + x as i32,
            f.lo[1] + y as i32,
            f.lo[2] + z as i32,
        ];
        let b = &mut out.bounds[id as usize];
        for k in 0..3 {
            b.lo[k] = b.lo[k].min(c[k]);
            b.hi[k] = b.hi[k].max(c[k]);
        }
        out.counts[id as usize] += 1;
    }
    out.label = label;
    out
}

/// Распространить минимальную метку по 6 соседям.
///
/// `write` — узлы, которым разрешено менять метку (фронт уходит только по ним),
/// `seed` — откуда распространение начинается (помеченные узлы-источники).
/// Узел принимает метку соседа, если она МЕНЬШЕ его собственной; метки только
/// убывают, поэтому процесс сходится к минимуму компоненты (глобальному индексу
/// её узла). Возвращает число слоёв фронта — это же число итераций
/// распространения на устройстве.
fn propagate_min(label: &mut [u32], write: &[bool], seed: &[bool], dims: [usize; 3]) -> usize {
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let n = nx * ny * nz;
    let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for i in 0..n {
        if seed[i] && label[i] != u32::MAX {
            q.push_back(i);
        }
    }
    let mut layers = 0usize;
    let cap = 2 * (nx + ny + nz) + 8;
    while !q.is_empty() {
        layers += 1;
        if layers > cap {
            /* Фронт шире габарита поля: этого не бывает у настоящих полей,
             * но зависнуть на кривом входе нельзя. */
            break;
        }
        let layer = q.len();
        for _ in 0..layer {
            let i = q.pop_front().unwrap();
            let l = label[i];
            let x = i % nx;
            let y = (i / nx) % ny;
            let z = i / (nx * ny);
            let mut push = |j: usize, label: &mut [u32], q: &mut std::collections::VecDeque<usize>| {
                if write[j] && label[j] > l {
                    label[j] = l;
                    q.push_back(j);
                }
            };
            if x > 0 {
                push(i - 1, label, &mut q);
            }
            if x + 1 < nx {
                push(i + 1, label, &mut q);
            }
            if y > 0 {
                push(i - nx, label, &mut q);
            }
            if y + 1 < ny {
                push(i + nx, label, &mut q);
            }
            if z > 0 {
                push(i - nx * ny, label, &mut q);
            }
            if z + 1 < nz {
                push(i + nx * ny, label, &mut q);
            }
        }
    }
    layers
}

/// Разбить тело на связные части; пустой список — тело связное.
pub fn split_disconnected(body: &TetBody) -> Vec<BodySeed> {
    split_with_labels(body, &label_field(body.sdf(), DEFAULT_ERODE), DEFAULT_PAD)
}

/// То же с явными параметрами: сколько слоёв эрозии и сколько узлов пустоты
/// оставить вокруг каждой части.
pub fn split_disconnected_with(body: &TetBody, erode: u32, pad: i32) -> Vec<BodySeed> {
    split_with_labels(body, &label_field(body.sdf(), erode), pad)
}

/// Разбиение по ГОТОВЫМ меткам: их может дать и CPU (`label_field`), и
/// устройство (`GpuSim::run_components`) — логика разбиения одна.
///
/// Теты делятся по центроиду, поэтому каждый родительский тет попадает ровно
/// в одну часть: сумма объёмов частей равна объёму родителя.
pub fn split_with_labels(body: &TetBody, lab: &Labels, pad: i32) -> Vec<BodySeed> {
    let f = body.sdf();
    if lab.count() < 2 || lab.label.len() != f.val.len() {
        return Vec::new();
    }
    let mut buckets: Vec<Vec<Tet>> = vec![Vec::new(); lab.count()];
    for t in &body.tets {
        buckets[part_of_tet(t, f, lab)].push(*t);
    }
    let pad = pad.max(1);
    let mut out: Vec<BodySeed> = Vec::with_capacity(lab.count());
    let mut noise_areas = 0usize;
    let mut noise_nodes = 0usize;
    for (k, tets) in buckets.into_iter().enumerate() {
        if tets.is_empty() {
            /* Область без тетов — шум поля (одиночные узлы у среза): массы у
             * неё нет, тела из неё не сделать. Считаем разом, чтобы не
             * заливать лог строкой на каждый осколок. */
            noise_areas += 1;
            noise_nodes += lab.counts[k];
            continue;
        }
        let b = lab.bounds[k];
        /* Бокс части: её габарит плюс `pad` узлов пустоты (обрезка по
         * границам родителя — внутри `field::crop`). */
        let lo = [b.lo[0] - pad, b.lo[1] - pad, b.lo[2] - pad];
        let dims = [
            b.dims()[0] + 2 * pad as usize,
            b.dims()[1] + 2 * pad as usize,
            b.dims()[2] + 2 * pad as usize,
        ];
        let mut part = field::crop(f, lo, dims);
        /* Чужие области, попавшие в бокс, гасим в ПЛЮС: иначе их
         * отрицательные узлы дадут в поле части вторую изоповерхность (меш с
         * дырой и лишним куском). Знак переворачиваем, а не ставим «далеко»:
         * так в поле не появляется НОВОГО перехода через ноль. */
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
                    let l = lab.label[si];
                    if l != u32::MAX && l != k as u32 {
                        let i = (z * pny + y) * pnx + x;
                        part.val[i] = part.val[i].abs().max(0.5);
                    }
                }
            }
        }
        out.push(BodySeed {
            label: k as u32,
            bounds: b,
            tets,
            sdf: part,
        });
    }
    if out.len() >= 2 && noise_areas > 0 && noise_nodes > 8 {
        eprintln!(
            "DIAG parts: {noise_areas} областей без тетов ({noise_nodes} узлов) погашены как шум поля"
        );
    }
    if out.len() < 2 {
        return Vec::new();
    }
    out
}

/// Номер компоненты, к которой относится тет: по его ЦЕНТРОИДУ.
///
/// Центроид считается в координатах УЗЛОВ ПОЛЯ (`node_step(level) / step`), как
/// в `shatter_fields`: у тета и поля шаги разные, без пересчёта раздача тетов
/// уезжает. Если центроид попал в пустой узел (тет крупнее зазора), берётся
/// ближайший помеченный узел в радиусе трёх, иначе — ближайший центр габарита.
fn part_of_tet(t: &Tet, f: &NodeField, lab: &Labels) -> usize {
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
        let l = lab.label[i];
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
                    let l = lab.label[i];
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
    let mut bi = 0usize;
    let mut bd = f32::MAX;
    for (k, b) in lab.bounds.iter().enumerate() {
        let c = b.center();
        let d = (c[0] - cx).powi(2) + (c[1] - cy).powi(2) + (c[2] - cz).powi(2);
        if d < bd {
            bd = d;
            bi = k;
        }
    }
    bi
}

#[cfg(test)]
mod tests {
    //! Аналитические случаи связности и сохранение массы при разбиении.
    use super::*;
    use std::collections::HashSet;

    fn ball_nodes(c: (i32, i32, i32), r: i32) -> Vec<(i32, i32, i32)> {
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

    fn box_nodes(lo: (i32, i32, i32), hi: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
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

    fn field_of(nodes: &[(i32, i32, i32)], pad: i32) -> NodeField {
        field::from_occupied(nodes.iter().copied(), pad)
    }

    /// Теты по занятым узлам: 24 тета вокруг узла плюс дедуп по ключу.
    fn tets_of(nodes: &[(i32, i32, i32)]) -> Vec<Tet> {
        use crate::tet::{bcc_tets_at_dedup, Tet};
        let mut seen: HashSet<[(i32, i32, i32); 4]> = HashSet::new();
        let mut out = Vec::new();
        for &n in nodes {
            for t in bcc_tets_at_dedup(n) {
                let mut k = t;
                k.sort();
                if seen.insert(k) {
                    out.push(Tet { level: 0, v: t });
                }
            }
        }
        out
    }

    /// Тело из занятых узлов: поле строится по ним же, шаг поля равен шагу
    /// узлов тетов (как у настоящего тела, иначе центроиды тетов уедут).
    fn body_of(nodes: &[(i32, i32, i32)], pad: i32) -> TetBody {
        let mut f = field_of(nodes, pad);
        f.step = node_step(0);
        let b = TetBody {
            tets: tets_of(nodes),
            level_cap: 0,
            origin: [0.0; 3],
            sdf: Default::default(),
        };
        let _ = b.sdf.set(f);
        b
    }

    /// Два шара, соединённые перемычкой ТОНЬШЕ ШАГА поля (по решётке перемычка
    /// не занимает ни одного узла) → две компоненты.
    #[test]
    fn thin_bridge_two_components() {
        let mut occ = ball_nodes((10, 16, 16), 5);
        occ.extend(ball_nodes((28, 16, 16), 5));
        let f = field_of(&occ, 3);
        for erode in [0, 1] {
            let c = components(&f, erode);
            assert_eq!(c.len(), 2, "erode={erode}: перемычка тоньше шага склеила шары");
            assert_ne!(c[0].1, c[1].1);
        }
    }

    /// Куб С ДЫРКОЙ (полость внутри и сквозной канал) → одна компонента.
    #[test]
    fn cube_with_hole_one_component() {
        let mut occ = box_nodes((2, 2, 2), (20, 20, 20));
        occ.retain(|&(x, y, z)| !(x > 6 && x < 16 && y > 6 && y < 16 && z > 6 && z < 16));
        occ.retain(|&(_, y, z)| !(y == 10 && z == 10));
        let f = field_of(&occ, 3);
        for erode in [0, 1] {
            assert_eq!(components(&f, erode).len(), 1, "erode={erode}: дырка разорвала куб");
        }
    }

    /// Два шара без перемычки → две компоненты.
    #[test]
    fn two_balls_two_components() {
        let mut occ = ball_nodes((8, 12, 12), 5);
        occ.extend(ball_nodes((30, 12, 12), 5));
        let f = field_of(&occ, 3);
        for erode in [0, 1] {
            assert_eq!(components(&f, erode).len(), 2, "erode={erode}");
        }
    }

    /// Угловой контакт (разность (1,1,0)) — перемычка нулевой толщины: связью
    /// не считается даже без эрозии (6-связность узлов).
    #[test]
    fn corner_contact_is_not_a_link() {
        let mut occ = box_nodes((2, 2, 2), (9, 9, 9));
        occ.extend(box_nodes((10, 10, 9), (17, 17, 16)));
        let f = field_of(&occ, 3);
        let lab = label_field(&f, 0);
        assert_eq!(lab.count(), 2, "угловой контакт склеил куски");
        assert_ne!(lab.at(&f, [2, 2, 2]), lab.at(&f, [17, 17, 16]));
    }

    /// Нитка В ОДИН УЗЕЛ: при `erode=0` держит тело целым, при `erode=1`
    /// (эрозия на узел) — уже нет. Это и есть порог толщины перемычки.
    #[test]
    fn one_node_neck_depends_on_erode() {
        let mut occ = box_nodes((2, 2, 2), (9, 9, 9));
        occ.extend(box_nodes((10, 9, 9), (11, 9, 9)));
        occ.extend(box_nodes((12, 9, 9), (19, 16, 16)));
        let f = field_of(&occ, 3);
        assert_eq!(components(&f, 0).len(), 1, "без эрозии нитка в узел — связь");
        assert_eq!(components(&f, 1).len(), 2, "эрозия на узел обязана её разорвать");
    }

    /// Разбиение сохраняет объём: сумма частей == объём родителя.
    #[test]
    fn split_keeps_volume() {
        let mut occ = ball_nodes((8, 12, 12), 5);
        occ.extend(ball_nodes((28, 12, 12), 5));
        let body = body_of(&occ, 4);
        let v0 = body.volume();
        let seeds = split_disconnected(&body);
        assert_eq!(seeds.len(), 2, "два шара — две части");
        let v: f64 = seeds.iter().map(|s| tets_volume(&s.tets)).sum();
        println!(
            "объём родителя {v0:.6}, сумма частей {v:.6} (расхождение {:.4}%)",
            (v - v0).abs() / v0 * 100.0
        );
        assert!(
            (v - v0).abs() <= v0 * 0.01,
            "объём не сохранился: {v} против {v0} ({:.2}%)",
            (v - v0).abs() / v0 * 100.0
        );
        assert!(seeds.iter().all(|s| !s.tets.is_empty()), "часть без тетов");
    }

    fn tets_volume(tets: &[Tet]) -> f64 {
        tets.iter().map(|t| t.volume() as f64).sum()
    }

    /// У каждой части СВОЙ бокс поля: чужой шар в него не попадает, а в самом
    /// поле части ровно одна занятая область (чужое погашено).
    #[test]
    fn parts_have_own_boxes() {
        let mut occ = ball_nodes((8, 12, 12), 5);
        occ.extend(ball_nodes((26, 12, 12), 5));
        let body = body_of(&occ, 4);
        let seeds = split_disconnected(&body);
        assert_eq!(seeds.len(), 2);
        let parent_nodes = body.sdf().val.len();
        for (k, s) in seeds.iter().enumerate() {
            assert!(s.sdf.val.len() < parent_nodes, "часть {k} несёт бокс родителя");
            assert_eq!(
                label_field(&s.sdf, 1).count(),
                1,
                "в поле части {k} осталось больше одной области"
            );
        }
        let inside = |f: &NodeField, p: [i32; 3]| {
            (0..3).all(|k| p[k] >= f.lo[k] && p[k] <= f.lo[k] + f.dims[k] as i32 - 1)
        };
        assert!(
            !inside(&seeds[0].sdf, [26, 12, 12]) && !inside(&seeds[1].sdf, [8, 12, 12]),
            "бокс одной части накрывает чужой шар"
        );
    }

    /// Замкнутость меша: каждое ребро принадлежит ровно двум треугольникам.
    fn closedness(m: &crate::tet::Mesh) -> usize {
        use std::collections::HashMap;
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

    /// Меши частей замкнуты: второй изоповерхности в поле части нет.
    #[test]
    fn part_meshes_are_closed() {
        let mut occ = ball_nodes((8, 12, 12), 5);
        occ.extend(ball_nodes((26, 12, 12), 5));
        let body = body_of(&occ, 6);
        let seeds = split_disconnected_with(&body, 0, 5);
        assert_eq!(seeds.len(), 2);
        for (k, s) in seeds.iter().enumerate() {
            let part = TetBody {
                tets: s.tets.clone(),
                level_cap: body.level_cap,
                origin: body.origin,
                sdf: Default::default(),
            };
            let _ = part.sdf.set(s.sdf.clone());
            let m = part.surface();
            let open = closedness(&m);
            assert_eq!(open, 0, "меш части {k} не замкнут: {open} рёбер непарные");
        }
    }

    /// Связное тело не делится; цельный куб — тоже.
    #[test]
    fn connected_bodies_are_not_split() {
        let cube = TetBody::cube([0.0, 0.0, 0.0], 4, 1);
        assert!(split_disconnected(&cube).is_empty(), "цельный куб разрезан");
        let mut occ = box_nodes((2, 2, 2), (18, 18, 18));
        occ.retain(|&(x, y, z)| !(x > 6 && x < 14 && y > 6 && y < 14 && z > 6 && z < 14));
        let holed = body_of(&occ, 4);
        assert!(split_disconnected(&holed).is_empty(), "куб с дыркой разрезан");
    }

    /// Цена CPU-оракула: 64³ и 128³ узлов.
    #[test]
    fn cpu_cost_64_and_128() {
        let mut line = String::from("CPU-компоненты: ");
        for dim in [64usize, 128] {
            let f = blob_field(dim);
            let mut best = f32::MAX;
            let mut got = 0;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let lab = label_field(&f, 1);
                best = best.min(t.elapsed().as_secs_f32() * 1e3);
                got = lab.count();
            }
            assert_eq!(got, 2, "{dim}³: должно быть две области");
            line.push_str(&format!("{dim}³ = {best:.2} мс; "));
        }
        println!("{line}");
    }

    /// Аналитическое поле из двух шаров (без BFS) — для замера цены.
    pub(crate) fn blob_field(dim: usize) -> NodeField {
        let mut val = vec![1.0f32; dim * dim * dim];
        let c = dim as f32 * 0.5;
        let balls = [([-10.0f32, 0.0, 0.0], 9.0f32), ([10.0, 0.0, 0.0], 9.0)];
        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    let p = [x as f32 - c, y as f32 - c, z as f32 - c];
                    let mut d = f32::MAX;
                    for (bc, r) in balls {
                        let q = [p[0] - bc[0], p[1] - bc[1], p[2] - bc[2]];
                        d = d.min((q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() - r);
                    }
                    val[(z * dim + y) * dim + x] = d;
                }
            }
        }
        NodeField { lo: [0; 3], dims: [dim, dim, dim], val, step: 1.0 }
    }
}
