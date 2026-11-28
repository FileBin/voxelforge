//! Непрерывное поле занятости по узлам решётки — вход для marching tetrahedra.
//!
//! Тело задаётся набором тетов, то есть двоичной занятостью. Марчингу нужно
//! ЧИСЛО в каждом узле, иначе он не интерполирует и поверхность выходит
//! гранёной (именно это и делал `extract_faces`).
//!
//! Поле — знаковое расстояние до границы занятой области в единицах узла:
//! внутри отрицательное, снаружи положительное, ноль проходит между соседними
//! занятым и пустым узлом. Расстояния считаются точным BFS по шести соседям,
//! затем поле сглаживается, чтобы изоповерхность не повторяла квантование сетки.
//!
//! Занятость берётся ПО ОБЪЁМУ тетов: заполнение одними вершинами оставляет
//! внутри дыры, ноль поля на границе не встречается, и марчинг выдаёт ноль
//! треугольников (проверено тестами замкнутости).

use crate::tet::Node;

/// Поле на прямоугольном блоке узлов.
#[derive(Clone, Debug)]
pub struct NodeField {
    /// Узел с минимальными координатами блока.
    pub lo: [i32; 3],
    /// Размер блока в узлах.
    pub dims: [usize; 3],
    /// Значения по индексу `((z * dims[1]) + y) * dims[0] + x`.
    /// В МЕТРАХ: это запечённые расстояния для физики (минус внутри).
    pub val: Vec<f32>,
    /// Сколько метров в одном шаге узла (для градиента и перевода координат).
    pub step: f32,
}

impl NodeField {
    #[inline]
    pub fn at(&self, p: [i32; 3]) -> f32 {
        let cl = |v: i32, lo: i32, n: usize| (v - lo).clamp(0, n as i32 - 1) as usize;
        let x = cl(p[0], self.lo[0], self.dims[0]);
        let y = cl(p[1], self.lo[1], self.dims[1]);
        let z = cl(p[2], self.lo[2], self.dims[2]);
        self.val[(z * self.dims[1] + y) * self.dims[0] + x]
    }

    #[inline]
    fn set(&mut self, i: usize, v: f32) {
        self.val[i] = v;
    }

    /// Сгладить поле расстояний: тетраэдры задают СТУПЕНЧАТУЮ границу, и её
    /// ноль повторяет клетки. Усреднение по шести соседям убирает эти ступеньки,
    /// оставляя общую форму (сглаживается только величина, знак не трогаем —
    /// иначе тонкие тела исчезают).
    pub fn smooth(&mut self, iterations: usize, lambda: f32) {
        let dims = self.dims;
        for _ in 0..iterations {
            let src = self.val.clone();
            for z in 1..dims[2].saturating_sub(1) {
                for y in 1..dims[1].saturating_sub(1) {
                    for x in 1..dims[0].saturating_sub(1) {
                        let i = (z * dims[1] + y) * dims[0] + x;
                        let sx = dims[0];
                        let sy = dims[0] * dims[1];
                        let avg = (src[i - 1] + src[i + 1] + src[i - sx] + src[i + sx]
                            + src[i - sy] + src[i + sy])
                            / 6.0;
                        let v = src[i] + lambda * (avg - src[i]);
                        let keep_sign = if src[i] < 0.0 {
                            -v.abs().max(1e-6)
                        } else {
                            v.abs().max(1e-6)
                        };
                        self.val[i] = keep_sign;
                    }
                }
            }
        }
    }

    /// Значение поля в МЕТРАХ — то, что читает физика: |v| это глубина
    /// проникновения, градиент — нормаль контакта.
    pub fn sample_m(&self, p: [f32; 3]) -> f32 {
        self.sample(p) * self.step
    }

    /// Трилинейная выборка поля в произвольной точке (в координатах узлов).
    pub fn sample(&self, p: [f32; 3]) -> f32 {
        let x0 = p[0].floor();
        let y0 = p[1].floor();
        let z0 = p[2].floor();
        let fx = p[0] - x0;
        let fy = p[1] - y0;
        let fz = p[2] - z0;
        let (x0, y0, z0) = (x0 as i32, y0 as i32, z0 as i32);
        let mut acc = 0.0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let w = (if dx == 1 { fx } else { 1.0 - fx })
                        * (if dy == 1 { fy } else { 1.0 - fy })
                        * (if dz == 1 { fz } else { 1.0 - fz });
                    acc += w * self.at([x0 + dx, y0 + dy, z0 + dz]);
                }
            }
        }
        acc
    }

    /// Градиент поля — нормаль изоповерхности (смотрит наружу: значение растёт
    /// от минуса внутри к плюсу снаружи).
    pub fn gradient(&self, p: [f32; 3]) -> [f32; 3] {
        let h = 0.5f32;
        let gx = self.sample([p[0] + h, p[1], p[2]]) - self.sample([p[0] - h, p[1], p[2]]);
        let gy = self.sample([p[0], p[1] + h, p[2]]) - self.sample([p[0], p[1] - h, p[2]]);
        let gz = self.sample([p[0], p[1], p[2] + h]) - self.sample([p[0], p[1], p[2] - h]);
        let len = (gx * gx + gy * gy + gz * gz).sqrt().max(1e-9);
        [gx / len, gy / len, gz / len]
    }

    /// Ближайший к нулю узел — проверка «поверхность есть».
    pub fn min_abs(&self) -> f32 {
        self.val.iter().fold(f32::MAX, |m, v| m.min(v.abs()))
    }
}

/// Разметка связности поля: каждая НЕСВЯЗНАЯ занятая область (`val < 0`)
/// получает свой номер.
///
/// Занятость читается по ЗНАКУ поля: сглаживание знак не меняет (см.
/// `build_from_solid`), поэтому разметка описывает ровно ту форму, которую
/// потом марчит мешер.
#[derive(Clone, Debug, Default)]
pub struct Components {
    /// Метка на каждый узел блока: `u32::MAX` — узел пуст.
    pub label: Vec<u32>,
    /// Габарит каждой области в координатах узлов, включительно.
    pub bounds: Vec<([i32; 3], [i32; 3])>,
    /// Сколько занятых узлов в каждой области.
    pub sizes: Vec<usize>,
}

impl Components {
    /// Число областей: 1 — тело связное.
    pub fn count(&self) -> usize {
        self.bounds.len()
    }
}

/// Поставить узлу метку области и положить его в очередь обхода.
#[inline]
fn link(label: &mut [u32], val: &[f32], stack: &mut Vec<usize>, j: usize, id: u32) {
    if label[j] == u32::MAX && val[j] < 0.0 {
        label[j] = id;
        stack.push(j);
    }
}

/// Разбить занятость поля на связные области по 6-связности УЗЛОВ.
///
/// `min_gap` — сколько ПУСТЫХ узлов должно лежать между областями, чтобы они
/// считались РАЗНЫМИ телами. `min_gap = 1` — обычная 6-связность: занятые
/// узлы соединяются, только если они соседи ПО ГРАНИ. Поэтому перемычка
/// тоньше шага поля (угловой контакт узел-в-узел, диагональ) связью НЕ
/// считается и два куска не склеиваются в одно тело. Больше 1 — допуск:
/// области, разделённые меньше чем `min_gap` пустыми узлами, остаются одним
/// телом (грубая сетка не должна рвать тело по одному слою пустоты).
///
/// Стоимость линейна по числу узлов блока: каждый занятый узел посещается
/// ровно один раз, на узел 6 проверок.
pub fn components(f: &NodeField, min_gap: i32) -> Components {
    let gap = min_gap.max(1) as usize;
    let (nx, ny, nz) = (f.dims[0], f.dims[1], f.dims[2]);
    let n = nx * ny * nz;
    let mut label = vec![u32::MAX; n];
    let mut bounds: Vec<([i32; 3], [i32; 3])> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let (sx, sy) = (1usize, nx);
    let sz = nx * ny;
    for start in 0..n {
        if f.val[start] >= 0.0 || label[start] != u32::MAX {
            continue;
        }
        let id = bounds.len() as u32;
        label[start] = id;
        stack.clear();
        stack.push(start);
        let mut lo = [i32::MAX; 3];
        let mut hi = [i32::MIN; 3];
        let mut size = 0usize;
        while let Some(i) = stack.pop() {
            size += 1;
            let x = i % nx;
            let y = (i / nx) % ny;
            let z = i / sz;
            let c = [f.lo[0] + x as i32, f.lo[1] + y as i32, f.lo[2] + z as i32];
            for k in 0..3 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
            for d in 1..=gap {
                if x + d < nx {
                    link(&mut label, &f.val, &mut stack, i + d * sx, id);
                }
                if x >= d {
                    link(&mut label, &f.val, &mut stack, i - d * sx, id);
                }
                if y + d < ny {
                    link(&mut label, &f.val, &mut stack, i + d * sy, id);
                }
                if y >= d {
                    link(&mut label, &f.val, &mut stack, i - d * sy, id);
                }
                if z + d < nz {
                    link(&mut label, &f.val, &mut stack, i + d * sz, id);
                }
                if z >= d {
                    link(&mut label, &f.val, &mut stack, i - d * sz, id);
                }
            }
        }
        bounds.push((lo, hi));
        sizes.push(size);
    }
    Components { label, bounds, sizes }
}

/// Границы блока узлов по набору узлов плюс запас `pad`.
fn bounds_of<'a, I: Iterator<Item = &'a Node>>(it: I, pad: i32) -> ([i32; 3], [usize; 3]) {
    let mut lo = [i32::MAX; 3];
    let mut hi = [i32::MIN; 3];
    let mut any = false;
    for n in it {
        any = true;
        let c = [n.0, n.1, n.2];
        for k in 0..3 {
            lo[k] = lo[k].min(c[k]);
            hi[k] = hi[k].max(c[k]);
        }
    }
    assert!(any, "поле без узлов");
    for k in 0..3 {
        lo[k] -= pad;
        hi[k] += pad;
    }
    let dims = [
        (hi[0] - lo[0] + 1) as usize,
        (hi[1] - lo[1] + 1) as usize,
        (hi[2] - lo[2] + 1) as usize,
    ];
    (lo, dims)
}

/// Построить поле по занятым узлам (вершинам тетов).
///
/// `pad` — сколько узлов пустоты добавить вокруг занятой области, чтобы
/// марчинг видел внешние теты и поверхность была замкнутой.
pub fn from_occupied<I: Iterator<Item = Node>>(occupied: I, pad: i32) -> NodeField {
    let nodes: Vec<Node> = occupied.collect();
    assert!(!nodes.is_empty(), "поле без занятых узлов");
    let (lo, dims) = bounds_of(nodes.iter(), pad);
    let n = dims[0] * dims[1] * dims[2];
    let mut solid = vec![false; n];
    for nd in &nodes {
        let x = (nd.0 - lo[0]) as usize;
        let y = (nd.1 - lo[1]) as usize;
        let z = (nd.2 - lo[2]) as usize;
        solid[(z * dims[1] + y) * dims[0] + x] = true;
    }
    build_from_solid(&solid, lo, dims)
}

/// Знак объёма тетраэдра (a,b,c,d): меняет знак при переходе через грань.
fn tet_vol(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> f32 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
}

/// Построить поле ПО ОБЪЁМУ тетов: занятым считается каждый целый узел внутри
/// любого тета.
///
/// Заполнение одними вершинами даёт дырявую занятость: в плотном теле между
/// вершинами остаются узлы, не принадлежащие ни одному тету, и ноль поля на
/// границе не встречается — марчинг выдаёт ноль треугольников.
pub fn from_tets(tets: &[[Node; 4]], pad: i32) -> NodeField {
    assert!(!tets.is_empty(), "поле без тетов");
    let (lo, dims) = match bounds_of(tets.iter().flat_map(|t| t.iter()), pad) {
        (l, d) => (l, d),
    };
    let (blo, bhi) = (
        [lo[0], lo[1], lo[2]],
        [
            lo[0] + dims[0] as i32 - 1,
            lo[1] + dims[1] as i32 - 1,
            lo[2] + dims[2] as i32 - 1,
        ],
    );
    let n = dims[0] * dims[1] * dims[2];
    let mut solid = vec![false; n];
    for t in tets {
        let pts: [[f32; 3]; 4] =
            std::array::from_fn(|i| [t[i].0 as f32, t[i].1 as f32, t[i].2 as f32]);
        let mut tlo = [i32::MAX; 3];
        let mut thi = [i32::MIN; 3];
        for q in &pts {
            for k in 0..3 {
                tlo[k] = tlo[k].min(q[k].floor() as i32);
                thi[k] = thi[k].max(q[k].ceil() as i32);
            }
        }
        for z in tlo[2].max(blo[2])..=thi[2].min(bhi[2]) {
            for y in tlo[1].max(blo[1])..=thi[1].min(bhi[1]) {
                for x in tlo[0].max(blo[0])..=thi[0].min(bhi[0]) {
                    /* Проверка «точка внутри тета» обязана быть СОГЛАСОВАНА по
                     * ориентации: у граней (0,1,2) и (0,1,3) нормали смотрят в
                     * разные стороны, поэтому «все под-объёмы одного знака» даёт
                     * ложное «снаружи» даже для центра тета (проверено). Берём
                     * знак самого тета и сравниваем с ним знаки четырёх под-тетов
                     * «грань + точка». */
                    let p = [x as f32, y as f32, z as f32];
                    let s0 = tet_vol(pts[0], pts[1], pts[2], pts[3]);
                    let d1 = tet_vol(p, pts[1], pts[2], pts[3]);
                    let d2 = tet_vol(pts[0], p, pts[2], pts[3]);
                    let d3 = tet_vol(pts[0], pts[1], p, pts[3]);
                    let d4 = tet_vol(pts[0], pts[1], pts[2], p);
                    let inside = (d1 * s0 >= 0.0 && d2 * s0 >= 0.0
                        && d3 * s0 >= 0.0 && d4 * s0 >= 0.0)
                        || (d1 * s0 <= 0.0 && d2 * s0 <= 0.0
                            && d3 * s0 <= 0.0 && d4 * s0 <= 0.0);
                    if inside {
                        let i = (((z - lo[2]) as usize) * dims[1] + ((y - lo[1]) as usize))
                            * dims[0]
                            + ((x - lo[0]) as usize);
                        solid[i] = true;
                    }
                }
            }
        }
    }
    build_from_solid(&solid, lo, dims)
}

/// По маске занятости: BFS-расстояния до границы, знак, затем сглаживание.
fn build_from_solid(solid: &[bool], lo: [i32; 3], dims: [usize; 3]) -> NodeField {
    let n = solid.len();
    /* 1. Расстояние до ближайшего ЗАНЯТОГО узла — BFS от всех занятых. */
    let mut d_in = vec![f32::MAX; n];
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for i in 0..n {
        if solid[i] {
            d_in[i] = 0.0;
            queue.push_back(i);
        }
    }
    bfs(&mut d_in, &mut queue, dims);
    /* 2. И до ближайшего ПУСТОГО — BFS от всех пустых. */
    let mut d_out = vec![f32::MAX; n];
    queue.clear();
    for i in 0..n {
        if !solid[i] {
            d_out[i] = 0.0;
            queue.push_back(i);
        }
    }
    bfs(&mut d_out, &mut queue, dims);
    /* 3. Знаковое расстояние: внутри минус, снаружи плюс. Поверхность — между
     * занятым и пустым узлом, поэтому ноль приходится на середину шага. */
    let mut f = NodeField { lo, dims, val: vec![0.0; n], step: 1.0 };
    for i in 0..n {
        let v = if solid[i] { -(d_out[i] - 0.5) } else { d_in[i] - 0.5 };
        f.set(i, v);
    }
    /* 4. Сглаживание: BFS даёт расстояние по сетке, и изоповерхность повторяет
     * её квантование. Несколько проходов усреднения убирают ступеньки, не
     * сдвигая границу (ноль остаётся между теми же узлами). */
    /* Знак запоминаем ДО сглаживания и восстанавливаем после: усреднение по
     * шести соседям на ТОНКИХ телах (осколки в один-два тета) перемывает
     * внутренность в плюс, и марчинг теряет тело целиком. Сглаживаем только
     * величину — ступеньки сетки уходят, а занятость остаётся точной. */
    let sign: Vec<f32> = f.val.iter().map(|v| if *v < 0.0 { -1.0 } else { 1.0 }).collect();
    for _ in 0..3 {
        let src = f.val.clone();
        for z in 1..dims[2].saturating_sub(1) {
            for y in 1..dims[1].saturating_sub(1) {
                for x in 1..dims[0].saturating_sub(1) {
                    let i = (z * dims[1] + y) * dims[0] + x;
                    let sx = dims[0];
                    let sy = dims[0] * dims[1];
                    let avg = (src[i] + src[i - 1] + src[i + 1] + src[i - sx] + src[i + sx]
                        + src[i - sy] + src[i + sy])
                        / 7.0;
                    f.set(i, avg.abs() * sign[i]);
                }
            }
        }
    }
    for i in 0..n {
        let v = f.val[i];
        if sign[i] < 0.0 {
            f.set(i, -v.abs());
        } else {
            f.set(i, v.abs());
        }
    }
    f
}

/// BFS по шести соседям: заполняет расстояния от уже помеченных нулями узлов.
fn bfs(dist: &mut [f32], queue: &mut std::collections::VecDeque<usize>, dims: [usize; 3]) {
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let mut head = 0usize;
    while head < queue.len() {
        let i = queue[head];
        head += 1;
        let x = i % nx;
        let y = (i / nx) % ny;
        let z = i / (nx * ny);
        let d = dist[i] + 1.0;
        let mut push = |j: usize, dist: &mut [f32], queue: &mut std::collections::VecDeque<usize>| {
            if dist[j] > d {
                dist[j] = d;
                queue.push_back(j);
            }
        };
        if x + 1 < nx {
            push(i + 1, dist, queue);
        }
        if x > 0 {
            push(i - 1, dist, queue);
        }
        if y + 1 < ny {
            push(i + nx, dist, queue);
        }
        if y > 0 {
            push(i - nx, dist, queue);
        }
        if z + 1 < nz {
            push(i + nx * ny, dist, queue);
        }
        if z > 0 {
            push(i - nx * ny, dist, queue);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Узлы шара радиусом `r` (в узлах) с центром `c`.
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

    /// Узлы куба `lo..=hi`.
    fn cube_nodes(lo: (i32, i32, i32), hi: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
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
        from_occupied(nodes.iter().copied(), pad)
    }

    /// Метка области в узле с мировыми координатами.
    fn label_at(f: &NodeField, c: &Components, p: [i32; 3]) -> u32 {
        let i = (((p[2] - f.lo[2]) as usize) * f.dims[1] + (p[1] - f.lo[1]) as usize)
            * f.dims[0]
            + (p[0] - f.lo[0]) as usize;
        c.label[i]
    }

    /// Два шара БЕЗ перемычки — две компоненты (базовый случай).
    #[test]
    fn two_balls_without_bridge_are_two_components() {
        let mut occ = ball_nodes((10, 16, 16), 5);
        occ.extend(ball_nodes((28, 16, 16), 5));
        let f = field_of(&occ, 3);
        let c = components(&f, 1);
        assert_eq!(c.count(), 2, "два раздельных шара обязаны дать 2 компоненты");
        assert!(c.sizes.iter().all(|&s| s > 100), "области не должны быть пустыми: {:?}", c.sizes);
        assert_ne!(
            label_at(&f, &c, [10, 16, 16]),
            label_at(&f, &c, [28, 16, 16]),
            "центры шаров попали в одну компоненту"
        );
    }

    /// Перемычка ТОНЬШЕ ШАГА поля: узлы шаров соприкасаются только УГЛОМ
    /// (разность (1,1,0)). Физически это перемычка нулевой толщины по
    /// решётке — связью она быть не должна: две части, а не одно тело.
    /// 26-связность склеила бы их, 6-связность — нет.
    #[test]
    fn corner_thin_neck_does_not_join_parts() {
        /* Два блока, у которых ближайшие узлы — диагональ: (9,9,9) и (10,10,9). */
        let mut occ = cube_nodes((2, 2, 2), (9, 9, 9));
        occ.extend(cube_nodes((10, 10, 9), (17, 17, 16)));
        let f = field_of(&occ, 3);
        let c = components(&f, 1);
        assert!(
            c.count() >= 2,
            "угловой контакт склеил куски в одно тело: count={}",
            c.count()
        );
        assert_ne!(
            label_at(&f, &c, [2, 2, 2]),
            label_at(&f, &c, [17, 17, 16]),
            "два куска оказались в одной компоненте через угол"
        );
    }

    /// Перемычка ШИРИНОЙ В ОДИН УЗЕЛ (грань-в-грань) — это уже связь:
    /// тело одно, и разбивать его нельзя.
    #[test]
    fn one_node_neck_joins_parts() {
        let mut occ = cube_nodes((2, 2, 2), (9, 9, 9));
        /* перемычка ровно в один узел по x на уровне y=z=9 */
        occ.extend(cube_nodes((10, 9, 9), (11, 9, 9)));
        occ.extend(cube_nodes((12, 9, 9), (19, 16, 16)));
        let f = field_of(&occ, 3);
        let c = components(&f, 1);
        assert_eq!(c.count(), 1, "перемычка в узел обязана держать тело целым");
    }

    /// Куб с ДЫРКОЙ (полостью внутри и сквозным каналом) — одна компонента:
    /// дырка не разрывает тело.
    #[test]
    fn cube_with_hole_is_one_component() {
        let lo = (2, 2, 2);
        let hi = (20, 20, 20);
        let mut occ = cube_nodes(lo, hi);
        /* полость внутри */
        occ.retain(|&(x, y, z)| {
            !(x > 6 && x < 16 && y > 6 && y < 16 && z > 6 && z < 16)
        });
        /* сквозной канал по x — тоже дырка, а не разрыв */
        occ.retain(|&(_, y, z)| !(y == 10 && z == 10));
        let f = field_of(&occ, 3);
        let c = components(&f, 1);
        assert_eq!(c.count(), 1, "куб с дыркой обязан остаться одним телом");
    }

    /// Порог связности: области, разделённые ОДНИМ пустым узлом, при
    /// `min_gap = 1` разные тела, при `min_gap = 2` — одно.
    #[test]
    fn gap_threshold_merges_parts() {
        let mut occ = cube_nodes((2, 2, 2), (9, 9, 9));
        /* второй куб начинается через ОДИН пустой узел (x = 11) */
        occ.extend(cube_nodes((11, 2, 2), (18, 9, 9)));
        let f = field_of(&occ, 3);
        assert_eq!(components(&f, 1).count(), 2, "слой пустоты разделяет при gap=1");
        assert_eq!(components(&f, 2).count(), 1, "gap=2 обязан терпеть один пустой узел");
    }

    /// Цена поиска компонент: 64³ и 128³ узлов, по три прогона.
    #[test]
    fn components_cost_at_64_and_128() {
        let mut line = String::from("компоненты: ");
        for dim in [64usize, 128] {
            let f = blob_field(dim, &[([-10.0, 0.0, 0.0], 9.0), ([10.0, 0.0, 0.0], 9.0)]);
            let mut best = f32::MAX;
            let mut got = 0;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let c = components(&f, 1);
                let ms = t.elapsed().as_secs_f32() * 1e3;
                best = best.min(ms);
                got = c.count();
            }
            assert_eq!(got, 2, "{dim}³: должно быть две области");
            line.push_str(&format!("{dim}³ = {best:.2} мс; "));
        }
        println!("{line}");
    }

    /// Аналитическое поле из шаров (без BFS): нужно только для замера цены.
    fn blob_field(dim: usize, balls: &[([f32; 3], f32)]) -> NodeField {
        let mut val = vec![1.0f32; dim * dim * dim];
        let c = dim as f32 * 0.5;
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

    /// Куб из узлов 0..=4: внутри минус, снаружи плюс, переход через ноль есть.
    #[test]
    fn cube_field_is_signed() {
        let mut occ = Vec::new();
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    occ.push((x, y, z));
                }
            }
        }
        let f = from_occupied(occ.into_iter(), 2);
        assert!(f.at([2, 2, 2]) < 0.0, "центр обязан быть внутри");
        assert!(f.at([-1, 2, 2]) > 0.0, "снаружи обязан быть плюс");
        assert!(f.min_abs() <= 0.51, "нет перехода через ноль: {}", f.min_abs());
    }

    /// Шар: расстояние растёт от центра к краю — проверяем знак на радиусе.
    #[test]
    fn ball_field_grows_outward() {
        let mut occ = Vec::new();
        let r = 4i32;
        for z in -r..=r {
            for y in -r..=r {
                for x in -r..=r {
                    if x * x + y * y + z * z <= r * r {
                        occ.push((x, y, z));
                    }
                }
            }
        }
        let f = from_occupied(occ.into_iter(), 2);
        let c = f.at([0, 0, 0]);
        let mid = f.at([2, 0, 0]);
        let out = f.at([r + 1, 0, 0]);
        assert!(c < mid, "в центре должно быть глубже: {c} vs {mid}");
        assert!(mid < 0.0 && out > 0.0, "знак сломан: {mid}, {out}");
    }

    /// Поле по объёму тетов: у куба из двух тетов переход через ноль есть даже
    /// тогда, когда узлы-вершины не образуют сплошной сетки.
    #[test]
    fn field_from_tet_volume_has_zero_crossing() {
        /* Два больших тета по противоположным углам: у них есть узлы строго
         * внутри (на тонких сколах таких узлов нет, и тест был бы пустым). */
        let tets = [
            [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4)],
            [(4, 4, 4), (0, 4, 4), (4, 0, 4), (4, 4, 0)],
        ];
        let f = from_tets(&tets, 3);
        assert!(f.min_abs() <= 0.51, "нет перехода через ноль: {}", f.min_abs());
        assert!(f.val.iter().any(|v| *v < 0.0), "нет внутренности поля");
    }
}

/// Расстояние от точки до треугольника (точка может лежать вне его плоскости).
pub(crate) fn point_tri_dist(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2];
    let d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2];
    if d1 <= 0.0 && d2 <= 0.0 {
        return (ap[0] * ap[0] + ap[1] * ap[1] + ap[2] * ap[2]).sqrt();
    }
    let bp = [p[0] - b[0], p[1] - b[1], p[2] - b[2]];
    let d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2];
    let d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2];
    if d3 >= 0.0 && d4 <= d3 {
        return (bp[0] * bp[0] + bp[1] * bp[1] + bp[2] * bp[2]).sqrt();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3).max(1e-20);
        let q = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
        return ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
    }
    let cp = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    let d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2];
    let d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2];
    if d6 >= 0.0 && d5 <= d6 {
        return (cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2]).sqrt();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6).max(1e-20);
        let q = [a[0] + ac[0] * t, a[1] + ac[1] * t, a[2] + ac[2] * t];
        return ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6)).max(1e-20);
        let q = [
            b[0] + (c[0] - b[0]) * t,
            b[1] + (c[1] - b[1]) * t,
            b[2] + (c[2] - b[2]) * t,
        ];
        return ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
    }
    /* внутри проекции треугольника — расстояние до плоскости */
    let n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-20);
    ((ap[0] * n[0] + ap[1] * n[1] + ap[2] * n[2]) / len).abs()
}

/// Точное расстояние от точки до поверхности набора тетов (минимум по 4 граням
/// каждого тета). Это и есть SDF дистанция «до меша тела», без сеточной ошибки.
pub(crate) fn dist_to_tets(tets: &[[[f32; 3]; 4]], p: [f32; 3]) -> f32 {
    let mut best = f32::MAX;
    for t in tets {
        for (i, j, k) in [(0usize, 1usize, 2usize), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
            let d = point_tri_dist(p, t[i], t[j], t[k]);
            if d < best {
                best = d;
            }
        }
    }
    best
}

/// Поле с ТОЧНЫМИ расстояниями до поверхности тела: в каждом узле блока лежит
/// евклидово расстояние до ближайшей грани тетов со знаком по занятости.
///
/// Этим и режется ребро при марчинге: доля `t = d0 / (d0 - d1)` даёт точку
/// пересечения реальной поверхности, а не середину ребра сетки. BFS-поле
/// (`from_tets`) остаётся как быстрый черновик: оно меряет расстояние шагами
/// сетки и на диагоналях врёт до √2.
pub fn from_tets_exact(tets: &[[Node; 4]], pad: i32) -> NodeField {
    use std::collections::HashMap as H;
    assert!(!tets.is_empty(), "поле без тетов");
    let (lo, dims) = bounds_of(tets.iter().flat_map(|t| t.iter()), pad);
    let n = dims[0] * dims[1] * dims[2];
    let idx = |x: i32, y: i32, z: i32| -> usize {
        (((z - lo[2]) as usize) * dims[1] + ((y - lo[1]) as usize)) * dims[0]
            + ((x - lo[0]) as usize)
    };

    /* 1. Занятость — по объёму тетов. */
    let mut solid = vec![false; n];
    for t in tets {
        let pts: [[f32; 3]; 4] =
            std::array::from_fn(|i| [t[i].0 as f32, t[i].1 as f32, t[i].2 as f32]);
        let s0 = tet_vol(pts[0], pts[1], pts[2], pts[3]);
        let (mut tlo, mut thi) = ([i32::MAX; 3], [i32::MIN; 3]);
        for q in &pts {
            for k in 0..3 {
                tlo[k] = tlo[k].min(q[k].floor() as i32);
                thi[k] = thi[k].max(q[k].ceil() as i32);
            }
        }
        for z in tlo[2]..=thi[2] {
            for y in tlo[1]..=thi[1] {
                for x in tlo[0]..=thi[0] {
                    let p = [x as f32, y as f32, z as f32];
                    let d1 = tet_vol(p, pts[1], pts[2], pts[3]) * s0;
                    let d2 = tet_vol(pts[0], p, pts[2], pts[3]) * s0;
                    let d3 = tet_vol(pts[0], pts[1], p, pts[3]) * s0;
                    let d4 = tet_vol(pts[0], pts[1], pts[2], p) * s0;
                    if (d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0 && d4 >= 0.0)
                        || (d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0 && d4 <= 0.0)
                    {
                        solid[idx(x, y, z)] = true;
                    }
                }
            }
        }
    }

    /* 2. ГРАНИЧНЫЕ грани: те, что встречаются у одного тета. Внутренние
     * (общие для двух тетов) лежат внутри тела, и расстояние до них — не
     * расстояние до поверхности: считать по всем граням значит опустить поле
     * внутри тела и получить ложные срезы. */
    let mut faces: H<[u64; 3], ([[f32; 3]; 3], u32)> = H::new();
    for t in tets {
        let p: [[f32; 3]; 4] =
            std::array::from_fn(|i| [t[i].0 as f32, t[i].1 as f32, t[i].2 as f32]);
        for (a, b, c) in [(0usize, 1usize, 2usize), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
            let mut key = [0u64; 3];
            for (k, v) in [a, b, c].iter().enumerate() {
                let nd = t[*v];
                key[k] = (((nd.0 as i64 + (1 << 20)) as u64) << 42)
                    | (((nd.1 as i64 + (1 << 20)) as u64) << 21)
                    | ((nd.2 as i64 + (1 << 20)) as u64);
            }
            key.sort();
            faces
                .entry(key)
                .and_modify(|e| e.1 += 1)
                .or_insert(([p[a], p[b], p[c]], 1));
        }
    }
    let boundary: Vec<[[f32; 3]; 3]> = faces
        .values()
        .filter(|(_, cnt)| *cnt == 1)
        .map(|(tri, _)| *tri)
        .collect();

    /* 3. Расстояние до граничных граней — только для узлов рядом с ними. */
    let mut val = vec![f32::MAX; n];
    for tri in &boundary {
        let (mut tlo, mut thi) = ([i32::MAX; 3], [i32::MIN; 3]);
        /* Запас 2 узла вокруг грани: узлы рядом с поверхностью обязаны получить
         * настоящее расстояние, иначе они остаются «далеко» и ломают
         * интерполяцию на соседних рёбрах. */
        for q in tri {
            for k in 0..3 {
                tlo[k] = tlo[k].min(q[k].floor() as i32 - 2).max(lo[k]);
                thi[k] = thi[k].max(q[k].ceil() as i32 + 2).min(lo[k] + dims[k] as i32 - 1);
            }
        }
        for z in tlo[2]..=thi[2] {
            for y in tlo[1]..=thi[1] {
                for x in tlo[0]..=thi[0] {
                    let p = [x as f32, y as f32, z as f32];
                    let d = point_tri_dist(p, tri[0], tri[1], tri[2]);
                    let v = &mut val[idx(x, y, z)];
                    if d < *v {
                        *v = d;
                    }
                }
            }
        }
    }
    /* Узлы, до которых грани не достали, далеко снаружи: значение обязано быть
     * положительным и заведомо большим шага, чтобы не давать ложных срезов. */
    /* Соглашение о нуле: он лежит МЕЖДУ узлами, а не на узлах. У тела на
     * решётке поверхность проходит точно через узлы, и без сдвига значения в
     * них равны нулю — срез вырождается, и поверхность рвётся (проверено:
     * 1184 ребра кратности != 2). Сдвиг полшага узла — та же величина, что была
     * у BFS-поля; меняется только ФОРМА поля: расстояния теперь евклидовы,
     * поэтому интерполяция вдоль ребра идёт по реальной метрике. */
    let far = (dims[0].max(dims[1]).max(dims[2])) as f32;
    for i in 0..n {
        let d = if val[i] == f32::MAX { far } else { val[i] };
        val[i] = if solid[i] { -(d + 0.5) } else { d + 0.5 };
    }
    NodeField { lo, dims, val, step: 1.0 }
}

/// Поле расстояний, построенное ТОЛЬКО по занятости узлов.
///
/// Тетраэдры здесь — лишь способ расставить узлы (какие узлы внутри тела).
/// Грани тетов в расстоянии НЕ участвуют: иначе поверхность повторяет
/// ступеньки тетраэдров. Расстояние считается по 26 соседям с метрическими
/// шагами (1, √2, √3) от границы занятости, затем поле сглаживается.
pub fn from_occupancy_metric(tets: &[[Node; 4]], pad: i32, smooth_iters: usize, step_m: f32) -> NodeField {
    assert!(!tets.is_empty(), "поле без тетов");
    let (lo, dims) = bounds_of(tets.iter().flat_map(|t| t.iter()), pad);
    let n = dims[0] * dims[1] * dims[2];
    let idx = |x: i32, y: i32, z: i32| -> usize {
        (((z - lo[2]) as usize) * dims[1] + ((y - lo[1]) as usize)) * dims[0]
            + ((x - lo[0]) as usize)
    };
    /* 1. Занятость по объёму тетов. */
    let mut solid = vec![false; n];
    for t in tets {
        let pts: [[f32; 3]; 4] =
            std::array::from_fn(|i| [t[i].0 as f32, t[i].1 as f32, t[i].2 as f32]);
        let s0 = tet_vol(pts[0], pts[1], pts[2], pts[3]);
        let (mut tlo, mut thi) = ([i32::MAX; 3], [i32::MIN; 3]);
        for q in &pts {
            for k in 0..3 {
                tlo[k] = tlo[k].min(q[k].floor() as i32);
                thi[k] = thi[k].max(q[k].ceil() as i32);
            }
        }
        for z in tlo[2]..=thi[2] {
            for y in tlo[1]..=thi[1] {
                for x in tlo[0]..=thi[0] {
                    let p = [x as f32, y as f32, z as f32];
                    let d1 = tet_vol(p, pts[1], pts[2], pts[3]) * s0;
                    let d2 = tet_vol(pts[0], p, pts[2], pts[3]) * s0;
                    let d3 = tet_vol(pts[0], pts[1], p, pts[3]) * s0;
                    let d4 = tet_vol(pts[0], pts[1], pts[2], p) * s0;
                    if (d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0 && d4 >= 0.0)
                        || (d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0 && d4 <= 0.0)
                    {
                        solid[idx(x, y, z)] = true;
                    }
                }
            }
        }
    }

    /* 2. Расстояние до ГРАНИЦЫ ЗАНЯТОСТИ: от узлов, у которых есть сосед
     * с другим знаком. Обход по 26 направлениям с метрической длиной шага. */
    let mut dirs: Vec<(i32, i32, i32, f32)> = Vec::with_capacity(26);
    for dx in -1i32..=1 {
        for dy in -1i32..=1 {
            for dz in -1i32..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                dirs.push((dx, dy, dz, ((dx * dx + dy * dy + dz * dz) as f32).sqrt()));
            }
        }
    }
    let dirs = dirs;
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let mut dist = vec![f32::MAX; n];
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = (z * ny + y) * nx + x;
                let mut border = false;
                for (dx, dy, dz, _) in dirs.iter() {
                    let (xx, yy, zz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                    if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i32 || yy >= ny as i32 || zz >= nz as i32 {
                        continue;
                    }
                    let j = ((zz as usize) * ny + yy as usize) * nx + xx as usize;
                    if solid[j] != solid[i] {
                        border = true;
                        break;
                    }
                }
                if border {
                    dist[i] = 0.0;
                    queue.push_back(i);
                }
            }
        }
    }
    /* волна: Dijkstra по малой сетке (шаги 1, √2, √3) */
    while let Some(i) = queue.pop_front() {
        let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
        let d0 = dist[i];
        for (dx, dy, dz, w) in dirs.iter() {
            let (xx, yy, zz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
            if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i32 || yy >= ny as i32 || zz >= nz as i32 {
                continue;
            }
            let j = ((zz as usize) * ny + yy as usize) * nx + xx as usize;
            let nd = d0 + w;
            if nd < dist[j] - 1e-6 {
                dist[j] = nd;
                queue.push_back(j);
            }
        }
    }

    let mut f = NodeField { lo, dims, val: vec![0.0; n], step: 1.0 };
    for i in 0..n {
        let d = if dist[i] == f32::MAX { 0.0 } else { dist[i] };
        f.val[i] = if solid[i] { -(d + 0.5) } else { d + 0.5 };
    }
    /* Значения — в шагах узла, поэтому шаг обязан доехать до поля: от него
     * зависят sample_m, intersect_plane и перевод узлов в мир в физике.
     * Раньше здесь оставалась единица (параметр не использовался), и поле
     * осколка оказывалось в 14 раз крупнее самого осколка. */
    f.step = step_m;
    f.smooth(smooth_iters, 0.6);
    f
}

/// Отсечь полупространством: CSG-пересечение поля с плоскостью `n·p <= d`.
///
/// Это и есть «дробление в любом соотношении»: режем НЕ по границам тетов, а
/// само поле, поэтому скол идёт по произвольной плоскости, а новая поверхность
/// получается гладкой и точной — вершины марчинга лягут ровно на плоскость.
///
/// Знак `max` даёт верный, а величину — нет: у ребра среза он берёт ДАЛЬНЮЮ из
/// двух поверхностей. Проверено (см. тест `shard_teeth_are_mesher_not_field`):
/// на форме меша это не сказывается — у узлов, между которыми проходит ноль,
/// значение и так равно расстоянию до ближней поверхности, а зубцы вдоль ребра
/// среза даёт марчинг (ячейка в два узла), а не формула пересечения: поле,
/// посчитанное как ТОЧНОЕ расстояние до геометрии осколка, даёт ровно те же
/// зубцы и то же отклонение вершин.
pub fn intersect_plane(f: &NodeField, n: [f32; 3], d: f32) -> NodeField {
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
    let n = [n[0] / len, n[1] / len, n[2] / len];
    let d = d / len;
    let mut out = f.clone();
    let (nx, ny, nz) = (f.dims[0], f.dims[1], f.dims[2]);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = (z * ny + y) * nx + x;
                let p = [
                    (f.lo[0] + x as i32) as f32,
                    (f.lo[1] + y as i32) as f32,
                    (f.lo[2] + z as i32) as f32,
                ];
                /* расстояние до плоскости — В ЕДИНИЦАХ УЗЛА, как и значения поля:
                 * раньше здесь ещё умножалось на шаг, и для полей в узловых
                 * единицах срез уезжал в 14 раз глубже (шаг 0.07 м) */
                let plane = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - d;
                out.val[i] = f.val[i].max(plane);
            }
        }
    }
    out
}

/// Обрезать поле до блока узлов `lo..lo+dims` (пересечение с исходным блоком).
///
/// Нужно осколкам: каждый несёт ТОЛЬКО свой кусок поля родителя, а не весь его
/// бокс. Иначе поле осколка — сотни тысяч узлов (не влезает в слот SDF на
/// устройстве), а марчинг перебирает всю сетку родителя на каждый осколок.
pub fn crop(f: &NodeField, lo: [i32; 3], dims: [usize; 3]) -> NodeField {
    let mut out_lo = [0i32; 3];
    let mut out_dims = [0usize; 3];
    for k in 0..3 {
        let a = lo[k].max(f.lo[k]);
        let hi_src = f.lo[k] + f.dims[k] as i32 - 1;
        let b = (lo[k] + dims[k] as i32 - 1).min(hi_src);
        assert!(b >= a, "crop: пустое пересечение по оси {k} ({a}..{b})");
        out_lo[k] = a;
        out_dims[k] = (b - a + 1) as usize;
    }
    let n = out_dims[0] * out_dims[1] * out_dims[2];
    let mut val = Vec::with_capacity(n);
    for z in 0..out_dims[2] {
        for y in 0..out_dims[1] {
            for x in 0..out_dims[0] {
                let sx = (out_lo[0] + x as i32 - f.lo[0]) as usize;
                let sy = (out_lo[1] + y as i32 - f.lo[1]) as usize;
                let sz = (out_lo[2] + z as i32 - f.lo[2]) as usize;
                val.push(f.val[(sz * f.dims[1] + sy) * f.dims[0] + sx]);
            }
        }
    }
    NodeField {
        lo: out_lo,
        dims: out_dims,
        val,
        step: f.step,
    }
}

/// Добавить `layers` узлов ПУСТОТЫ вокруг блока.
///
/// Нужно для мелких осколков: у них занятая область почти вплотную подходит к
/// границе блока, и марчингу не хватает внешних узлов для интерполяции — срез
/// получается «впуклым». Лишние слои дают ему с чем работать.
pub fn expand(f: &NodeField, layers: i32) -> NodeField {
    if layers <= 0 {
        return f.clone();
    }
    let dims = [
        f.dims[0] + 2 * layers as usize,
        f.dims[1] + 2 * layers as usize,
        f.dims[2] + 2 * layers as usize,
    ];
    /* Значения поля — в шагах узла, поэтому и «далеко снаружи» тоже: раньше
     * здесь стояли метры, и у полей в узловых единицах подложка оказывалась
     * меньше типичных значений — марчинг строил поверхность на её границе. */
    let far = f.dims[0].max(f.dims[1]).max(f.dims[2]) as f32 * 2.0;
    let mut out = NodeField {
        lo: [f.lo[0] - layers, f.lo[1] - layers, f.lo[2] - layers],
        dims,
        val: vec![far; dims[0] * dims[1] * dims[2]],
        step: f.step,
    };
    for z in 0..f.dims[2] {
        for y in 0..f.dims[1] {
            for x in 0..f.dims[0] {
                let src = (z * f.dims[1] + y) * f.dims[0] + x;
                let dz = z + layers as usize;
                let dy = y + layers as usize;
                let dx = x + layers as usize;
                let dst = (dz * dims[1] + dy) * dims[0] + dx;
                out.val[dst] = f.val[src];
            }
        }
    }
    out
}
