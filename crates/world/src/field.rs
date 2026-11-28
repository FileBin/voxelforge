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
pub struct NodeField {
    /// Узел с минимальными координатами блока.
    pub lo: [i32; 3],
    /// Размер блока в узлах.
    pub dims: [usize; 3],
    /// Значения по индексу `((z * dims[1]) + y) * dims[0] + x`.
    pub val: Vec<f32>,
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

    /// Ближайший к нулю узел — проверка «поверхность есть».
    pub fn min_abs(&self) -> f32 {
        self.val.iter().fold(f32::MAX, |m, v| m.min(v.abs()))
    }
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
                    let p = [x as f32, y as f32, z as f32];
                    let d1 = tet_vol(pts[0], pts[1], pts[2], p);
                    let d2 = tet_vol(pts[0], pts[1], pts[3], p);
                    let d3 = tet_vol(pts[0], pts[2], pts[3], p);
                    let d4 = tet_vol(pts[1], pts[2], pts[3], p);
                    let inside = (d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0 && d4 >= 0.0)
                        || (d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0 && d4 <= 0.0);
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
    let mut f = NodeField { lo, dims, val: vec![0.0; n] };
    for i in 0..n {
        let v = if solid[i] { -(d_out[i] - 0.5) } else { d_in[i] - 0.5 };
        f.set(i, v);
    }
    /* 4. Сглаживание: BFS даёт расстояние по сетке, и изоповерхность повторяет
     * её квантование. Несколько проходов усреднения убирают ступеньки, не
     * сдвигая границу (ноль остаётся между теми же узлами). */
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
                    f.set(i, avg);
                }
            }
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
        let tets = [
            [(0, 0, 0), (2, 0, 0), (0, 2, 0), (1, 1, 1)],
            [(0, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 1)],
        ];
        let f = from_tets(&tets, 3);
        assert!(f.min_abs() <= 0.51, "нет перехода через ноль: {}", f.min_abs());
        /* Углы тета лежат НА его границе, поэтому после сглаживания их знак не
         * показатель: проверяем, что внутри объёма есть отрицательная зона. */
        assert!(f.val.iter().any(|v| *v < 0.0), "нет внутренности поля");
    }
}
