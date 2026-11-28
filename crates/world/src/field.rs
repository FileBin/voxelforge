//! Непрерывное поле занятости по узлам решётки — вход для marching tetrahedra.
//!
//! Тело задаётся набором тетов, то есть двоичной занятостью. Марчингу нужно
//! ЧИСЛО в каждом узле, иначе он не интерполирует и поверхность выходит
//! гранёной (именно это и делал `extract_faces`).
//!
//! Поле строим как знаковое расстояние до границы занятой области, в единицах
//! узла: узлы-углы тетов считаем занятыми, расстояние до ближайшего занятого и
//! до ближайшего пустого узла берём точным BFS по шести соседям, а не
//! приближённым двухпроходным шаблоном — на сетке в десятки узлов это дешевле,
//! чем шум в нормалях поверхности.
//!
//! Ноль поля лежит между занятым и пустым узлом, то есть поверхность проходит
//! посередине между ними. Так марчинг даёт сглаженную изоповерхность без
//! «ступенек» на полшага, которые получались от граней тетов.

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
        let x = (p[0] - self.lo[0]) as usize;
        let y = (p[1] - self.lo[1]) as usize;
        let z = (p[2] - self.lo[2]) as usize;
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

/// Построить поле по занятым узлам.
///
/// `pad` — сколько узлов пустоты добавить вокруг занятой области, чтобы
/// марчинг видел внешние теты и поверхность была замкнутой.
pub fn from_occupied<I: Iterator<Item = Node>>(occupied: I, pad: i32) -> NodeField {
    let nodes: Vec<Node> = occupied.collect();
    assert!(!nodes.is_empty(), "поле без занятых узлов");
    let mut lo = [i32::MAX; 3];
    let mut hi = [i32::MIN; 3];
    for n in &nodes {
        let c = [n.0, n.1, n.2];
        for k in 0..3 {
            lo[k] = lo[k].min(c[k]);
            hi[k] = hi[k].max(c[k]);
        }
    }
    for k in 0..3 {
        lo[k] -= pad;
        hi[k] += pad;
    }
    let dims = [
        (hi[0] - lo[0] + 1) as usize,
        (hi[1] - lo[1] + 1) as usize,
        (hi[2] - lo[2] + 1) as usize,
    ];
    let n = dims[0] * dims[1] * dims[2];
    let idx = |p: [i32; 3]| -> usize {
        (((p[2] - lo[2]) as usize) * dims[1] + ((p[1] - lo[1]) as usize)) * dims[0]
            + ((p[0] - lo[0]) as usize)
    };

    /* 1. Занятость (1) и пустота (0). */
    let mut solid = vec![false; n];
    for p in &nodes {
        solid[idx([p.0, p.1, p.2])] = true;
    }

    /* 2. Точное расстояние (в шагах узла) до ближайшего занятого узла — BFS от
     * всех занятых. */
    let mut d_in = vec![f32::MAX; n];
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for i in 0..n {
        if solid[i] {
            d_in[i] = 0.0;
            queue.push_back(i);
        }
    }
    bfs(&mut d_in, &mut queue, dims);

    /* 3. И расстояние до ближайшего ПУСТОГО узла — BFS от всех пустых. */
    let mut d_out = vec![f32::MAX; n];
    queue.clear();
    for i in 0..n {
        if !solid[i] {
            d_out[i] = 0.0;
            queue.push_back(i);
        }
    }
    bfs(&mut d_out, &mut queue, dims);

    /* 4. Знаковое расстояние: внутри минус, снаружи плюс. Поверхность — между
     * занятым и пустым узлом, поэтому 0 приходится на середину шага. */
    let mut f = NodeField { lo, dims, val: vec![0.0; n] };
    for i in 0..n {
        /* У занятого узла расстояние до ближайшего ПУСТОГО (d_out), у пустого —
         * до ближайшего занятого (d_in): иначе у всех занятых узлов расстояние
         * выходило нулевым и поле было плоским внутри тела. */
        let v = if solid[i] { -(d_out[i] - 0.5) } else { d_in[i] - 0.5 };
        f.set(i, v);
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
        if x + 1 < nx { push(i + 1, dist, queue); }
        if x > 0 { push(i - 1, dist, queue); }
        if y + 1 < ny { push(i + nx, dist, queue); }
        if y > 0 { push(i - nx, dist, queue); }
        if z + 1 < nz { push(i + nx * ny, dist, queue); }
        if z > 0 { push(i - nx * ny, dist, queue); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Куб из узлов 0..=4 (5x5x5): внутри значения отрицательные, снаружи
    /// положительные, и между ними есть переход через ноль.
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
        assert_eq!(f.at([2, 2, 2]) < 0.0, true, "центр обязан быть внутри");
        assert_eq!(f.at([-1, 2, 2]) > 0.0, true, "снаружи обязан быть плюс");
        // переход через ноль есть: минимальное |значение| меньше половины шага
        assert!(f.min_abs() <= 0.51, "нет перехода через ноль: {}", f.min_abs());
        // и поле непостоянно — иначе марчингу нечего интерполировать
        let mut uniq = 0.0f32;
        for v in &f.val {
            uniq += v.abs();
        }
        assert!(uniq > 0.0);
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
}
