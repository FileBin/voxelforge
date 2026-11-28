//! T1 — тетраэдр Сомирвилля №1, ячейка BCC-тетраэдральной решётки.
//!
//! Перенос эталона (`SandboxGame/src/world/T1Lattice.hpp`, коммит 19b58dc) на
//! нашу решётку: узлы `Node` — целые координаты в шагах узла своего уровня.
//!
//! ГЕОМЕТРИЯ. Одна ячейка, четыре вершины, в каноническом порядке A, B, D, C:
//!
//!     A = (-1, 0, 0)      D = (0,  1, 1)
//!     B = ( 1, 0, 0)      C = (0, -1, 1)
//!
//!     AB = 2,  DC = 2                  два ДЛИННЫХ противоположных ребра
//!     AC = AD = BC = BD = √3           четыре КОРОТКИХ ребра
//!
//! Значит все четыре грани — равнобедренные (основание 2, бок √3, площадь √2),
//! а объём 2/3 в единицах уровня (6V = 4). Всё это проверяется тестами отсюда
//! же, не «на слово».
//!
//! ПОЧЕМУ ПОРЯДОК A, B, D, C, А НЕ A, B, C, D: для этих координат алфавитный
//! порядок ЛЕВЫЙ (6V = −4). Канонический — правый (6V > 0), длинное ребро в
//! слотах 0-1, второе длинное в 2-3. При фиксированных слотах таблица граней
//! закручена наружу для ЛЮБОЙ канонической ячейки любого уровня — отсюда
//! правильная ориентация граней без пересчёта нормалей на каждой.
//!
//! RED-ДРОБЛЕНИЕ. Бисекция всех шести рёбер: 4 угловых ячейки + центральный
//! октаэдр, разрезанный по диагонали M_AB–M_CD. Все ВОСЕМЬ детей — конгруэнтные
//! T1 вдвое мельче, то есть ячейка самоподобна (у tet-oct-соты этого нет:
//! тетраэдр там дробится в 4 тетраэдра + октаэдр, два типа ячеек навсегда).
//! `red_refine` отдаёт детей в порядке слотов 0..7, `parent_of` — точный
//! целочисленный обратный ход по слоту.
//!
//! ЭТО ДАННЫЕ И ТЕСТЫ. Меш строится не по тетам, а по регулярной сетке поля
//! (см. `TetBody::extract_marching`), поэтому таблица дробления на картинку не
//! влияет: она задаёт разметку физики и проверки инвариантов.

use crate::tet::Node;

pub const VERTICES: usize = 4;
pub const FACES: usize = 4;
pub const EDGES: usize = 6;
pub const CHILDREN: usize = 8;

/// Опорная ячейка в каноническом порядке слотов: A, B, D, C.
pub const REFERENCE: [Node; 4] = [(-1, 0, 0), (1, 0, 0), (0, 1, 1), (0, -1, 1)];

/// Шесть рёбер как пары слотов: сначала два длинных (0-1 = AB, 2-3 = DC),
/// затем четыре коротких.
pub const EDGE: [[usize; 2]; EDGES] = [[0, 1], [2, 3], [0, 2], [0, 3], [1, 2], [1, 3]];

/// Ребро `e` — длинное (длина 2), иначе короткое (√3).
pub const fn edge_is_long(e: usize) -> bool {
    e < 2
}

pub const fn sub(a: Node, b: Node) -> Node {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

pub const fn add(a: Node, b: Node) -> Node {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

pub const fn cross(a: Node, b: Node) -> Node {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

pub const fn dot(a: Node, b: Node) -> i64 {
    a.0 as i64 * b.0 as i64 + a.1 as i64 * b.1 as i64 + a.2 as i64 * b.2 as i64
}

/// Квадрат длины ребра в единицах уровня: 4 у длинного, 3 у короткого.
pub const fn edge_dist2(a: Node, b: Node) -> i64 {
    let d = sub(a, b);
    dot(d, d)
}

/// 6·V тетраэдра (a,b,c,d) — знак задаёт ориентацию, модуль — объём.
pub const fn volume6(v: &[Node; 4]) -> i64 {
    dot(sub(v[1], v[0]), cross(sub(v[2], v[0]), sub(v[3], v[0])))
}

/// Таблица граней, закрученных НАРУЖУ относительно центроида опорной ячейки.
/// Считается при компиляции, как `makeFaces()` эталона.
const fn face_table() -> [[usize; 3]; FACES] {
    let tri = [[0usize, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    let mut f = [[0usize; 3]; FACES];
    /* 4·центроид в целых (координаты REFERENCE целые) */
    let mut c = [0i64; 3];
    let mut i = 0;
    while i < VERTICES {
        c[0] += REFERENCE[i].0 as i64;
        c[1] += REFERENCE[i].1 as i64;
        c[2] += REFERENCE[i].2 as i64;
        i += 1;
    }
    let mut t = 0;
    while t < FACES {
        let i0 = tri[t][0];
        let mut i1 = tri[t][1];
        let mut i2 = tri[t][2];
        let p0 = REFERENCE[i0];
        let p1 = REFERENCE[i1];
        let p2 = REFERENCE[i2];
        let n = cross(sub(p1, p0), sub(p2, p0));
        /* (центроид грани − центроид ячейки) · 12, чтобы остаться в целых */
        let s = (
            4 * (p0.0 as i64 + p1.0 as i64 + p2.0 as i64) - 3 * c[0],
            4 * (p0.1 as i64 + p1.1 as i64 + p2.1 as i64) - 3 * c[1],
            4 * (p0.2 as i64 + p1.2 as i64 + p2.2 as i64) - 3 * c[2],
        );
        let side = n.0 as i64 * s.0 + n.1 as i64 * s.1 + n.2 as i64 * s.2;
        if side < 0 {
            let tmp = i1;
            i1 = i2;
            i2 = tmp;
        }
        f[t] = [i0, i1, i2];
        t += 1;
    }
    f
}

pub const FACE: [[usize; 3]; FACES] = face_table();

/// Канонический порядок любой четвёрки вершин: единственное длинное ребро
/// (d² = 4 против 3 у короткого) в слоты 0-1, второе длинное 2-3, затем правый
/// обход (6V > 0). Длинное ребро строго длиннее, поэтому выбор однозначен для
/// настоящей T1-ячейки.
pub fn canonical(v: [Node; 4]) -> [Node; 4] {
    let (mut li, mut lj, mut best) = (0usize, 1usize, -1i64);
    for i in 0..VERTICES {
        for j in (i + 1)..VERTICES {
            let d = edge_dist2(v[i], v[j]);
            if d > best {
                best = d;
                li = i;
                lj = j;
            }
        }
    }
    let mut rest = [0usize; 2];
    let mut n = 0;
    for k in 0..VERTICES {
        if k != li && k != lj {
            rest[n] = k;
            n += 1;
        }
    }
    let mut t = [v[li], v[lj], v[rest[0]], v[rest[1]]];
    if volume6(&t) < 0 {
        t.swap(2, 3);
    }
    t
}

/// Инвариант, который обязан держать каждый конструктор: 6V > 0, длинные рёбра
/// в слотах 0-1 и 2-3 равны, все четыре поперечных ребра короткие, и при этом
/// 4·(короткое)² = 3·(длинное)² — то есть грани равнобедренные (2, √3, √3).
pub fn is_canonical(v: &[Node; 4]) -> bool {
    if volume6(v) <= 0 {
        return false;
    }
    let long01 = edge_dist2(v[0], v[1]);
    let long23 = edge_dist2(v[2], v[3]);
    if long01 != long23 || long01 <= 0 {
        return false;
    }
    let short02 = edge_dist2(v[0], v[2]);
    if 4 * short02 != 3 * long01 {
        return false;
    }
    let crosses = [
        edge_dist2(v[0], v[3]),
        edge_dist2(v[1], v[2]),
        edge_dist2(v[1], v[3]),
    ];
    crosses.iter().all(|d| *d == short02)
}

/// Ориентировать произвольную четвёрку вершин.
pub fn oriented(a: Node, b: Node, c: Node, d: Node) -> [Node; 4] {
    canonical([a, b, c, d])
}

/// Red-дробление: дети в порядке слотов 0..7, в координатах СЛЕДУЮЩЕГО уровня
/// (углы удвоены, середины — суммы; всё целое). Первый элемент пары — сырой
/// порядок постройки (он нужен `parent_of`), второй — каноническая ячейка.
///
///     слот 0  [A,    M_AB, M_AC, M_AD]  слот 2  [C,    M_CD, M_AC, M_BC]
///     слот 1  [B,    M_AB, M_BC, M_BD]  слот 3  [D,    M_CD, M_AD, M_BD]
///     слот 4  [M_AB, M_CD, M_AC, M_BC]  слот 6  [M_AB, M_CD, M_BD, M_AD]
///     слот 5  [M_AB, M_CD, M_BC, M_BD]  слот 7  [M_AB, M_CD, M_AD, M_AC]
pub fn red_refine(v: &[Node; 4]) -> [([Node; 4], [Node; 4]); CHILDREN] {
    let dbl = |n: Node| (2 * n.0, 2 * n.1, 2 * n.2);
    let p = [dbl(v[0]), dbl(v[1]), dbl(v[2]), dbl(v[3])];
    let u1 = add(v[0], v[1]); // M_AB
    let u2 = add(v[3], v[2]); // M_CD
    let u3 = add(v[0], v[3]); // M_AC
    let u4 = add(v[0], v[2]); // M_AD
    let u5 = add(v[1], v[3]); // M_BC
    let u6 = add(v[1], v[2]); // M_BD
    let raw: [[Node; 4]; CHILDREN] = [
        [p[0], u1, u3, u4],
        [p[1], u1, u5, u6],
        [p[3], u2, u3, u5],
        [p[2], u2, u4, u6],
        [u1, u2, u3, u5],
        [u1, u2, u5, u6],
        [u1, u2, u6, u4],
        [u1, u2, u4, u3],
    ];
    core::array::from_fn(|i| (raw[i], canonical(raw[i])))
}

/// Точный обратный ход: `raw` ребёнка (порядок постройки) + слот → ячейка
/// родителя в каноническом порядке, в целых координатах.
pub fn parent_of(raw: &[Node; 4], slot: usize) -> [Node; 4] {
    let c = *raw;
    let mid = |a: Node, b: Node, c: Node| -> Node {
        ((a.0 + b.0 - c.0) / 2, (a.1 + b.1 - c.1) / 2, (a.2 + b.2 - c.2) / 2)
    };
    let half = |a: Node| -> Node { (a.0 / 2, a.1 / 2, a.2 / 2) };
    let mut p = [(0i32, 0i32, 0i32); 4];
    match slot {
        0 => {
            p[0] = half(c[0]);
            p[1] = sub(c[1], p[0]);
            p[3] = sub(c[2], p[0]);
            p[2] = sub(c[3], p[0]);
        }
        1 => {
            p[1] = half(c[0]);
            p[0] = sub(c[1], p[1]);
            p[3] = sub(c[2], p[1]);
            p[2] = sub(c[3], p[1]);
        }
        2 => {
            p[3] = half(c[0]);
            p[2] = sub(c[1], p[3]);
            p[0] = sub(c[2], p[3]);
            p[1] = sub(c[3], p[3]);
        }
        3 => {
            p[2] = half(c[0]);
            p[3] = sub(c[1], p[2]);
            p[0] = sub(c[2], p[2]);
            p[1] = sub(c[3], p[2]);
        }
        4 => {
            p[3] = mid(c[2], c[3], c[0]);
            p[0] = sub(c[2], p[3]);
            p[1] = sub(c[0], p[0]);
            p[2] = sub(c[1], p[3]);
        }
        5 => {
            p[1] = mid(c[2], c[3], c[1]);
            p[3] = sub(c[2], p[1]);
            p[2] = sub(c[3], p[1]);
            p[0] = sub(c[0], p[1]);
        }
        6 => {
            p[2] = mid(c[2], c[3], c[0]);
            p[1] = sub(c[2], p[2]);
            p[0] = sub(c[3], p[2]);
            p[3] = sub(c[1], p[2]);
        }
        _ => {
            p[0] = mid(c[2], c[3], c[1]);
            p[2] = sub(c[2], p[0]);
            p[3] = sub(c[3], p[0]);
            p[1] = sub(c[0], p[0]);
        }
    }
    canonical(p)
}

/// Ключ ячейки: четыре вершины без порядка (ячейка и есть свой набор вершин на
/// своём уровне, поэтому имя уникально).
pub fn cell_key(v: &[Node; 4]) -> [Node; 4] {
    let mut k = *v;
    k.sort_by_key(|n| crate::tet::pack(*n));
    k
}

/// Ключ грани — отсортированная тройка вершинных кодов.
pub fn face_key(v: &[Node; 4], f: usize) -> [Node; 3] {
    let mut k = [v[FACE[f][0]], v[FACE[f][1]], v[FACE[f][2]]];
    k.sort_by_key(|n| crate::tet::pack(*n));
    k
}

/// Ключ ребра — отсортированная пара.
pub fn edge_key(v: &[Node; 4], e: usize) -> [Node; 2] {
    let mut k = [v[EDGE[e][0]], v[EDGE[e][1]]];
    k.sort_by_key(|n| crate::tet::pack(*n));
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(a: Node, b: Node) -> f32 {
        let d = sub(a, b);
        ((d.0 * d.0 + d.1 * d.1 + d.2 * d.2) as f32).sqrt()
    }

    /// Рёбра: 2, 2, √3 ×4; объём 2/3; грани равнобедренные (2, √3, √3) площади √2.
    #[test]
    fn reference_cell_geometry() {
        let v = REFERENCE;
        assert!(is_canonical(&v), "опорная ячейка обязана быть канонической");
        assert_eq!(volume6(&v), 4, "6V опорной ячейки = 4 (объём 2/3)");
        let want = [2.0f32, 2.0, 3f32.sqrt(), 3f32.sqrt(), 3f32.sqrt(), 3f32.sqrt()];
        for e in 0..EDGES {
            let d = dist(v[EDGE[e][0]], v[EDGE[e][1]]);
            assert!((d - want[e]).abs() < 1e-6, "ребро {e}: {d} вместо {}", want[e]);
            assert_eq!(edge_is_long(e), e < 2);
        }
        /* площадь грани: |(p1-p0)×(p2-p0)|/2 = √2 */
        for f in 0..FACES {
            let (a, b, c) = (v[FACE[f][0]], v[FACE[f][1]], v[FACE[f][2]]);
            let n = cross(sub(b, a), sub(c, a));
            let area = 0.5 * ((n.0 * n.0 + n.1 * n.1 + n.2 * n.2) as f32).sqrt();
            assert!((area - 2f32.sqrt()).abs() < 1e-6, "грань {f}: площадь {area}");
        }
    }

    /// Таблица граней замкнута (каждое ребро ровно в двух гранях) и закручена
    /// наружу — проверка на самом опорном тетраэдре.
    #[test]
    fn facet_table_is_closed_and_outward() {
        let v = REFERENCE;
        let mut count = std::collections::HashMap::new();
        for f in 0..FACES {
            for k in 0..3 {
                let a = FACE[f][k];
                let b = FACE[f][(k + 1) % 3];
                *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert_eq!(count.len(), EDGES, "рёбер в таблице граней: {}", count.len());
        assert!(count.values().all(|c| *c == 2), "ребро не в двух гранях: {count:?}");
        let c = (
            (v[0].0 + v[1].0 + v[2].0 + v[3].0) as f32 / 4.0,
            (v[0].1 + v[1].1 + v[2].1 + v[3].1) as f32 / 4.0,
            (v[0].2 + v[1].2 + v[2].2 + v[3].2) as f32 / 4.0,
        );
        for f in 0..FACES {
            let (a, b, d) = (v[FACE[f][0]], v[FACE[f][1]], v[FACE[f][2]]);
            let n = cross(sub(b, a), sub(d, a));
            let m = (
                (a.0 + b.0 + d.0) as f32 / 3.0 - c.0,
                (a.1 + b.1 + d.1) as f32 / 3.0 - c.1,
                (a.2 + b.2 + d.2) as f32 / 3.0 - c.2,
            );
            let nl = ((n.0 * n.0 + n.1 * n.1 + n.2 * n.2) as f32).sqrt();
            let ml = (m.0 * m.0 + m.1 * m.1 + m.2 * m.2).sqrt();
            let cos = (n.0 as f32 * m.0 + n.1 as f32 * m.1 + n.2 as f32 * m.2) / (nl * ml);
            assert!(cos > 0.1, "грань {f} смотрит внутрь: cos={cos}");
        }
    }

    /// Канонический порядок: алфавитный A,B,C,D — левый, каноника — правый.
    #[test]
    fn canonical_order_is_right_handed() {
        let alpha = [REFERENCE[0], REFERENCE[1], REFERENCE[3], REFERENCE[2]];
        assert!(volume6(&alpha) < 0, "A,B,C,D обязан быть левым");
        let c = canonical(alpha);
        assert!(volume6(&c) > 0, "канонический обязан быть правым");
        assert!(is_canonical(&c));
        assert_eq!(edge_dist2(c[0], c[1]), 4, "длинное ребро — в слотах 0-1");
        assert_eq!(edge_dist2(c[2], c[3]), 4, "второе длинное — в 2-3");
    }

    /// ВОСЕМЬ детей: все конгруэнтны T1 и вдвое мельче (самоподобие 1/2).
    #[test]
    fn red_refine_gives_eight_congruent_children() {
        for level_offset in 0..2 {
            let mut parent = REFERENCE;
            for _ in 0..level_offset {
                parent = red_refine(&parent)[0].1;
            }
            let kids = red_refine(&parent);
            assert_eq!(kids.len(), CHILDREN);
            let mut sum = 0i64;
            for (i, (raw, cell)) in kids.iter().enumerate() {
                assert!(is_canonical(cell), "ребёнок {i} не канонический");
                /* 6V ребёнка в ЕДИНИЦАХ РЕБЁНКА тоже 4 — «конгруэнтен T1» */
                assert_eq!(volume6(cell), 4, "ребёнок {i}: 6V != 4");
                /* сырой порядок постройки может быть ЛЕВЫМ (6V = −4) — это
                 * порядок таблицы эталона, а не ячейка; канонизация его
                 * разворачивает, объём при этом тот же по модулю */
                assert_eq!(volume6(raw).abs(), 4, "сырьё {i}: |6V| != 4");
                assert_eq!(canonical(*raw), *cell, "сырьё канонизируется не в cell");
                sum += 4;
            }
            /* 8 детей по 1/8 объёма родителя; родитель — тоже T1 (6V = 4) */
            assert_eq!(sum, 8 * 4, "сумма объёмов детей");
        }
    }

    /// Дети покрывают родителя по объёму: сумма 6V детей в единицах родителя
    /// равна 6V родителя (координаты вдвое мельче → объём в 8 раз меньше).
    #[test]
    fn red_refine_tiles_the_parent_by_volume() {
        let parent = REFERENCE;
        let kids = red_refine(&parent);
        let sum_parent_units: i64 = kids.iter().map(|(_, cell)| volume6(cell)).sum();
        assert_eq!(sum_parent_units / 8, volume6(&parent), "объём родителя");
    }

    /// parent_of — точный целочисленный обратный ход для каждого слота.
    #[test]
    fn parent_of_round_trips_every_slot() {
        let parent = REFERENCE;
        let kids = red_refine(&parent);
        for (i, (raw, _)) in kids.iter().enumerate() {
            assert_eq!(parent_of(raw, i), canonical(parent), "слот {i}");
            let again = red_refine(&parent_of(raw, i));
            assert_eq!(again[i].0, *raw, "слот {i} не воспроизводится");
        }
    }

    /// Ячейка — свой набор вершин: ключ не зависит от порядка и уникален.
    #[test]
    fn cell_key_is_order_independent() {
        let v = REFERENCE;
        let mut shuffled = [v[3], v[1], v[0], v[2]];
        assert_eq!(cell_key(&v), cell_key(&shuffled));
        shuffled[0] = (7, 0, 0);
        assert_ne!(cell_key(&v), cell_key(&shuffled));
    }
}
