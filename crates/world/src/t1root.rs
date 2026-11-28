//! T1Root — кубическая раскладка T1 на пространство (фаза 2 эталона).
//!
//! Перенос `SandboxGame/src/world/T1Root.hpp` (коммит 3b25a0a) на наши
//! координаты узла.
//!
//! РОМБОДОДЕКАЭДР — ячейка Вороного FCC. В этих единицах у него 8 вершин
//! (±1,±1,±1) и 6 вершин (±2,0,0), 12 ромбических граней с короткой диагональю
//! 2 и длинной 2√2, ребро √3, объём 16. Каждую грань режем по КОРОТКОЙ
//! диагонали на два равнобедренных треугольника и ставим конус из ЦЕНТРА
//! додекаэдра: у конуса рёбра {2, 2, √3, √3, √3, √3}, 6V = 4 — значит каждый
//! конус ЕСТЬ T1-ячейка. 12 граней × 2 = 24 ячейки на додекаэдр, объём
//! 24 · 2/3 = 16 — ровно объём додекаэдра, без щелей и наложений.
//!
//! ЦЕНТРЫ додекаэдров — подрешётка FCC `4Z³ ∪ ((2,2,0)+4Z³)`, поэтому на куб
//! 4×4×4 приходится 4 центра и 96 ячеек (3/2 ячейки на кубическую единицу
//! объёма). Вершины всех ячеек лежат в узлах BCC-решётки.
//!
//! ЗАЧЕМ ЭТО ЗДЕСЬ: это разметка (какие T1 стоят в каких местах), а меш
//! строится по сетке поля и от неё не зависит. Модуль даёт данные и
//! инварианты; ни марчинг, ни физика его пока не используют.

use crate::tet::Node;
use crate::t1;

/// 8 вершин «три ребра» — (±1,±1,±1).
pub fn dodeca_three_valent() -> Vec<Node> {
    let mut out = Vec::with_capacity(8);
    for sx in [-1i32, 1] {
        for sy in [-1i32, 1] {
            for sz in [-1i32, 1] {
                out.push((sx, sy, sz));
            }
        }
    }
    out
}

/// 6 вершин «четыре ребра» — (±2,0,0) и перестановки.
pub fn dodeca_four_valent() -> Vec<Node> {
    vec![
        (-2, 0, 0),
        (2, 0, 0),
        (0, -2, 0),
        (0, 2, 0),
        (0, 0, -2),
        (0, 0, 2),
    ]
}

fn dist2(a: Node, b: Node) -> i64 {
    t1::edge_dist2(a, b)
}

/// 12 ромбических граней додекаэдра: каждая — четвёрка вершин (P, s1, Q, s2)
/// в порядке обхода. Пара P,Q — соседние «четырёхрёберные» вершины на
/// расстоянии 2√2, s1,s2 — две трёхрёберные, равноудалённые от обеих на √3.
pub fn dodeca_faces() -> Vec<[Node; 4]> {
    let four = dodeca_four_valent();
    let three = dodeca_three_valent();
    let mut faces = Vec::with_capacity(12);
    for i in 0..four.len() {
        for j in (i + 1)..four.len() {
            let (p, q) = (four[i], four[j]);
            if dist2(p, q) != 8 {
                continue; // 2√2 в квадрате
            }
            let mid: Vec<Node> = three
                .iter()
                .copied()
                .filter(|s| dist2(*s, p) == 3 && dist2(*s, q) == 3)
                .collect();
            assert_eq!(mid.len(), 2, "у ромба обязано быть две трёхрёберные вершины");
            // обход: P → s1 → Q → s2 (s1/s2 по разные стороны от PQ)
            faces.push([p, mid[0], q, mid[1]]);
        }
    }
    faces
}

/// 24 T1-ячейки додекаэдра с центром в `origin`: по два конуса на грань,
/// разрез по КОРОТКОЙ диагонали (s1–s2). Порядок детерминированный: грань за
/// гранью, сначала конус к P, потом к Q.
pub fn dodecahedron_cells_at(origin: Node) -> Vec<[Node; 4]> {
    let mut out = Vec::with_capacity(24);
    for f in dodeca_faces() {
        let (p, s1, q, s2) = (f[0], f[1], f[2], f[3]);
        let shift = |v: Node| (v.0 + origin.0, v.1 + origin.1, v.2 + origin.2);
        out.push(t1::oriented(origin, shift(p), shift(s1), shift(s2)));
        out.push(t1::oriented(origin, shift(q), shift(s1), shift(s2)));
    }
    out
}

/// Центры додекаэдров в кубе `[0,n)³`: подрешётка FCC `4Z³ ∪ ((2,2,0)+4Z³)`.
/// Углы считаются один раз, поэтому на куб 4×4×4 приходится РОВНО 4 центра.
pub fn centers_in_cube(n: i32) -> Vec<Node> {
    let mut out = Vec::new();
    let mut z = 0;
    while z < n {
        let mut y = 0;
        while y < n {
            let mut x = 0;
            while x < n {
                if x % 4 == 0 && y % 4 == 0 && z % 4 == 0 {
                    out.push((x, y, z));
                } else if (x % 4, y % 4, z % 4) == (2, 2, 0)
                    || (x % 4, y % 4, z % 4) == (2, 0, 2)
                    || (x % 4, y % 4, z % 4) == (0, 2, 2)
                {
                    out.push((x, y, z));
                }
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }
    out
}

/// Все T1-ячейки куба `[0,n)³`: 24 на каждый центр раскладки.
pub fn cells_in_cube(n: i32) -> Vec<[Node; 4]> {
    let mut out = Vec::new();
    for c in centers_in_cube(n) {
        out.extend(dodecahedron_cells_at(c));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Ромбододекаэдр: 12 граней, у каждой короткая диагональ 2 и ребро √3.
    #[test]
    fn dodecahedron_has_twelve_rhombi() {
        let faces = dodeca_faces();
        assert_eq!(faces.len(), 12, "граней додекаэдра");
        for f in &faces {
            let (p, s1, q, s2) = (f[0], f[1], f[2], f[3]);
            assert_eq!(dist2(p, s1), 3, "ребро ромба √3");
            assert_eq!(dist2(s1, q), 3);
            assert_eq!(dist2(q, s2), 3);
            assert_eq!(dist2(s2, p), 3);
            assert_eq!(dist2(p, q), 8, "длинная диагональ 2√2");
            assert_eq!(dist2(s1, s2), 4, "короткая диагональ 2");
        }
    }

    /// 24 ячейки на додекаэдр, все — канонические T1 (6V = 4).
    #[test]
    fn dodecahedron_holds_24_t1_cells() {
        let cells = dodecahedron_cells_at((0, 0, 0));
        assert_eq!(cells.len(), 24, "24 T1 на додекаэдр");
        for (i, c) in cells.iter().enumerate() {
            assert!(t1::is_canonical(c), "ячейка {i} не каноническая T1");
            assert_eq!(t1::volume6(c), 4, "ячейка {i}: 6V != 4");
        }
        /* три вершины 8·(±1,±1,±1) + 6·(±2,0,0) + центр = 15 вершин */
        let mut verts: Vec<Node> = cells.iter().flat_map(|c| c.iter().copied()).collect();
        verts.sort_by_key(|v| crate::tet::pack(*v));
        verts.dedup();
        assert_eq!(verts.len(), 15, "вершин додекаэдра + центр");
    }

    /// Объём 24 конусов = объём додекаэдра (16) — «без щелей и наложений».
    #[test]
    fn cells_fill_the_dodecahedron_by_volume() {
        let cells = dodecahedron_cells_at((0, 0, 0));
        let sum: i64 = cells.iter().map(t1::volume6).sum();
        assert_eq!(sum, 24 * 4, "сумма 6V всех конусов");
        /* 6V додекаэдра = 6 · 16 = 96; 24 · 4 = 96 ✓ */
        assert_eq!(sum, 96, "объём додекаэдра: 6V = 96 (V = 16)");
    }

    /// Грани: 24 наружных (половинки ромбов) и 36 внутренних пар
    /// (96 слотов = 24 + 36·2).
    #[test]
    fn facet_census_is_24_surface_and_36_interior() {
        let cells = dodecahedron_cells_at((0, 0, 0));
        let mut owners: HashMap<(u64, u64, u64), usize> = HashMap::new();
        for c in &cells {
            for f in 0..t1::FACES {
                let k = t1::face_key(c, f);
                let key = (
                    crate::tet::pack(k[0]),
                    crate::tet::pack(k[1]),
                    crate::tet::pack(k[2]),
                );
                *owners.entry(key).or_insert(0) += 1;
            }
        }
        let surface = owners.values().filter(|n| **n == 1).count();
        let interior = owners.values().filter(|n| **n == 2).count();
        assert_eq!(surface, 24, "наружных половинок");
        assert_eq!(interior, 36, "внутренних пар");
        assert!(owners.values().all(|n| *n <= 2), "грань у трёх ячеек — щель/наложение");
        assert_eq!(surface + 2 * interior, 24 * 4, "слоты граней");
    }

    /// Раскладка кубическая: 4 центра и 96 T1 на куб 4×4×4.
    #[test]
    fn four_by_four_by_four_cube_holds_96_cells() {
        let centers = centers_in_cube(4);
        assert_eq!(centers.len(), 4, "центров в кубе 4×4×4: {centers:?}");
        let cells = cells_in_cube(4);
        assert_eq!(cells.len(), 96, "96 T1 на куб 4×4×4");
        for c in &cells {
            assert!(t1::is_canonical(c), "ячейка куба не каноническая T1");
        }
        /* и это ровно 3/2 ячейки на единицу объёма: 96 / 64 */
        assert_eq!(cells.len() as f32 / 64.0, 1.5);
    }

    /// Сдвиг центра сдвигает всю ячейку: раскладка периодична.
    #[test]
    fn cells_translate_with_their_center() {
        let a = dodecahedron_cells_at((0, 0, 0));
        let b = dodecahedron_cells_at((4, 0, 0));
        assert_eq!(a.len(), b.len());
        for (ca, cb) in a.iter().zip(b.iter()) {
            let moved: [Node; 4] = ca.map(|v| (v.0 + 4, v.1, v.2));
            assert_eq!(moved, *cb, "ячейка не сдвинулась вместе с центром");
        }
    }
}
