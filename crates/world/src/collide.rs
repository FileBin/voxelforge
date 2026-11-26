//! Контакты по тетраэдрам: столкновение считается не по AABB тела, а по
//! самим вокселям.
//!
//! Каждый тетраэдр — выпуклый, поэтому пара «тетраэдр × тетраэдр» решается
//! SAT по нормалям их граней: если ни одна не разделяет, берётся ось с
//! минимальным перекрытием — это и есть нормаль контакта, а перекрытие —
//! глубина. Пары ищутся по равномерной сетке, так что цена зависит от числа
//! соседей, а не от квадрата числа тетраэдров.
//!
//! Контакты уходят в солвер на видеокарте: он принимает тот же формат
//! (`(индекс партнёра, нормаль, проникновение)`), что и его собственные
//! проходы, поэтому интегрирование и импульсы остаются на GPU.

use std::collections::HashMap;

/// Контакт тела `body` с партнёром `other` (`other = -1` — пол или стенка).
#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub other: i32,
    /// Нормаль смотрит ОТ тела К партнёру (как ждёт солвер).
    pub n: [f32; 3],
    pub pen: f32,
    /// Точка приложения импульса в мире. Без неё солвер берёт плечо как
    /// `n * half_shape`, и любое тело ведёт себя как сфера в ориджине.
    pub p: [f32; 3],
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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
fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// Нормали четырёх граней тетраэдра, ориентированные наружу.
fn face_normals(t: &[[f32; 3]; 4]) -> [[f32; 3]; 4] {
    let c = [
        (t[0][0] + t[1][0] + t[2][0] + t[3][0]) * 0.25,
        (t[0][1] + t[1][1] + t[2][1] + t[3][1]) * 0.25,
        (t[0][2] + t[1][2] + t[2][2] + t[3][2]) * 0.25,
    ];
    let mut out = [[0.0f32; 3]; 4];
    for (i, (a, b, c2, d)) in [
        (0usize, 1usize, 2usize, 3usize),
        (0, 1, 3, 2),
        (0, 2, 3, 1),
        (1, 2, 3, 0),
    ]
    .iter()
    .enumerate()
    {
        let _ = i;
        let n = norm(cross(sub(t[*b], t[*a]), sub(t[*c2], t[*a])));
        // наружу — от центра
        let mid = [
            (t[*a][0] + t[*b][0] + t[*c2][0]) / 3.0,
            (t[*a][1] + t[*b][1] + t[*c2][1]) / 3.0,
            (t[*a][2] + t[*b][2] + t[*c2][2]) / 3.0,
        ];
        let out_dir = sub(mid, c);
        out[i] = if dot(n, out_dir) < 0.0 {
            [-n[0], -n[1], -n[2]]
        } else {
            n
        };
        let _ = *d;
    }
    out
}

fn project(pts: &[[f32; 3]; 4], axis: [f32; 3]) -> (f32, f32) {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for p in pts {
        let d = dot(*p, axis);
        lo = lo.min(d);
        hi = hi.max(d);
    }
    (lo, hi)
}

/// SAT двух тетраэдров. Возвращает (нормаль от a к b, глубина) или None.
pub fn tet_tet(a: &[[f32; 3]; 4], b: &[[f32; 3]; 4]) -> Option<([f32; 3], f32, [f32; 3])> {
    let mut best_axis = [0.0f32; 3];
    let mut best_overlap = f32::MAX;

    // Оси: нормали граней обоих тетраэдров ПЛЮС попарные векторные
    // произведения рёбер. Без последних SAT пропускает разделяющую ось и
    // выдаёт фантомное проникновение (наблюдалось 0.07 м у стоящих рядом
    // осколков).
    let mut axes: Vec<[f32; 3]> = Vec::with_capacity(8 + 36);
    for n in face_normals(a).iter().chain(face_normals(b).iter()) {
        axes.push(*n);
    }
    const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for (i, j) in EDGES {
        let ea = sub(a[j], a[i]);
        for (k, l) in EDGES {
            let eb = sub(b[l], b[k]);
            let c = cross(ea, eb);
            let len = dot(c, c).sqrt();
            if len > 1e-4 {
                axes.push([c[0] / len, c[1] / len, c[2] / len]);
            }
        }
    }

    for n in axes.iter() {
        let (alo, ahi) = project(a, *n);
        let (blo, bhi) = project(b, *n);
        let overlap = ahi.min(bhi) - alo.max(blo);
        if overlap <= 0.0 {
            return None;
        }
        if overlap < best_overlap {
            best_overlap = overlap;
            // ось должна смотреть от a к b
            let ca = [
                (a[0][0] + a[1][0] + a[2][0] + a[3][0]) * 0.25,
                (a[0][1] + a[1][1] + a[2][1] + a[3][1]) * 0.25,
                (a[0][2] + a[1][2] + a[2][2] + a[3][2]) * 0.25,
            ];
            let cb = [
                (b[0][0] + b[1][0] + b[2][0] + b[3][0]) * 0.25,
                (b[0][1] + b[1][1] + b[2][1] + b[3][1]) * 0.25,
                (b[0][2] + b[1][2] + b[2][2] + b[3][2]) * 0.25,
            ];
            best_axis = if dot(sub(cb, ca), *n) < 0.0 {
                [-n[0], -n[1], -n[2]]
            } else {
                *n
            };
        }
    }
    if best_overlap == f32::MAX {
        return None;
    }
    // Точка контакта: середина между самыми глубокими вершинами обоих
    // тетраэдров вдоль найденной оси.
    let n = best_axis;
    let mut va = a[0];
    let mut best = dot(a[0], n);
    for v in a.iter() {
        if dot(*v, n) > best {
            best = dot(*v, n);
            va = *v;
        }
    }
    let mut vb = b[0];
    let mut worst = dot(b[0], n);
    for v in b.iter() {
        if dot(*v, n) < worst {
            worst = dot(*v, n);
            vb = *v;
        }
    }
    let p = [
        0.5 * (va[0] + vb[0]),
        0.5 * (va[1] + vb[1]),
        0.5 * (va[2] + vb[2]),
    ];
    Some((best_axis, best_overlap, p))
}

/// Все контакты мира: тетраэдры против тетраэдров других тел, пола и стенок.
///
/// `shapes[i]` — тетраэдры тела `i` в локальных координатах, `pos[i]` — его
/// позиция в мире. Один тетраэдр может дать несколько контактов, но в список
/// тела попадают только самые глубокие (солвер держит 24 на тело).
pub fn build(
    shapes: &[Vec<[[f32; 3]; 4]>],
    pos: &[[f32; 3]],
    floor_y: f32,
    walls: ([f32; 3], [f32; 3]),
    cell: f32,
) -> Vec<Vec<Contact>> {
    let mut out: Vec<Vec<Contact>> = vec![Vec::new(); shapes.len()];

    // мировые тетраэдры и их AABB
    let mut world: Vec<Vec<[[f32; 3]; 4]>> = Vec::with_capacity(shapes.len());
    let mut grid: HashMap<(i32, i32, i32), Vec<(usize, usize)>> = HashMap::new();
    let key_of = |p: [f32; 3]| -> (i32, i32, i32) {
        (
            (p[0] / cell).floor() as i32,
            (p[1] / cell).floor() as i32,
            (p[2] / cell).floor() as i32,
        )
    };

    for (bi, tets) in shapes.iter().enumerate() {
        let p = pos[bi];
        let w: Vec<[[f32; 3]; 4]> = tets
            .iter()
            .map(|t| {
                let mut v = [[0.0f32; 3]; 4];
                for k in 0..4 {
                    v[k] = [t[k][0] + p[0], t[k][1] + p[1], t[k][2] + p[2]];
                }
                v
            })
            .collect();
        for (ti, t) in w.iter().enumerate() {
            // тетраэдр попадает в клетки своего AABB
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for v in t {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
            let (a, b) = (key_of(lo), key_of(hi));
            for x in a.0..=b.0 {
                for y in a.1..=b.1 {
                    for z in a.2..=b.2 {
                        grid.entry((x, y, z)).or_default().push((bi, ti));
                    }
                }
            }
        }
        world.push(w);
    }

    // пол и стенки: пол — до четырёх точек опоры (иначе коробка балансирует
    // на одной точке и «катится»), стенки — по одной на стенку
    for (bi, w) in world.iter().enumerate() {
        let mut floor_pen = 0.0f32;
        let mut wall_pen = [0.0f32; 4];
        for t in w {
            for v in t {
                if v[1] < floor_y {
                    floor_pen = floor_pen.max(floor_y - v[1]);
                }
                if v[0] < walls.0[0] {
                    wall_pen[0] = wall_pen[0].max(walls.0[0] - v[0]);
                }
                if v[0] > walls.1[0] {
                    wall_pen[1] = wall_pen[1].max(v[0] - walls.1[0]);
                }
                if v[2] < walls.0[2] {
                    wall_pen[2] = wall_pen[2].max(walls.0[2] - v[2]);
                }
                if v[2] > walls.1[2] {
                    wall_pen[3] = wall_pen[3].max(v[2] - walls.1[2]);
                }
            }
        }
        if std::env::var("VF_DEBUG").is_ok() && bi == 0 {
            let mut lowest = f32::MAX;
            for t in w {
                for v in t {
                    lowest = lowest.min(v[1]);
                }
            }
            println!("    [collide] тело 0: pos y {:.2}, низ тетраэдров {:.2}, floor_pen {:.3}", pos[bi][1], lowest, floor_pen);
        }
        let _ = floor_pen;
        if std::env::var("VF_DEBUG").is_ok() {
            let mut lowest = f32::MAX;
            for t in w { for v in t { lowest = lowest.min(v[1]); } }
            println!("    [collide] би {bi}: floor_y {floor_y}, низ {lowest:.4}, floor_pen {floor_pen:.4}");
        }
        let mut cand: Vec<([f32; 3], f32)> = Vec::new();
        for t in w {
            for v in t {
                if v[1] < floor_y {
                    cand.push((*v, floor_y - v[1]));
                }
            }
        }
        if !cand.is_empty() {
            cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut chosen: Vec<([f32; 3], f32)> = vec![cand[0]];
            while chosen.len() < 4 {
                let mut best: Option<([f32; 3], f32)> = None;
                let mut best_d = 0.0f32;
                for (p, d) in &cand {
                    let mut m = f32::MAX;
                    for (q, _) in &chosen {
                        let dx = p[0] - q[0];
                        let dz = p[2] - q[2];
                        m = m.min((dx * dx + dz * dz).sqrt());
                    }
                    if m > best_d {
                        best_d = m;
                        best = Some((*p, *d));
                    }
                }
                // новая опора должна быть заметно в стороне, иначе это та же точка
                match best {
                    Some(b) if best_d > 0.03 => chosen.push(b),
                    _ => break,
                }
            }
            for (p, d) in chosen {
                out[bi].push(Contact {
                    other: -1,
                    n: [0.0, 1.0, 0.0],
                    pen: d,
                    p,
                });
            }
        }
        let wall_dirs = [
            ([1.0, 0.0, 0.0], 0usize),
            ([-1.0, 0.0, 0.0], 1),
            ([0.0, 0.0, 1.0], 2),
            ([0.0, 0.0, -1.0], 3),
        ];
        for (n, idx) in wall_dirs {
            if wall_pen[idx] > 0.0 {
                let mut at = pos[bi];
                let mut deepest = 0.0f32;
                for t in w {
                    for v in t {
                        let d = match idx {
                            0 => walls.0[0] - v[0],
                            1 => v[0] - walls.1[0],
                            2 => walls.0[2] - v[2],
                            _ => v[2] - walls.1[2],
                        };
                        if d > deepest {
                            deepest = d;
                            at = *v;
                        }
                    }
                }
                out[bi].push(Contact {
                    other: -1,
                    n,
                    pen: wall_pen[idx],
                    p: at,
                });
            }
        }
    }

    if std::env::var("VF_TIME").is_ok() {
        let total: usize = world.iter().map(|w| w.len()).sum();
        println!("    [collide] тетраэдров {total}, клеток {}, пар-кандидатов считается", grid.len());
    }
    let t_pairs = std::time::Instant::now();
    let mut pair_checks = 0usize;
    // пары тетраэдров из соседних клеток
    let mut seen: std::collections::HashSet<(usize, usize, usize, usize)> =
        std::collections::HashSet::new();
    for list in grid.values() {
        // сортировка по телу: пары внутри одного тела пропускаются пачкой,
        // а не перебором
        let mut l = list.clone();
        l.sort_by_key(|(b, _)| *b);
        let mut start = 0;
        while start < l.len() {
            let mut end = start;
            while end < l.len() && l[end].0 == l[start].0 {
                end += 1;
            }
            // тело l[start].0 против всех следующих тел в клетке
            for i in start..end {
                for j in (end)..l.len() {
                    let (ba, ta) = l[i];
                    let (bb, tb) = l[j];
                    let (b1, t1, b2, t2) = if ba < bb {
                        (ba, ta, bb, tb)
                    } else {
                        (bb, tb, ba, ta)
                    };
                    if !seen.insert((b1, t1, b2, t2)) {
                        continue;
                    }
                    pair_checks += 1;
                if let Some((n, pen, pt)) = tet_tet(&world[b1][t1], &world[b2][t2]) {
                        out[b1].push(Contact {
                            other: b2 as i32,
                            n,
                            pen,
                            p: pt,
                        });
                        out[b2].push(Contact {
                            other: b1 as i32,
                            n: [-n[0], -n[1], -n[2]],
                            pen,
                            p: pt,
                        });
                    }
                }
            }
            start = end;
        }
    }

    if std::env::var("VF_TIME").is_ok() {
        println!(
            "    [collide] пар проверено {pair_checks}, фаза {:.1} мс",
            t_pairs.elapsed().as_secs_f64() * 1000.0
        );
    }
    // Агрегация по паре тел: десять тетраэдров одного осколка, вошедших в
    // другой, — это ОДИН контакт, а не десять. Иначе солвер складывает
    // одинаковые импульсы и разносит мир (проверено: скорости до 1e17).
    if std::env::var("VF_DEBUG").is_ok() {
        for (bi, list) in out.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            println!("    [collide] тело {bi}: сырых контактов {}", list.len());
            for c in list.iter().take(6) {
                println!("      other {} pen {:.4} n {:?} p {:?}", c.other, c.pen, c.n, c.p);
            }
        }
    }
    let mut merged: Vec<Vec<Contact>> = vec![Vec::new(); shapes.len()];
    for (bi, list) in out.iter().enumerate() {
        let mut by: HashMap<i32, Vec<Contact>> = HashMap::new();
        let mut normal_sum: HashMap<i32, [f32; 3]> = HashMap::new();
        for c in list {
            by.entry(c.other).or_default().push(*c);
            let ns = normal_sum.entry(c.other).or_insert([0.0; 3]);
            for k in 0..3 {
                ns[k] += c.n[k] * c.pen.max(1e-6);
            }
        }
        for (other, mut cs) in by {
            cs.sort_by(|a, b| b.pen.partial_cmp(&a.pen).unwrap_or(std::cmp::Ordering::Equal));
            // Несколько точек опоры на пару: с одной точкой куб, стоящий на
            // кубе, опрокидывается, а не лежит. Берём самые глубокие и
            // разнесённые в пространстве.
            let mut keep: Vec<Contact> = Vec::new();
            for c in cs {
                if keep.len() >= 4 {
                    break;
                }
                let far = keep.iter().all(|k| {
                    let dx = k.p[0] - c.p[0];
                    let dy = k.p[1] - c.p[1];
                    let dz = k.p[2] - c.p[2];
                    (dx * dx + dy * dy + dz * dz).sqrt() > 0.06
                });
                if keep.is_empty() || far {
                    keep.push(c);
                }
            }
            let mut n = keep.first().map(|c| c.n).unwrap_or([0.0, 1.0, 0.0]);
            if other >= 0 {
                if let Some(ns) = normal_sum.get(&other) {
                    let l = (ns[0] * ns[0] + ns[1] * ns[1] + ns[2] * ns[2]).sqrt();
                    if l > 1e-9 {
                        n = [ns[0] / l, ns[1] / l, ns[2] / l];
                    }
                }
            }
            for c in keep {
                merged[bi].push(Contact {
                    other,
                    n,
                    pen: c.pen,
                    p: c.p,
                });
            }
        }
        merged[bi].sort_by(|a, b| b.pen.partial_cmp(&a.pen).unwrap_or(std::cmp::Ordering::Equal));
    }
    merged
}
