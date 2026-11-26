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
pub fn tet_tet(a: &[[f32; 3]; 4], b: &[[f32; 3]; 4]) -> Option<([f32; 3], f32)> {
    let mut best_axis = [0.0f32; 3];
    let mut best_overlap = f32::MAX;

    for n in face_normals(a).iter().chain(face_normals(b).iter()) {
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
    Some((best_axis, best_overlap))
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

    // пол и стенки: самый глубокий тетраэдр тела задаёт контакт
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
        if floor_pen > 0.0 {
            out[bi].push(Contact {
                other: -1,
                n: [0.0, 1.0, 0.0],
                pen: floor_pen,
            });
        }
        let wall_dirs = [
            ([1.0, 0.0, 0.0], 0usize),
            ([-1.0, 0.0, 0.0], 1),
            ([0.0, 0.0, 1.0], 2),
            ([0.0, 0.0, -1.0], 3),
        ];
        for (n, idx) in wall_dirs {
            if wall_pen[idx] > 0.0 {
                out[bi].push(Contact {
                    other: -1,
                    n,
                    pen: wall_pen[idx],
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
                if let Some((n, pen)) = tet_tet(&world[b1][t1], &world[b2][t2]) {
                        out[b1].push(Contact {
                            other: b2 as i32,
                            n,
                            pen,
                        });
                        out[b2].push(Contact {
                            other: b1 as i32,
                            n: [-n[0], -n[1], -n[2]],
                            pen,
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
    let mut merged: Vec<Vec<Contact>> = vec![Vec::new(); shapes.len()];
    for (bi, list) in out.iter().enumerate() {
        let mut best: HashMap<i32, Contact> = HashMap::new();
        let mut normal_sum: HashMap<i32, [f32; 3]> = HashMap::new();
        for c in list {
            let e = best.entry(c.other).or_insert(*c);
            if c.pen > e.pen {
                *e = *c;
            }
            let ns = normal_sum.entry(c.other).or_insert([0.0; 3]);
            for k in 0..3 {
                ns[k] += c.n[k] * c.pen.max(1e-6);
            }
        }
        for (other, c) in best {
            let mut n = c.n;
            if other >= 0 {
                if let Some(ns) = normal_sum.get(&other) {
                    let l = (ns[0] * ns[0] + ns[1] * ns[1] + ns[2] * ns[2]).sqrt();
                    if l > 1e-9 {
                        n = [ns[0] / l, ns[1] / l, ns[2] / l];
                    }
                }
            }
            merged[bi].push(Contact { other, n, pen: c.pen });
        }
        merged[bi].sort_by(|a, b| b.pen.partial_cmp(&a.pen).unwrap_or(std::cmp::Ordering::Equal));
    }
    merged
}
