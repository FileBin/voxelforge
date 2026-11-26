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

/// Узел в единицах половины шага решётки на своём уровне.
pub type Node = (i32, i32, i32);

/// Тетраэдр: 4 узла плюс уровень (всегда степень двойки от базы).
#[derive(Clone, Copy, Debug)]
pub struct Tet {
    pub level: u8,
    pub v: [Node; 4],
}

fn pack(n: Node) -> u64 {
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

/// 12 тетраэдров кубической ячейки (углы — биты `x + 2y + 4z`, центр — 8).
const CUBE_TETS: [[u8; 4]; 12] = [
    [0, 2, 6, 8], [0, 6, 4, 8],
    [1, 5, 7, 8], [1, 7, 3, 8],
    [0, 4, 5, 8], [0, 5, 1, 8],
    [2, 3, 7, 8], [2, 7, 6, 8],
    [0, 1, 3, 8], [0, 3, 2, 8],
    [4, 6, 7, 8], [4, 7, 5, 8],
];

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
    pub fn refine(&self) -> [Tet; 8] {
        let [a, b, c, d] = self.v;
        let lv = self.level + 1;
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
}

impl TetBody {
    /// Куб из `n³` кубических ячеек, каждая — 12 тетраэдров.
    pub fn cube(center: [f32; 3], n: i32, level: u8) -> Self {
        let mut tets = Vec::with_capacity((n * n * n * 12) as usize);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let base = (2 * i - n, 2 * j - n, 2 * k - n);
                    let nodes = cube_nodes(base);
                    for t in CUBE_TETS.iter() {
                        tets.push(Tet {
                            level,
                            v: [
                                nodes[t[0] as usize],
                                nodes[t[1] as usize],
                                nodes[t[2] as usize],
                                nodes[t[3] as usize],
                            ],
                        });
                    }
                }
            }
        }
        Self {
            tets,
            level_cap: level + 2,
            origin: center,
        }
    }

    /// Шар из тетраэдров: всё, что ближе радиуса к центру.
    pub fn ball(center: [f32; 3], radius: f32, level: u8) -> Self {
        let a = BASE / (1u32 << level) as f32;
        let n = (radius / a).ceil() as i32 + 1;
        let mut tets = Vec::new();
        for i in -n..n {
            for j in -n..n {
                for k in -n..n {
                    let base = (2 * i, 2 * j, 2 * k);
                    let nodes = cube_nodes(base);
                    for t in CUBE_TETS.iter() {
                        let tet = Tet {
                            level,
                            v: [
                                nodes[t[0] as usize],
                                nodes[t[1] as usize],
                                nodes[t[2] as usize],
                                nodes[t[3] as usize],
                            ],
                        };
                        // центроид локальный (узлы — без origin), центр шара в
                        // локальных координатах — это ноль
                        let c = tet.centroid();
                        let d = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                        if d <= radius as f64 {
                            tets.push(tet);
                        }
                    }
                }
            }
        }
        Self {
            tets,
            level_cap: level,
            origin: center,
        }
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

    /// Раскол: тетраэдры раздаются ближайшей затравке (Вороной). Осколки
    /// выходят неровными, потому что граница режется по центрам тетраэдров.
    pub fn shatter(&self, seeds: usize, seed: u64) -> Vec<TetBody> {
        // центроиды тетраэдров локальные (без origin) — затравки тоже локальные,
        // иначе все тетраэдры достанутся одной-двум ближайшим затравкам
        let (lo, hi) = self.bounds();
        let lo = [lo[0] - self.origin[0], lo[1] - self.origin[1], lo[2] - self.origin[2]];
        let hi = [hi[0] - self.origin[0], hi[1] - self.origin[1], hi[2] - self.origin[2]];
        let mut rng = Lcg::new(seed);
        let pts: Vec<[f64; 3]> = (0..seeds)
            .map(|_| {
                [
                    lo[0] as f64 + (hi[0] - lo[0]) as f64 * rng.unit() as f64,
                    lo[1] as f64 + (hi[1] - lo[1]) as f64 * rng.unit() as f64,
                    lo[2] as f64 + (hi[2] - lo[2]) as f64 * rng.unit() as f64,
                ]
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
        buckets
            .into_iter()
            .filter(|v| !v.is_empty())
            .map(|tets| TetBody {
                tets,
                level_cap: self.level_cap,
                origin: self.origin,
            })
            .collect()
    }

    /// Марчинг по тетраэдрам: граничные треугольники поверхности.
    /// Вершины общие для соседних треугольников — меш водонепроницаемый.
    pub fn surface(&self) -> Mesh {
        // грани, встречающиеся один раз, — поверхность
        let mut faces: HashMap<[u64; 3], (u32, [Node; 3], u8, [f32; 3])> = HashMap::new();
        let origin = self.centroid();
        for t in &self.tets {
            let p: [[f32; 3]; 4] = std::array::from_fn(|i| node_pos(t.v[i], t.level, origin));
            // центр тетраэдра — чтобы ориентировать нормали наружу
            let c = [
                (p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4.0,
                (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4.0,
                (p[0][2] + p[1][2] + p[2][2] + p[3][2]) / 4.0,
            ];
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
            // порядок обхода — наружу от тетраэдра
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
                    positions.push(if !flip {
                        [a, b, c][*idx]
                    } else {
                        [a, b, c][*idx]
                    });
                    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
                    normals.push([n[0] / len, n[1] / len, n[2] / len]);
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
        mesh
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

    // Добивка: остались пары, идущие по ребру в одну сторону — разворачиваем.
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
        for (shard, opposite) in owners {
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
