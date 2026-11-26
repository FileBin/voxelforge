//! BCC-воксели: ячейка — усечённый октаэдр (ячейка Вороного решётки BCC),
//! а не куб. У ячейки есть **уровень**: шаг решётки `base / 2^level`, объём
//! `a³/2`. Подразбиение — 8 детей с шагом вдвое меньше: объём, а значит и
//! масса, при дроблении сохраняется точно.
//!
//! Решётка BCC в координатах `a/2`: узлы — целые тройки с чётной суммой.
//! Отсюда два узла на кубическую ячейку: угол и центр тела.

/// Крупная ячейка уровня 0 (шаг решётки BCC, м).
pub const BASE: f32 = 0.28;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    pub level: u8,
    /// Центр ячейки в мире.
    pub c: [f32; 3],
}

impl Cell {
    /// Шаг решётки (ребро кубической ячейки BCC) на этом уровне.
    pub fn edge(&self) -> f32 {
        BASE / (1u32 << self.level) as f32
    }
    /// Объём ячейки: в кубической ячейке BCC два узла, значит `a³/2`.
    pub fn volume(&self) -> f32 {
        let a = self.edge();
        0.5 * a * a * a
    }
}

/// Воксельное тело: набор ячеек плюс предел дробления.
#[derive(Clone, Debug)]
pub struct VoxelBody {
    pub cells: Vec<Cell>,
    pub level_cap: u8,
}

impl VoxelBody {
    /// Куб из BCC-ячеек уровня `level`: `n` кубических ячеек по ребру.
    pub fn cube(center: [f32; 3], n: i32, level: u8) -> Self {
        let a = BASE / (1u32 << level) as f32;
        let half = 0.5 * n as f32 * a;
        let mut cells = Vec::new();
        // координаты в единицах a/2: чётная сумма = узел BCC
        for i in 0..=2 * n {
            for j in 0..=2 * n {
                for k in 0..=2 * n {
                    // BCC в единицах a/2: у узла все три координаты одной чётности
                    // (чётные — угол куба, нечётные — центр тела). «Чётная сумма»
                    // здесь неверна: она даёт центры граней, то есть FCC.
                    if i % 2 != j % 2 || j % 2 != k % 2 {
                        continue;
                    }
                    cells.push(Cell {
                        level,
                        c: [
                            center[0] - half + 0.5 * a * i as f32,
                            center[1] - half + 0.5 * a * j as f32,
                            center[2] - half + 0.5 * a * k as f32,
                        ],
                    });
                }
            }
        }
        Self { cells, level_cap: 2 }
    }

    /// Шар из BCC-ячеек: всё, что ближе радиуса. Снаряд в движке — тоже
    /// воксельное тело, чтобы в мире не было ни одного не-BCC объекта.
    pub fn ball(center: [f32; 3], radius: f32, level: u8) -> Self {
        let a = BASE / (1u32 << level) as f32;
        let n = (radius / a).ceil() as i32 + 1;
        let mut cells = Vec::new();
        for i in -2 * n..=2 * n {
            for j in -2 * n..=2 * n {
                for k in -2 * n..=2 * n {
                    if i.rem_euclid(2) != j.rem_euclid(2) || j.rem_euclid(2) != k.rem_euclid(2) {
                        continue;
                    }
                    let c = [
                        center[0] + 0.5 * a * i as f32,
                        center[1] + 0.5 * a * j as f32,
                        center[2] + 0.5 * a * k as f32,
                    ];
                    let d = ((c[0] - center[0]).powi(2)
                        + (c[1] - center[1]).powi(2)
                        + (c[2] - center[2]).powi(2))
                    .sqrt();
                    if d <= radius {
                        cells.push(Cell { level, c });
                    }
                }
            }
        }
        Self { cells, level_cap: level }
    }

    /// Раздробить все ячейки на один уровень: каждая → 8 детей.
    /// Возвращает `false`, если уже достигнут предел.
    pub fn subdivide(&mut self) -> bool {
        if self.cells.iter().all(|c| c.level >= self.level_cap) {
            return false;
        }
        let mut out = Vec::with_capacity(self.cells.len() * 8);
        for cell in &self.cells {
            if cell.level >= self.level_cap {
                out.push(*cell);
                continue;
            }
            let a = cell.edge();
            let lvl = cell.level + 1;
            // 8 детей вокруг центра: шаг вдвое меньше, объём ровно 1/8
            for sx in [-1.0f32, 1.0] {
                for sy in [-1.0f32, 1.0] {
                    for sz in [-1.0f32, 1.0] {
                        out.push(Cell {
                            level: lvl,
                            c: [
                                cell.c[0] + 0.25 * a * sx,
                                cell.c[1] + 0.25 * a * sy,
                                cell.c[2] + 0.25 * a * sz,
                            ],
                        });
                    }
                }
            }
        }
        self.cells = out;
        true
    }

    /// Суммарный объём — сохраняется при дроблении, по нему считается масса.
    pub fn volume(&self) -> f32 {
        self.cells.iter().map(|c| c.volume()).sum()
    }

    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for c in &self.cells {
            let r = 0.5 * c.edge();
            for k in 0..3 {
                lo[k] = lo[k].min(c.c[k] - r);
                hi[k] = hi[k].max(c.c[k] + r);
            }
        }
        (lo, hi)
    }

    /// Центр масс ячеек.
    pub fn centroid(&self) -> [f32; 3] {
        let mut s = [0.0f64; 3];
        let mut w = 0.0f64;
        for c in &self.cells {
            let v = c.volume() as f64;
            for k in 0..3 {
                s[k] += c.c[k] as f64 * v;
            }
            w += v;
        }
        [(s[0] / w) as f32, (s[1] / w) as f32, (s[2] / w) as f32]
    }

    /// Раскол: ячейки раздаются ближайшей затравке (Вороной в евклидовой
    /// метрике). Осколки выходят неровными — это и видно в кадре.
    pub fn shatter(&self, seeds: usize, seed: u64) -> Vec<VoxelBody> {
        let (lo, hi) = self.bounds();
        let mut rng = Lcg::new(seed);
        let pts: Vec<[f32; 3]> = (0..seeds)
            .map(|_| {
                [
                    lo[0] + (hi[0] - lo[0]) * rng.unit(),
                    lo[1] + (hi[1] - lo[1]) * rng.unit(),
                    lo[2] + (hi[2] - lo[2]) * rng.unit(),
                ]
            })
            .collect();

        let mut buckets: Vec<Vec<Cell>> = vec![Vec::new(); seeds];
        for c in &self.cells {
            let mut best = 0;
            let mut best_d = f32::MAX;
            for (i, p) in pts.iter().enumerate() {
                let d = (c.c[0] - p[0]).powi(2) + (c.c[1] - p[1]).powi(2) + (c.c[2] - p[2]).powi(2);
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            buckets[best].push(*c);
        }
        buckets
            .into_iter()
            .filter(|v| !v.is_empty())
            .map(|cells| VoxelBody {
                cells,
                level_cap: self.level_cap,
            })
            .collect()
    }
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
