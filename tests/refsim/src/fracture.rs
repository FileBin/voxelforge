//! Настоящий раскол воксельного тела.
//!
//! Тело — куб из вокселей. Пре-фрактура: K затравок расставляются внутри
//! объёма, каждая занятая ячейка достаётся ближайшей затравке (диаграмма
//! Вороного). Осколки получаются неровные: у них разное число ячеек, разная
//! форма и рваные границы — именно это и видно, когда тело ломается.
//!
//! Физика осколка приближена его AABB: твёрдое тело с массой по числу ячеек.
//! Точная коллизия по форме осколка — отдельная задача (нужны выпуклые
//! оболочки), а для «тело раскололось и разлетелось» этого достаточно; зато
//! картинка честная — воксели двигаются вместе со своим осколком.

use crate::{Body, Shape};

#[derive(Clone, Debug)]
pub struct Shard {
    /// Ячейки в системе координат тела (целые индексы решётки).
    pub cells: Vec<[i32; 3]>,
    /// Центр осколка в мире.
    pub center: [f32; 3],
    /// Полуразмеры AABB для физики.
    pub half: [f32; 3],
    pub mass: f64,
}

pub struct VoxelFracture {
    /// Размер ячейки, м.
    pub cell: f32,
    /// Размер тела в ячейках по ребру.
    pub dim: u32,
    /// Мировой центр тела.
    pub center: [f32; 3],
    pub shards: Vec<Shard>,
}

/// Детерминированный ГПСЧ (LCG) — иначе раскол не воспроизводится.
struct Rng(u64);
impl Rng {
    fn next_f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
}

impl VoxelFracture {
    /// Куб `dim³` из вокселей, расколотый на `seeds` осколков.
    pub fn cube(dim: u32, cell: f32, seeds: usize, center: [f32; 3], density: f64) -> Self {
        let d = dim as i32;
        let half_world = 0.5 * dim as f32 * cell;
        let seed_pts: Vec<[f32; 3]> = {
            let mut rng = Rng(0x5EED_1234_ABCD_9F01);
            (0..seeds)
                .map(|_| {
                    [
                        (rng.next_f() - 0.5) * 0.94 * dim as f32 * cell,
                        (rng.next_f() - 0.5) * 0.94 * dim as f32 * cell,
                        (rng.next_f() - 0.5) * 0.94 * dim as f32 * cell,
                    ]
                })
                .collect()
        };

        let mut buckets: Vec<Vec<[i32; 3]>> = vec![Vec::new(); seeds];
        for z in 0..d {
            for y in 0..d {
                for x in 0..d {
                    // центр ячейки в локальных координатах
                    let p = [
                        (x as f32 + 0.5) * cell - half_world,
                        (y as f32 + 0.5) * cell - half_world,
                        (z as f32 + 0.5) * cell - half_world,
                    ];
                    let mut best = 0usize;
                    let mut best_d = f32::MAX;
                    for (i, s) in seed_pts.iter().enumerate() {
                        let dd = (p[0] - s[0]).powi(2) + (p[1] - s[1]).powi(2) + (p[2] - s[2]).powi(2);
                        if dd < best_d {
                            best_d = dd;
                            best = i;
                        }
                    }
                    buckets[best].push([x, y, z]);
                }
            }
        }

        let mut shards = Vec::new();
        for cells in buckets.into_iter().filter(|c| !c.is_empty()) {
            let n = cells.len();
            let mut lo = [i32::MAX; 3];
            let mut hi = [i32::MIN; 3];
            let mut sum = [0f64; 3];
            for c in &cells {
                for k in 0..3 {
                    lo[k] = lo[k].min(c[k]);
                    hi[k] = hi[k].max(c[k]);
                    sum[k] += c[k] as f64 + 0.5;
                }
            }
            let local_center = [
                (sum[0] / n as f64) as f32 * cell - half_world,
                (sum[1] / n as f64) as f32 * cell - half_world,
                (sum[2] / n as f64) as f32 * cell - half_world,
            ];
            let half = [
                0.5 * (hi[0] - lo[0] + 1) as f32 * cell,
                0.5 * (hi[1] - lo[1] + 1) as f32 * cell,
                0.5 * (hi[2] - lo[2] + 1) as f32 * cell,
            ];
            shards.push(Shard {
                mass: density * n as f64 * (cell as f64).powi(3),
                cells,
                center: [
                    center[0] + local_center[0],
                    center[1] + local_center[1],
                    center[2] + local_center[2],
                ],
                half,
            });
        }

        Self {
            cell,
            dim,
            center,
            shards,
        }
    }

    /// Тела для физики: по одному AABB на осколок.
    pub fn bodies(&self) -> Vec<Body> {
        self.shards
            .iter()
            .map(|s| {
                let mut b = Body::block(
                    [s.center[0] as f64, s.center[1] as f64, s.center[2] as f64],
                    [s.half[0] as f64, s.half[1] as f64, s.half[2] as f64],
                );
                b.mass = s.mass;
                b.shape = Shape::Block;
                b
            })
            .collect()
    }

    /// Локальные координаты ячеек осколка относительно ЕГО центра: вьюер
    /// складывает их с мировой позицией осколка на текущем кадре.
    pub fn local_cells(&self) -> Vec<Vec<[f32; 3]>> {
        let half_world = 0.5 * self.dim as f32 * self.cell;
        self.shards
            .iter()
            .map(|s| {
                let dx = s.center[0] - self.center[0];
                let dy = s.center[1] - self.center[1];
                let dz = s.center[2] - self.center[2];
                s.cells
                    .iter()
                    .map(|c| {
                        [
                            (c[0] as f32 + 0.5) * self.cell - half_world - dx,
                            (c[1] as f32 + 0.5) * self.cell - half_world - dy,
                            (c[2] as f32 + 0.5) * self.cell - half_world - dz,
                        ]
                    })
                    .collect()
            })
            .collect()
    }
}

/// Формат `.vfx` (voxelforge voxel fracture):
/// `"VFX1" | u32 shards | u32 frames | u32 total_cells | f32 cell`
/// далее по осколку: `u32 n`, затем `n × (f32 x,y,z)` — локальные смещения ячеек
/// далее кадры: `shards × (f32 x,y,z)` — мировые центры осколков.
pub fn encode_vfx(fr: &VoxelFracture, frames: &[Vec<[f32; 3]>]) -> Vec<u8> {
    let local = fr.local_cells();
    let total: usize = local.iter().map(|v| v.len()).sum();
    let mut out = Vec::new();
    out.extend_from_slice(b"VFX1");
    out.extend_from_slice(&(fr.shards.len() as u32).to_le_bytes());
    out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&fr.cell.to_le_bytes());
    for cells in &local {
        out.extend_from_slice(&(cells.len() as u32).to_le_bytes());
        for c in cells {
            for k in 0..3 {
                out.extend_from_slice(&c[k].to_le_bytes());
            }
        }
    }
    for f in frames {
        for p in f {
            for k in 0..3 {
                out.extend_from_slice(&p[k].to_le_bytes());
            }
        }
    }
    out
}
