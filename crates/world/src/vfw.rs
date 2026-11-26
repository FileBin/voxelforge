//! Формат `.vfw` — мир воксельных тел по кадрам.
//!
//! ```text
//! "VFW1"
//! u32 frames | u32 shapes | u32 cells_total | f32 base
//! shape:  u32 level | u32 n | n × (f32 x, y, z)   — смещения ячеек от центра тела
//! frame:  u32 n_bodies | n_bodies × (u32 shape | f32 pos×3 | f32 quat×4)
//! ```
//!
//! Таблица форм растёт по ходу записи: осколок, раздробившийся на кадре 40,
//! получает новую форму, и кадры после него ссылаются уже на неё. Так видно
//! ровно то, что происходит: крупные ячейки становятся мелкими.

use crate::bcc::VoxelBody;

pub struct Shape {
    pub level: u8,
    /// Смещения ячеек относительно центра масс тела.
    pub offsets: Vec<[f32; 3]>,
    /// Полуразмеры — для отладки и камеры.
    pub half: [f32; 3],
}

pub struct BodyFrame {
    pub shape: u32,
    pub pos: [f32; 3],
    pub quat: [f32; 4],
}

pub fn shape_of(body: &VoxelBody) -> Shape {
    let c = body.centroid();
    let level = body.cells.iter().map(|x| x.level).max().unwrap_or(0);
    let offsets = body
        .cells
        .iter()
        .map(|x| [x.c[0] - c[0], x.c[1] - c[1], x.c[2] - c[2]])
        .collect();
    let (lo, hi) = body.bounds();
    Shape {
        level,
        offsets,
        half: [
            0.5 * (hi[0] - lo[0]),
            0.5 * (hi[1] - lo[1]),
            0.5 * (hi[2] - lo[2]),
        ],
    }
}

pub fn encode(frames: &[Vec<BodyFrame>], shapes: &[Shape]) -> Vec<u8> {
    let total: usize = shapes.iter().map(|s| s.offsets.len()).sum();
    let mut out = Vec::new();
    out.extend_from_slice(b"VFW1");
    out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    out.extend_from_slice(&(shapes.len() as u32).to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&crate::bcc::BASE.to_le_bytes());

    for s in shapes {
        out.extend_from_slice(&(s.level as u32).to_le_bytes());
        out.extend_from_slice(&(s.offsets.len() as u32).to_le_bytes());
        for o in &s.offsets {
            for k in 0..3 {
                out.extend_from_slice(&o[k].to_le_bytes());
            }
        }
    }

    for f in frames {
        out.extend_from_slice(&(f.len() as u32).to_le_bytes());
        for b in f {
            out.extend_from_slice(&b.shape.to_le_bytes());
            for k in 0..3 {
                out.extend_from_slice(&b.pos[k].to_le_bytes());
            }
            for k in 0..4 {
                out.extend_from_slice(&b.quat[k].to_le_bytes());
            }
        }
    }
    out
}
