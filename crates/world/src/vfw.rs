//! Формат `.vfw` — мир воксельных тел по кадрам.
//!
//! ```text
//! "VFW2"
//! u32 frames | u32 shapes | u32 verts_total | f32 base
//! shape:  u32 level | u32 nv | u32 ni | nv × (f32 x,y,z) | ni × u32
//! frame:  u32 m | m × (u32 shape | f32 pos×3 | f32 quat×4)
//! ```
//!
//! Меш формы считается марчингом по тетраэдрам BCC (см. `tet::mesh_body`) и
//! хранится один раз; кадр задаёт только жёсткое движение. Когда осколок
//! дробит ячейки, у него появляется новая форма с более подробным мешем —
//! поэтому в кадре видно, как крупные воксели становятся мелкими.

use crate::tet::{Mesh, TetBody};

pub struct Shape {
    pub level: u8,
    pub mesh: Mesh,
    /// Решётка тела: откуда и с каким шагом идут ячейки (для сетки вокселей).
    pub grid_lo: [f32; 3],
    pub grid_hi: [f32; 3],
    pub grid_step: f32,
    /// Каркас самих тетраэдров (пары точек) — вьюер показывает его вместо меша.
    pub edges: Vec<[f32; 3]>,
}

pub struct BodyFrame {
    pub shape: u32,
    pub pos: [f32; 3],
    pub quat: [f32; 4],
}

pub fn shape_of(body: &TetBody) -> Shape {
    let (mut mesh, grid) = body.surface_grid();
    // Меш строится в мировых координатах (узлы решётки), а кадр задаёт
    // позицию тела — значит вершины должны быть ЛОКАЛЬНЫМИ.
    let c = body.world_centroid();
    for p in mesh.positions.iter_mut() {
        p[0] -= c[0];
        p[1] -= c[1];
        p[2] -= c[2];
    }
    Shape {
        level: mesh.level,
        mesh,
        grid_lo: grid.lo,
        grid_hi: grid.hi,
        grid_step: grid.step,
        edges: body.edge_lines(),
    }
}

pub fn encode(frames: &[Vec<BodyFrame>], shapes: &[Shape]) -> Vec<u8> {
    let total: usize = shapes.iter().map(|s| s.mesh.positions.len()).sum();
    let mut out = Vec::new();
    out.extend_from_slice(b"VFW4");
    out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    out.extend_from_slice(&(shapes.len() as u32).to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&crate::bcc::BASE.to_le_bytes());

    for s in shapes {
        out.extend_from_slice(&(s.level as u32).to_le_bytes());
        out.extend_from_slice(&(s.mesh.positions.len() as u32).to_le_bytes());
        out.extend_from_slice(&(s.mesh.indices.len() as u32).to_le_bytes());
        for k in 0..3 {
            out.extend_from_slice(&s.grid_lo[k].to_le_bytes());
        }
        for k in 0..3 {
            out.extend_from_slice(&s.grid_hi[k].to_le_bytes());
        }
        out.extend_from_slice(&s.grid_step.to_le_bytes());
        for p in &s.mesh.positions {
            for k in 0..3 {
                out.extend_from_slice(&p[k].to_le_bytes());
            }
        }
        for i in &s.mesh.indices {
            out.extend_from_slice(&i.to_le_bytes());
        }
        out.extend_from_slice(&(s.edges.len() as u32).to_le_bytes());
        for p in &s.edges {
            for k in 0..3 {
                out.extend_from_slice(&p[k].to_le_bytes());
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
