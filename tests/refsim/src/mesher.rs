//! BCC-мешер: марширующие тетраэдры по полю.
//!
//! Решётка BCC разбивается на тетраэдры шестью тетраэдрами на ячейку (вдоль
//! главной диагонали) — это и есть та самая декомпозиция Соммервилля, ради
//! которой BCC выбран: таблица случаев у неё одна на все уровни, а уточнение
//! red 8→1 просто уменьшает ячейку, не меняя логику мешера.
//!
//! Реализация без таблицы на 16 случаев: для каждого тетраэдра находятся все
//! рёбра со сменой знака поля, из точек пересечения строится веер. Это чуть
//! больше работы, но нет шанса перепутать случай — а их легко перепутать.

use std::collections::HashMap;

use crate::voxel::VoxelBody;

pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    /// Сварка вершин по позиции: без неё каждый тетраэдр рожает свои вершины,
    /// и поверхность формально «не замкнута», хотя геометрически замкнута.
    weld: HashMap<(i64, i64, i64), u32>,
}

/// Шесть тетраэдров на куб (индексы углов по битам x|y<<1|z<<2).
const TETS: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
];

fn grad(body: &VoxelBody, p: [f32; 3], h: f32) -> [f32; 3] {
    let g = [
        body.sample([p[0] + h, p[1], p[2]]) - body.sample([p[0] - h, p[1], p[2]]),
        body.sample([p[0], p[1] + h, p[2]]) - body.sample([p[0], p[1] - h, p[2]]),
        body.sample([p[0], p[1], p[2] + h]) - body.sample([p[0], p[1], p[2] - h]),
    ];
    let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    if len < 1e-9 {
        [0.0, 1.0, 0.0]
    } else {
        [g[0] / len, g[1] / len, g[2] / len]
    }
}

impl Mesh {
    fn vertex(&mut self, body: &VoxelBody, p: [f32; 3], cell: f32) -> u32 {
        const Q: f32 = 1e5;
        let key = (
            (p[0] * Q).round() as i64,
            (p[1] * Q).round() as i64,
            (p[2] * Q).round() as i64,
        );
        if let Some(i) = self.weld.get(&key) {
            return *i;
        }
        let n = grad(body, p, cell * 0.5);
        let i = self.vertices.len() as u32;
        self.vertices.push(p);
        self.normals.push(n);
        self.weld.insert(key, i);
        i
    }
}

/// Построить поверхность нулевого уровня поля.
pub fn mesh_isosurface(body: &VoxelBody, base_cell: f32) -> Mesh {
    let dim = body.base.dim as usize;
    let cell = body.base.cell;
    let origin = body.base.origin;
    let mut out = Mesh {
        vertices: Vec::new(),
        normals: Vec::new(),
        indices: Vec::new(),
        weld: HashMap::new(),
    };

    let at = |x: usize, y: usize, z: usize| -> [f32; 3] {
        [
            origin[0] + x as f32 * cell,
            origin[1] + y as f32 * cell,
            origin[2] + z as f32 * cell,
        ]
    };

    for z in 0..dim - 1 {
        for y in 0..dim - 1 {
            for x in 0..dim - 1 {
                let corner = |i: usize| -> ([f32; 3], f32) {
                    let (cx, cy, cz) = (x + (i & 1), y + ((i >> 1) & 1), z + ((i >> 2) & 1));
                    let p = at(cx, cy, cz);
                    (p, body.sample(p))
                };
                let c: [([f32; 3], f32); 8] = [
                    corner(0),
                    corner(1),
                    corner(2),
                    corner(3),
                    corner(4),
                    corner(5),
                    corner(6),
                    corner(7),
                ];
                for tet in TETS {
                    emit_tet(&mut out, body, cell, base_cell, &[c[tet[0]], c[tet[1]], c[tet[2]], c[tet[3]]]);
                }
            }
        }
    }
    out
}

fn emit_tet(
    out: &mut Mesh,
    body: &VoxelBody,
    cell: f32,
    base_cell: f32,
    tet: &[([f32; 3], f32); 4],
) {
    // Рёбра тетраэдра.
    const EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
    let mut pts: Vec<[f32; 3]> = Vec::with_capacity(4);
    for e in EDGES {
        let (pa, va) = tet[e[0]];
        let (pb, vb) = tet[e[1]];
        if (va < 0.0) == (vb < 0.0) {
            continue;
        }
        let t = va / (va - vb);
        pts.push([
            pa[0] + (pb[0] - pa[0]) * t,
            pa[1] + (pb[1] - pa[1]) * t,
            pa[2] + (pb[2] - pa[2]) * t,
        ]);
    }
    if pts.len() < 3 {
        return;
    }

    /* Канонический порядок точек квада. Без него соседние тетраэдры строят
     * веер от разных вершин, получают разные диагонали — и хотя дыры нет,
     * поверхность перестаёт быть манифолдом (рёбра не парятся). Порядок по
     * углу вокруг центроида плюс поворот к лексикографически меньшей точке
     * даёт обеим сторонам ровно один и тот же веер. */
    let n = grad(body, pts[0], cell.max(base_cell));
    if pts.len() == 4 {
        let c = [
            (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) * 0.25,
            (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) * 0.25,
            (pts[0][2] + pts[1][2] + pts[2][2] + pts[3][2]) * 0.25,
        ];
        let helper = if n[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let u = [
            n[1] * helper[2] - n[2] * helper[1],
            n[2] * helper[0] - n[0] * helper[2],
            n[0] * helper[1] - n[1] * helper[0],
        ];
        let ul = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-9);
        let u = [u[0] / ul, u[1] / ul, u[2] / ul];
        let v = [
            n[1] * u[2] - n[2] * u[1],
            n[2] * u[0] - n[0] * u[2],
            n[0] * u[1] - n[1] * u[0],
        ];
        let mut order: Vec<usize> = (0..4).collect();
        let ang = |p: &[f32; 3]| -> f32 {
            let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            let a = d[0] * u[0] + d[1] * u[1] + d[2] * u[2];
            let b = d[0] * v[0] + d[1] * v[1] + d[2] * v[2];
            b.atan2(a)
        };
        order.sort_by(|&i, &j| ang(&pts[i]).partial_cmp(&ang(&pts[j])).unwrap());
        // Поворот к лексикографически меньшей точке — общий выбор для обеих сторон.
        let mut best = 0;
        for k in 1..4 {
            let a = pts[order[k]];
            let b = pts[order[best]];
            if (a[0], a[1], a[2]) < (b[0], b[1], b[2]) {
                best = k;
            }
        }
        order.rotate_left(best);
        let reordered: Vec<[f32; 3]> = order.iter().map(|&i| pts[i]).collect();
        pts = reordered;
    }

    let idx: Vec<u32> = pts
        .iter()
        .map(|p| out.vertex(body, *p, cell.max(base_cell)))
        .collect();
    for k in 1..idx.len() - 1 {
        let (a, b, c) = (idx[0], idx[k], idx[k + 1]);
        let va = out.vertices[a as usize];
        let vb = out.vertices[b as usize];
        let vc = out.vertices[c as usize];
        let e1 = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
        let e2 = [vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]];
        let nn = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let g = out.normals[a as usize];
        if nn[0] * g[0] + nn[1] * g[1] + nn[2] * g[2] < 0.0 {
            out.indices.push(a);
            out.indices.push(c);
            out.indices.push(b);
        } else {
            out.indices.push(a);
            out.indices.push(b);
            out.indices.push(c);
        }
    }
}

/// Каркас BCC-ячеек: рёбра кубов решётки, которые пересекают поверхность или
/// лежат рядом с ней. Нужен, чтобы в вьюере было видно саму решётку, а не
/// только гладкую поверхность.
pub fn lattice_wireframe(body: &VoxelBody, max_lines: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
    let dim = body.base.dim as usize;
    let cell = body.base.cell;
    let origin = body.base.origin;
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();
    let push_edge = |a: [f32; 3], b: [f32; 3], verts: &mut Vec<[f32; 3]>, idx: &mut Vec<u32>| {
        if idx.len() / 2 >= max_lines {
            return;
        }
        let i = verts.len() as u32;
        verts.push(a);
        verts.push(b);
        idx.push(i);
        idx.push(i + 1);
    };

    for z in 0..dim {
        for y in 0..dim {
            for x in 0..dim {
                let p = [
                    origin[0] + x as f32 * cell,
                    origin[1] + y as f32 * cell,
                    origin[2] + z as f32 * cell,
                ];
                let d = body.sample(p).abs();
                if d > cell * 1.5 {
                    continue; // далеко от поверхности — не рисуем
                }
                if x + 1 < dim {
                    push_edge(p, [p[0] + cell, p[1], p[2]], &mut verts, &mut idx);
                }
                if y + 1 < dim {
                    push_edge(p, [p[0], p[1] + cell, p[2]], &mut verts, &mut idx);
                }
                if z + 1 < dim {
                    push_edge(p, [p[0], p[1], p[2] + cell], &mut verts, &mut idx);
                }
            }
        }
    }
    (verts, idx)
}

/// Формат `.vfm` (voxelforge mesh):
/// `"VFM1" | u32 verts | u32 tris | u32 line_verts`
/// затем `verts × (f32 xyz)`, `tris × (u32 a,b,c)`, `line_verts × (f32 xyz)`.
pub fn encode_vfm(mesh: &Mesh, lines: &[([f32; 3], [f32; 3])]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"VFM1");
    out.extend_from_slice(&(mesh.vertices.len() as u32).to_le_bytes());
    out.extend_from_slice(&(mesh.indices.len() as u32).to_le_bytes());
    out.extend_from_slice(&((lines.len() * 2) as u32).to_le_bytes());
    for v in &mesh.vertices {
        for k in 0..3 {
            out.extend_from_slice(&v[k].to_le_bytes());
        }
    }
    for i in &mesh.indices {
        out.extend_from_slice(&i.to_le_bytes());
    }
    for (a, b) in lines {
        for p in [a, b] {
            for k in 0..3 {
                out.extend_from_slice(&p[k].to_le_bytes());
            }
        }
    }
    out
}
