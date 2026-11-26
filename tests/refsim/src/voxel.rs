//! Воксельные тела: поле на решётке + уточняемый патч.
//!
//! Тело хранится как знаковое поле расстояния (SDF) на равномерной решётке.
//! «Воксель разного размера» здесь — это патч с вдвое меньшей ячейкой поверх
//! базовой сетки: тот же объём описан точнее, а не «добавлен новый кубик».
//! Именно это и означает red 8→1: одна базовая ячейка уточняется в восемь
//! вложенных, и поверхность внутри неё перестаёт быть ступенчатой.
//!
//! Контакт тела с полем — это выборка поля в точке: внутри поверхности
//! расстояние отрицательное, и тело выталкивается по градиенту. Поэтому
//! «мелкий воксель проваливается сквозь крупный» — не вопрос размера шага:
//! выборка интерполируется, и дыр в поле нет.

/// Заголовок одной сетки.
#[derive(Clone, Copy, Debug)]
pub struct GridHead {
    pub dim: u32,
    /// Мировая координата узла (0,0,0).
    pub origin: [f32; 3],
    /// Размер ячейки.
    pub cell: f32,
    /// Смещение в общем буфере данных, в элементах f32.
    pub offset: u32,
}

#[derive(Clone)]
pub struct VoxelBody {
    /// Сначала базовые значения (dim³), затем уточнённый патч.
    pub data: Vec<f32>,
    pub base: GridHead,
    pub fine: Option<GridHead>,
}

impl VoxelBody {
    fn node_count(dim: u32) -> usize {
        (dim as usize).pow(3)
    }

    /// Сфера как поле: `|p − c| − r`. Отрицательно внутри.
    pub fn sphere(center: [f32; 3], radius: f32, dim: u32, cell: f32) -> Self {
        let origin = [
            center[0] - 0.5 * (dim - 1) as f32 * cell,
            center[1] - 0.5 * (dim - 1) as f32 * cell,
            center[2] - 0.5 * (dim - 1) as f32 * cell,
        ];
        let mut data = vec![0.0f32; Self::node_count(dim)];
        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    let p = [
                        origin[0] + x as f32 * cell,
                        origin[1] + y as f32 * cell,
                        origin[2] + z as f32 * cell,
                    ];
                    let d = ((p[0] - center[0]).powi(2)
                        + (p[1] - center[1]).powi(2)
                        + (p[2] - center[2]).powi(2))
                    .sqrt()
                        - radius;
                    let i = ((z * dim + y) * dim + x) as usize;
                    data[i] = d;
                }
            }
        }
        Self {
            data,
            base: GridHead {
                dim,
                origin,
                cell,
                offset: 0,
            },
            fine: None,
        }
    }

    /// Коробка как поле: `max(|p − c| − half)` по осям.
    pub fn cube(center: [f32; 3], half: [f32; 3], dim: u32, cell: f32) -> Self {
        let origin = [
            center[0] - 0.5 * (dim - 1) as f32 * cell,
            center[1] - 0.5 * (dim - 1) as f32 * cell,
            center[2] - 0.5 * (dim - 1) as f32 * cell,
        ];
        let mut data = vec![0.0f32; Self::node_count(dim)];
        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    let p = [
                        origin[0] + x as f32 * cell,
                        origin[1] + y as f32 * cell,
                        origin[2] + z as f32 * cell,
                    ];
                    let q = [
                        (p[0] - center[0]).abs() - half[0],
                        (p[1] - center[1]).abs() - half[1],
                        (p[2] - center[2]).abs() - half[2],
                    ];
                    let outside = [
                        q[0].max(0.0),
                        q[1].max(0.0),
                        q[2].max(0.0),
                    ];
                    let d = (outside[0] * outside[0] + outside[1] * outside[1] + outside[2] * outside[2])
                        .sqrt()
                        + q[0].max(q[1]).max(q[2]).min(0.0);
                    data[((z * dim + y) * dim + x) as usize] = d;
                }
            }
        }
        Self {
            data,
            base: GridHead {
                dim,
                origin,
                cell,
                offset: 0,
            },
            fine: None,
        }
    }

    /// Сколько узлов внутри поверхности (отрицательное значение поля).
    pub fn inside_nodes(&self) -> usize {
        self.data.iter().filter(|v| **v < 0.0).count()
    }

    /// Пре-фрактура: разбить тело на 8 октантов по его габаритам.
    ///
    /// Это ровно то, что в плане называется пре-фрактурой: разбиение считается
    /// заранее, до удара, поэтому в момент удара не нужно ничего делить —
    /// обломки уже существуют и спят. Октанты точно замощают исходную коробку:
    /// объём и масса сохраняются, щелей между осколками нет.
    pub fn fracture_octants(&self, center: [f32; 3], half: [f32; 3], density: f64) -> Vec<([f32; 3], [f32; 3], f64)> {
        let mut out = Vec::with_capacity(8);
        let h = [half[0] * 0.5, half[1] * 0.5, half[2] * 0.5];
        for sz in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sx in [-1.0f32, 1.0] {
                    let c = [
                        center[0] + sx * h[0],
                        center[1] + sy * h[1],
                        center[2] + sz * h[2],
                    ];
                    let volume = 8.0 * (h[0] as f64) * (h[1] as f64) * (h[2] as f64);
                    out.push((c, h, density * volume));
                }
            }
        }
        out
    }

    /// Уточнить область: пересобрать поле внутри `[lo, hi]` с ячейкой вдвое
    /// мельче. Форма не меняется — меняется только точность её описания.
    pub fn refine_region(
        &mut self,
        center: [f32; 3],
        radius: f32,
        lo: [f32; 3],
        hi: [f32; 3],
    ) -> GridHead {
        let fine_cell = self.base.cell * 0.5;
        let dim_f = ((hi[0] - lo[0]) / fine_cell).ceil() as u32 + 1;
        let offset = self.data.len() as u32;
        let mut patch = vec![0.0f32; Self::node_count(dim_f)];
        for z in 0..dim_f {
            for y in 0..dim_f {
                for x in 0..dim_f {
                    let p = [
                        lo[0] + x as f32 * fine_cell,
                        lo[1] + y as f32 * fine_cell,
                        lo[2] + z as f32 * fine_cell,
                    ];
                    let d = ((p[0] - center[0]).powi(2)
                        + (p[1] - center[1]).powi(2)
                        + (p[2] - center[2]).powi(2))
                    .sqrt()
                        - radius;
                    patch[((z * dim_f + y) * dim_f + x) as usize] = d;
                }
            }
        }
        self.data.extend_from_slice(&patch);
        let head = GridHead {
            dim: dim_f,
            origin: lo,
            cell: fine_cell,
            offset,
        };
        self.fine = Some(head);
        head
    }

    fn sample_grid(&self, g: &GridHead, p: [f32; 3]) -> f32 {
        let inv = 1.0 / g.cell;
        let fx = (p[0] - g.origin[0]) * inv;
        let fy = (p[1] - g.origin[1]) * inv;
        let fz = (p[2] - g.origin[2]) * inv;
        let n = g.dim as f32 - 1.0;
        if fx < 0.0 || fy < 0.0 || fz < 0.0 || fx > n || fy > n || fz > n {
            return f32::MAX; // вне сетки — «далеко снаружи»
        }
        let (x0, y0, z0) = (fx.floor(), fy.floor(), fz.floor());
        let (tx, ty, tz) = (fx - x0, fy - y0, fz - z0);
        let (x0, y0, z0) = (x0 as u32, y0 as u32, z0 as u32);
        let (x1, y1, z1) = ((x0 + 1).min(g.dim - 1), (y0 + 1).min(g.dim - 1), (z0 + 1).min(g.dim - 1));
        let at = |x: u32, y: u32, z: u32| {
            let i = g.offset as usize + ((z * g.dim + y) * g.dim + x) as usize;
            self.data[i]
        };
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(at(x0, y0, z0), at(x1, y0, z0), tx);
        let c10 = lerp(at(x0, y1, z0), at(x1, y1, z0), tx);
        let c01 = lerp(at(x0, y0, z1), at(x1, y0, z1), tx);
        let c11 = lerp(at(x0, y1, z1), at(x1, y1, z1), tx);
        lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz)
    }

    /// Значение поля в точке: если точка попадает в уточнённый патч — берём его.
    pub fn sample(&self, p: [f32; 3]) -> f32 {
        if let Some(f) = &self.fine {
            let s = self.sample_grid(f, p);
            if s != f32::MAX {
                return s;
            }
        }
        self.sample_grid(&self.base, p)
    }

    /// То же, но только по точкам внутри области: уточняем мы участок, и
    /// сравнивать ошибку надо на нём же, иначе её «съедает» остальная сфера.
    pub fn surface_error_in(
        &self,
        center: [f32; 3],
        radius: f32,
        lo: [f32; 3],
        hi: [f32; 3],
        samples: u32,
    ) -> f64 {
        let mut worst = 0.0f64;
        let n = samples.max(4);
        for i in 0..n {
            for j in 0..n {
                let theta = std::f32::consts::PI * (i as f32 + 0.5) / n as f32;
                let phi = 2.0 * std::f32::consts::PI * (j as f32 + 0.5) / n as f32;
                let p = [
                    center[0] + radius * theta.sin() * phi.cos(),
                    center[1] + radius * theta.cos(),
                    center[2] + radius * theta.sin() * phi.sin(),
                ];
                if (0..3).any(|k| p[k] < lo[k] || p[k] > hi[k]) {
                    continue;
                }
                worst = worst.max(self.sample(p).abs() as f64);
            }
        }
        worst
    }

    /// Максимальная ошибка поля на идеальной поверхности сферы.
    /// На уточнённом патче она обязана падать — иначе уточнение бессмысленно.
    pub fn surface_error(&self, center: [f32; 3], radius: f32, samples: u32) -> f64 {
        let mut worst = 0.0f64;
        let n = samples.max(4);
        for i in 0..n {
            for j in 0..n {
                let theta = std::f32::consts::PI * (i as f32 + 0.5) / n as f32;
                let phi = 2.0 * std::f32::consts::PI * (j as f32 + 0.5) / n as f32;
                let p = [
                    center[0] + radius * theta.sin() * phi.cos(),
                    center[1] + radius * theta.cos(),
                    center[2] + radius * theta.sin() * phi.sin(),
                ];
                worst = worst.max(self.sample(p).abs() as f64);
            }
        }
        worst
    }
}
