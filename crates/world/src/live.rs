// Живой мир: цельный объект, выстрелы, раскол по удару. Всё считается
// на ходу — ни одного заранее заготовленного осколка.
use crate::collide;
use crate::sim::{GpuSim, GpuBody, Params};
use crate::tet::{node_pos, Tet, TetBody};
use crate::types::Body;
use crate::vfw::Shape;

pub const MAX_BODIES: usize = 64;
pub const DENSITY: f64 = 420.0;
pub const MIN_TETS: usize = 20;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Target,
    Shard,
    Bullet,
}

pub struct LiveBody {
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub vel: [f64; 3],
    pub mass: f64,
    pub radius: f32,
    pub tets: Vec<Tet>,
    pub origin: [f32; 3],
    pub level_cap: u8,
    local: Vec<[[f32; 3]; 4]>,
    pub mesh: i32,
    pub lines: i32,
    pub kind: Kind,
    pub alive: bool,
}

impl LiveBody {
    fn rebuild(&mut self) {
        let tb = TetBody {
            tets: self.tets.clone(),
            level_cap: self.level_cap,
            origin: self.origin,
        };
        let c = tb.world_centroid();
        self.local = self
            .tets
            .iter()
            .map(|t| {
                std::array::from_fn(|k| {
                    let p = node_pos(t.v[k], t.level, self.origin);
                    [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
                })
            })
            .collect();
    }

    pub fn radius_of(&self) -> f32 {
        let mut r = 0.0f32;
        for t in &self.local {
            for p in t {
                r = r.max((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt());
            }
        }
        r
    }

    fn world_tets(&self) -> Vec<[[f32; 3]; 4]> {
        self.local
            .iter()
            .map(|t| {
                std::array::from_fn(|k| {
                    let p = rot(self.quat, [t[k][0] as f64, t[k][1] as f64, t[k][2] as f64]);
                    [
                        (p[0] + self.pos[0]) as f32,
                        (p[1] + self.pos[1]) as f32,
                        (p[2] + self.pos[2]) as f32,
                    ]
                })
            })
            .collect()
    }
}

fn rot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[1], q[2], q[3]];
    let cx = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    let t = [cx[0] + q[0] * v[0], cx[1] + q[0] * v[1], cx[2] + q[0] * v[2]];
    let c2 = [u[1] * t[2] - u[2] * t[1], u[2] * t[0] - u[0] * t[2], u[0] * t[1] - u[1] * t[0]];
    [v[0] + 2.0 * c2[0], v[1] + 2.0 * c2[1], v[2] + 2.0 * c2[2]]
}

fn rot_inv(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    rot([q[0], -q[1], -q[2], -q[3]], v)
}

/// Матрица модели (column-major) из позиции и кватерниона.
pub fn model_matrix(pos: [f64; 3], quat: [f64; 4]) -> [f32; 16] {
    let q = quat;
    let (xx, yy, zz) = (q[1] * q[1], q[2] * q[2], q[3] * q[3]);
    let (xy, xz, yz) = (q[1] * q[2], q[1] * q[3], q[2] * q[3]);
    let (wx, wy, wz) = (q[0] * q[1], q[0] * q[2], q[0] * q[3]);
    [
        (1.0 - 2.0 * (yy + zz)) as f32, (2.0 * (xy + wz)) as f32, (2.0 * (xz - wy)) as f32, 0.0,
        (2.0 * (xy - wz)) as f32, (1.0 - 2.0 * (xx + zz)) as f32, (2.0 * (yz + wx)) as f32, 0.0,
        (2.0 * (xz + wy)) as f32, (2.0 * (yz - wx)) as f32, (1.0 - 2.0 * (xx + yy)) as f32, 0.0,
        pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0,
    ]
}

pub struct Live {
    pub bodies: Vec<LiveBody>,
    pub gpu: GpuSim,
    pub mirror: Vec<GpuBody>,
    pub params: Params,
    pub cracks: usize,
    pub shards_made: usize,
    pub shots: usize,
    pub last_log: String,
    pub lines_on: bool,
    pub near: f32,
    pub far: f32,
}

impl Live {
    pub fn new(shader_dir: &std::path::Path) -> Live {
        // Плита сразу лежит на полу: центр на высоте полуразмера, иначе она
        // падает с 1.2 м и к первому выстрелу оказывается не там, куда целятся.
        let plate0 = TetBody::cube([0.0, 0.0, 0.0], 4, 0);
        let (lo0, hi0) = plate0.bounds();
        let rest_y = 0.5 * (hi0[1] - lo0[1]) + 0.001;
        let plate = TetBody::cube([0.0, rest_y, 0.0], 4, 0);
        let params = Params {
            lo: [-8.0, 0.0, -8.0],
            hi: [8.0, 10.0, 8.0],
            cell: 0.30,
            friction: 0.45,
            max_push: 0.22,
            ..Params::default()
        };

        let mut body = LiveBody {
            pos: [0.0, rest_y as f64, 0.0],
            quat: [1.0, 0.0, 0.0, 0.0],
            vel: [0.0; 3],
            mass: DENSITY * plate.volume(),
            radius: 0.5,
            tets: plate.tets.clone(),
            origin: plate.origin,
            level_cap: plate.level_cap,
            local: Vec::new(),
            mesh: -1,
            lines: -1,
            kind: Kind::Target,
            alive: true,
        };
        body.rebuild();

        let mut gpu_bodies: Vec<GpuBody> = Vec::with_capacity(MAX_BODIES);
        gpu_bodies.push(gpu_body(&body));
        for _ in 1..MAX_BODIES {
            let parked = Body::new([0.0, -400.0, 0.0], [0.0; 3], 1.0, 0.05);
            let mut g = GpuBody::from_cpu(&parked);
            g.misc[1] = 1.0;
            gpu_bodies.push(g);
        }
        let gpu = GpuSim::new(gpu_bodies.clone(), shader_dir, params);

        Live {
            bodies: vec![body],
            gpu,
            mirror: gpu_bodies,
            params,
            cracks: 0,
            shards_made: 0,
            shots: 0,
            last_log: String::new(),
            lines_on: false,
            near: 0.0,
            far: 60.0,
        }
    }

    pub fn alive_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.alive).count()
    }

    fn free_slot(&mut self) -> Option<usize> {
        if let Some(i) = self.bodies.iter().position(|b| !b.alive) {
            return Some(i);
        }
        if self.bodies.len() < MAX_BODIES {
            let _ = Body::new([0.0, -400.0, 0.0], [0.0; 3], 1.0, 0.05);
            self.bodies.push(LiveBody {
                pos: [0.0, -400.0, 0.0],
                quat: [1.0, 0.0, 0.0, 0.0],
                vel: [0.0; 3],
                mass: 1.0,
                radius: 0.05,
                tets: Vec::new(),
                origin: [0.0; 3],
                level_cap: 0,
                local: Vec::new(),
                mesh: -1,
                lines: -1,
                kind: Kind::Shard,
                alive: false,
            });
            return Some(self.bodies.len() - 1);
        }
        None
    }

    pub fn step(&mut self, dt: f32, substeps: u32, iterations: u32) {
        // Контакты нельзя снимать один раз на кадр: за 1/60 с снаряд на 26 м/с
        // пролетает 0.43 м — больше тетраэдра, и просто проходит сквозь тело.
        // Дробим кадр так, чтобы самое быстрое тело смещалось не больше
        // трети ячейки за кусок, и пересчитываем контакты на каждом куске.
        let vmax = self
            .bodies
            .iter()
            .filter(|b| b.alive)
            .map(|b| (b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2]).sqrt())
            .fold(0.0f64, f64::max);
        let cell = (self.params.cell * 0.5) as f64;
        let chunks = ((vmax * dt as f64 / (cell * 0.35)).ceil() as u32).clamp(2, 6);
        let per = (substeps / chunks).max(1);
        let sdt = dt / chunks as f32;

        for _ in 0..chunks {
            let mut tet_shapes: Vec<Vec<[[f32; 3]; 4]>> = Vec::with_capacity(MAX_BODIES);
            for b in &self.bodies {
                tet_shapes.push(if b.alive { b.world_tets() } else { Vec::new() });
            }
            while tet_shapes.len() < MAX_BODIES {
                tet_shapes.push(Vec::new());
            }
            let positions: Vec<[f32; 3]> = self
                .mirror
                .iter()
                .map(|b| [b.pos_r[0], b.pos_r[1], b.pos_r[2]])
                .collect();
            let mut contacts = collide::build(
                &tet_shapes,
                &positions,
                0.0,
                (self.params.lo, self.params.hi),
                self.params.cell * 0.5,
            );
            for i in self.bodies.len()..MAX_BODIES {
                contacts[i].clear();
            }

            self.depenetrate(&contacts);
            self.fracture(&contacts);

            self.gpu.write_contacts(&contacts);
            self.gpu.frame_with_contacts(sdt, per, iterations);
            let snap = self.gpu.read_bodies();

            for i in 0..self.bodies.len() {
                if !self.bodies[i].alive {
                    continue;
                }
                let g = &snap[i];
                let b = &mut self.bodies[i];
                b.pos = [g.pos_r[0] as f64, g.pos_r[1] as f64, g.pos_r[2] as f64];
                b.quat = [
                    g.quat[0] as f64,
                    g.quat[1] as f64,
                    g.quat[2] as f64,
                    g.quat[3] as f64,
                ];
                b.vel = [g.vel_m[0] as f64, g.vel_m[1] as f64, g.vel_m[2] as f64];
            }
            let n = self.bodies.len();
            self.mirror[..n].copy_from_slice(&snap[..n]);
        }

        // улетевшее за пределы ящика убираем
        let mut moved = false;
        for i in 0..self.bodies.len() {
            if !self.bodies[i].alive {
                continue;
            }
            let p = self.bodies[i].pos;
            if p[1] < -2.0
                || p[0].abs() > self.params.hi[0] as f64 + 1.5
                || p[2].abs() > self.params.hi[2] as f64 + 1.5
            {
                self.bodies[i].alive = false;
                let parked = Body::new([0.0, -400.0, 0.0], [0.0; 3], 1.0, 0.05);
                let mut g2 = GpuBody::from_cpu(&parked);
                g2.misc[1] = 1.0;
                self.mirror[i] = g2;
                moved = true;
            }
        }
        if moved {
            self.gpu.write_bodies(&self.mirror);
        }
    }

    /// Позиционная развязка — мягко и с потолком: жёсткий «выброс» на всю
    /// глубину контакта в кадр и давал дрожание, будто коллайдер шире меша.
    fn depenetrate(&mut self, contacts: &[Vec<collide::Contact>]) {
        let mut delta: Vec<[f32; 3]> = vec![[0.0; 3]; self.mirror.len()];
        let mut any = false;
        for i in 0..self.bodies.len() {
            if !self.bodies[i].alive {
                continue;
            }
            for c in &contacts[i] {
                if c.pen < 0.006 {
                    continue;
                }
                let k = (0.6 * c.pen).min(0.015);
                if c.other < 0 {
                    for a in 0..3 {
                        delta[i][a] += c.n[a] * k;
                    }
                    any = true;
                } else if (c.other as usize) < self.bodies.len() {
                    let j = c.other as usize;
                    if !self.bodies[j].alive {
                        continue;
                    }
                    let (wi, wj) = (self.mirror[i].vel_m[3], self.mirror[j].vel_m[3]);
                    if wi <= 0.0 || wj <= 0.0 {
                        continue;
                    }
                    let share = wi / (wi + wj);
                    for a in 0..3 {
                        delta[i][a] -= c.n[a] * k * share;
                        delta[j][a] += c.n[a] * k * share;
                    }
                    any = true;
                }
            }
        }
        if !any {
            return;
        }
        for i in 0..self.bodies.len() {
            if delta[i] == [0.0; 3] {
                continue;
            }
            for a in 0..3 {
                self.mirror[i].pos_r[a] += delta[i][a];
                // Якорь сна сдвигаем вместе с телом: иначе GPU видит «прыжок»
                // каждый кадр, таймер покоя сбрасывается и осколки дрожат
                // вечно вместо того, чтобы уснуть.
                self.mirror[i].anchor[a] += delta[i][a];
            }
        }
        self.gpu.write_bodies(&self.mirror);
    }

    /// Наибольшее проникновение в текущей позе (м) — для проверок и отладки.
    pub fn max_pen(&self) -> f32 {
        let mut tet_shapes: Vec<Vec<[[f32; 3]; 4]>> = Vec::with_capacity(MAX_BODIES);
        for b in &self.bodies {
            tet_shapes.push(if b.alive { b.world_tets() } else { Vec::new() });
        }
        while tet_shapes.len() < MAX_BODIES {
            tet_shapes.push(Vec::new());
        }
        let positions: Vec<[f32; 3]> = self
            .mirror
            .iter()
            .map(|b| [b.pos_r[0], b.pos_r[1], b.pos_r[2]])
            .collect();
        let contacts = collide::build(
            &tet_shapes,
            &positions,
            0.0,
            (self.params.lo, self.params.hi),
            self.params.cell * 0.5,
        );
        contacts
            .iter()
            .flatten()
            .map(|c| c.pen)
            .fold(0.0f32, f32::max)
    }

    /// Наибольшая скорость среди живых тел.
    pub fn max_speed(&self) -> f64 {
        self.bodies
            .iter()
            .filter(|b| b.alive)
            .map(|b| (b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2]).sqrt())
            .fold(0.0, f64::max)
    }

    fn fracture(&mut self, contacts: &[Vec<collide::Contact>]) {
        let mut jobs: Vec<(usize, [f32; 3], f32)> = Vec::new();
        for (i, list) in contacts.iter().enumerate() {
            if i >= self.bodies.len() {
                break;
            }
            for c in list {
                if c.other < 0 || c.other as usize >= self.bodies.len() {
                    continue;
                }
                let j = c.other as usize;
                if i >= j || !self.bodies[i].alive || !self.bodies[j].alive {
                    continue;
                }
                let a = self.bodies[i].vel;
                let b = self.bodies[j].vel;
                let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                let n = [c.n[0] as f64, c.n[1] as f64, c.n[2] as f64];
                let vrel = (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]).abs();
                if vrel < 2.0 {
                    continue;
                }
                // цель — не снаряд: снаряд мелкий и колется только он сам,
                // а нам нужен тот, в кого попали
                let (ti, oi) = match (self.bodies[i].kind == Kind::Bullet, self.bodies[j].kind == Kind::Bullet) {
                    (false, true) => (i, j),
                    (true, false) => (j, i),
                    (true, true) => continue,
                    (false, false) => {
                        if self.bodies[i].tets.len() >= self.bodies[j].tets.len() {
                            (i, j)
                        } else {
                            (j, i)
                        }
                    }
                };
                if self.bodies[ti].tets.len() < MIN_TETS || self.bodies[ti].kind == Kind::Bullet {
                    continue;
                }
                let mu = 1.0 / (1.0 / self.bodies[ti].mass + 1.0 / self.bodies[oi].mass);
                let e = 0.5 * mu * vrel * vrel;
                if e < 250.0 {
                    continue;
                }
                let pi = self.bodies[ti].pos;
                let pn = [c.n[0] as f64, c.n[1] as f64, c.n[2] as f64];
                let point = [
                    (pi[0] - pn[0] * 0.2) as f32,
                    (pi[1] - pn[1] * 0.2) as f32,
                    (pi[2] - pn[2] * 0.2) as f32,
                ];
                jobs.push((ti, point, e as f32));
            }
        }
        for (ti, point, energy) in jobs {
            if !self.bodies[ti].alive {
                continue;
            }
            let free = self.bodies.iter().filter(|b| !b.alive).count()
                + (MAX_BODIES - self.bodies.len());
            if free < 2 {
                continue;
            }
            self.split(ti, point, energy, free.min(8));
        }
    }

    /// Разбить тело ti в точке point. Число осколков и разлёт — от энергии.
    pub fn split(&mut self, ti: usize, point: [f32; 3], energy: f32, max_new: usize) {
        let (tq, tpos, tvel) = {
            let t = &self.bodies[ti];
            (t.quat, t.pos, t.vel)
        };
        let tb = TetBody {
            tets: self.bodies[ti].tets.clone(),
            level_cap: self.bodies[ti].level_cap,
            origin: self.bodies[ti].origin,
        };
        let c = tb.world_centroid();
        // точка удара в координатах решётки этого тела
        let p_local = rot_inv(
            tq,
            [
                point[0] as f64 - tpos[0],
                point[1] as f64 - tpos[1],
                point[2] as f64 - tpos[2],
            ],
        );
        let focus = [
            (p_local[0] + c[0] as f64) as f32,
            (p_local[1] + c[1] as f64) as f32,
            (p_local[2] + c[2] as f64) as f32,
        ];

        let n = (2.0 + energy / 700.0).clamp(3.0, max_new as f32) as usize;
        let mut frags = tb.shatter_biased(n, 0x51ED_2A17 ^ (self.cracks as u64 + 1), focus);
        frags.truncate(max_new);
        if frags.len() < 2 {
            return;
        }

        let mass_total: f64 = frags.iter().map(|f| DENSITY * f.volume()).sum();
        let v_rms = (2.0 * 0.45 * energy as f64 / mass_total.max(1e-6))
            .sqrt()
            .min(20.0);

        let mut rng = crate::tet::Lcg::new(0xB17E_5EED ^ (self.cracks as u64));
        let mut created = 0usize;

        for (k, fr) in frags.iter().enumerate() {
            let lc = fr.world_centroid();
            // мировой центр осколка (учитывая поворот тела)
            let wp = rot(tq, [lc[0] as f64, lc[1] as f64, lc[2] as f64]);
            let wp = [wp[0] + tpos[0], wp[1] + tpos[1], wp[2] + tpos[2]];
            let d = [wp[0] - point[0] as f64, wp[1] - point[1] as f64, wp[2] - point[2] as f64];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3);
            let speed = v_rms * (0.5 + 0.9 * rng.unit() as f64);

            let slot = if k == 0 { Some(ti) } else { self.free_slot() };
            let Some(slot) = slot else { break };

            let mut body = LiveBody {
                pos: wp,
                quat: tq,
                vel: [
                    d[0] / len * speed + tvel[0] * 0.4,
                    d[1] / len * speed + tvel[1] * 0.4,
                    d[2] / len * speed + tvel[2] * 0.4,
                ],
                mass: DENSITY * fr.volume(),
                radius: 0.2,
                tets: fr.tets.clone(),
                origin: fr.origin,
                level_cap: fr.level_cap,
                local: Vec::new(),
                mesh: -1,
                lines: -1,
                kind: Kind::Shard,
                alive: true,
            };
            body.rebuild();
            body.radius = body.radius_of();
            let gb = gpu_body(&body);
            self.mirror[slot] = gb;
            if slot < self.bodies.len() {
                self.bodies[slot] = body;
            } else {
                self.bodies.push(body);
            }
            created += 1;
        }

        if created >= 2 {
            self.gpu.write_bodies(&self.mirror);
            self.cracks += 1;
            self.shards_made += created;
            self.last_log = format!(
                "трещина #{}: {} осколков, удар {:.1} кДж",
                self.cracks,
                created,
                energy as f64 / 1000.0
            );
        }
    }

    /// Выстрел из точки from в направлении dir.
    pub fn shoot(&mut self, from: [f64; 3], dir: [f64; 3], speed: f64) -> bool {
        let Some(slot) = self.free_slot() else { return false };
        // снаряд — гранёный кусок, а не шар
        let ball = TetBody::cube([0.0, 0.0, 0.0], 2, 1);
        let mut body = LiveBody {
            pos: from,
            quat: [1.0, 0.0, 0.0, 0.0],
            vel: [dir[0] * speed, dir[1] * speed, dir[2] * speed],
            mass: 90.0,
            radius: 0.14,
            tets: ball.tets.clone(),
            origin: ball.origin,
            level_cap: ball.level_cap,
            local: Vec::new(),
            mesh: -1,
            lines: -1,
            kind: Kind::Bullet,
            alive: true,
        };
        body.rebuild();
        body.radius = body.radius_of();
        let gb = gpu_body(&body);
        self.mirror[slot] = gb;
        if slot < self.bodies.len() {
            self.bodies[slot] = body;
        } else {
            self.bodies.push(body);
        }
        self.gpu.write_bodies(&self.mirror);
        self.shots += 1;
        true
    }

    /// Сброс: снова один цельный объект.
    pub fn reset(&mut self) {
        let fresh = Live::new(
            std::path::Path::new(
                std::env::var("VF_SHADERS").unwrap_or_else(|_| "shaders/spv".into()).as_str(),
            ),
        );
        self.bodies = fresh.bodies;
        self.mirror = fresh.mirror;
        self.gpu = fresh.gpu;
        self.cracks = 0;
        self.shards_made = 0;
        self.shots = 0;
        self.last_log.clear();
    }

    /// Формы для окна: у каждого тела свой меш и каркас тетраэдров.
    pub fn shape_for(&self, i: usize) -> Option<Shape> {
        let b = &self.bodies[i];
        if !b.alive || b.tets.is_empty() {
            return None;
        }
        let tb = TetBody {
            tets: b.tets.clone(),
            level_cap: b.level_cap,
            origin: b.origin,
        };
        let c = tb.world_centroid();
        // без сглаживания: у скола должны быть грани, а не блоб
        let (mut mesh, grid) = tb.surface_grid_smoothed(0, 0.0);
        for p in mesh.positions.iter_mut() {
            p[0] -= c[0];
            p[1] -= c[1];
            p[2] -= c[2];
        }
        // edge_lines() уже отдаёт точки относительно центроида — второй раз
        // вычитать нельзя, иначе каркас уезжает от меша
        let edges = tb.edge_lines();
        Some(Shape {
            level: mesh.level,
            mesh,
            grid_lo: grid.lo,
            grid_hi: grid.hi,
            grid_step: grid.step,
            edges,
        })
    }
}

fn gpu_body(b: &LiveBody) -> GpuBody {
    let mut cpu = Body::new(b.pos, b.vel, b.mass, b.radius);
    cpu.rot = b.quat;
    let mut g = GpuBody::from_cpu(&cpu);
    // Настоящие полуразмеры, а не радиус: солвер берёт по ним плечо для
    // контактов, которые считает сам GPU. Точка контакта из SAT теперь
    // приходит отдельно, но габариты всё равно должны быть честными.
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for t in &b.local {
        for p in t {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    let half = if b.local.is_empty() {
        [b.radius, b.radius, b.radius]
    } else {
        [
            0.5 * (hi[0] - lo[0]),
            0.5 * (hi[1] - lo[1]),
            0.5 * (hi[2] - lo[2]),
        ]
    };
    g.half_shape = [half[0], half[1], half[2], 1.0];
    g
}
