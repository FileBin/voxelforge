//! Плотная куча: сферы, скоростной контактный слой, сон, сетка broadphase.
//!
//! История решения, чтобы не повторять ошибку. Первой версией был
//! **позиционный** солвер (XPBD-стиль: сдвигаем позиции, скорость выводим как
//! `v = Δx/h`). На разреженных контактах он работает (демки D1/D2), но в
//! плотной куче каждая коррекция проникновения превращается в скорость:
//! коррекции не успевают разрешиться за фиксированные итерации, накапливаются
//! — тела разлетаются до 25 м/с, куча расползается в один слой.
//!
//! Здесь наоборот: **сначала скорости, потом позиции**. Контакты решаются
//! импульсами с bias (Баумгарте), позиции интегрируются из уже исправленных
//! скоростей. Тогда исправление проникновения — импульс, а не телепорт, и он
//! не может «стать» скоростью.
//!
//! Вращения нет: качение — следующий инкремент вместе с опрокидыванием блоков.

use std::collections::HashMap;

use crate::Body;

const GROUND: usize = usize::MAX;

/// Контакт сферы с сферой или с полом (скоростная формулировка).
#[derive(Clone, Copy)]
pub struct PileContact {
    pub a: usize,
    pub b: usize,
    /// Нормаль от `a` к `b`.
    pub n: [f64; 3],
    pub pen: f64,
    /// Накопленный нормальный импульс за подшаг.
    pub lambda_n: f64,
    /// Накопленный касательный импульс.
    pub lambda_t: f64,
    /// Направление касательной, в котором гасится скольжение.
    pub t: [f64; 3],
}

pub struct PileSim {
    pub bodies: Vec<Body>,
    pub gravity: f64,
    pub ground_y: f64,
    pub substeps: usize,
    pub iterations: usize,
    /// Контактный зазор: проникновение мельче него не выталкивается.
    pub margin: f64,
    /// Доля проникновения, выталкиваемая за подшаг (0..1).
    pub beta: f64,
    /// Коэффициент трения Кулона.
    pub friction: f64,
    /// Пары с отношением масс выше этого решаются односторонне.
    pub mass_ratio_limit: f64,
    pub sleep_linear: f64,
    pub sleep_time: f64,
    pub cell: f64,
    pub contacts: Vec<PileContact>,
    pub asleep: Vec<bool>,
    sleep_for: Vec<f64>,
    sleep_anchor: Vec<[f64; 3]>,
    pub awake: usize,
    grid: HashMap<(i64, i64, i64), Vec<u32>>,
}

impl PileSim {
    pub fn new(bodies: Vec<Body>) -> Self {
        let n = bodies.len();
        let cell = bodies
            .iter()
            .map(|b| 2.0 * b.radius as f64)
            .fold(0.1f64, f64::max);
        let anchors: Vec<[f64; 3]> = bodies.iter().map(|b| b.pos).collect();
        Self {
            bodies,
            gravity: 9.81,
            ground_y: 0.0,
            substeps: 4,
            iterations: 8,
            margin: 5e-4,
            beta: 0.2,
            friction: 0.7,
            mass_ratio_limit: 1000.0,
            sleep_linear: 0.05,
            sleep_time: 0.5,
            cell,
            contacts: Vec::new(),
            asleep: vec![false; n],
            sleep_for: vec![0.0; n],
            sleep_anchor: anchors,
            awake: n,
            grid: HashMap::new(),
        }
    }

    fn key(&self, p: [f64; 3]) -> (i64, i64, i64) {
        (
            (p[0] / self.cell).floor() as i64,
            (p[1] / self.cell).floor() as i64,
            (p[2] / self.cell).floor() as i64,
        )
    }

    /// Обратная масса: спящее тело и пол — бесконечная масса.
    fn inv_mass_of(&self, i: usize) -> f64 {
        if i == GROUND || self.asleep[i] || self.bodies[i].mass <= 0.0 {
            0.0
        } else {
            1.0 / self.bodies[i].mass
        }
    }

    /// Обратные массы пары + правило `κ > limit`: тяжёлый партнёр становится
    /// бесконечномассовым для лёгкого, но его скорость читается.
    fn pair_inv_mass(&self, a: usize, b: usize) -> (f64, f64) {
        let mut wa = self.inv_mass_of(a);
        let mut wb = self.inv_mass_of(b);
        if a != GROUND && wa > 0.0 && wb > 0.0 {
            let (ma, mb) = (self.bodies[a].mass, self.bodies[b].mass);
            let ratio = (ma / mb).max(mb / ma);
            if ratio > self.mass_ratio_limit {
                if ma > mb {
                    wa = 0.0;
                } else {
                    wb = 0.0;
                }
            }
        }
        (wa, wb)
    }

    fn build_grid(&mut self) {
        self.grid.clear();
        for i in 0..self.bodies.len() {
            let k = self.key(self.bodies[i].pos);
            self.grid.entry(k).or_default().push(i as u32);
        }
    }

    fn collect(&mut self) {
        self.contacts.clear();

        for i in 0..self.bodies.len() {
            let b = &self.bodies[i];
            let pen = self.ground_y - (b.pos[1] - b.radius as f64);
            if pen > -self.margin {
                self.contacts.push(PileContact {
                    a: GROUND,
                    b: i,
                    n: [0.0, 1.0, 0.0],
                    pen,
                    lambda_n: 0.0,
                    lambda_t: 0.0,
                    t: [0.0, 0.0, 0.0],
                });
            }
        }

        // Детерминированный обход сетки: ключи сортируем, иначе порядок пар
        // зависит от RandomState и прогоны разойдутся.
        let mut keys: Vec<(i64, i64, i64)> = self.grid.keys().copied().collect();
        keys.sort_unstable();
        for k in keys {
            let here = self.grid[&k].clone();
            for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    for dz in -1..=1i64 {
                        let nk = (k.0 + dx, k.1 + dy, k.2 + dz);
                        if nk < k {
                            continue; // пару считаем один раз
                        }
                        let Some(other) = self.grid.get(&nk).map(|v| v.clone()) else {
                            continue;
                        };
                        for &ia in &here {
                            for &ib in &other {
                                let (i, j) = (ia as usize, ib as usize);
                                if i >= j {
                                    continue;
                                }
                                if self.asleep[i] && self.asleep[j] {
                                    continue;
                                }
                                let (pa, pb) = (self.bodies[i].pos, self.bodies[j].pos);
                                let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                                let dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                                let radii =
                                    self.bodies[i].radius as f64 + self.bodies[j].radius as f64;
                                if dist2 > radii * radii || dist2 == 0.0 {
                                    continue;
                                }
                                let dist = dist2.sqrt();
                                let n = [d[0] / dist, d[1] / dist, d[2] / dist];
                                self.contacts.push(PileContact {
                                    a: i,
                                    b: j,
                                    n,
                                    pen: radii - dist,
                                    lambda_n: 0.0,
                                    lambda_t: 0.0,
                                    t: [0.0, 0.0, 0.0],
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// Пробуждение: спящего будит бодрствующий сосед, который реально едет.
    fn wake_pass(&mut self) {
        let mut to_wake: Vec<usize> = Vec::new();
        for c in &self.contacts {
            if c.a == GROUND {
                continue;
            }
            let (a, b) = (c.a, c.b);
            let sp = |s: &Self, i: usize| {
                let v = s.bodies[i].vel;
                (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
            };
            if self.asleep[a] && !self.asleep[b] && sp(self, b) > 4.0 * self.sleep_linear {
                to_wake.push(a);
            } else if self.asleep[b] && !self.asleep[a] && sp(self, a) > 4.0 * self.sleep_linear {
                to_wake.push(b);
            }
        }
        for i in to_wake {
            self.asleep[i] = false;
            self.sleep_for[i] = 0.0;
            self.sleep_anchor[i] = self.bodies[i].pos;
        }
    }

    /// Скоростное решение одного контакта: нормаль с bias, затем трение.
    fn solve(&mut self, ci: usize, h: f64) {
        let mut c = self.contacts[ci];
        let (wa, wb) = self.pair_inv_mass(c.a, c.b);
        let w = wa + wb;
        if w <= 0.0 {
            return;
        }

        let va = if c.a == GROUND {
            [0.0; 3]
        } else {
            self.bodies[c.a].vel
        };
        let vb = self.bodies[c.b].vel;
        let rel = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];

        // --- нормаль -------------------------------------------------------
        let vn = rel[0] * c.n[0] + rel[1] * c.n[1] + rel[2] * c.n[2];
        let push = ((c.pen - self.margin).max(0.0) * self.beta) / h;
        let d_lambda = (push - vn) / w;
        let new_n = (c.lambda_n + d_lambda).max(0.0);
        let d = new_n - c.lambda_n;
        c.lambda_n = new_n;
        if d != 0.0 {
            if wa > 0.0 {
                for k in 0..3 {
                    self.bodies[c.a].vel[k] -= c.n[k] * wa * d;
                }
            }
            for k in 0..3 {
                self.bodies[c.b].vel[k] += c.n[k] * wb * d;
            }
        }

        // --- трение --------------------------------------------------------
        if self.friction > 0.0 && c.a != GROUND && c.lambda_n > 0.0 {
            let va = self.bodies[c.a].vel;
            let vb = self.bodies[c.b].vel;
            let mut rel = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
            let dot = rel[0] * c.n[0] + rel[1] * c.n[1] + rel[2] * c.n[2];
            for k in 0..3 {
                rel[k] -= c.n[k] * dot;
            }
            let len = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            if len > 1e-9 {
                let t = [rel[0] / len, rel[1] / len, rel[2] / len];
                let d_lambda_t = -len / w;
                let cap = self.friction * c.lambda_n;
                let new_t = (c.lambda_t + d_lambda_t).clamp(-cap, cap);
                let dt = new_t - c.lambda_t;
                c.lambda_t = new_t;
                c.t = t;
                if dt != 0.0 {
                    if wa > 0.0 {
                        for k in 0..3 {
                            self.bodies[c.a].vel[k] -= t[k] * wa * dt;
                        }
                    }
                    for k in 0..3 {
                        self.bodies[c.b].vel[k] += t[k] * wb * dt;
                    }
                }
            }
        }

        self.contacts[ci] = c;
    }

    fn substep(&mut self, h: f64) {
        // 1. скорости
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if self.asleep[i] {
                continue;
            }
            b.vel[1] -= self.gravity * h;
        }

        // 2. контакты и скоростное решение
        self.build_grid();
        self.collect();
        self.wake_pass();
        for _ in 0..self.iterations {
            for ci in 0..self.contacts.len() {
                self.solve(ci, h);
            }
        }

        // 3. позиции — из уже исправленных скоростей
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if self.asleep[i] {
                continue;
            }
            for k in 0..3 {
                b.pos[k] += b.vel[k] * h;
            }
        }

        // 4. сон по накопленному смещению: тело может колебаться вокруг нуля,
        //    поэтому смещение надёжнее мгновенной скорости
        let mut awake = 0usize;
        for i in 0..self.bodies.len() {
            if self.asleep[i] {
                continue;
            }
            let p = self.bodies[i].pos;
            let a = self.sleep_anchor[i];
            let d = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
            let moved = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if moved > self.sleep_linear * self.sleep_time {
                self.sleep_anchor[i] = p;
                self.sleep_for[i] = 0.0;
            } else {
                self.sleep_for[i] += h;
                if self.sleep_for[i] >= self.sleep_time {
                    self.asleep[i] = true;
                    self.bodies[i].vel = [0.0; 3];
                    self.sleep_anchor[i] = p;
                    continue;
                }
            }
            awake += 1;
        }
        self.awake = awake;
    }

    pub fn step(&mut self, dt: f64) {
        let h = dt / self.substeps as f64;
        for _ in 0..self.substeps {
            self.substep(h);
        }
    }

    pub fn max_penetration(&mut self) -> f64 {
        self.build_grid();
        self.collect();
        let mut worst: f64 = 0.0;
        for c in &self.contacts {
            if c.pen > worst {
                worst = c.pen;
            }
        }
        worst
    }

    pub fn max_speed(&self) -> f64 {
        self.bodies
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.asleep[*i])
            .map(|(_, b)| (b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2]).sqrt())
            .fold(0.0f64, f64::max)
    }

    pub fn top(&self) -> f64 {
        self.bodies
            .iter()
            .map(|b| b.pos[1] + b.radius as f64)
            .fold(f64::NEG_INFINITY, f64::max)
    }
}
